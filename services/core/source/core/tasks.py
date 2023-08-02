import datetime
import json
import logging
import os
import time

import numpy as np
import typing
import threading

import pika
from celery import Celery, chord
from celery.signals import after_setup_logger, setup_logging
from celery.signals import after_setup_task_logger
from celery.app.log import TaskFormatter
from celery.utils.log import get_task_logger

from core import logger, logformat
import utils
from db import db_helper as db

from forecasting.load_forecast_service import ReferenceBasedLoadForecaster, ThermalLoadForecaster
from forecasting.pv_forecast_service import ShortTermForecaster, QuotaBlocksForecaster
from forecasting.ev_forecast_service import ReferenceBasedEVForecaster, UserInputBasedEVForecaster, \
    UserAndMeanBasedConnectedEVForecaster, ConnectedEVForecaster, DisconnectedEVForecaster
from optimization.scheduler import Scheduler, GeneralScheduler, GridPowerSetpointScheduler, MockupRandomScheduler
from forecasting import pv_forecast_service
from optimization import test_optimization


# Rest of config is defined in environment in docker-compose file
celery_app = Celery()
celery_app.config_from_object('celeryconfig')

logger = get_task_logger(logger.name)


@celery_app.on_after_finalize.connect
def hello(sender, **kwargs):
    logger.info(f'Hello from core!')


'''
Overall setup
'''


@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    """
    Configure format for all log messages by setting the format of all handlers of the celery
    task logger (which has been initialized above).
    Example log message:
    [2022-02-22 17:15:28,469 - bem-core - task:core.forecast_pv - forecast_pv - INFO]: Forecast of pv active
    power from 2022-02-22T17:21:00+00:00 to 2022-02-22T18:20:00+00:00 with resolution 0:01:00 created.
    """
    for handler in logger.handlers:
        handler.setFormatter(TaskFormatter(logformat))
    logger.info(f'Logger is configured.')


@celery_app.task(name='core.setup')
def setup(ignore_schedule_deviations: bool):
    # Save the current weather forecast configuration with a timestamp in the settings database for tracking purposes
    pv_forecast_service.save_weather_forecast_config()

    # Start listening for primary and final quotas published by the grid management service
    thread_1 = threading.Thread(target=wait_for_quotas)
    thread_1.start()

    # Start listening for the event that the user has provided EV charging info, published by the gridmanagement service
    thread_2 = threading.Thread(target=wait_for_ev_charging_input)
    thread_2.start()

    # Start listening for the event that the EV has been disconnected, published by the device management service
    thread_3 = threading.Thread(target=wait_for_ev_disconnection)
    thread_3.start()

    # Start listening for grid connection setpoints, published by the modbus server connected to the grid control unit
    thread_4 = threading.Thread(target=wait_for_gcp_setpoint)
    thread_4.start()

    if not ignore_schedule_deviations:
        # Start listening for the event that some device or the entire household (measured at the grid connection point)
        # produces or consumes more than scheduled (incl. tolerance)
        thread_5 = threading.Thread(target=wait_for_schedule_deviations)
        thread_5.start()

    return "Success"


'''
Forecasting
'''


@celery_app.task(name='core.forecast_ev', expires=120)
def forecast_ev(window_start: datetime.datetime, window_size: datetime.timedelta,
                user_input: typing.Dict[str, str] = None, triggering_event: str = None) -> typing.Dict[
    str, typing.Dict[str, float]]:
    evse_installed = utils.are_devices_of_subcategory_installed(os.getenv('EVSE_KEY'))
    if not evse_installed:
        return {}

    if triggering_event == "ev_disconnection":
        logger.info("EV has been disconnected. Get connection forecast based on mean disconnection duration and "
                    "reference-based forecast.")
        ev_forecaster = DisconnectedEVForecaster(window_start, window_size)

    else:
        connected = utils.get_latest_device_measurement(os.getenv('EVSE_KEY'), fields='connected')
        logger.debug(f"EV connected: {connected}")

        if not connected and user_input is None:
            ev_forecaster = ReferenceBasedEVForecaster(window_start, window_size)
        elif user_input:
            wait_for_minutes = float(os.getenv('EXPECT_EV_CONNECTED_X_MIN_AFTER_USER_INPUT'))
            check_interval_seconds = 30
            waited_for = 0
            while not connected and (waited_for < wait_for_minutes):
                # User has provided charging input before connecting the vehicle to the charging station
                # Wait for 30s, then check again. Continue loop for some minutes, then give up.
                if waited_for == 0:
                    logger.info(f'EV is not yet connected. Check again every {check_interval_seconds}s.')
                elif waited_for % 2 == 0:
                    logger.info(
                        f'EV still not connected after {waited_for} minutes. Continue checking every {check_interval_seconds}s.')

                time.sleep(check_interval_seconds)
                waited_for += check_interval_seconds / 60
                connected = utils.get_latest_device_measurement(os.getenv('EVSE_KEY'), fields='connected')

            if connected:
                logger.info('Get connection forecast based on user planned departure')
                ev_forecaster = UserInputBasedEVForecaster(window_start, window_size, user_input['scheduled_departure'])
            else:
                logger.info(f'EV user input was provided, but EV is still not connected after {wait_for_minutes} '
                            f'minutes. Get connection forecast based on mean disconnection duration and '
                            'reference-based forecast.')
                ev_forecaster = DisconnectedEVForecaster(window_start, window_size)
        else:
            # (Still) Connected, but no new user input
            logger.info('Get connection forecast based on stored user planned departure and mean state durations.')
            ev_forecaster = ConnectedEVForecaster(window_start, window_size)

    ev_forecast = ev_forecaster.make_forecast()
    logger.info(
        f'Forecast of EV availability from {min(ev_forecast[os.getenv("EVSE_KEY")].keys())} to {max(ev_forecast[os.getenv("EVSE_KEY")].keys())} with resolution {ev_forecaster.resolution} created.')
    return ev_forecast


@celery_app.task(name='core.forecast_load_el', expires=120)
def forecast_load_el(window_start: datetime.datetime, window_size: datetime.timedelta, update=False) -> typing.Dict[
    str, typing.Dict[str, float]]:
    load_forecaster = ReferenceBasedLoadForecaster(window_start, window_size)
    load_forecast = load_forecaster.make_forecast()
    logger.info(
        f'Forecast of el. load from {min(load_forecast[os.getenv("LOAD_EL_KEY")].keys())} to {max(load_forecast[os.getenv("LOAD_EL_KEY")].keys())} with resolution {load_forecaster.resolution} created.')
    return load_forecast


@celery_app.task(name='core.forecast_load_th', expires=120)
def forecast_load_th(window_start: datetime.datetime, window_size: datetime.timedelta, update=False) -> typing.Dict[
    str, typing.Dict[str, float]]:
    load_forecaster = ThermalLoadForecaster(window_start, window_size)
    load_forecast = load_forecaster.make_forecast()
    logger.info(
        f'Forecast of th. load from {window_start} to {load_forecaster.forecast_window_end} with resolution {load_forecaster.resolution} created.')
    return load_forecast


@celery_app.task(name='core.forecast_pv', expires=120)
def forecast_pv(window_start: datetime.datetime, window_size: datetime.timedelta, lead_time: datetime.timedelta = None,
                update=False) -> typing.Dict[str, typing.Dict[str, float]]:
    pv_installed = utils.are_devices_of_subcategory_installed(os.getenv('PV_KEY'))
    if not pv_installed:
        return {}

    assert window_start or lead_time, 'Neither forecast window start nor lead time provided.'
    if not window_start:
        window_start = (datetime.datetime.now(tz=datetime.timezone.utc) + lead_time).replace(second=0, microsecond=0)

    pv_specifications: typing.List[dict] = utils.get_devices_of_subcategory(os.getenv('PV_KEY'))
    pv_forecasts = {}
    for pv_spec in pv_specifications:
        if window_size == datetime.timedelta(hours=1):
            pv_forecaster = ShortTermForecaster(pv_specification=pv_spec, window_start=window_start,
                                                window_size=window_size)
        else:
            pv_forecaster = QuotaBlocksForecaster(pv_specification=pv_spec, window_start=window_start,
                                                  window_size=window_size)

        pv_forecast: typing.Dict[str, typing.Dict[str, float]] = pv_forecaster.make_forecast()
        logger.info(
            f'Forecast of {pv_spec["key"]} active power from {min(pv_forecast[pv_spec["key"]].keys())} to {max(pv_forecast[pv_spec["key"]].keys())} with resolution {pv_forecaster.resolution} created.')
        # Will contain all PV forecasts by device key (e.g. {'pv1': {...}, 'pv2': {...}}
        pv_forecasts.update(pv_forecast)
    return pv_forecasts


'''
Scheduling of device operation
'''


@celery_app.task(name='core.create_schedules', expires=120)
def create_schedules(forecasts: typing.Union[typing.List[dict], None],
                     window_start: datetime.datetime,
                     window_size: datetime.timedelta,
                     grid_quotas_category: str = None,
                     grid_power_setpoint: int = None,
                     grid_quotas: typing.Dict[str, int] = None,
                     init_states: typing.Dict[str, typing.Dict[str, float]] = None,
                     triggering_event: str = None,
                     triggering_source: str = None,
                     **kwargs) -> typing.Dict[
    datetime.datetime, int]:
    """
    Start the schedule optimization and publish the resulting power schedules.
    :param forecasts: Forecasts of sources/devices.
    :param window_start: Start of the scheduling horizon
    :param window_size: Scheduling horizon
    :param grid_quotas_category: (Optional) Category of grid quotas, e.g. "final"
    :param grid_quotas: (Optional) Time series of grid quotas as dict with keys=ISO-formatted timestamps
    :param grid_power_setpoint: (Optional) Grid setpoint [W]
    :param init_states: (Optional) Initial states of sources/devices.
    :param triggering_event: (Optional) The event that triggered this scheduling process.
    :param triggering_source: (Optional) The source (e.g. some device) that triggered this scheduling process.
    :param kwargs: Other options.
    """

    if forecasts:
        # Convert list of forecasts to dict to make forecasts directly accessible by device key
        forecasts = {list(f.keys())[0]: list(f.values())[0] for f in forecasts if len(f.values()) > 0}

    if grid_power_setpoint is not None:
        # Case: grid connection setpoint received
        logger.debug(f'Start: {window_start}, size {window_size}, GCP setpoint={grid_power_setpoint}')
        scheduler = GridPowerSetpointScheduler(
            window_start=window_start, window_size=window_size, grid_power_setpoint=grid_power_setpoint
        )
        schedules_are_setpoints = True

    else:
        # Case: regular scheduling, primary quotas, final quotas, EV arrival,... (any other case than a GCP setpoint)
        logger.debug(f'Start: {window_start}, size {window_size}, restriction category "{grid_quotas_category}"')
        scheduler = GeneralScheduler(
            window_start=window_start,
            window_size=window_size,
            forecasts=forecasts,
            grid_quotas_category=grid_quotas_category,
            grid_quotas=grid_quotas,
            init_states=init_states,
            triggering_event=triggering_event,
            triggering_source=triggering_source,
            **kwargs
        )
        schedules_are_setpoints = False
    #TESTING VON JONAS
    consum_limit_active=f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_consum_limit_active"
    feedin_limit_active=f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_feedin_limit_active"
    if grid_power_setpoint is None: 
    #if grid_power_setpoint is None and (consum_limit_active.count("1") >= 1 or feedin_limit_active.count("1") >= 1):
        logger.debug("bin bei resovle")
        schedules: typing.Dict[str, typing.Dict[
        str, int]] = scheduler.resolve ()

    else:
        logger.debug("bin bei schedule")
        schedules: typing.Dict[str, typing.Dict[
        str, int]] = scheduler.schedule ()

    if schedules:
        publish_schedules.delay(schedules=schedules, schedules_are_setpoints=schedules_are_setpoints)


@celery_app.task(name='core.scheduling')
def scheduling(window_start: datetime.datetime,
               window_size: datetime.timedelta,
               lead_time: datetime.timedelta = None,
               update_forecasts: typing.Union[list, str] = 'all',
               grid_quotas_category: str = None,
               grid_quotas: typing.Dict[str, int] = None,
               grid_power_setpoint: int = None,
               user_input: dict = None,
               init_states: typing.Dict[str, typing.Dict[str, float]] = None,
               triggering_event: str = None,
               triggering_source: str = None,
               **kwargs):
    """
    A concatenation of the creation of the forecasts that are required for the device schedule optimization and
    the schedule optimization itself.
    :param window_start: Start of the scheduling horizon
    :param window_size: Scheduling horizon
    :param lead_time: (Optional) Time delta between now and the start of the scheduling horizon
    :param update_forecasts: (Optional) List of sources/devices whose forecast has to be updated or created.
    Default is "all", i.e. all forecasts are updated/created.
    :param grid_quotas_category: (Optional) Category of grid quotas, e.g. "final"
    :param grid_quotas: (Optional) Time series of grid quotas as dict with keys=ISO-formatted timestamps
    :param grid_power_setpoint: (Optional) Grid setpoint [W]
    :param user_input: (Optional) Input provided by users
    :param init_states: (Optional) Initial states of sources/devices.
    :param triggering_event: (Optional) The event that triggered this scheduling process.
    :param triggering_source: (Optional) The source (e.g. some device) that triggered this scheduling process.
    :param kwargs: Other options.
    """
    logger.info(f'Scheduling {"" if triggering_event is None else "due to " + str(triggering_event)}!')

    if window_start is None and window_size == datetime.timedelta(hours=24):
        # When executed as periodic task, the window start cannot be provided as argument, hence set here to
        # next full hour + lead time
        lead_time = datetime.timedelta(hours=0) if lead_time is None else lead_time
        window_start = datetime.datetime.now(tz=datetime.timezone.utc).replace(minute=0, second=0, microsecond=0) \
                       + datetime.timedelta(hours=1) + lead_time

    # Mapping of sources/devices to their respective forecast function (celery task signature)
    forecast_func_sig_map = {
        os.getenv('PV_KEY'): forecast_pv.s(window_start, window_size),
        os.getenv('LOAD_EL_KEY'): forecast_load_el.s(window_start, window_size),
        os.getenv('EVSE_KEY'): forecast_ev.s(window_start, window_size, user_input, triggering_event)
    }

    if update_forecasts == 'all':
        forecast_func_sigs = list(forecast_func_sig_map.values())

    else:
        # Filter for the sources whose forecast shall be updated.
        forecast_func_sigs = [forecast_func_sig_map[source] for source in update_forecasts]

    if len(forecast_func_sigs) > 0:
        # Calls pv forecast, load and EV forecast tasks in parallel, then calls schedule optimization
        # with their forecasts
        chord(
            forecast_func_sigs
        )(create_schedules.s(window_start, window_size, grid_quotas_category=grid_quotas_category,
                             grid_power_setpoint=grid_power_setpoint, grid_quotas=grid_quotas,
                             init_states=init_states, triggering_event=triggering_event,
                             triggering_source=triggering_source, **kwargs))
    else:
        # No forecasts to create. Call schedule optimization directly.
        create_schedules.delay(forecasts=None, window_start=window_start, window_size=window_size,
                               grid_quotas_category=grid_quotas_category,
                               grid_power_setpoint=grid_power_setpoint, grid_quotas=grid_quotas,
                               init_states=init_states, triggering_event=triggering_event,
                               triggering_source=triggering_source, **kwargs)


@celery_app.task(name='core.repeat_gcp_setpoint_scheduling')
def repeat_gcp_setpoint_scheduling(power_setpoint: int, countdown: int):
    """
    Repeat scheduling w.r.t. the same GCP power setpoint every <countdown> seconds until this task gets cancelled
    when a new GCP setpoint is received.
    :param power_setpoint: Currently active power setpoint
    :param countdown: Delay between two scheduling runs (a little shorter than the scheduling horizon)
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)

    scheduling.delay(
        window_start=now + datetime.timedelta(seconds=5),
        window_size=datetime.timedelta(seconds=int(os.getenv('GCP_SETPOINT_SCHEDULE_WINDOW_SIZE_SEC'))),
        update_forecasts=[],
        grid_power_setpoint=power_setpoint
    )
    repeat_gcp_setpoint_scheduling.apply_async(kwargs=dict(power_setpoint=power_setpoint, countdown=countdown),
                                               countdown=countdown)


@celery_app.task(name='core.mockup_scheduling')
def mockup_scheduling(window_start: str = None, lead_time: int = 0):
    """
    :param window_start: Start of schedule time window (first period) in ISO-format
    :param lead_time: Additional hours between calculation of schedule and window_start
    """

    if not window_start:
        # Next full hour + lead time
        window_start = datetime.datetime.now(tz=datetime.timezone.utc).replace(minute=0, second=0, microsecond=0) \
                       + datetime.timedelta(hours=1 + lead_time)  # utc
    else:
        window_start = datetime.datetime.fromisoformat(window_start)  # utc

    print(f'Start random scheduling as of {window_start}')
    scheduler = MockupRandomScheduler(window_start)
    scheduler.random_gcp_schedule()


'''
Message publishing and subscribing
'''


# @celery_app.task(name='core.await_quotas')
def wait_for_quotas():
    """
    Sets up a queue bound to the message exchange "inbox" to subscribe to the topics "quotas.*".
    When quota information is received, rescheduling with these limits is triggered.
    """

    def trigger_scheduling(ch, method, properties, body):
        """
        Executed when receiving message for topic 'quotas.primary'.
        Deserializes payload to dict and extracts the absolute power limits and the window start.
        """
        quota_data = json.loads(body)

        category = method.routing_key.split('.')[-1]
        logger.debug(f'{category} quota data received: {quota_data}')

        if category == 'primary':
            window_start = quota_data.pop('start')
            update_forecasts = []
        elif category == 'final':
            window_start = quota_data.pop('start')
            update_forecasts = 'all'
        else:
            # Ignore messages from other topics than quotas.primary and quotas.final
            return
        abs_power_limits = {t: values['abs_power_limit'] for t, values in quota_data.items()}
        triggering_event = f"{category}_quotas"

        scheduling.delay(
            window_start=datetime.datetime.fromisoformat(window_start),
            window_size=datetime.timedelta(hours=24),
            update_forecasts=update_forecasts,
            grid_quotas_category=category,
            grid_quotas=abs_power_limits,
            triggering_event=triggering_event,
            triggering_source=os.getenv('GRID_CONNECTION_POINT_KEY')
        )

    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()
    try:
        # Declare queue, bind it to the exchange 'inbox' and start consuming messages of the topic 'quotas.*'
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        channel.queue_bind(queue=queue_name, exchange=os.getenv('RABBITMQ_BEM_INBOX'), routing_key='quotas.*')
        channel.basic_consume(queue=queue_name, on_message_callback=trigger_scheduling, auto_ack=True)
        logger.info('Queue declared for topic "quotas.*" -> start consuming...')
        channel.start_consuming()
    finally:
        connection.close()


@celery_app.task(name='core.await_gcp_setpoint')
def wait_for_gcp_setpoint():
    """
    Sets up a queue to receive broadcast messages with grid connection setpoints.
    When a setpoint is received, the callback function to trigger the scheduling processes is called.
    :return:
    """

    def reset_repetition_task(power_setpoint: int, countdown: int):
        """
        Initiate the scheduling task again with the same power setpoint (except for clearance).
        It will only be executed if no new setpoint is received before the countdown expires. Otherwise (new setpoint
        before countdown), the existing task is cancelled and the task is re-initiated with a fresh countdown.
        In the case of the clear signal (setpoint=None), the task is cancelled without being renewed.
        :param power_setpoint: The last received GCP power setpoint
        :param countdown: Seconds until the scheduling task is executed again with this power setpoint
        """
        worker_host = "celery@core_worker"
        # Get all scheduled tasks of the core worker
        scheduled_tasks: typing.List[dict] = celery_app.control.inspect().scheduled()[worker_host]
        for task in scheduled_tasks:
            task_name = task["request"]["name"]
            if task_name == 'core.repeat_gcp_setpoint_scheduling':
                # Cancel existing task that repeats the setpoint scheduling with the prior setpoint
                task_id = task["request"]["id"]
                logger.debug(f'Revoke task {task_name} ({task_id})')
                celery_app.control.revoke(task_id=task_id, destination=[worker_host])

        if power_setpoint is not None:
            # Initiate the task with a fresh countdown if setpoint has not been cleared
            repeat_gcp_setpoint_scheduling.apply_async((power_setpoint, countdown,), countdown=countdown)

    def trigger_scheduling(ch, method, properties, body):
        """
        Executed when receiving message with grid connection setpoints.
        Deserializes payload to dict and provides it to the trading task.
        """
        payload = json.loads(body)
        logger.debug(f'GCP setpoint received: {payload}')
        power_setpoint = payload['value']
        timestamp: str = payload['timestamp']  # ISO format
        now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)

        if power_setpoint is None:
            # Setpoint is cleared!
            # Reschedule (cost min.) with current quotas and reference schedules
            scheduling.delay(
                window_start=utils.get_next_schedule_window_start(now, datetime.timedelta(
                    seconds=int(os.getenv('SCHEDULER_TEMP_RESOLUTION_SEC')))),
                window_size=datetime.timedelta(hours=24),
                update_forecasts='all',
                grid_quotas_category='final',
                triggering_event=f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_setpoint_cleared",
                triggering_source=os.getenv('GRID_CONNECTION_POINT_KEY')
            )
        else:
            # New setpoint!
            # Find schedules for next 1 minute with min. deviation to GCP setpoint
            assert now - datetime.timedelta(minutes=1) < datetime.datetime.fromisoformat(
                timestamp), f'GCP setpoint is too old (timestamp: {timestamp}) and will be ignored.'
            scheduling.delay(
                window_start=now + datetime.timedelta(seconds=5),
                window_size=datetime.timedelta(seconds=int(os.getenv('GCP_SETPOINT_SCHEDULE_WINDOW_SIZE_SEC'))),
                update_forecasts=[],
                triggering_event=f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_setpoint",
                grid_power_setpoint=power_setpoint
            )
        reset_repetition_task(power_setpoint, countdown=int(os.getenv('GCP_SETPOINT_SCHEDULE_WINDOW_SIZE_SEC')) - 5)

    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()
    try:
        # Declare queue to receive broadcast messages with grid connection setpoints
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        channel.queue_bind(queue=queue_name, exchange=os.getenv('RABBITMQ_GRID_SETPOINT'))
        channel.basic_consume(queue=queue_name, on_message_callback=trigger_scheduling, auto_ack=True)
        logger.info(
            f"Queue declared for broadcast message on {os.getenv('RABBITMQ_GRID_SETPOINT')} exchange -> start consuming...")
        channel.start_consuming()
    finally:
        connection.close()


# @celery_app.task(name='core.await_ev_charging_input')
def wait_for_ev_charging_input():
    def trigger_scheduling(ch, method, properties, body):
        """
        Executed when receiving message for topic ''.
        Deserializes payload to dict and ....
        """
        input = json.loads(body)
        assert (
                       'scheduled_departure' and 'soc') in input.keys(), f'EV charging input from user with keys "scheduled_departure" and "soc" expected. Got {list(input.keys())}'
        soc_init = input.pop('soc')
        fake_user_input: bool = input.get("fake", False)
        if fake_user_input:
            input.pop("fake")
            triggering_event = "ev_max_charging_delay"
        else:
            triggering_event = "ev_user_input"

        now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
        window_start = utils.get_next_schedule_window_start(now, datetime.timedelta(
            seconds=int(os.getenv('SCHEDULER_TEMP_RESOLUTION_SEC'))))
        logger.debug(f'Window start: {window_start}, departure={input["scheduled_departure"]}, init_soc={soc_init}')
        scheduling.delay(
            window_start=window_start,
            window_size=datetime.timedelta(hours=24),
            update_forecasts='all',
            grid_quotas_category='final',
            user_input=input,
            init_states={os.getenv("EVSE_KEY"): {'soc': soc_init if soc_init <= 1 else soc_init / 100}},
            triggering_event=triggering_event,
            triggering_source=os.getenv('EVSE_KEY')
        )

    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()
    try:
        # Declare queue, bind it to the exchange 'inbox' and start consuming messages of the topic ''
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        channel.queue_bind(queue=queue_name, exchange=os.getenv('RABBITMQ_BEM_INBOX'), routing_key='user.input.ev')
        channel.basic_consume(queue=queue_name, on_message_callback=trigger_scheduling, auto_ack=True)
        logger.info('Queue declared for topic "user.input.ev" -> start consuming...')
        channel.start_consuming()
    finally:
        connection.close()


def wait_for_ev_disconnection():
    def trigger_scheduling(ch, method, properties, body):
        """
        Executed when receiving message for topic 'evse.connection'.
        Deserializes payload to dict and triggers scheduling if EV is now disconnected.
        """
        status = json.loads(body)
        if status['connected'] == 0:
            now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
            window_start = utils.get_next_schedule_window_start(now, datetime.timedelta(
                seconds=int(os.getenv('SCHEDULER_TEMP_RESOLUTION_SEC'))))
            logger.debug(f'Window start: {window_start}')
            scheduling.delay(
                window_start=window_start,
                window_size=datetime.timedelta(hours=24),
                update_forecasts='all',
                grid_quotas_category='final',
                triggering_event="ev_disconnection",
                triggering_source=os.getenv('EVSE_KEY')
            )

    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()
    try:
        # Declare queue, bind it to the exchange 'control' and start consuming messages of the topic ''
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        channel.queue_bind(queue=queue_name, exchange=os.getenv('RABBITMQ_BEM_CONTROL'),
                           routing_key=f'{os.getenv("EVSE_KEY")}.connection')
        channel.basic_consume(queue=queue_name, on_message_callback=trigger_scheduling, auto_ack=True)
        logger.info(f'Queue declared for topic "{os.getenv("EVSE_KEY")}.connection" -> start consuming...')
        channel.start_consuming()
    finally:
        connection.close()


def wait_for_schedule_deviations():
    """
    Sets up a queue bound to the message exchange "control" to subscribe to the topics "*.deviation".
    Rescheduling is triggered to create a new feasible schedule (for all devices).
    """

    def evaluate_deviation_message(device: str, message: dict):
        """
        Message looks like this (if no setpoint is currently active):

            payload = {
            "feature": "active_power",
            'sensor_value': measured_value,
            'target_value': schedule_value,
            'target_since': None,
            'timestamp': timestamp.isoformat(timespec='seconds'),
            'setpoint_active': False
        }
        """
        reschedule = True
        triggering_event = os.getenv('DEVIATION_MESSAGE_SUBTOPIC')

        if device == os.getenv('EVSE_KEY'):
            # Check if deviation was signalled due to the EV being disconnected while charging, because
            # this information must be handed to the EV forecast task
            if message["feature"] == "active_power" and float(message["target_value"]) > 0 and float(
                    message["sensor_value"]) == 0:
                connected = utils.get_latest_device_measurement(os.getenv('EVSE_KEY'), fields='connected')
                if connected == 0 or connected is False:
                    logger.info("Set triggering event to ev_disconnection")
                    triggering_event = "ev_disconnection"

            if triggering_event != "ev_disconnection":
                logger.debug(f"Deviation from {device} schedule not due to EV disconnection.")

        try:
            if message["setpoint_active"] in [True, "True"]:
                # TODO: how to handle this case: just ignore and await new setpoint (like now) or re-schedule?
                triggering_event = "setpoint_deviation"
                reschedule = False
        except KeyError:
            # Some deviation message contain this key, others don't (currently, only GCP deviation messages contain it)
            pass

        return triggering_event, reschedule

    def trigger_scheduling(triggering_event: str, device: str):
        now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0, second=0)
        window_start = utils.get_next_schedule_window_start(now, datetime.timedelta(
            seconds=int(os.getenv('SCHEDULER_TEMP_RESOLUTION_SEC'))))
        logger.debug(f'Window start: {window_start}')
        # Set the end of the new schedule to the end of the last one. This allows to re-use existing forecasts.
        # Whole 24-hour schedules are created by other time- or event-triggered optimization runs.
        try:
            window_end = utils.latest_schedule_meta_data(source=device,
                                                         query_start_time=now,
                                                         query_end_time=now + datetime.timedelta(hours=24))['end']

            # Get list of sources/devices for which the forecast shall be updated prior to schedule optimization
            update_forecast_of: typing.List[str] = utils.forecast_update_necessary(
                window_start=window_start,
                window_end=window_end,
                sources=Scheduler.forecast_needed_subcat.copy(),
                scheduling_trigger=dict(event=triggering_event, source=device),
                update_every_seconds=3600
            )
            logger.debug(f"Forecast update needed for: {update_forecast_of}")
        except KeyError:
            # No schedule found, therefore an empty dict was returned -> schedule over 24 hours and update all forecasts
            window_end = window_start + datetime.timedelta(hours=24)
            update_forecast_of = 'all'

        scheduling.delay(
            window_start=window_start,
            window_size=window_end - window_start,
            update_forecasts=update_forecast_of,
            grid_quotas_category='final',
            triggering_event=triggering_event,
            triggering_source=device
        )

    def process_deviation_message(ch, method, properties, body):
        """
        Executed when receiving message for topic '*.deviation' for some device * ('gcp' can also be a device)
        """
        device = method.routing_key.split('.')[0]
        message = json.loads(body)
        logger.info(f'Deviation message received from {device}: {message}')

        # Store the deviation message including source in the database
        insert_result = db.save_dict_to_db(db=os.getenv('MONGO_EVENTS_DB_NAME'),
                                           data_category=os.getenv('MONGO_SCHEDULE_DEVIATIONS_COLL_NAME'),
                                           data={'source': device, **message})
        logger.debug(
            f"Deviation message {'successfully' if insert_result.acknowledged else 'NOT'} stored in the "
            f"database ({os.getenv('MONGO_EVENTS_DB_NAME')}.{os.getenv('MONGO_SCHEDULE_DEVIATIONS_COLL_NAME')}, "
            f"doc ID: {insert_result.inserted_id})")

        triggering_event, reschedule = evaluate_deviation_message(device, message)
        if reschedule:
            trigger_scheduling(triggering_event, device)

    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()
    try:
        # Declare queue, bind it to the exchange 'control' and start consuming messages of the topic '*.deviation'
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        channel.queue_bind(queue=queue_name, exchange=os.getenv('RABBITMQ_BEM_CONTROL'),
                           routing_key=f'*.{os.getenv("DEVIATION_MESSAGE_SUBTOPIC")}')
        channel.basic_consume(queue=queue_name, on_message_callback=process_deviation_message, auto_ack=True)
        logger.info(f'Queue declared for topic *.{os.getenv("DEVIATION_MESSAGE_SUBTOPIC")} -> start consuming...')
        channel.start_consuming()
    finally:
        connection.close()


@celery_app.task(name='core.publish_schedules')
def publish_schedules(schedules: typing.Dict[str, typing.Dict[str, float]], schedules_are_setpoints: bool):
    """
    Publish each device's schedule to the message exchange to be received by its controller.
    Publish the GCP schedule to inform other services (e.g. grid management) about it.
    :param schedules: Power schedule for each device (or the grid connection point) (keys=device keys or "gcp",
    values=dict(timestamp=active_power_value); timestamps are ISO-format str)
    :param schedules_are_setpoints: If the schedules are setpoints, i.e. only a single value

    """
    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()
    try:
        for device_key, schedule in schedules.items():
            try:
                if schedules_are_setpoints:
                    assert len(
                        schedule) == 1, f"Setpoint schedule of {device_key} contains more than 1 period: {schedule}"
                    if device_key == os.getenv('GRID_CONNECTION_POINT_KEY'):
                        # To not confuse it with the received GCP setpoint
                        message_type = "schedule"
                    else:
                        message_type = "setpoint"
                    payload = json.dumps({'active_power': list(schedule.values())[0]})

                else:
                    # Normal schedule with multiple periods
                    assert len(
                        schedule) > 1, f"Schedule of {device_key} is empty or contains only a single period: {schedule}"
                    if device_key == os.getenv('GRID_CONNECTION_POINT_KEY'):
                        # No need to publish it in this case
                        continue
                    message_type = "schedule"
                    payload = json.dumps({'active_power': schedule})

                logger.debug(f'Publish {message_type} for {device_key}.')
                channel.basic_publish(exchange=os.getenv('RABBITMQ_BEM_CONTROL'),
                                      routing_key=f'{device_key}.{message_type}',
                                      body=payload)
            except AssertionError as e:
                logger.error(e)
                continue

            except Exception:
                logger.exception(f'Error publishing {message_type} for {device_key}.')
                continue
    finally:
        connection.close()


'''
TESTING
'''


@celery_app.task(name='core.test_optimization')
def test_optimization_task():
    test_optimization.run(call_task=True, test_opt_with_primary_quotas=True)
