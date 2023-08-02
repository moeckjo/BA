import json
import os
import logging
import datetime
import random
import typing
import pandas as pd
import pika

from celery import Celery, chain
from celery.signals import after_setup_task_logger, after_setup_logger
from celery.app.log import TaskFormatter
from celery.utils.log import get_task_logger

from db import db_helper as db

from gridmanagement import logger, logformat
import grid_manager, communication, utils
from gridconnection_model import GridConnectionPoint

# Rest is defined in environment in docker-compose file
celery_app = Celery()
celery_app.config_from_object('celeryconfig')

logger = get_task_logger(logger.name)


@celery_app.on_after_finalize.connect
def hello(sender, **kwargs):
    logger.info('Hello from grid management!')


'''
Initial setup
'''


@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    """
    Configure format for all log messages by setting the format of all handlers of the celery
    task logger (which has been initialized above).
    Example log message:
    [2022-02-22 15:02:42,031 - bem-gridmanagement - task:gridmgmt.get_meter_data - make_request - INFO]:
    GET request to https://inubit.flexqgrid.online/api/gems/messwerte/_ce52507d-b850-53f4-9b70-fa36b2d56789
    """
    for handler in logger.handlers:
        handler.setFormatter(TaskFormatter(logformat))
    logger.info('Logger is configured.')


@celery_app.task(name='gridmgmt.grid_setup')
def setup_grid_management() -> dict:
    grid_connection_db_id, grid_connection_dso_uuid = grid_manager.create_grid_connection()

    # Save the current grid connection configuration with a timestamp in the settings database for tracking purposes
    grid_manager.save_gridconnection_config()

    # Subscribe to secondary quota messages
    wait_for_secondary_quotas.delay()
    # Subscribe to GCP setpoints
    wait_for_gcp_setpoint.delay()
    # Subscribe to GCP schedules realizing the setpoint
    wait_for_gcp_setpoint_schedule.delay()

    return grid_connection_db_id, grid_connection_dso_uuid


'''
Requesting current grid state data measured at the grid connection point and subsequent processing
'''


@celery_app.task(name='gridmgmt.get_meter_data')
def get_meter_data(uuid: str):
    esb_connector = communication.ESBConnector(communication.load_esb_config())
    message = esb_connector.get_message(
        message_type='meter_data',
        uuid=uuid,
        retry_after=5,
        stop_retry_after=20,
        catch_exceptions=(AssertionError,)
    )

    meter_active_power: typing.Dict[str, float] = grid_manager.process_meter_data_message(message)

    if meter_active_power:
        process_meter_values.apply_async(kwargs=dict(meter_active_power=meter_active_power))

    return meter_active_power


@celery_app.task(name='gridmgmt.process_meter_values')
def process_meter_values(meter_active_power: typing.Dict[str, float]):
    gcp_active_power, inflexible_load_active_power = grid_manager.process_meter_values(meter_active_power)

    if inflexible_load_active_power is not None:
        db.save_measurement(
            source=os.getenv('LOAD_EL_KEY'),
            data=inflexible_load_active_power
        )
    if gcp_active_power is not None:
        check_limit_violation.apply_async(kwargs=dict(gcp_active_power=gcp_active_power))

        db.save_measurement(
            source=os.getenv('GRID_CONNECTION_POINT_KEY'),
            data=gcp_active_power
        )

    return f"GCP power={gcp_active_power}, inflex. load={inflexible_load_active_power}"


@celery_app.task(name='gridmgmt.check_limit_violation')
def check_limit_violation(gcp_active_power: typing.Dict[str, typing.Dict[str, float]]):
    # This task is only executed if new meter data was received to calculate the current active
    # power at the grid connection point
    grid_manager.check_for_tolerated_deviation(gcp_active_power)


'''
Requesting user input for the EV charging process
'''


@celery_app.task(name='gridmgmt.get_ev_charging_user_input')
def get_ev_charging_user_input(dso_uuid: str):
    esb_connector = communication.ESBConnector(communication.load_esb_config())
    message = esb_connector.get_message(
        message_type='ev_charging_user_input',
        uuid=dso_uuid
    )
    if message:
        grid_manager.process_ev_charging_user_input_message(message, dso_uuid)


'''
Sending latest device and load measurements (for visualization in user app)
'''


@celery_app.task(name='gridmgmt.provide_latest_measurements')
def provide_measurements(resolution: int, window_size: datetime.timedelta, dso_uuid: str):
    """
    Task chain to get the last measurements for the defined passed time window.
    :param resolution: Requested resolution of the data in seconds. Data will be resampled if necessary.
    :param window_size: Requested interval of the recent past. Determines time range = now - window_size
    :param dso_uuid: UUID of this grid connection point defined by the DSO
    """
    filter_fields_general = set(os.getenv('MEASUREMENTS_FIELDS_GENERAL').split(','))
    filter_fields_bess = set(os.getenv('MEASUREMENTS_FIELDS_STORAGE').split(','))
    filter_fields = {key: filter_fields_general for key in
                     [os.getenv('PV_KEY'), os.getenv('EVSE_KEY'), os.getenv('HEAT_PUMP_KEY'),
                      os.getenv('LOAD_EL_KEY').split('_')[0]]
                     }
    filter_fields[os.getenv('BESS_KEY')] = filter_fields_bess

    data = grid_manager.get_device_measurements_aggregated_per_subcategory_formatted(
        resolution=resolution,
        ignore_nan_in_resampled_data=True,
        window_size=window_size,
        include_load=True,
        filter_fields=filter_fields,
        return_readable_value_format=True,
        iso_timestamps=True
    )

    send_measurements.apply_async(kwargs=dict(measurements=data, uuid=dso_uuid))


@celery_app.task(name='gridmgmt.send_measurements')
def send_measurements(measurements: typing.Dict[str, pd.DataFrame], uuid: str):
    """
    Construct the message (json payload), then send it to ESB.
    """
    message = grid_manager.construct_measurements_message(measurements, uuid=uuid)

    connector = communication.ESBConnector(
        specifications=communication.load_esb_config()
    )
    logger.debug(f'Measurements: {message}')
    connector.send_message(message_type='measurements', message=message, uuid=uuid)


'''
Calculation and sending of flexibility at the grid connection point
'''


@celery_app.task(name='gridmgmt.get_gcp_flex')
def get_gcp_flexibility(
        gcp_db_id: str, device_flexibilities: typing.Dict[str, typing.List[int]]
) -> typing.Tuple[int, int, dict]:
    logger.debug('CALCULATE GCP FLEX')
    grid_connection_point = grid_manager.get_model_from_db(gcp_db_id)
    gcp_attributes = {"cluster_id": grid_connection_point.cluster_id,
                      "uuid": grid_connection_point.dso_uuid,
                      "unconditional_consumption": grid_connection_point.unconditional_consumption}
    try:
        lower_bound, upper_bound = grid_connection_point.flexibility(**device_flexibilities)
        return int(lower_bound), int(upper_bound), gcp_attributes

    except (TypeError, ValueError):
        # Happens if there are None values because there were no recent measurements (e.g. in the last 10 minutes)
        # for a device or load
        logger.warning(f"Could not calculate GCP flexibility due to missing device or load measurements.")
        return None, None, gcp_attributes



@celery_app.task(name='gridmgmt.send_gcp_flex')
def send_gcp_flexibility(info: typing.Tuple[int, int, dict], timestamp: datetime.datetime):
    lower_bound, upper_bound, gcp_attributes = info
    connector = communication.ESBConnector(
        specifications=communication.load_esb_config()
    )
    message = grid_manager.construct_flexibility_message(lower_bound, upper_bound, gcp_attributes, timestamp)
    logger.debug(f'Flex message: {message}')
    connector.send_message(message_type='flexibility', message=message, uuid=gcp_attributes["uuid"])


@celery_app.task(name='gridmgmt.provide_instant_gcp_flex')
def provide_instant_gcp_flexibility(device_flexibilities: typing.Dict[str, typing.List[int]], gcp_db_id: str):
    """
    Task chain to calculate and send the current flexibility at the grid connection point.
    It is triggered automatically after calculating the flexibility of all devices.
    :param device_flexibilities: Flexibilities (active power) of all generation, storage and consumption devices
    :param gcp_db_id: (DB) ID of grid connection instance
    """
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc)

    # Define task chain
    flex_chain = chain(
        # Based on the device's flexibilities, calculate the resulting flex. at the grid point (exchangeable power)
        get_gcp_flexibility.s(gcp_db_id, device_flexibilities),
        # Send out the resulting lower and upper bound to the interested external party
        send_gcp_flexibility.s(timestamp)
    )
    # Execute task chain
    # FYI: Execution of whole chain takes about 1 second.
    flex_chain()


'''
Handling of power setpoints upon reception
'''


@celery_app.task(name='gridmgmt.send_gcp_setpoint_realization')
def send_gcp_power_realization(power: int):
    connector = communication.ESBConnector(communication.load_esb_config())
    message = grid_manager.construct_realized_setpoint_message(power)
    connector.send_message(
        message_type="realization_setpoint",
        message=message,
        uuid=grid_manager.load_grid_config()["specifications"]["uuid"]
    )


@celery_app.task(name='gridmgmt.send_gcp_setpoint_ack')
def send_gcp_power_setpoint_ack(timestamp: str, feedin_setpoint: int, consumption_setpoint: int, clear: bool):
    connector = communication.ESBConnector(communication.load_esb_config())
    message = grid_manager.construct_ack_setpoint_message(timestamp, feedin_setpoint, consumption_setpoint, clear)

    connector.send_message(
        message_type="ack_setpoint",
        message=message,
        uuid=grid_manager.load_grid_config()["specifications"]["uuid"]
    )
    return message


@celery_app.task(name='gridmgmt.handle_gcp_limit')
def handle_gcp_power_setpoint(message: dict):
    """
    Upon reception of a mandatory, immediate power setpoint of grid consumption or feedin, send out required message.
    :param message: Message containing the setpoint
    """
    power_setpoint = message['value']
    timestamp: str = message['timestamp']  # ISO format

    if power_setpoint is None:
        feedin_setpoint = consumption_setpoint = None
        clear = True
    else:
        feedin_setpoint = abs(power_setpoint) if power_setpoint <= 0 else None
        consumption_setpoint = power_setpoint if power_setpoint > 0 else None
        clear = False
    # Send confirmation to decentral grid operator system that (and which) limit has been received
    send_gcp_power_setpoint_ack.apply_async((timestamp, feedin_setpoint, consumption_setpoint, clear,))


@celery_app.task(name='gridmgmt.await_gcp_setpoint')
def wait_for_gcp_setpoint():
    """
    Sets up a queue to receive broadcast messages with grid connection setpoints.
    When a setpoint is received, the callback function to trigger the further processes is called.
    """

    def process_message(ch, method, properties, body):
        """
        Executed when receiving message with a grid connection setpoint.
        Deserializes payload to dict and initiates further handling.
        """
        payload = json.loads(body)
        logger.debug(f'Message received for topic {method.routing_key}: {body} -> data: {payload}')
        handle_gcp_power_setpoint.delay(payload)

    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()
    try:
        # Declare queue to receive broadcast messages with grid connection setpoints
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        channel.queue_bind(queue=queue_name, exchange=os.getenv('RABBITMQ_GRID_SETPOINT'))
        channel.basic_consume(queue=queue_name, on_message_callback=process_message, auto_ack=True)
        logger.info(
            f"Queue declared for broadcast message on {os.getenv('RABBITMQ_GRID_SETPOINT')} exchange -> start consuming...")
        channel.start_consuming()
    finally:
        connection.close()


@celery_app.task(name='gridmgmt.await_gcp_schedule')
def wait_for_gcp_setpoint_schedule():
    """
    Sets up a queue bound to the message exchange "control" to subscribe to the topic 'gcp.schedule'.
    When a GCP schedule is received, the callback function to trigger the further processes is called.
    """

    def trigger_sending_process(ch, method, properties, body):
        """
        Executed when receiving message for topics '*.schedule'.
        If it's a schedule that realizes a previously received setpoint, it calls the function
        to send the corresponding realization. Otherwise ("normal" schedule), nothing happens.
        """
        payload = json.loads(body)
        schedule_power: int = payload.pop('active_power')
        try:
            assert isinstance(schedule_power,
                              int), f"ERROR: GCP setpoint schedule with single value (int) expected, got {schedule_power}"
            logger.debug(f'GCP setpoint schedule received with value={schedule_power}')
            send_gcp_power_realization.delay(power=schedule_power)
        except AssertionError as e:
            logger.error(e)

    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()

    try:
        # Declare queue, bind it to the exchange 'control' and start consuming messages with schedules
        result = channel.queue_declare(queue='', exclusive=True)
        # Assign automatically generated queue name
        queue_name = result.method.queue
        channel.queue_bind(queue=queue_name, exchange=os.getenv('RABBITMQ_BEM_CONTROL'),
                           routing_key=f'{os.getenv("GRID_CONNECTION_POINT_KEY")}.schedule')
        channel.basic_consume(queue=queue_name, on_message_callback=trigger_sending_process, auto_ack=True)
        logger.info('Queue declared for topic "gcp.schedule" -> start consuming...')
        channel.start_consuming()
    finally:
        connection.close()


'''
Handling of reference schedule at the grid connection point
'''


@celery_app.task(name='gridmgmt.send_and_save_gcp_schedule')
def send_and_save_gcp_reference_schedule(schedule: typing.Dict[str, int], window_start: datetime.datetime, uuid: str,
                                         updated_at: datetime.datetime):
    message = grid_manager.construct_gcp_schedule_message(
        window_start, schedule, uuid
    )

    esb_connector = communication.ESBConnector(communication.load_esb_config())

    esb_connector.send_message(message_type='schedule', message=message, uuid=uuid)
    # Sending was successful, now store this schedule as reference schedule for the latest quotas to be received
    grid_manager.store_reference_schedule(schedule, updated_at)
    # Also publish it internally, because other services need it as well
    grid_manager.publish_reference_schedule_to_ems(schedule)


'''
Handling of quota request and further processing
'''


@celery_app.task(name='gridmgmt.request_and_process_primary_quotas')
def request_and_process_primary_quotas(uuid: str, quota_market_enabled: bool):
    """
    Request quotas from the DSO and process the message.
    :param uuid: DSO-specified UUID of this system
    :param quota_market_enabled: Boolean indicating if there's an active quota market where we can trade primary
    quotas and get secondary quotas as market result.
    :return: None
    """
    quota_chain = chain(
        # Request quotas
        get_primary_quotas.s(uuid),
        # Process the quota message (implicitly provided to the function as first argument)
        process_primary_quotas.s(uuid, quota_market_enabled)
    )
    quota_chain()


@celery_app.task(name='gridmgmt.get_primary_quotas')
def get_primary_quotas(uuid: str) -> dict:
    esb_connector = communication.ESBConnector(communication.load_esb_config())

    modified_since = datetime.datetime.now() - datetime.timedelta(minutes=5)
    quota_message: dict = esb_connector.get_message(
        message_type='quotas',
        uuid=uuid,
        initial_delay=random.randint(0, 15), retry_after=15, stop_retry_after=7 * 60,
        headers={'If-Modified-Since': modified_since},
        catch_exceptions=(AssertionError,)
    )
    logger.info(f"Raw quota message: {quota_message}")
    return quota_message


@celery_app.task(name='gridmgmt.process_primary_quotas')
def process_primary_quotas(quota_message: typing.Dict[str, dict], uuid: str, quota_market_enabled: bool) -> \
        typing.Dict[str, float]:
    # Extract quota time series from message
    # Returned quota information is a dict that contains the mandatory primary (key='primary') and
    # preliminary (key='preliminary') quotas and limits separately
    # window_start is returned as ISO-format string
    quota_information: typing.Dict[str, typing.Dict[str, dict]]
    calculation_method: str
    window_start: str
    quota_information, calculation_method, window_start = grid_manager.process_quota_message(quota_message, uuid)
    for quota_category, quota_data in quota_information.items():
        grid_manager.save_quotas_to_db(quota_data, quota_category=quota_category, calculation_method=calculation_method)

        if quota_category == 'primary':
            if quota_market_enabled:
                grid_manager.publish_quotas_to_ems({**quota_data, 'start': window_start}, quota_category=quota_category)
            else:
                # No market, therefore there will be no secondary quotas -> determine final quotas based on primary
                # quotas and reference schedule
                # Note: Final quotas are needed because they always define the grid restrictions when
                # scheduling during an ongoing quota period block (in case of power schedule deviations, for instance)
                get_final_quotas.apply_async(kwargs=dict(quota_window_start=window_start, secondary_quota_limits=dict(),
                                                         primary_quota_limits=quota_data,
                                                         calculation_method=calculation_method
                                                         ))


@celery_app.task(name='gridmgmt.get_final_quotas')
def get_final_quotas(quota_window_start: str, secondary_quota_limits: typing.Dict[str, int],
                     primary_quota_limits: typing.Union[None, typing.Dict[str, int]] = None,
                     calculation_method: str = None):
    # Calculate, store and and publish the final quotas and limits.
    grid_manager.determine_final_quotas(
        period_block_start=datetime.datetime.fromisoformat(quota_window_start),
        secondary_power_limits=secondary_quota_limits,
        primary_power_limits=primary_quota_limits,
        calculation_method=calculation_method
    )


@celery_app.task(name='gridmgmt.await_secondary_quotas')
def wait_for_secondary_quotas():
    """
    Sets up a queue bound to the message exchange "inbox" to subscribe to the topic "quotas.secondary.active_power".
    When quota information is received, the callback function to trigger the trading process is called.
    :return:
    """

    def trigger_final_quota_calculation(ch, method, properties, body):
        """
        Executed when receiving message for topic 'quotas.secondary.active_power'.
        Deserializes payload to dict and provides it to the trading task.
        """
        secondary_quota_limits = json.loads(body)
        period_block_start: str = secondary_quota_limits.pop('period_block_start')
        logger.debug(f'Message received for topic {method.routing_key}: {body} -> data: {secondary_quota_limits}')
        get_final_quotas.apply_async(
            kwargs=dict(quota_window_start=period_block_start, secondary_quota_limits=secondary_quota_limits)
        )

    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()
    try:
        # Declare queue, bind it to the exchange 'inbox' and start consuming messages of the topic 'quotas.secondary.power'
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        channel.queue_bind(queue=queue_name, exchange=os.getenv('RABBITMQ_BEM_INBOX'),
                           routing_key='quotas.secondary.active_power')
        channel.basic_consume(queue=queue_name, on_message_callback=trigger_final_quota_calculation, auto_ack=True)
        logger.info('Queue declared for topic "quotas.secondary.active_power" -> start consuming...')
        channel.start_consuming()
    finally:
        connection.close()


'''
Sending orders for secondary market (to ESB for persistence)
'''


@celery_app.task(name='gridmgmt.send_market_orders')
def send_market_orders(order_period_start: datetime.datetime, order_sets: typing.List[typing.List[dict]]):
    """
    :param order_period_start: Start of the period block for which these orders are placed (can precede first period contained in orders)
    :param order_sets: List of order sets. Each order set is again a list of orders.
                        An order is a dict specifying the period (timestamp), order type, quantity and price.
    :return:
    """
    uuid = grid_manager.load_grid_config()["specifications"]["uuid"]
    message = grid_manager.construct_market_orders_message(order_period_start, order_sets, uuid)
    logger.debug(f'Market order message: {message}')

    connector = communication.ESBConnector(
        specifications=communication.load_esb_config()
    )
    connector.send_message(
        message_type='market_orders',
        message=message,
        uuid=uuid
    )


'''
Task chain for sending out the reference schedule at the grid connection point, getting the quota information in return 
and processing it. 
'''


@celery_app.task(name='gridmgmt.send_gcp_reference_schedule')
def send_gcp_reference_schedule(uuid: str):
    """
    This function is periodically executed whenever a new reference schedule has to be
    sent to the DSO (a few minutes before the deadline).
    :param uuid: DSO-specified UUID of this system
    """
    # Determine the time filter for the reference schedule (the upcoming quota period blocks)
    now = datetime.datetime.now(
        tz=datetime.timezone.utc)  # system time -> e.g. ~7:55 UTC for block starting at 10 CET/9 UTC
    next_full_hour = now.replace(minute=0, second=0, microsecond=0, tzinfo=datetime.timezone.utc) + datetime.timedelta(
        hours=1)  # e.g. 8:00 UTC
    window_start = next_full_hour + datetime.timedelta(
        hours=int(os.getenv('SCHEDULE_SENDING_LEAD_TIME')))  # e.g. 9:00 UTC / 10:00 CET
    window_size = datetime.timedelta(
        hours=int(os.getenv('QUOTA_WINDOW_SIZE')) * int(os.getenv('QUOTA_WINDOWS_PER_SCHEDULE')))  # 6h*4=24h
    window_end = window_start + window_size
    # Retrieve the latest schedule for the grid connection for this time filter
    schedule: typing.Dict[datetime.datetime, int] = grid_manager.get_latest_gcp_schedule(
        start_time=window_start,
        end_time=window_end,
        return_timestamps_in_isoformat=False
    )
    if schedule is None:
        return f"Could not send a GCP reference schedule as of {window_start}, " \
               f"because there's no corresponding schedule."

    # Downsample the schedule to the target resolution
    reference_schedule = utils.resample_time_series(schedule,
                                                    target_resolution=int(os.getenv('GCP_SCHEDULE_RESOLUTION')),
                                                    conversion='down',
                                                    resample_kwargs=dict(closed='right', label='right')
                                                    )

    # Send this schedule, then store it separately as "reference schedule", because the just retrieved schedule
    # will be replaced in the database when re-optimizing
    send_and_save_gcp_reference_schedule.apply_async(kwargs=dict(
        schedule=reference_schedule, window_start=window_start, uuid=uuid, updated_at=now)
    )
