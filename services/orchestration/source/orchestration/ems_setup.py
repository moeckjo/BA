"""
Note:
    - on_after_finalize.connect only works for functions in module where Celery instance is created (i.e., here)
    - add_periodic_task only works in module where Celery instance is created (i.e., here)
"""
import json
import os
import sys

import typing
import time

import pytz
import pika
from celery import Celery, beat, schedules
from celery.utils import abstract
from celery.signals import task_success, task_postrun, after_task_publish
import datetime
import logging
from dotenv import load_dotenv

from celeryconfig import task_routes

import db.db_helper as db

logger = logging.getLogger('bem-orchestration')
logformat = "[%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s]:  %(message)s"
loglevel = logging.DEBUG if os.getenv("DEBUG").lower() == "true" else logging.INFO
logging.basicConfig(stream=sys.stdout, level=loglevel, format=logformat)
logging.getLogger("pika").setLevel(logging.WARNING)


class MyCelery(Celery):

    # Method overridden to enable adding periodic tasks to beat's schedule by task name (not only by signature)
    def add_periodic_task(self, schedule, sig=None, task_name=None,
                          args=(), kwargs=None, schedule_entry_name=None, **opts):
        """
        Add a task to the celery beat schedule to execute it periodically.
        :param schedule: Execeution interval (timedelta or seconds)
        :param sig: Task signature (must be provided if no task name)
        :param task_name: Task name (must be provided if no task signature)
        :param args: Arguments of the task
        :param kwargs: Keyword arguments of the periodic task
        :param schedule_entry_name: Unique name of the entry in the celery beat schedule
        :param opts: Further options for the task execution
        :return:
        """
        logger.info(f'Add periodic task {schedule_entry_name} with schedule {schedule} and options {opts}')

        if isinstance(sig, abstract.CallableSignature):
            # Default method call
            super().add_periodic_task(schedule, sig, args, kwargs, schedule_entry_name)
        elif isinstance(task_name, str):
            # Create beat schedule entry
            schedule_entry = {
                'schedule': schedule,
                'task': task_name,
                'args': args,
                'kwargs': {} if not kwargs else kwargs,
                'options': {**opts},
            }
            # Add entry to beat schedule file
            self._add_periodic_task(schedule_entry_name, schedule_entry)
        else:
            raise ValueError('Either task signature or task name must be provided. You provided none.')


logger.debug("Initializing celery object.")
celery = MyCelery()  # Rest is defined in environment in docker-compose file
celery.config_from_object('celeryconfig')

'''
Helper functions
'''


def determine_upcoming_block_start(buffer: datetime.timedelta) -> datetime.datetime:
    """
    :param buffer: Time buffer to consider before the block starts (e.g. for computations or external requirements)
    :return: start of next block (UTC)
    """
    # Determine the start of the upcoming time block
    # Initiate with first possible block of a day
    now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
    now_local = now_utc.astimezone(pytz.timezone(os.getenv('LOCAL_TIMEZONE')))
    next_block = now_local.replace(hour=int(os.getenv('FIRST_QUOTA_WINDOW_HOUR')), minute=0, second=0,
                                   microsecond=0)  # local time
    while now_local > next_block - buffer:
        next_block += datetime.timedelta(hours=int(os.getenv('QUOTA_WINDOW_SIZE')))

    return next_block.astimezone(datetime.timezone.utc)  # utc


def hours_of_day_list(reference: datetime.datetime, hour_delta: int):
    """
    Computes sorted list of hours equally spaced by hour_delta, distributed around the given start time.
    :param reference: Reference timestamp
    :param hour_delta: Distance to the reference timestamp in hours
    :return:
    """
    timestamps = [reference]
    while len(timestamps) < 24 / hour_delta:
        next = timestamps[-1] + datetime.timedelta(hours=hour_delta)
        timestamps.append(next)
    # Return only the hours (int) of the timestamps
    return sorted([t.hour for t in timestamps])


def quota_block_crontab_schedule(buffer: datetime.timedelta, lead_time: datetime.timedelta):
    first_execution = determine_upcoming_block_start(buffer + lead_time) - buffer - lead_time
    cron_hours = hours_of_day_list(reference=first_execution, hour_delta=int(os.getenv('QUOTA_WINDOW_SIZE')))
    crontab_schedule = schedules.crontab(
        hour=str(cron_hours).strip('[]'),
        minute=str(first_execution.minute)
    )
    return crontab_schedule


def save_environment():
    # Store current settings defined in .env in the database
    # settings = {k: v.rstrip('\n') for k, v in (l.split('=') for l in open('.env') if ('#' not in l) and ('=' in l))}
    environment = dict(os.environ)
    environment['timestamp'] = datetime.datetime.now(tz=pytz.utc).isoformat(timespec='seconds')
    db.save_dict_to_db(
        db=os.getenv('MONGO_SETTINGS_DB_NAME'),
        data_category=os.getenv('MONGO_SETTINGS_ENVIRONMENT_COLL_NAME'),
        data=environment
    )
    logger.debug(f"Stored the following environment:\n {environment}")

'''
Setup message exchanges
'''


def setup_message_exchanges():
    """
    Declares the given message exchanges, using RabbitMQ as message broker.
    :param exchanges: List of tuples with (exchange_name, exchange_type)
    """
    logger.debug('Load config for message exchanges.')
    with open(os.path.join(os.getenv('BEM_ROOT_DIR'), 'config', 'message_exchange_config.json')) as config_file:
        exchanges = json.load(config_file)
    logger.debug(f'Exchange config: {exchanges}')

    # Establish connection to RabbitMQ server
    logger.debug('Connect to RabbitMQ Host and set up message exchanges.')

    # It can take 10-30s until the RabbitMQ server is really up and connections can be established
    # -> retry until connected.
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=os.getenv('RABBITMQ_HOSTNAME'),
        connection_attempts=100, retry_delay=2))

    channel = connection.channel()

    # Create the exchanges
    for exchange in exchanges:
        if exchange['name'] in ["", None, "None"]:
            exchange['name'] = os.getenv(exchange["name_env_var"])
            logger.debug(f'Declaring exchange "{exchange["name"]}" of type "{exchange["type"]}".')

            assert exchange['name'] != "", f"No exchange name has been defined. " \
                                           f"Variable '{exchange['name_env_var']}' has no value."
        assert exchange['type'] in ['direct', 'topic',
                                    'headers',
                                    'fanout'], f"Invalid message exchange type {exchange['type']} for exchange '{exchange['name']}'."

        channel.exchange_declare(exchange=exchange['name'], exchange_type=exchange['type'], durable=True)
        logger.info(f"Declared {exchange['type']} exchange '{exchange['name']}'.")

    connection.close()


'''
Setup device management service
'''


def setup_periodic_devmgmt_tasks(devmgmt_systems_ids: typing.Dict[str, str]):
    # # Add management task that is executed with the given frequency (parameter 'schedule') for each device
    # # Schedule entries ('name' parameter) must differ
    # for category, dms_id in devmgmt_systems_ids.items():
    #     # TODO: Remove condition to create periodic task for all DMSs
    #     if category in ['pv', 'bess']:
    #         # Recording of device measurements
    #         # TODO: this may be obsolete when doing read requests in separate service and different for some device?
    #         celery.add_periodic_task(
    #             schedule=120,  # TODO: change to datetime.timedelta(seconds=int(os.getenv('DEVICE_RECORDING_FREQ'))),
    #             task_name=f'devmgmt.record_device_state',
    #             args=(dms_id,),
    #             schedule_entry_name=f'record_{category}_state'
    #         )
    #         # Management/control task chain
    #         celery.add_periodic_task(
    #             schedule=datetime.timedelta(seconds=int(os.getenv('DEVICE_CHECK_FREQ_SEC'))),
    #             task_name=f'devmgmt.manage',
    #             args=(dms_id,),
    #             expires=int(int(os.getenv('DEVICE_CHECK_FREQ_SEC')) * 0.9),
    #             schedule_entry_name=f'manage_{category}_system'
    #         )

    # Add task for checking if EV has to be charged, i.e. if max. charging delay after having been connected has passed
    celery.add_periodic_task(
        schedule=datetime.timedelta(hours=float(os.getenv('EV_CHARGING_DELAY_CHECK_FREQUENCY'))),
        task_name='devmgmt.ensure_ev_charging',
        schedule_entry_name=f'ensure_ev_charging',
        expires=int(
            datetime.timedelta(hours=float(os.getenv('EV_CHARGING_DELAY_CHECK_FREQUENCY'))).total_seconds() * 0.9),
    )


def setup_device_management():
    # Send device setup task to devmgmt-worker to receive the IDs of the created device management systems (DMS)
    logger.info('Send task for setup of device management service.')
    result = celery.send_task(name='devmgmt.device_setup')
    devmgmt_systems_ids = result.get()  # dict(device_key, database ID)

    setup_periodic_devmgmt_tasks(devmgmt_systems_ids)

    logger.debug('Setup of device management completed!')
    return devmgmt_systems_ids


'''
Setup grid management service
'''


def setup_periodic_gridmgmt_tasks(internal_grid_connection_id, dso_uuid, devmgmt_systems_ids, quota_market_enabled):
    # Add task for calculation and sending of instant grid-point flexibility every 30 seconds
    # Note: The actual task of calculating and sending the GCP flexibility is called after getting the flexibilities
    #       from all devices. (see 'link' argument)
    celery.add_periodic_task(
        schedule=datetime.timedelta(seconds=int(os.getenv('INSTANT_FLEX_FREQUENCY'))),
        # Get device flexibilities first
        task_name='devmgmt.get_devices_flex',
        args=(devmgmt_systems_ids,),
        expires=int(int(os.getenv('INSTANT_FLEX_FREQUENCY')) * 0.9),
        schedule_entry_name=f'provide_instant_gcp_flex_from_dev_flex',
        # Now call actual GCP flex task, which – as linked task – automatically takes the received device
        # flexibilities as first input argument (GCP ID becomes 2nd arg)
        link=celery.signature('gridmgmt.provide_instant_gcp_flex',
                              args=(internal_grid_connection_id,),
                              queue=task_routes['gridmgmt.*']['queue']
                              )
    )

    # Schedule task for sending out the latest 24h schedule and request the latest quotas

    # Determine crontab schedule based on temporal parameters
    # Consider 5 min for internal processing to ensure that schedule is sent on time
    crontab_schedule = quota_block_crontab_schedule(
        buffer=datetime.timedelta(minutes=5),
        lead_time=datetime.timedelta(hours=int(os.getenv('SCHEDULE_SENDING_LEAD_TIME')))
    )
    # Add periodic task with determined crontab schedule
    celery.add_periodic_task(
        schedule=crontab_schedule,
        task_name='gridmgmt.send_gcp_reference_schedule',
        kwargs=dict(uuid=dso_uuid),
        expires=int(datetime.timedelta(hours=int(os.getenv('SCHEDULE_SENDING_LEAD_TIME'))).total_seconds() / 2),
        schedule_entry_name=f'send_gcp_reference_schedule',
    )

    # Schedule task for requesting the latest quotas (shortly after sending the reference schedule)

    # Determine crontab schedule based on temporal parameters
    # Quotas are expected to be ready x minutes after the full hour before the quota block
    crontab_schedule = quota_block_crontab_schedule(
        buffer=datetime.timedelta(minutes=0),
        lead_time=datetime.timedelta(hours=int(os.getenv('SCHEDULE_SENDING_LEAD_TIME')),
                                     seconds=-int(os.getenv('QUOTA_COMPUTATION_TIME_SEC')))
    )
    # Add periodic task with determined crontab schedule
    celery.add_periodic_task(
        schedule=crontab_schedule,
        task_name='gridmgmt.request_and_process_primary_quotas',
        kwargs=dict(uuid=dso_uuid, quota_market_enabled=quota_market_enabled),
        expires=55 * 60,
        schedule_entry_name=f'quota_request_chain',
    )

    # Add task for requesting the latest grid state data from the DSO for this grid connection point
    celery.add_periodic_task(
        schedule=datetime.timedelta(seconds=int(os.getenv('METER_DATA_REQUEST_FREQUENCY'))),
        task_name='gridmgmt.get_meter_data',
        args=(dso_uuid,),
        schedule_entry_name=f'get_latest_meter_data',
        expires=int(int(os.getenv('METER_DATA_REQUEST_FREQUENCY')) * 0.9),

    )

    # Add task for sending out the latest measurements of devices and load, e.g. from the last 15 minutes
    window_size = datetime.timedelta(seconds=int(os.getenv('MEASUREMENTS_SENDING_FREQUENCY')))
    cron_minutes = [i * int(window_size.seconds / 60) for i in range(int(3600 / window_size.seconds))]
    celery.add_periodic_task(
        schedule=schedules.crontab(minute=str(cron_minutes).strip('[]')),
        task_name='gridmgmt.provide_latest_measurements',
        kwargs=dict(
            resolution=int(os.getenv('OUTGOING_MEASUREMENTS_RESOLUTION')),
            window_size=window_size,
            dso_uuid=dso_uuid
        ),
        expires=int(int(os.getenv('MEASUREMENTS_SENDING_FREQUENCY')) * 0.9),
        schedule_entry_name=f'provide_latest_measurements',
    )

    # Add task for requesting the latest user input for the EV charging process
    celery.add_periodic_task(
        schedule=datetime.timedelta(seconds=int(os.getenv('EV_CHARGING_USER_INPUT_REQUEST_FREQUENCY'))),
        task_name='gridmgmt.get_ev_charging_user_input',
        args=(dso_uuid,),
        schedule_entry_name=f'get_ev_charging_user_input',
        expires=int(int(os.getenv('EV_CHARGING_USER_INPUT_REQUEST_FREQUENCY')) * 0.9),
    )


def setup_grid_management(devmgmt_systems_ids, quota_market_enabled):
    logger.debug(f'Setup grid management (at {datetime.datetime.now()})')

    # Send grid setup task to gridmgmt-worker to receive the DB ID of the created grid connection model
    logger.info('Send task for setup of grid management service.')
    result = celery.send_task(name='gridmgmt.grid_setup')
    internal_grid_connection_id, dso_uuid = result.get()  # str

    # Set up all periodic (time-based) tasks (e.g. sending current flexibility,
    # sending schedules, requesting quotas etc.)
    setup_periodic_gridmgmt_tasks(internal_grid_connection_id, dso_uuid, devmgmt_systems_ids, quota_market_enabled)
    logger.info('Setup of gridmanagement service completed.')


'''
Setup trading
'''


def setup_periodic_trading_tasks():
    logger.info(f'Set up period trading task')

    # Add the request of market results as periodic task
    # Determine crontab schedule based on temporal parameters
    crontab_schedule = quota_block_crontab_schedule(
        buffer=datetime.timedelta(minutes=0),
        lead_time=datetime.timedelta(hours=float(os.getenv('MARKET_RESULT_LEAD_TIME')))
    )

    # Add task with determined crontab schedule
    celery.add_periodic_task(
        schedule=crontab_schedule,
        task_name='trading.market_result',
        expires=int(datetime.timedelta(minutes=float(os.getenv('MARKET_RESULT_LEAD_TIME')) * 60 + 5).total_seconds()),
        schedule_entry_name=f'get_market_result',
    )


def setup_trading():
    logger.info('Send task for setup of trading service.')

    celery.send_task(name='trading.setup')
    setup_periodic_trading_tasks()

    logger.info('Setup of trading service completed.')


'''
Setup core
'''


def setup_core(ignore_schedule_deviations: bool):
    logger.info('Send task for setup of core service.')
    result = celery.send_task(name='core.setup', kwargs=dict(ignore_schedule_deviations=ignore_schedule_deviations))
    success: str = result.get()
    logger.info(f'Setup of core service completed successfully.')


'''
Initiate (daily) 24h-scheduling
'''


def start_daily_scheduling():
    """
    Determine time schedule for device schedule optimization process based on fixed daily hour.
    Start first scheduling as of next possible period, then periodically according to time schedule.
    """
    now = datetime.datetime.now(tz=pytz.timezone(os.getenv('LOCAL_TIMEZONE')))
    # now = pytz.timezone(os.getenv('LOCAL_TIMEZONE')).localize(datetime.datetime(2021, 4, 20, 10, 15))
    logger.debug(f'Now: {now}')

    todays_regular_schedule_start = now.replace(hour=int(os.getenv('DAILY_SCHEDULE_START_HOUR')), minute=0,
                                                second=0, microsecond=0)
    todays_regular_scheduling_time = todays_regular_schedule_start - datetime.timedelta(minutes=15)
    next_full_hour = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    last_full_hour = now.replace(minute=0, second=0, microsecond=0)

    if now < todays_regular_scheduling_time:
        next_regular_scheduling_time = todays_regular_scheduling_time

        if todays_regular_scheduling_time - now >= datetime.timedelta(minutes=5):
            # At least 5 minutes until regular scheduling is triggered. Schedule now and then again at regular time.
            schedule_now = True
            schedule_start = next_full_hour if now.minute >= 45 else last_full_hour
        else:
            # Simply wait those >5 min to start scheduling at the regular time.
            schedule_now = False
    else:
        # Time of scheduling for current horizon has already passed
        # Next regular scheduling is 1 day later
        next_regular_scheduling_time = todays_regular_scheduling_time + datetime.timedelta(days=1)
        if next_regular_scheduling_time - now >= datetime.timedelta(minutes=5):
            # At least 5 minutes until regular scheduling is triggered. Schedule now and then again at regular time.
            if now == todays_regular_scheduling_time:
                # Ensure that moment of todays_regular_scheduling_time passes and will not be additionally triggered by crontab
                time.sleep(1)
            schedule_now = True
            schedule_start = next_full_hour if now.minute >= 45 else last_full_hour
        elif datetime.timedelta(minutes=0) <= next_regular_scheduling_time - now < datetime.timedelta(minutes=5):
            # Simply wait those >5 min to start scheduling at the regular time.
            schedule_now = False
        else:  # now > next_regular_scheduling_time
            # Both last scheduling times have passed
            # Can happen in the time between scheduling start time (e.g. 23:45) and schedule start time (0:00) if it spans midnight
            schedule_now = True
            schedule_start = next_full_hour
            next_regular_scheduling_time += datetime.timedelta(days=1)

    logger.info(f'Start regular scheduling as of  {next_regular_scheduling_time.astimezone(pytz.utc)}')
    daily_scheduling_time_utc = next_regular_scheduling_time.astimezone(pytz.utc)
    celery.add_periodic_task(
        schedule=schedules.crontab(hour=str(daily_scheduling_time_utc.hour),
                                   minute=str(daily_scheduling_time_utc.minute)),
        task_name='core.scheduling',
        kwargs=dict(window_start=None, window_size=datetime.timedelta(hours=24)),
        expires=3600,
        schedule_entry_name='daily_scheduling'
    )

    if schedule_now:
        logger.info(
            f'Schedule now as of {schedule_start} and then later regularly as of {next_regular_scheduling_time.astimezone(pytz.utc)}.')
        # Schedule now starting with next or last full hour
        logger.debug(f'Create 24-hour schedule as of {schedule_start}.')
        celery.send_task(name='core.scheduling',
                         # args=(schedule_start.astimezone(pytz.utc), datetime.timedelta(hours=1),))
                         kwargs=dict(window_start=schedule_start.astimezone(pytz.utc),
                                     window_size=datetime.timedelta(hours=24)))

    daily_scheduling_time_utc = todays_regular_scheduling_time.astimezone(pytz.utc)
    celery.add_periodic_task(
        schedule=schedules.crontab(hour=str(daily_scheduling_time_utc.hour),
                                   minute=str(daily_scheduling_time_utc.minute)),
        task_name='core.scheduling',
        kwargs=dict(window_start=None, window_size=datetime.timedelta(hours=24)),
        expires=3600,
        schedule_entry_name='daily_scheduling'
    )


def start_scheduling():
    """
    Determine time schedule for device schedule optimization process based on quota blocks.
    Start first scheduling as of next possible period, then periodically according to time schedule.
    """
    # now = pytz.timezone(os.getenv('LOCAL_TIMEZONE')).localize(datetime.datetime.now())  # local
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    logger.debug(f'Now raw: {datetime.datetime.now()}, tz: {time.tzname}')
    # Buffer for computation time
    buffer = datetime.timedelta(seconds=int(os.getenv('SCHEDULE_COMPUTATION_BUFFER_SEC')))
    schedule_window_size = datetime.timedelta(hours=24)

    # Determine start of next scheduling period
    # Initiate with start of current hour
    next_period = now.replace(minute=0, second=0, microsecond=0)
    while now >= next_period - buffer:
        next_period += datetime.timedelta(seconds=float(os.getenv('SCHEDULER_TEMP_RESOLUTION_SEC')))
    logger.info(f'Start first schedule as of {next_period}')

    # Send task to create the device schedules as of the next possible period, w.r.t. to possible grid limits
    celery.send_task(name='core.scheduling', kwargs=dict(
        window_start=next_period, window_size=schedule_window_size,
        grid_quotas_category="final", triggering_event="start-up"
    ))

    # Create periodic scheduling task

    # Time between start of the schedule and the time the schedule has to be sent to the DSO
    lead_time = datetime.timedelta(hours=int(os.getenv('SCHEDULE_SENDING_LEAD_TIME')))
    # Get crontab schedule for 24h-scheduling before having to send out the schedule
    crontab_schedule = quota_block_crontab_schedule(
        buffer=buffer + datetime.timedelta(minutes=5),
        # add 5 minutes because schedule sending task has also a buffer of 5 min
        lead_time=lead_time
    )
    scheduling_trigger = "new_quota_period_block" if os.getenv('SCHEDULING_PERIODICITY_REF') == "quota" else None
    # Add periodic task, starting just before the upcoming block + requested schedule sending lead time, every 6 hours
    celery.add_periodic_task(
        schedule=crontab_schedule,
        task_name='core.scheduling',
        schedule_entry_name='scheduling',
        kwargs=dict(window_start=None,
                    window_size=schedule_window_size,
                    lead_time=lead_time,
                    triggering_event=scheduling_trigger
                    )
    )


def start_hourly_pv_forecasting():
    lead_time = datetime.timedelta(seconds=int(os.getenv('PV_FORECAST_SHORTTERM_LEAD_TIME_SEC')))
    window_size = datetime.timedelta(hours=int(os.getenv('PV_FORECAST_SHORTTERM_WINDOW_SIZE')))
    now = datetime.datetime.now(tz=datetime.timezone.utc).replace(second=0, microsecond=0)
    window_start = now + datetime.timedelta(minutes=1) + lead_time
    celery.send_task(name='core.forecast_pv', kwargs=dict(window_start=window_start, window_size=window_size))

    # next_full_hour = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)

    # Add periodic task to update forecast shortly before each hour for the following hour
    celery.add_periodic_task(
        schedule=schedules.crontab(hour='*/1', minute=int(60 - lead_time.total_seconds() / 60)),
        task_name='core.forecast_pv',
        kwargs=dict(window_start=None,
                    window_size=window_size,
                    lead_time=lead_time),
        expires=int(3600 * 0.9),
        schedule_entry_name='short_term_pv_forecast',
        # eta=next_full_hour - lead_time
    )


'''
Mockup functions
'''


def mockup_start_scheduling():
    # now = pytz.timezone(os.getenv('LOCAL_TIMEZONE')).localize(datetime.datetime.now())  # local
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    print(f'Now raw: {datetime.datetime.now()}, tz: {time.tzname}')
    # Buffer for computation time
    buffer = datetime.timedelta(seconds=int(os.getenv('SCHEDULE_COMPUTATION_BUFFER_SEC')))

    # Determine start of next 15 min period
    # Initiate with start of current hour
    next_period = now.replace(minute=0, second=0, microsecond=0)
    print(f'init next period with {next_period} (now={now})')
    while now > next_period - buffer:
        next_period += datetime.timedelta(hours=float(os.getenv('QUOTA_TEMP_RESOLUTION')))
    # next_period_utc_iso = next_period.astimezone(datetime.timezone.utc).isoformat()
    print(f'Start first schedule as of {next_period}')

    # Schedule as of next period
    celery.send_task(name='core.mockup_scheduling', args=(next_period.isoformat(),))

    # Get crontab schedule for 24h-scheduling
    crontab_schedule = quota_block_crontab_schedule(
        buffer=buffer,
        lead_time=datetime.timedelta(hours=int(os.getenv('SCHEDULE_SENDING_LEAD_TIME')))
    )

    print(f'Crontab schedule for 24h-scheduling (UTC): {crontab_schedule}')
    # Add periodic task, starting just before the upcoming block, every 6 hours
    celery.add_periodic_task(
        schedule=crontab_schedule,
        task_name='core.mockup_scheduling',
        schedule_entry_name='mockup_scheduling',
        kwargs=dict(window_size=None, lead_time=int(os.getenv('SCHEDULE_SENDING_LEAD_TIME')))
    )


'''
Overall EMS setup
Calls the functions defined above.
'''


@celery.on_after_finalize.connect
def setup_ems(sender, **kwargs):
    """
    Calls setup functions, i.e. integrates into this EMS  ...
    1. all connected devices
    2. the connected (distribution) grid with corresponding IT systems
    3. Forecast and optimization services
    4. Trading services
    ... and initiates all necessary periodic tasks.
    """
    logger.info('Starting setup of EMS.')

    setup_message_exchanges()
    logger.info("Wait 20 seconds to ensure that all bem services are up and ready for setup.")
    time.sleep(20)
    logger.info("Done sleeping, let's go!")

    # Get scenario settings
    quota_market_enabled = False
    try:
        if int(os.getenv('QUOTA_MARKET_ENABLED')) == 1:
            quota_market_enabled = True
    except (TypeError, AttributeError):
        logger.warning("The variable QUOTA_MARKET_ENABLED is not defined. It is set to False. "
                       "If the quota market is actually enabled, add QUOTA_MARKET_ENABLED=1 to the environment file.")

    ignore_schedule_deviations = False
    try:
        if int(os.getenv('IGNORE_SCHEDULE_DEVIATIONS')) == 1:
            ignore_schedule_deviations = True
    except (TypeError, AttributeError):
        logger.warning("The variable IGNORE_SCHEDULE_DEVIATIONS is not defined. It is set to False. "
                       "If schedule deviations should be ignored (i.e. no re-scheduling), add "
                       "IGNORE_SCHEDULE_DEVIATIONS=1 to the environment file.")

    logger.info(f"Scenario settings: "
                f"quota_market_enabled={quota_market_enabled}; "
                f"ignore_schedule_deviations={ignore_schedule_deviations}")

    # Store global settings, i.e. environment variables, in the database
    save_environment()

    setup_core(ignore_schedule_deviations)
    devmgmt_systems_ids = setup_device_management()
    setup_grid_management(devmgmt_systems_ids, quota_market_enabled)
    if quota_market_enabled:
        setup_trading()

    start_hourly_pv_forecasting()

    if not int(os.getenv('TEST_OPTIMIZATION')):

        logger.info(
            f'Setup finished. Start {"mockup" if int(os.getenv("MOCKUP")) else os.getenv("SCHEDULING_PERIODICITY_REF")} scheduling.')

        if int(os.getenv('MOCKUP')):
            mockup_start_scheduling()
        else:
            if os.getenv('SCHEDULING_PERIODICITY_REF') == "daily":
                start_daily_scheduling()
            elif os.getenv('SCHEDULING_PERIODICITY_REF') == "quota":
                start_scheduling()
            else:
                logger.error(
                    f"Invalid value '{os.getenv('SCHEDULING_PERIODICITY_REF')}' for variable SCHEDULING_PERIODICITY_REF (options: daily/quota)")

        if int(os.getenv('SEND_SCHEDULE_AT_INIT')):
            celery.send_task(name='gridmgmt.send_gcp_reference_schedule')
            os.putenv('SEND_SCHEDULE_AT_INIT', str(0))

        logger.info('EMS setup completed!')

    else:
        # Call task that runs optimization test module
        logger.info('Send scheduling test task')
        celery.send_task(name='core.test_optimization')