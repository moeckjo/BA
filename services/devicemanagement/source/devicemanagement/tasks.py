import json
import logging
import threading
import time

import typing
import pandas as pd
import pika
from celery import Celery, chain
from celery.result import GroupResult
from celery.signals import after_setup_task_logger
from celery.app.log import TaskFormatter
from celery.utils.log import get_task_logger

import datetime
import os

from devicemanagement import logger, logformat
import device_manager
import utils
from device_manager import DeviceManagementSystem
from management import flexibility_manager

from db.mongodb_handler import MongoConnector
import db.db_helper as db

# Rest is defined in environment in docker-compose file
celery_app = Celery()
celery_app.config_from_object('celeryconfig')

logger = get_task_logger(logger.name)


@celery_app.on_after_finalize.connect
def hello(sender, **kwargs):
    logger.info('Hello from device management!')


'''
Initial setup
'''


@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    """
    Configure format for all log messages by setting the format of all handlers of the celery
    task logger (which has been initialized above).
    Example log message:
    [2022-02-22 17:38:41,888 - bem-devicemanagement - task:devmgmt.get_devices_flex - get_el_device_flexibilities
    - INFO]: Get flexibility of these devices: ['pv', 'bess', 'evse']
    """
    for handler in logger.handlers:
        handler.setFormatter(TaskFormatter(logformat))
    logger.info(f'Logger is configured.')


@celery_app.task(name='devmgmt.device_setup')
def setup_device_management() -> dict:
    """
    Call create_device_management_systems() to create a device management system instance for each device to
    be integrated (i.e., connected, modelled, observed and controlled).
    Each instance is also stored in a database, which in turn assigns the instance a unique (database) ID.
    :return Dictionary of key=device_subcategory, value=db_id pairs for all integrated devices.
    """
    # TODO: remove cleaning of entire collection before going into production (or not?)
    #  -> all info is reproduced on container start up, hence why not clean to avoid duplicate entries?
    mongo_connector = MongoConnector(db=os.getenv('MONGO_DEVICE_DB_NAME'))
    mongo_connector.clear_collection(os.getenv('MONGO_DEVICE_MGMT_SYSTEM_COLL_NAME'))
    mongo_connector.clear_collection(os.getenv('MONGO_DEVICE_COLL_NAME'))

    device_management_systems: typing.List[object] = device_manager.create_device_management_systems()
    logger.info(f'Returned DMSs: {device_management_systems}')

    db_ids = {}
    for dms in device_management_systems:
        # Mapping of device key to ID in database
        db_ids[str(dms.model)] = dms.get_db_id()

    # Save the current device configuration with a timestamp in the settings database for tracking purposes
    device_manager.save_device_config()

    # Start listening for the event that some device produces or consumes more than scheduled (incl. tolerance),
    # published by the connector interface
    thread = threading.Thread(target=wait_for_schedule_deviations)
    thread.start()

    return db_ids


'''
Observe and control
'''


@celery_app.task(name='devmgmt.observe', ignore_result=False)
def observe(observer, connector) -> typing.Tuple[datetime.datetime, dict]:
    timestamp, measurements = observer.observe(connector)
    return timestamp, measurements


@celery_app.task(name='devmgmt.record', ignore_result=False)
def record(data, observer):
    timestamp, measurements = data
    observer.record(timestamp, measurements)


@celery_app.task(name='devmgmt.check', ignore_result=False)
def check(data, observer, model):
    # TODO: Return useful result
    timestamp, measurements = data
    result = observer.check(timestamp, measurements, model)
    return result


@celery_app.task(name='devmgmt.react', ignore_result=False)
def react(check_result, controller, model):
    # TODO: Implement reaction for all devices
    # Remove None item form list that was returned by the record task (no other solution known)
    # check_result = list(filter(None, check_result))[0]
    reaction, adjust_schedules = controller.react(check_result, model)
    if isinstance(adjust_schedules, dict):
        print('Need to ajdust schedules.')
        # TODO: Uncomment when scheduling is implemented and delete sending of test task below
        # window_start = adjust_schedules['from']
        # window_size = adjust_schedules['to'] - window_start
        # celery_app.send_task(
        #     name='core.scheduling',
        #     kwargs={
        #         'window_start': window_start,
        #         'window_size': window_size,
        #         'operational': True
        #     })
        celery_app.send_task(name='core.test_rescheduling', args=(repr(model),))

    return reaction


@celery_app.task(name='devmgmt.update')
def update_dms(dms_id, model, observer, controller, connector, start_time):
    dms_components = {
        'model': model,
        'observer': observer,
        'controller': controller,
        'connector': connector
    }
    DeviceManagementSystem.update_objects_in_db(dms_id, dms_components)
    print(f'Management process took {time.perf_counter() - start_time} sec.')


@celery_app.task(name='devmgmt.record_device_state', ignore_results=False)
def record_device_state(dms_id):
    dms_components = DeviceManagementSystem.get_from_db(dms_id)
    observer = dms_components['observer']
    connector = dms_components['connector']

    data = connector.request_measurements()
    record(data, observer)


@celery_app.task(name='devmgmt.manage', ignore_results=True)
def manage(dms_id) -> GroupResult:
    """
    Chain of tasks to observe the device operation, record current measurements,
    check for deviations and react if necessary.
    :param dms_id: Database ID of the device management system.
    :return:
    """
    start_time = time.perf_counter()

    # Get device management system with all components from the database
    dms_components = DeviceManagementSystem.get_from_db(dms_id)
    mid = time.process_time()
    model = dms_components['model']
    observer = dms_components['observer']
    controller = dms_components['controller']
    connector = dms_components['connector']

    # Define task chain
    management_chain = chain(
        observe.s(observer, connector),
        # Execute check with data returned from observe
        # Execute react after check with returned check result
        chain(check.s(observer, model), react.s(controller, model)),
        # Update components in DB, in case some instance attributes have changed.
        # Otherwise these changes do not persist.
        update_dms.si(dms_id, model, observer, controller, connector, start_time)
    )

    # Execute task chain
    results = management_chain()
    return results


def control(dms_id):
    dms_components = DeviceManagementSystem.get_from_db(dms_id)
    mid = time.process_time()
    model = dms_components['model']
    observer = dms_components['observer']
    controller = dms_components['controller']
    connector = dms_components['connector']

    controller.control(connector)


'''
Flexibility calculation
'''


@celery_app.task(name='devmgmt.get_devices_flex')
def get_el_device_flexibilities(dms_ids) -> typing.Dict[str, typing.List[int]]:
    """
    :param dms_ids: dict(device_subcategory, ID)
    :return: Max. and min. values for all generation, storage and consumption devices, respectively
    """
    device_models = {}
    t_delta = int(os.getenv('INSTANT_FLEX_DURATION'))
    # Get all device models from DB
    for device_key, dms_id in dms_ids.items():
        try:
            result = DeviceManagementSystem.get_from_db(dms_id)
            model = result['model']
            device_models[device_key] = model
        except TypeError as e:
            logger.debug(f'Query result for {device_key}: {result} (or as list: {list(result)}')
            e.with_traceback(tb=e.__traceback__)

    # Get the respective flexibilities regarding active power
    logger.info(f"Get flexibility of these devices: {list(device_models.keys())}")
    flexibilities: typing.Dict[str, list] = flexibility_manager.electric_flexibilities(device_models, t_delta)
    del device_models

    return flexibilities


'''
EV charging necessity check and triggering
'''


@celery_app.task(name='devmgmt.ensure_ev_charging')
def ensure_ev_charging():
    """
    Usually the charging process is triggered soon after connection when the user enters its charging
    preferences (incl. current SOC) in an app. This function ensures eventual charging if either the EV has not
    been connected soon afterwards or if this input is missing completely.

    It works as follows:

    First, check if the EV is connected. If not, nothing to do here.

    Then check if the EV has been charged since it was connected. If so, nothing to do.

    If not, check if an input with a future departure was received from the user, but the EV has not been
    charged since then. If so, publish this user input again to trigger scheduling.

    Otherwise, check if the max. allowed time delay (e.g. 2 hours) has passed since connection without being
    charged and trigger the scheduling/charging process if that's the case by publishing a fake user input.

    """

    def publish_ev_charging_input_to_ems(input):
        payload = json.dumps(input)
        # Establish connection to RabbitMQ server
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
        channel = connection.channel()
        channel.basic_publish(exchange=os.getenv('RABBITMQ_BEM_INBOX'), routing_key=f'user.input.ev',
                              body=payload)
        connection.close()
        logger.debug(f'Fake user input has been published: {input}')

    now = datetime.datetime.now(tz=datetime.timezone.utc)

    # Get connection data from the database
    query_start_time = now - datetime.timedelta(
        hours=(float(os.getenv('EV_CHARGING_MAX_DELAY')) + float(os.getenv('EV_CHARGING_DELAY_CHECK_FREQUENCY'))),
        minutes=5
    )
    connection_states: pd.Dataframe = device_manager.get_device_measurements(
        source=os.getenv('EVSE_KEY'),
        fields='connected',
        start_time=query_start_time,
    )
    logger.debug(f"Connection states: {connection_states}")

    current_connection_state = connection_states.loc[max(connection_states.index), 'connected']
    if current_connection_state == 0:
        return "EV is not connected."

    # Get the changes of the connection state
    # (diff=0 -> no change; diff=-1 -> from connected to unconnected; diff=1 from unconnected to connected)
    connection_states_diff = connection_states['connected'].diff()

    # Check if the connection state changed from 0 (unconnected) to 1 (connected) in the considered period
    if connection_states_diff.max(skipna=True) <= 0:
        # -> No
        return f"EV has been continuously connected since {query_start_time}."

    # EV got connected during the considered period -> get time of connection
    connected_at = connection_states_diff[connection_states_diff == 1].index[-1].to_pydatetime()

    # Check if it hast been charged since it was connected
    if device_manager.ev_charged_since(connected_at):
        return f"EV is being charged or has been charged since connected at {connected_at}."

    latest_user_input_with_future_departure: typing.Union[
        dict, None] = utils.get_latest_user_input_with_future_departure(now)
    logger.debug(f"Latest user input with future departure: {latest_user_input_with_future_departure}")

    if latest_user_input_with_future_departure is not None:
        # Probable case: User input has been provided, but the EV was not connected within
        # EXPECT_EV_CONNECTED_X_MIN_AFTER_USER_INPUT minutes after the the input was received. Consequently, the
        # schedule optimization was not executed.
        # This should not be penalized, hence check for user input with future departure and if the EV has
        # been charged since they entered their charging information.
        time_of_input = datetime.datetime.fromisoformat(latest_user_input_with_future_departure['timestamp'])
        if device_manager.ev_charged_since(time_of_input):
            return f"EV is being charged or has been charged since user input was provided at {time_of_input}."
        else:
            if now - time_of_input <= datetime.timedelta(seconds=3 * int(os.getenv('SCHEDULER_TEMP_RESOLUTION_SEC'))):
                # Wait some more time. It's possible that there's already a new schedule that will charge the
                # vehicle, but it's not active yet.
                return f"Input has just been provided at {time_of_input}. Check again in " \
                       f"{float(os.getenv('EV_CHARGING_DELAY_CHECK_FREQUENCY')) * 60} minutes if charging has been started."

            else:
                # Waited long enough.
                if latest_user_input_with_future_departure['soc'] < float(os.getenv('EV_SOC_PREFERRED')):
                    # If the scheduling process were triggered, charging would have started by now since charging
                    # is enforced if the SOC is below the (configured) preferred SOC. Publish user input again to
                    # trigger scheduling with charging.
                    publish_ev_charging_input_to_ems(latest_user_input_with_future_departure)
                    return f"EV charging user input from {time_of_input} has been published again to " \
                           f"trigger scheduling and charging."
                else:
                    # Above the preferred SOC, the optimization model decides if, when and with which power
                    # the vehicle is charged. Hence, the input might have triggered the scheduling, but charging
                    # has not been scheduled so far.
                    return f"The initial SOC (as stated by the user) is above the preferred SOC of " \
                           f"{float(os.getenv('EV_SOC_PREFERRED')) * 100}%. It's not possible to say if the vehicle " \
                           f"should have been charged by now."

    else:
        # No user input with a future departure available.
        # (Not yet – it might still be provided. If that happens in the next 2 hours, it's covered by the first case.)
        # If neither user input nor charging after EV_CHARGING_MAX_DELAY since connection happened, a fake user input
        # is published to trigger the scheduling process with a low SOC to force charging.

        time_since_connection = now - connected_at

        # We want to incentivize the users to enter the charging information in the app. Therefore, they are penalized
        # if they don't do it by delaying the charging of the vehicle for a certain time (EV_CHARGING_MAX_DELAY hours).
        # -> Check if max. charging delay has passed.
        if time_since_connection >= datetime.timedelta(hours=float(os.getenv('EV_CHARGING_MAX_DELAY'))):
            # Max. delay has passed. Publish a fake user input with a low SOC to enforce charging now.
            charging_input = dict(
                timestamp=now.isoformat(timespec='seconds'),
                scheduled_departure=(
                        now + datetime.timedelta(hours=int(os.getenv('EV_STANDING_TIME_DEFAULT')))).replace(
                    second=0).isoformat(timespec='seconds'),
                soc=0.2,
                soc_target=1,
                fake=True
            )
            publish_ev_charging_input_to_ems(charging_input)

            # Save fake input to database
            db.save_dict_to_db(
                db=os.getenv('MONGO_USERDATA_DB_NAME'),
                data_category=os.getenv('MONGO_EV_CHARGING_INPUT_COLL_NAME'),
                data=charging_input
            )
            return f"EV has been connected for more than {float(os.getenv('EV_CHARGING_MAX_DELAY'))} hour(s) without " \
                   f"being charged. A fake user input has been published."
        else:
            return f"EV has only been connected for {time_since_connection} hour(s) without being charged. " \
                   f"Max. is {int(os.getenv('EV_CHARGING_MAX_DELAY'))}, therefore no action is taken."


def wait_for_schedule_deviations():
    """
    Sets up a queue bound to the message exchange "control" to subscribe to the topics "*.deviation".
    Currently, it is mainly used to derive that the EV's battery is full and to store its SOC accordingly.
    (To prevent that re-scheduling is done with the assumption that SOC < 100% based on current schedule, because we
    don't get SOC measurements from the charging station.)
    It also outputs a warning in case of problematic charging states, e.g. when charging got suspended due to some problem.
    """

    def process_deviation_message(ch, method, properties, body):
        """
        Executed when receiving message for topic '*.deviation' for some device *
        ('gcp' can also be a device)

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
        device = method.routing_key.split('.')[0]
        message = json.loads(body)
        logger.info(f'Deviation message received from device {device}: {message}')

        if device == os.getenv('EVSE_KEY'):
            ev_full: bool = device_manager.evaluate_ev_schedule_deviation(
                target_value=float(message["target_value"]),
                sensor_value=float(message["sensor_value"]),
                timestamp=message['timestamp']
            )
            if ev_full:
                logger.info(f'EV is probably full. Saving SOC=1.0 as measurement.')
                # -> Save SOC=1.0 to prevent that charging is planned again with the next immediate schedule
                db.save_measurement(os.getenv('EVSE_KEY'),
                                    data={message['timestamp']: {'soc': 1.0}}
                                    )

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
