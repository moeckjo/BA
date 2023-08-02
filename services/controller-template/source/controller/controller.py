import os
import sys
import json
import logging
import math
import typing
from threading import Timer
from datetime import datetime
from datetime import timezone

from dotenv import load_dotenv, find_dotenv
from paho.mqtt.client import Client

logger = logging.getLogger(__name__)
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=log_format)


def timestamp_now():
    '''
    Computes the current timestamp in ms since epoch (UTC).

    Returns:
    ---------
        Timestamp in milliseconds
    '''

    dt_utcnow = datetime.now(timezone.utc)
    logger.debug(f'Local time at the controller-template: '
                 f'{dt_utcnow.isoformat()}')
    ts_seconds = datetime.timestamp(dt_utcnow)
    return round(ts_seconds * 1000)


class Controller():

    def __init__(self,
                 mqtt_client=Client,
                 timestamp_now=timestamp_now):

        # Below the normal startup and configuration of this class
        logger.info('Starting up Controller')

        self.timestamp_now = timestamp_now

        # dotenv allows us to load env variables from .env files which is
        # convient for developing. If you set override to True tests
        # may fail as the tests assume that the existing environ variables
        # have higher priority over ones defined in the .env file.
        logger.debug("Load environment variables")
        load_dotenv(find_dotenv(), verbose=True, override=False)

        self.CONTROLLER_NAME = os.getenv('CONTROLLER_NAME')
        self.DEVICE_KEY = os.getenv('DEVICE_KEY')
        self.DEBUG = bool(os.getenv('DEBUG'))
        self.MQTT_BROKER_HOST = os.getenv('MQTT_BROKER_HOST')
        self.MQTT_BROKER_PORT = int(os.getenv('MQTT_BROKER_PORT'))

        # Set the log level according to the DEBUG flag.
        if self.DEBUG:
            logger.info("Changing log level to DEBUG")
            for logger_name in logging.root.manager.loggerDict:
                logging.getLogger(logger_name).setLevel(logging.DEBUG)
        else:
            logger.info("Changing log level to INFO")
            for logger_name in logging.root.manager.loggerDict:
                logging.getLogger(logger_name).setLevel(logging.INFO)

        # Topic to send messages when a sensor value
        # does not correspond to the set point
        # self.MQTT_SENSOR_DEVIATION_TOPIC = \
        #     os.getenv('MQTT_SENSOR_DEVIATION_TOPIC')
        self.MQTT_SENSOR_DEVIATION_TOPIC = f"{self.DEVICE_KEY}/controller/{os.getenv('DEVIATION_MESSAGE_SUBTOPIC')}"

        # Topics to receive messages with setpoints or schedules
        # self.MQTT_SCHEDULE_TOPIC = os.getenv('MQTT_SCHEDULE_TOPIC')
        # self.MQTT_SETPOINT_TOPIC = os.getenv('MQTT_SETPOINT_TOPIC')
        self.MQTT_SCHEDULE_TOPIC = f"{self.DEVICE_KEY}/controller/{os.getenv('SCHEDULE_MESSAGE_SUBTOPIC')}"
        self.MQTT_SETPOINT_TOPIC = f"{self.DEVICE_KEY}/controller/{os.getenv('SETPOINT_MESSAGE_SUBTOPIC')}"

        # Values describing the maximum permissible difference between
        # the actual value and the target value
        self.REL_MAX_DIFFERENCE = float(os.getenv('REL_MAX_DIFFERENCE'))
        self.ABS_MAX_DIFFERENCE = float(os.getenv('ABS_MAX_DIFFERENCE'))
        self.REL_MAX_DIFFERENCE_APPLICATION_THRESHOLD = float(os.getenv('REL_MAX_DIFFERENCE_APPLICATION_THRESHOLD'))
        # Grant the device some time to adjust to new target values before checking for deviations (seconds)
        self.ALLOWED_RAMP_TIME = int(os.getenv('ALLOWED_RAMP_TIME'))

        logger.debug('Initialize necessary instance variables')
        self.dict_actuator_to_sensor_topic = {}
        # STRUCTURE
        # dict_sensor_topic_to_data = {
        #   <actuator_mqtt_topic> : <sensor_mqtt_topic>,
        #   <actuator_mqtt_topic> : <sensor_mqtt_topic>,
        #   ...
        # }

        self.dict_sensor_topic_to_data = {}
        # STRUCTURE
        # dict_sensor_topic_to_data = {
        #   <sensor_mqtt_topic> : {
        #       'timer':            <timer obj> or None,
        #       'actuator_topic':   <str>,
        #       'target_value':     <float> or math.nan,
        #       'target_since':     <timestamp in ms> or None,
        #       'subscribed':       <bool>,
        #       'deviation_reported': <bool>
        #   },
        #   ...
        # }

        logger.debug('Setting up connection to MQTT Broker')
        # The configuration for connecting to the mqtt broker.
        connect_kwargs = {
            "host": self.MQTT_BROKER_HOST,
            "port": self.MQTT_BROKER_PORT,
        }
        # The private userdata, used by the callbacks.
        userdata = {
            "connect_kwargs": connect_kwargs,
            "self": self,
        }

        self.client = mqtt_client(userdata=userdata)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message

        # Initial connection to broker
        self.client.connect(**connect_kwargs)

        self.client.subscribe(self.MQTT_SCHEDULE_TOPIC)
        logger.info(f'Subscribed to the topic for the schedule msgs. Topic: '
                    f'{self.MQTT_SCHEDULE_TOPIC}')

        # Setpoint messages aka red phase messages
        self.client.subscribe(self.MQTT_SETPOINT_TOPIC)
        logger.info(f'Subscribed to the topic for the setpoint msgs. Topic: '
                    f'{self.MQTT_SETPOINT_TOPIC}')

        # Start loop in background process.
        logger.info(f'Controller {self.CONTROLLER_NAME} is set up')
        self.client.loop_forever()

    @staticmethod
    def on_connect(client, userdata, flags, rc):
        logger.info(
            'Connected to MQTT broker tcp://%s:%s',
            userdata['connect_kwargs']['host'],
            userdata['connect_kwargs']['port'],
        )

    @staticmethod
    def on_disconnect(client, userdata, rc):
        '''
        Attempt reconnecting if disconnect was not called from a call to
        client.disconnect().
        '''
        if rc != 0:
            logger.info(
                'Lost connection to MQTT broker with controller-template %s. '
                'Reconnecting',
                rc
            )
            client.connect(**userdata['connect_kwargs'])

    @staticmethod
    def on_message(client, userdata, msg):
        """
        Handle incoming messages.

        For config messages:
            #TODO: Docu

        For schedule messages:
            #TODO: Docu
        """

        try:
            self = userdata['self']
            if msg.topic in self.MQTT_SCHEDULE_TOPIC:
                self.run_new_schedule_flow(schedule_topic=msg.topic,
                                           value_msg=msg.payload)
            elif msg.topic in self.dict_sensor_topic_to_data.keys():
                self.run_new_measurement_flow(sensor_topic=msg.topic,
                                              value_msg=msg.payload)
            elif msg.topic in self.MQTT_SETPOINT_TOPIC:
                self.run_new_setpoint_flow(setpoint_topic=msg.topic,
                                           value_msg=msg.payload)
            else:
                logger.info(f'Receiver MQTT msg w/ unknown topic {msg.topic}')

        except Exception:
            logger.exception(
                'Expection while processing MQTT message.\n'
                'Topic: %s\n'
                'Message:\n%s',
                *(msg.topic, msg.payload)
            )
            raise

    def run_new_schedule_flow(self, schedule_topic: str, value_msg: str):
        '''
        Method receives a schedule message and processes it. Processing
        includes the following steps:
            (1) Checking whether the message is valid or not
            (2) Extraction of the information from the message
                (especially the schedule)
            (3) Saving the extracted information in the central directory
                (if not already saved)
            (4) IF a timer has already been set up:
                    Stopping the timer, extracting the old schedule,
                    updating the old schedule with the new
            (5) Remove all commands within the schedule that are in the past
            (6) Subscribing to the sensor topic of the corresponding subject
            (7) Start a new timer
                    - until the next command within the schedule
                    - Timer contains schedule and information about the topics

        Parameters
        ----------
        schedule_topic : str , required
             A string containing the topic via which the MQTT message
             was received.
        value_msg : str , required
            Directory encoded as a string containing the content of the
            mqtt message. The directory created on it must have the following
            format:
                value_msg = { actuator_topic : "",
                              sensor_topic : "",
                              schedule : { unix_timestamp: value,
                                           unix_timestamp: value,
                                           ...
                                          }
                            }
            The timestamps must be encoded as strings. They must follow
            the following structure:    2021-06-18T09:29:30+02:00
                                        YYYY-MM-DDTHH:MM:SS+tz
        '''
        logger.info(f'Receive schedule-msg with topic {schedule_topic}')
        payload = json.loads(value_msg)

        # Check whether the directory contains all the necessary keys
        if not all(k in payload.keys() for k in
                   ('actuator_topic', 'sensor_topic', 'schedule')):
            logger.info('Schedule message does not have the correct format. '
                        'Will be discarded.')
            return

        # Save the messages content
        msg_actuator_topic = payload['actuator_topic']
        msg_sensor_topic = payload['sensor_topic']
        msg_schedule = payload['schedule']

        # Convert string timestamps to unix timestamps
        msg_schedule_time = self.convert_str_timestamps(msg_schedule)
        # Ensure ascending order of the schedule
        msg_schedule_time = dict(sorted(msg_schedule_time.items()))

        # Check whether the transferred sensor topic is already known:
        # If not: Create an entry (initial_dict) in central directory
        if not (msg_sensor_topic in self.dict_sensor_topic_to_data.keys()):
            logger.debug(f'So far unknown sensor topic {msg_sensor_topic}. '
                         f'Sensor topic will be included in internal data '
                         f'structure.')
            # new sensor_topic
            # maintain class variables (dicts)
            initial_dict = {
                'timer': None,
                'actuator_topic': msg_actuator_topic,
                'target_value': math.nan,
                'target_since': None,
                'subscribed': False,
                'deviation_reported': False
            }
            self.dict_sensor_topic_to_data[msg_sensor_topic] = initial_dict
            self.dict_actuator_to_sensor_topic[msg_actuator_topic] = \
                msg_sensor_topic

        if self.dict_sensor_topic_to_data[msg_sensor_topic]['timer']:
            logger.debug(f'There already exists a schedule (& timer) '
                         f'for topic: {msg_sensor_topic}. '
                         f'Will update the schedule.')

            # Stop existing timer obj and export its schedule
            old_kwargs = self.stop_timer_get_kwargs(msg_sensor_topic)
            existing_schedule = old_kwargs['schedule']

            # Update existing schedule with new schedule
            existing_schedule.update(msg_schedule_time)
            becoming_schedule = dict(sorted(existing_schedule.items()))
        else:
            # dict_sensor_topic_to_data[msg_sensor_topic]['timer'] == None
            logger.debug(f'There is no existing schedule (& timer) for topic: '
                         f'{msg_sensor_topic}. Will create a new timer '
                         f'w/ that new schedule.')

            becoming_schedule = msg_schedule_time

        # If the schedule has elements that are in the past,
        # these elements are discarded
        timestamp_now = self.timestamp_now()
        earliest_timestamp = next(iter(becoming_schedule))

        while (earliest_timestamp <= timestamp_now) & bool(becoming_schedule):
            logger.debug(f'The schedule contains timestamps from the past. '
                         f'The corresponding schedule steps will be '
                         f'discarded.')
            becoming_schedule.pop(earliest_timestamp)
            if becoming_schedule:
                earliest_timestamp = next(iter(becoming_schedule))

        # check if becoming schedule is not empty
        if becoming_schedule:

            if not self.dict_sensor_topic_to_data[msg_sensor_topic][
                'subscribed']:
                # Subscribe to measured values for the object we're controlling
                self.start_mqtt_sensor_subscription(
                    sensor_topic=msg_sensor_topic)

            # Start up a timer instance that delays the call to
            # update_current_value until the time has come.
            # Also store the timer object so we can cancel it if new
            # schedules or setpoint messages arrive
            delay_ms = (earliest_timestamp - timestamp_now)
            delay_s = delay_ms / 1000.
            timer_kwargs = {
                'actuator_topic': msg_actuator_topic,
                'sensor_topic': msg_sensor_topic,
                'schedule': becoming_schedule,
            }

            timer = Timer(
                interval=delay_s,
                function=self.update_current_values,
                kwargs=timer_kwargs
            )
            timer.start()
            self.add_timer(msg_sensor_topic, timer)
        else:
            # case: becoming schedule is empty
            logger.debug(f'All timestamps of the schedule message are expired. '
                         f'The schedule message will be discarded.')

    def run_new_measurement_flow(self, sensor_topic: str, value_msg: str):
        '''
        Method receives a measurement message (aka sensor message)
        and processes it. Processing includes the following steps:
            (1) Checking whether the message is valid or not
            (2) Extraction of the information from the message
            (3) Check whether a target value has been set
            (4) Calculation of upper and lower borders
                (with the absolute and relative tolerance)
            (5) Check whether the measured value is within the borders
            (6) IF measured value is outside:
                    Publish a MQTT deviation message

        Parameters
        ----------
        sensor_topic : str , required
             A string containing the topic via which the MQTT message
             was received.
        value_msg: str , required

            payload = { 'value': xx,
                        'timestamp':yy
                       }
        '''

        def check_for_deviation():
            deviation = False
            # Calculate the upper and lower limits
            rel_upper = target_value * (1 + self.REL_MAX_DIFFERENCE)
            rel_lower = target_value * (1 - self.REL_MAX_DIFFERENCE)
            abs_upper = target_value + self.ABS_MAX_DIFFERENCE
            abs_lower = target_value - self.ABS_MAX_DIFFERENCE

            # Check whether the measured value is within the calculated tolerance
            if (abs(target_value) > self.REL_MAX_DIFFERENCE_APPLICATION_THRESHOLD) and \
                    (not (abs(rel_lower) <= abs(sensor_value) <= abs(rel_upper))):
                logger.debug(f'The sensor value {sensor_value} on the topic {sensor_topic} '
                             f'violates the maximum relative tolerance {(rel_lower, rel_upper)}')
                deviation = True
            if not (abs_lower <= sensor_value <= abs_upper):
                logger.debug(f'The sensor value {sensor_value} on the topic {sensor_topic} '
                             f'violates the maximum absolute tolerance {(abs_lower, abs_upper)}')
                deviation = True
            return deviation

        logger.info(f'Receive measurement-msg with topic {sensor_topic}')
        feature = sensor_topic.split('/')[-1]
        payload = json.loads(value_msg)

        if not all(k in payload.keys() for k in ('value', 'timestamp')):
            logger.info('Measurement message is incomplete. '
                        'Will be discarded.')
            return

        sensor_value = payload["value"]
        try:
            sensor_value = float(sensor_value)
        except ValueError:
            pass
        sensor_timestamp = payload["timestamp"]

        # Loading the target value for the corresponding measured value
        target_value = self.dict_sensor_topic_to_data[sensor_topic][
            'target_value']
        # Loading the timestamp of when this target value was sent
        target_since = self.dict_sensor_topic_to_data[sensor_topic][
            'target_since']

        if target_value is math.nan or target_value is None or target_value == "None":
            logger.info(f'No target value has been set for sensor topic '
                        f'{sensor_topic}, the measurement message will '
                        f'be discarded')
            return

        time_since_new_target = self.timestamp_now() - target_since
        if time_since_new_target / 1000 < self.ALLOWED_RAMP_TIME:
            logger.debug(f'Allowed ramp time of {self.ALLOWED_RAMP_TIME}s has not passed yet. '
                         f'Will not check for deviation yet. Target={target_value}, measured={sensor_value}.')
            return

        deviation_detected: bool = check_for_deviation()

        # If the measured value does not meet the tolerance:
        # Inform the Connector Interface via MQTT
        if deviation_detected:
            if not self.dict_sensor_topic_to_data[sensor_topic]['deviation_reported']:
                deviation_payload = {
                    "feature": feature,
                    'sensor_value': sensor_value,
                    'target_value': target_value,
                    'target_since': target_since,
                    'timestamp': sensor_timestamp
                }
                self.client.publish(topic=self.MQTT_SENSOR_DEVIATION_TOPIC,
                                    payload=json.dumps(deviation_payload),
                                    qos=1,
                                    retain=False)
                # Mark that the deviation from this current target value has been reported to prevent
                # that the message is sent over and over again when the next sensor value arrives in a
                # (few) second(s). Eventually, a new target value from an adjusted schedule or setpoint will
                # be set, but probably not before the next sensor value arrives.
                self.dict_sensor_topic_to_data[sensor_topic]['deviation_reported'] = True
                logger.info(f'Deviation message sent: {deviation_payload}')
            else:
                logger.debug(f'Deviation of measured={sensor_value} from target={target_value} has already been reported.')

    def run_new_setpoint_flow(self, setpoint_topic: str, value_msg: str):
        '''
        Method receives a measurement message (aka sensor message)
        and processes it. Processing includes the following steps:
            (1) Checking whether the message is valid or not
            (2) Extraction of the information from the message
            (3) Saving the extracted information in the central directory
                (if not already saved)
            (4) IF a timer has already been set up:
                    Stopping the timer, delete existing schedule
            (5) IF the Sensor-topic has been subscribed:
                    Unsubscribe from this sensor topic
            (6) Sending the new target value

        Parameters
        ----------
        setpoint_topic : str , required
             A string containing the topic via which the MQTT message
             was received.
        value_msg : str , required
            Directory encoded as a string containing the content of the
            mqtt message. The directory created on it must have the following
            format:
                value_msg = { actuator_topic : "",
                              sensor_topic : "",
                              setpoint : ""
                            }
        '''
        logger.info(f'Receive setpoint-msg with topic: {setpoint_topic}')
        payload = json.loads(value_msg)

        # Check whether the directory contains all the necessary keys
        if not all(k in payload.keys() for k in
                   ('actuator_topic', 'sensor_topic', 'setpoint')):
            logger.info('Setpoint message is incomplete. Will be discarded.')
            return

        # Save the messages content
        msg_actuator_topic = payload['actuator_topic']
        msg_sensor_topic = payload['sensor_topic']
        msg_setpoint = payload['setpoint']

        # Check whether the transferred sensor topic is already known:
        # If not: Create an entry (initial_dict) in central directory
        if not (msg_sensor_topic in self.dict_sensor_topic_to_data.keys()):
            logger.debug(f'So far unknown sensor topic {msg_sensor_topic}. '
                         f'Sensor topic will be included in internal data '
                         f'structure.')
            # new sensor_topic
            initial_dict = {
                'timer': None,
                'actuator_topic': msg_actuator_topic,
                'target_value': msg_setpoint,
                'target_since': None,
                'subscribed': False,
                'deviation_reported': False
            }
            self.dict_sensor_topic_to_data[msg_sensor_topic] = initial_dict
            self.dict_actuator_to_sensor_topic[msg_actuator_topic] = \
                msg_sensor_topic

        # Existing timers are deleted to prevent the setpoint value from
        # being overwritten with a schedule that still exists.
        if self.dict_sensor_topic_to_data[msg_sensor_topic]['timer']:
            logger.debug(f'There exists a schedule for topic: '
                         f'{msg_sensor_topic}. '
                         f'Timer and schedule will be deleted')
            self.stop_timer_get_kwargs(msg_sensor_topic)

        # The deviation control of the target value is stopped when
        # received a setpoint messages.
        # if self.dict_sensor_topic_to_data[msg_sensor_topic]['subscribed']:
        #     logger.debug(f'The controller-template is still subscribed to topic '
        #                  f'{msg_sensor_topic}. '
        #                  f'Subscription will be terminated')
        #     self.stop_mqtt_sensor_subscription(msg_sensor_topic)

        # Subscribe to measured values for the object we're controlling
        if not self.dict_sensor_topic_to_data[msg_sensor_topic][
            'subscribed']:
            # Subscribe to measured values for the object we're controlling
            self.start_mqtt_sensor_subscription(
                sensor_topic=msg_sensor_topic)

        self.update_actuator_value(msg_actuator_topic, msg_setpoint)

    def disconnect(self):
        '''
        Shutdown gracefully
        -> Disconnect from broker and stop background loop.
        '''
        self.client.disconnect()
        self.client.loop_stop()
        # Remove the client, so init can establish a new connection.
        del self.client

    def update_current_values(self,
                              actuator_topic: str,
                              sensor_topic: str,
                              schedule: dict):
        '''
        Method is called when a timer has expired. First, the instruction
        from the schedule is processed. Then a new timer obj is started
        that initiates the execution of the next instruction of the schedule.

        Parameters
        ----------
        actuator_topic : str , required
            String containing the MQTT topic via which messages can
            subsequently be sent to the device connector
        sensor_topic :str , required
            String containing the MQTT topic via which actual values can
            be received for a subject to be controlled.
        schedule : dict , required
            A directory containing the individual instructions and
            associated timestamps. In short: the schedule
        '''

        logger.info(f'Timer expired.'
                    f'A new instruction is sent via the '
                    f'topic {actuator_topic}.')

        # remove timer obj from central dictionary
        if sensor_topic in self.dict_sensor_topic_to_data.keys():
            self.dict_sensor_topic_to_data[sensor_topic]['timer'] = None

        # update device connector via mqtt
        # extract the current instruction and send it via method update_actuator_value
        timestamp_now = self.timestamp_now()
        next_timestamp = next(iter(schedule))
        next_instruction = schedule.pop(next_timestamp)
        self.update_actuator_value(actuator_topic=actuator_topic,
                                   value=next_instruction)

        # check if schedule contains further elements to set new time
        if bool(schedule):
            next_timestamp = next(iter(schedule))

            # Prepare new timer object for the next instruction
            delay_ms = (next_timestamp - timestamp_now)
            delay_s = delay_ms / 1000.
            timer_kwargs = {
                "actuator_topic": actuator_topic,
                "sensor_topic": sensor_topic,
                "schedule": schedule,
            }

            timer = Timer(
                interval=delay_s,
                function=self.update_current_values,
                kwargs=timer_kwargs
            )
            timer.start()
            self.add_timer(sensor_topic, timer)
        else:
            logger.info('Last period of schedule is running. '
                        'This last target value stays valid until reset by new schedule or setpoint.')


    def start_mqtt_sensor_subscription(self, sensor_topic: str):
        '''
        Method starts a mqtt subscription the given sensor topic. In addition,
        the method ensures that the central directory remains up to date.

        Parameters
        ----------
        sensor_topic : required , str
            A string with the MQTT topic via which we can receive sensor
            data from a device controller-template.
        '''

        logger.info(f'Start listening to sensor topic {sensor_topic}')

        # Check whether the central directory already knows the sensor
        # topic. If it is missing, it will be included with default values.
        if sensor_topic not in self.dict_sensor_topic_to_data.keys():
            logger.warning(f'Supposed to start a subscription to '
                           f'topic {sensor_topic}.'
                           f'However, this topic has not yet been saved '
                           f'in the internal data structure.')
            initial_dict = {
                'timer': None,
                'actuator_topic': '',
                'target_value': math.nan,
                'target_since': None,
                'subscribed': False,
                'deviation_reported': False
            }
            self.dict_sensor_topic_to_data[sensor_topic] = initial_dict

        # Set the "Subscribed" flag to avoid double subscriptions
        self.dict_sensor_topic_to_data[sensor_topic]['subscribed'] = True
        self.client.subscribe(topic=sensor_topic,
                              qos=1)

    def stop_mqtt_sensor_subscription(self, sensor_topic: str):
        '''
        method terminates the MQTT subscription associated
        with the passed topic.

        Parameters
        ----------
        sensor_topic : required , str
            A string with an MQTT topic via which we no longer
            want to receive MQTT messages.
        '''

        logger.info(f'Stop listening to sensor topic {sensor_topic}')

        # Check whether the central directory already knows the sensor
        # topic. If it is missing, it will be included with default values.
        if sensor_topic not in self.dict_sensor_topic_to_data.keys():
            logger.warning(f'Supposed to stop a subscription to '
                           f'topic {sensor_topic}. '
                           f'However, this topic has not yet been saved '
                           f'in the internal data structure.')
            initial_dict = {
                'timer': None,
                'actuator_topic': '',
                'target_value': math.nan,
                'target_since': None,
                'subscribed': False,
                'deviation_reported': False
            }
            self.dict_sensor_topic_to_data[sensor_topic] = initial_dict


        # Reset the target value and associated variables
        # self.dict_sensor_topic_to_data[sensor_topic]['target_value'] = math.nan
        # self.dict_sensor_topic_to_data[sensor_topic]['target_since'] = None
        # self.dict_sensor_topic_to_data[sensor_topic]['deviation_reported'] = False

        self.client.unsubscribe(topic=sensor_topic)
        # Release the "Subscribed" flag so that the controller can
        # subscribe to the topic again later.
        self.dict_sensor_topic_to_data[sensor_topic]['subscribed'] = False

    def update_actuator_value(self, actuator_topic: str, value: typing.Union[int, float, str]):
        '''
        Method updates the target value and associated variables in the central dictionary and
        sends the passed value (msg) via the passed MQTT topic (actuator_topic) to a device connector.

        Parameters
        ----------
        actuator_topic : str , required
            MQTT topic via which the message is to be sent.

        value :  , required
            Value (string, float or int) to be sent.

        '''

        # Save the target value in the central dictionary so that a later
        # target/actual comparison is possible.
        corr_sensor_topic = self.dict_actuator_to_sensor_topic[actuator_topic]
        self.dict_sensor_topic_to_data[corr_sensor_topic]['target_value'] = value
        self.dict_sensor_topic_to_data[corr_sensor_topic]['deviation_reported'] = False

        formatted_msg = {'value': value}

        logger.info(f'Send actuator value {value} via topic {actuator_topic}')
        self.client.publish(
            payload=json.dumps(formatted_msg),
            topic=actuator_topic,
            retain=False,
        )
        # Store the current timestamp to keep track of the time passed since sending the target value
        self.dict_sensor_topic_to_data[corr_sensor_topic]['target_since'] = self.timestamp_now()

    def add_timer(self, sensor_topic: str, timer: Timer):
        """
        Add a timer the collecting object.

        Parameters:
        -----------
        sensor_topic : str , required
            MQTT topic, via which sensor values are received for
            the domain to be controlled.

        timer : Threading.Timer object , required
            The timer object after it has been started.
        """
        logger.info(f'Add a timer obj for topic {sensor_topic} to internal'
                    f'data structure.')

        # Check whether the central directory already knows the sensor
        # topic. If it is missing, it will be included with default values.
        if sensor_topic not in self.dict_sensor_topic_to_data.keys():
            logger.warning(f'Supposed to add a time to  '
                           f'sensor topic {sensor_topic}.'
                           f'However, this topic has not yet been saved '
                           f'in the internal data structure.')
            initial_dict = {
                'timer': None,
                'actuator_topic': '',
                'target_value': math.nan,
                'target_since': None,
                'subscribed': False,
                'deviation_reported': False
            }
            self.dict_sensor_topic_to_data[sensor_topic] = initial_dict

        # Save the timer in the central directory so that it can be
        # accessed when a new schedule is received from the producer.
        self.dict_sensor_topic_to_data[sensor_topic]['timer'] = timer

    def stop_timer_get_kwargs(self, sensor_topic: str):
        '''
        method removes the timer associated with the passed sensor topic
        from the central directory. After the timer is removed, it
        extracts its kwargs and returns them to the caller.

        Parameters:
        -----------
        sensor_topic : str , required
            MQTT topic, via which sensor values are received for
            the domain to be controlled.
        '''

        if sensor_topic in self.dict_sensor_topic_to_data.keys():
            # get timer obj and remove it from the storing dict
            timer = self.dict_sensor_topic_to_data[sensor_topic]['timer']
            timer.cancel()
            timer_kwargs = timer.kwargs
            self.dict_sensor_topic_to_data[sensor_topic]['timer'] = None
        else:
            # If no timer was found for the given sensor topic,
            # only an empty directory is returned.
            timer_kwargs = {}

        return timer_kwargs

    def convert_str_timestamps(self, schedule: dict):
        '''
        method converts the ISO-timestamps within a schedule into milliseconds.
        The timestamp must comply with a defined format
        (see Parameters > schedule)

        Parameters:
        -----------
        schedule : dict , required
            A directory representing a schedule. Example follows at the end.
            The timestamps must be encoded as ISO-format strings
            - timestamps : str
                - %Y-%m-%dT%H:%M:%S%z
                - 2021-06-18T09:29:30+02:00
                - YYYY-MM-DDTHH:MM:SS+tz
            - schedule structure
                schedule : { iso_timestamp: value,
                             iso_timestamp: value,
                             ...
                            }

        Returns:
        -----------
        dict_result : dict
            The input directory, with the difference that the timestamps
             are now unix timestamps (in milliseconds).
        '''

        dict_result = {}
        logger.debug(f'schedule contains the following datetime ts: {schedule.keys()}')
        for str_timestamp in schedule.keys():
            dt_ts = datetime.fromisoformat(str_timestamp)
            second_ts = datetime.timestamp(dt_ts)
            milli_secs_ts = round(second_ts * 1000)
            dict_result[milli_secs_ts] = schedule[str_timestamp]

        return dict_result


if __name__ == "__main__":
    controller = Controller()
