import datetime
import json
import logging
import os
import time
import typing

import pika
from pika.exceptions import StreamLostError, AMQPConnectionError, ChannelClosedByBroker, ConnectionClosedByBroker
import requests

from paho.mqtt.client import Client
from dotenv import load_dotenv, find_dotenv

from db import db_helper as db
from dispatch import DispatchOnce, DispatchInInterval

# Log everything to stdout by default, i.e. to docker container logs.
LOGFORMAT = "[%(asctime)s - %(name)s - %(module)s - %(levelname)s]:  %(message)s"
logging.basicConfig(format=LOGFORMAT, level=logging.DEBUG)
logger = logging.getLogger("bem-connector-interface")
# Pika's logging goes nuts – better not set a level above warning
logging.getLogger("pika").setLevel(logging.WARNING)


class ConnectorInterface():

    def __init__(self):

        logger.info('Initiating Connector')
        self._DeviceDispatcher = DispatchInInterval
        self._MqttClient = Client
        self._RabbitMQClient = pika.BlockingConnection

        logger.info('Loading settings from environment variables')

        load_dotenv(find_dotenv(), verbose=True, override=False)
        self.MQTT_BROKER_HOST = os.getenv('MQTT_BROKER_HOST')
        self.MQTT_BROKER_PORT = int(os.getenv('MQTT_BROKER_PORT'))
        self.RABBIT_MQ_HOST = os.getenv('RABBITMQ_HOSTNAME')
        self.RABBIT_MQ_EXCHANGE = os.getenv('RABBITMQ_BEM_CONTROL')
        self.DEPEND_ON_RABBIT_MQ = (os.getenv('DEPEND_ON_RABBIT_MQ', 'True').lower() in ('true', '1', 't'))
        self.HEARTBEAT_INTERVAL = int(os.getenv('HEARTBEAT_INTERVAL'))
        self.DEBUG = (os.getenv('DEBUG', 'True').lower() in ('true', '1', 't'))
        self.DEBUG_INTERFACE_ONLY = (os.getenv('DEBUG_INTERFACE_ONLY', 'True').lower() in ('true', '1', 't'))
        self.READ_ONLY = (os.getenv('READ_ONLY', default='False').lower() in ('true', '1', 't'))

        logger.debug(f'Import configuration files from two of these: {os.listdir("./config")}')
        # load "internal to MQTT topic" dictionary
        _path_internal_to_mqtt_maps = './config/Internal_to_MQTT.json'
        with open(_path_internal_to_mqtt_maps, 'r') as read_file:
            self.dict_intern_to_mqtt = json.load(read_file)

        # load "internal to MQTT admin topic" dictionary
        _path_internal_to_mqtt_admin_maps = './config/mqtt_devices.json'
        with open(_path_internal_to_mqtt_admin_maps, 'r') as read_file:
            self.dict_intern_to_mqtt_admin: typing.Dict[str, dict] = json.load(read_file)

        # derive data structure dict_mqtt_sensor_to_intern from dict_intern_to_mqtt
        self.dict_mqtt_sensor_to_intern = self.get_mqtt_sensor_to_intern()

        # derive topic->device mappings from dict_intern_to_mqtt_admin for errors and deviations
        self.dict_mqtt_deviation_to_intern = self.get_dict_mqtt_deviation_to_intern()
        self.dict_mqtt_error_to_intern = self.get_dict_mqtt_error_to_intern()
        # derive device->controller topic mapping from dict_intern_to_mqtt_admin
        self.dict_intern_to_controller_topics = self.get_dict_intern_to_controller_topics()

        # Init class variables for communication
        self.mqtt_client = None
        self.mqtt_subscribed = False
        self.rabbitmq_connection = None
        self.rabbitmq_channel = None

        # Set the log level according to the DEBUG flag.
        loggers = logging.root.manager.loggerDict
        if self.DEBUG:
            for logger_name in loggers:
                if "pika" not in logger_name:
                    logger.warning(f"Setting log level to DEBUG")
                    logging.getLogger(logger_name).setLevel(logging.DEBUG)
        else:
            logger.warning(f"Setting log level to INFO")
            for logger_name in loggers:
                if "pika" not in logger_name:
                    logging.getLogger(logger_name).setLevel(logging.INFO)

        if self.DEBUG_INTERFACE_ONLY:
            logger.warning(f"Setting log level of {logger.name} to DEBUG")
            logger.setLevel(level=logging.DEBUG)

        logger.info("Finished connector interface init.")

    def run(self):
        '''
        Method provides for the internal and external connections.
        Internally, the method connects to a RabbitMQ exchange to receive
        control commands for the external devices. Externally, the method
        connects via an MQTT broker. The MQTT broker can be used to receive
        sensor data from the devices and to send control data to the devices.
        '''

        logger.debug("Entering Connector-Interface run method")
        # Setup and configure connection with MQTT message broker.
        # Also wire through a reference to the Connector instance (self)
        # as this allows _handle_incoming_mqtt_msg to call Connector methods.
        logger.info("Configuring MQTT connection")
        self.mqtt_client = self._MqttClient(userdata={"self": self})
        self.mqtt_client.on_message = self._handle_incoming_mqtt_msg
        self.mqtt_client.on_connect = self.subscribe_mqtt_topics
        self.mqtt_client.connect(
            host=self.MQTT_BROKER_HOST, port=self.MQTT_BROKER_PORT
        )

        # Execute the MQTT main loop in a dedicated thread. This is
        # similar to use loop_start of paho mqtt but allows us to use a
        # unfied concept to check whether all background processes are
        # still alive.
        def mqtt_worker(mqtt_client):
            try:
                mqtt_client.loop_forever()
            finally:
                # Gracefully terminate connection once the main program exits.
                mqtt_client.disconnect()

        logger.debug("Starting broker dispatcher with MQTT client loop.")
        broker_dispatcher = DispatchOnce(
            target_func=mqtt_worker,
            target_kwargs={"mqtt_client": self.mqtt_client}
        )
        broker_dispatcher.start()

        # This collects all running dispatchers. These are checked for health
        # in the main loop below.
        dispatchers = [broker_dispatcher]

        # Subscribe to all MQTT sensor and deviation topics
        #self.subscribe_mqtt_topics()

        # Only create a rabbitMQ client if it is to be a read-write interface
        if not self.READ_ONLY:
            # Setup and configure connection with RabbitMQ Exchange.
            logger.info("Configuring RabbitMQ connection")

            try:
                # Establish connection to RabbitMQ server
                self.rabbitmq_connection = self._RabbitMQClient(
                    pika.ConnectionParameters(
                        host=self.RABBIT_MQ_HOST,
                        connection_attempts=30, retry_delay=2
                    ))

                self.rabbitmq_channel = self.rabbitmq_connection.channel()

                # Declaring exchanges is idempotent. Hence declare it here in case the orchestration service
                # has not run yet, which usually declares all exchanges
                self.rabbitmq_channel.exchange_declare(
                    exchange=self.RABBIT_MQ_EXCHANGE, exchange_type='topic', durable=True
                )

                # Subscribe to all required topics of the RabbitMQ topic exchange
                self.subscribe_rabbitmq_topics(
                    exchange=self.RABBIT_MQ_EXCHANGE,
                    on_message=self._handle_incoming_rabbitmq_msg
                )

                # Execute the RabbitMQ consuming loop in a dedicated thread.
                def rabbitmq_worker(channel, connection):
                    try:
                        channel.start_consuming()
                    except (StreamLostError, ConnectionClosedByBroker):
                        logger.error(f"RabbitMQ broker is down.")
                    finally:
                        logger.error(f"RabbitMQ connection is closed: {connection.is_closed}.")
                        if not connection.is_closed:
                            # Gracefully terminate connection once the main program exits.
                            connection.close()

                logger.debug('Starting RabbitMQ dispatcher with consuming channel.')
                rabbit_dispatcher = DispatchOnce(
                    target_func=rabbitmq_worker,
                    target_kwargs={'channel': self.rabbitmq_channel,
                                   'connection': self.rabbitmq_connection}
                )
                rabbit_dispatcher.start()

                # This list collects all running dispatchers. These are checked for
                # health in the main loop below.
                dispatchers.append(rabbit_dispatcher)

            except:
                logger.error("RabbitMQ connection could not be set up.")
                if self.DEPEND_ON_RABBIT_MQ:
                    logger.error(f"Restarting connector interface, "
                                 f"because DEPEND_ON_RABBIT_MQ is set to {self.DEPEND_ON_RABBIT_MQ}.")
                    raise
                else:
                    logger.error("Starting connector interface without RabbitMQ connection, "
                                 f"because DEPEND_ON_RABBIT_MQ is set to {self.DEPEND_ON_RABBIT_MQ}. "
                                 "Restart connector interface manualley to create connection when RabbitMQ broker is up again.")
        else:
            logger.info(f'No RabbitMQ client has been set up.')

        # Start the main loop which we spend all the operation time in.
        logger.info('Connector interface online. Entering main loop.')
        try:
            while True:
                logger.info('Round and round it goes')
                # Check that all dispatchers are alive, and if this is the
                # case assume that the connector operations as expected.
                if not all([d.is_alive() for d in dispatchers]):
                    # If one is not alive, see if we encountered an exception
                    # and raise it, as exceptions in threads are not
                    # automatically forwarded to the main program.
                    for d in dispatchers:
                        if d.exception is not None:
                            raise d.exception
                    # If no exception is found raise a custom on.
                    raise RuntimeError(
                        'At least one dispatcher thread is not alive, but no '
                        'exception was caught.'
                    )

                    break

                if not self.mqtt_client.is_connected():
                    # Connection was/is lost
                    logger.warning(f"Connection to MQTT broker is lost.")
                    self.mqtt_subscribed = False

                time.sleep(self.HEARTBEAT_INTERVAL)

        except (KeyboardInterrupt, SystemExit):
            # This is the normal way to exit the Connector. No need to log the
            # exception.
            logger.info(
                'Connector received KeyboardInterrupt or SystemExit'
                ', shutting down.'
            )
        except:
            # This is execution when something goes really wrong.
            logger.exception(
                "Connector main loop has caused an unexpected exception. "
                "Shutting down."
            )
        finally:
            for dispatcher in dispatchers:
                # Ask the dispatcher (i.e. thread) to quit and give it
                # one second to execute any cleanup. Anything that takes
                # longer will be killed hard once the main program exits
                # as the dispatcher thread is expected to be a daemonic
                # thread.
                logger.debug("Terminating dispatcher %s", dispatcher)
                if dispatcher.is_alive():
                    dispatcher.terminate()
                dispatcher.join(1)
            logger.info("Connector shut down completed. Good bye.")

    @staticmethod
    def _handle_incoming_mqtt_msg(client, userdata, msg):
        '''
        Method will be executed when a message is received via MQTT.
        The message must contain one and only one of the following
        three information types:
            - Deviation         (sent by the controller-template)
            - Sensor reading    (sent by the connector)
            - Error message     (sent by the connector)

        Parameters
        ----------
        client :   , required
        userdata :   , required
        msg :   , required
        '''

        self = userdata["self"]
        msg_sender = msg.topic.split('/')[0]
        msg_kind = msg.topic.split('/')[1]
        try:
            msg_feature = msg.topic.split('/')[2]
            msg_feature_part = f", feature: {msg_feature}"
        except IndexError:
            msg_feature_part = ""
        logger.debug(f'Handling incoming MQTT message from {msg_sender} '
                     f'with kind: {msg_kind}{msg_feature_part}')

        if msg.topic in self.dict_mqtt_sensor_to_intern.keys():
            # Handle a sensor reading message
            self.run_mqtt_sensor_flow(
                topic=msg.topic,
                value_msg_json=msg.payload
            )
        elif msg.topic in self.dict_mqtt_deviation_to_intern.keys():
            # Handle a deviation message
            self.run_mqtt_deviation_flow(
                topic=msg.topic,
                value_msg_json=msg.payload
            )
        elif msg.topic in self.dict_mqtt_error_to_intern.keys():
            # Handle an error message
            self.run_mqtt_error_flow(
                topic=msg.topic,
                value_msg_json=msg.payload
            )
        else:
            logger.debug("Ignored incoming MQTT msg on topic: %s",
                         msg.topic)

    def _handle_incoming_rabbitmq_msg(self, ch, method, properties, body):
        '''
        Method will be executed when a message is received via RabbitMQ
        and via the routing key '*.schedule'. The received message will
        be deserialized and passed to the run_actuator_flow method for
        further processing.

        Parameters
        ----------
        ch : pika.Channel , required
        method :  , required
        properties :  , required
        body :  , required
        '''

        msg_payload = json.loads(body)
        msg_receiver = method.routing_key.split('.')[0]  # eg. 'wallbox' or 'bess'
        msg_kind = method.routing_key.split('.')[1]  # {schedule, setpoint}

        logger.debug(f'RabbitMQ: Message received for topic {method.routing_key}: '
                     f'{body} -> data: {msg_payload}')

        if msg_kind.lower() == 'schedule':
            try:
                self.run_rabbitmq_schedule_flow(msg_receiver=msg_receiver,
                                                schedules=msg_payload)
            except KeyError:
                logger.debug(f'Receiver "{msg_receiver}" of the {msg_kind.lower()} message is unknown.'
                             f' Message is discarded.')

        elif msg_kind.lower() == 'setpoint':
            try:
                self.run_rabbitmq_setpoint_flow(msg_receiver=msg_receiver,
                                                setpoints=msg_payload)
            except KeyError:
                logger.debug(f'Receiver "{msg_receiver}" of the {msg_kind.lower()} message is unknown.'
                             f' Message is discarded.')
        else:
            logger.debug(f'Ignored incoming RabbitMQ msg with '
                         f'routing key: {method.routing_key}')

    def run_rabbitmq_schedule_flow(self, msg_receiver: str, schedules: typing.Dict[str, dict]):
        '''
        The method processes a schedule msg received by RabbitMQ.
        These are commands that have been created within the BEMS.
        These commands are to be transmitted to the connected devices.

        Parameters
        ----------
        msg_receiver : str , required
            E.g. msg_receiver = 'wallbox'
        schedules : dict, required
            Data, which is to be transmitted to the device's controller
            E.g.:
            {
                "active_power": {
                    "2021-06-04T10:00:00+00:00": 3592,
                    ...
                    "2021-06-04T10:45:00+00:00": 2855
                }
            }
        '''

        for feature in schedules:
            actuator_topic = self.dict_intern_to_mqtt[msg_receiver][feature]['actuator']
            sensor_topic = self.dict_intern_to_mqtt[msg_receiver][feature]['sensor']
            controller_topic = self.dict_intern_to_controller_topics[msg_receiver] + "/" + os.getenv(
                'SCHEDULE_MESSAGE_SUBTOPIC')

            msg = {'actuator_topic': actuator_topic,
                   'sensor_topic': sensor_topic,
                   'schedule': schedules[feature]
                   }

            self.mqtt_client.publish(
                payload=json.dumps(msg),
                topic=controller_topic,
                retain=False,
            )
            logger.debug(f"MQTT: Sent message {msg} to {msg_receiver}'s controller "
                         f"via topic {controller_topic}")

    def run_rabbitmq_setpoint_flow(self, msg_receiver: str, setpoints: typing.Dict[str, int]):
        """
        The method processes a setpoint msg received by RabbitMQ.
        These are commands that have been created within the GEMS system.
        These commands are to be transmitted to the connected devices.

        Parameters
        ----------
        msg_receiver : str , required
            E.g. msg_receiver = 'wallbox'
        setpoints : dict, required
            Data, which is to be transmitted to the device's controller
            E.g.:
            { "active_power": 3592, 'reactive_power': 3000}

        """

        for feature in setpoints:
            actuator_topic = self.dict_intern_to_mqtt[msg_receiver][feature]['actuator']
            sensor_topic = self.dict_intern_to_mqtt[msg_receiver][feature]['sensor']
            controller_topic = self.dict_intern_to_controller_topics[msg_receiver] + "/" + os.getenv(
                'SETPOINT_MESSAGE_SUBTOPIC')

            msg = {'actuator_topic': actuator_topic,
                   'sensor_topic': sensor_topic,
                   'setpoint': setpoints[feature]}

            self.mqtt_client.publish(
                payload=json.dumps(msg),
                topic=controller_topic,
                retain=False,
            )

            logger.debug(f"MQTT: Sent message {msg} to {msg_receiver}'s controller "
                         f"via topic {controller_topic}")

    def run_mqtt_sensor_flow(self, topic: str, value_msg_json: str):
        '''
        The method processes messages passed by the callback method for
        MQTT messages. The MQTT message is adapted so that it can be stored
        in the Influx DB and is subsequently submitted to this same database.

        Following information will be passed to the Influx DB:
            db_source_data : str
                Name of the device (or connector) that emitted the date.
                E.g.: inverter, wallbox, ...

            db_payload_data : str
                Date which was transmitted with contextual information.
                Formatting:
                    {timestamp: {datapoint: reading}}

        Parameters
        ----------
        topic : str , required
            The MQTT topic where the date was received.
        value_msg_json : str , required
            The date what was received on the MQTT topic. Made of a
            JSON string that can be converted to a dictionary.
        '''

        if not (topic in self.dict_mqtt_sensor_to_intern.keys()):
            logger.debug(f"MQTT: Receive unknown topic: {topic}")
            return

        msg_sender_device = self.dict_mqtt_sensor_to_intern[topic]['device_id']  # eg. wallbox
        msg_feature = self.dict_mqtt_sensor_to_intern[topic]['feature']  # eg. active_power

        value_msg = json.loads(value_msg_json)
        datapoint_value = value_msg['value']
        try:
            # Try to convert the value from string to float
            datapoint_value = float(datapoint_value)
        except (ValueError, TypeError):
            pass
        datapoint_timestamp = value_msg['timestamp']  # Unix timestamp in milliseconds

        db_source_data = msg_sender_device
        db_payload_data = {
            datapoint_timestamp: {msg_feature: datapoint_value}}

        try:
            db.save_measurement(db_source_data, db_payload_data)
            logger.debug(f'DB: Stored msg {db_payload_data} from device '
                         f'{db_source_data} in db')
        except requests.exceptions.ConnectionError:
            logger.warning('InfluxDB is not up. Could not save the data.')

    def run_mqtt_error_flow(self, topic: str, value_msg_json: str):
        '''
        Text

        #TODO: Comment

        Following information will be passed to the Influx DB:
            db_source_data : str
                Name of the device (or connector) that emitted the date.
                E.g.: inverter, wallbox, ...

            db_payload_data : str
                Date which was transmitted with contextual information.
                Formatting:
                    {timestamp: {datapoint: reading}}

        Parameters
        ----------
        topic : str , required
            The MQTT topic where the date was received.
        value_msg_json : str , required
            The date what was received on the MQTT topic. Made of a
            JSON string that can be converted to a dictionary.
        '''

        logger.info('Ignored MQTT error msg (due to the read-only version)')
        '''
        COMMENTED OUT IN FAVOUR OF A READ-ONLY INTERFACE
        list_topic = topic.split('/')

        # Check if the topic is valid
        if not (len(list_topic) == 2):
            logger.debug(f'MQTT: Receive an invalid error topic: {topic}')
            return
        elif list_topic[1] != 'error':
            logger.debug(f'MQTT: Receive an invalid error topic: {topic}')
            return

        msg_sender_device = list_topic[0]   # eg. wallbox
        msg_error = list_topic[1]           # = error

        dict_error_msg = json.loads(value_msg_json)
        dict_error_msg_info = dict_error_msg['error']
        dict_error_msg_info_ts = dict_error_msg_info['timestamp']

        db_source_data = msg_sender_device
        db_payload_data = {
            dict_error_msg_info_ts: {msg_error: dict_error_msg_info}}

        db.save_measurement(db_source_data, db_payload_data)
        logger.debug(f'DB: Stored error msg from connector '
                     f'{db_source_data} in db')
        '''

    def run_mqtt_deviation_flow(self, topic: str, value_msg_json: str):
        '''
        Forward the deviation message to the RabbitMQ exchange

        Parameters
        ----------
        topic : str , required
            The MQTT topic where the message was received.
        value_msg_json : str , required
            The data what was received on the MQTT topic. Made of a
            JSON string that can be converted to a dictionary.

            E.g.:
            deviation_payload = {
                "feature": "active_power",
                'sensor_value': 4236,
                'target_value': 5000,
                'target_since': 1637241516425,
                'timestamp': 1637241536425
            }
        '''

        if not (topic in self.dict_mqtt_deviation_to_intern.keys()):
            logger.debug(f"MQTT: Received unknown topic: {topic}")
            return

        device_id = self.dict_mqtt_deviation_to_intern[topic]['device_id']  # e.g. evse
        msg_feature = self.dict_mqtt_deviation_to_intern[topic]['feature']  # = deviation
        deviation_msg = json.loads(value_msg_json)
        # Convert timestamps in milliseconds to ISO-format strings
        fields_with_ts = ['target_since', 'timestamp']
        for field in fields_with_ts:
            try:
                deviation_msg[field] = datetime.datetime.fromtimestamp(
                    deviation_msg[field] / 1000, tz=datetime.timezone.utc
                ).isoformat(timespec='seconds')
            except KeyError:
                logger.error(f"Key {field} not found in deviation message: {deviation_msg}")

        # Publish on RabbitMQ exchange
        # Establish connection to RabbitMQ server
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.RABBIT_MQ_HOST))
        try:
            channel = connection.channel()
            # Publish message without any transformation to the corresponding RabbitMQ topic (routing key)
            payload = json.dumps(deviation_msg)
            logger.info(
                f'Publish deviation message from {device_id} controller to RabbitMQ exchange "{self.RABBIT_MQ_EXCHANGE}": {payload}.')
            channel.basic_publish(exchange=self.RABBIT_MQ_EXCHANGE,
                                  routing_key=f'{device_id}.{msg_feature}',
                                  body=payload)

            logger.debug(f'Deviation message forwarded: {deviation_msg}')

        finally:
            connection.close()

    ####################################################################
    #                                                                  #
    #      |||      Methods that support initialization      |||       #
    #      VVV                                               VVV       #
    ####################################################################

    def get_mqtt_sensor_to_intern(self):
        '''
        Method converts the dictionary dict_intern_to_mqtt so that the
        internal identifiers (device id and feature) can be derived
        from MQTT sensor topics.

        dict_intern_to_mqtt = {
            <device_id 01>: {
                <feature 01>: {
                    "sensor": <mqtt sensor topic>,
                    "actuator": <mqtt actuator topic>
                },
                <feature 02>: {
                    "sensor": <mqtt sensor topic>,
                    "actuator": <mqtt actuator topic>
                },
                ...
            },
            <device_id 02>: {
            ...
            }
        }

        dict_mqtt_sensor_to_intern = {
            <mqtt sensor topic 01>: {
                "device_id": <device_id 01>,
                "feature": <feature 01>
            },
            <mqtt sensor topic 02>: {
                "device_id": <device_id 01>,
                "feature": <feature 02>
            },
            ...
            <mqtt sensor topic N>: {
                "device_id": <device_id M>,
                "feature": <feature P>
            }
        }
        '''

        dict_mqtt_sensor_to_intern = {}

        for curr_device in self.dict_intern_to_mqtt.keys():

            for curr_feature in self.dict_intern_to_mqtt[curr_device]:
                curr_mqtt_sensor = self.dict_intern_to_mqtt[curr_device][curr_feature]['sensor']

                # check whether there is a MQTT-Sensor-Topic for the feature (curr_feature)
                if not curr_mqtt_sensor:
                    # if there is none -> continue with next feature
                    continue

                dict_mqtt_sensor_to_intern[curr_mqtt_sensor] = {}
                dict_mqtt_sensor_to_intern[curr_mqtt_sensor]['device_id'] = curr_device
                dict_mqtt_sensor_to_intern[curr_mqtt_sensor]['feature'] = curr_feature

        return dict_mqtt_sensor_to_intern

    def get_dict_mqtt_deviation_to_intern(self):
        '''
        Method converts the dictionary dict_intern_to_mqtt so that the
        internal identifiers (device id and feature) can be derived
        from MQTT sensor topics.

        dict_intern_to_mqtt_admin = {
            <device_id 01>: {
                "controller": <device_id 01>/controller,
                "error": <device_id 01>/error,
                "deviation": <device_id 01>/controller/deviation,
            },
            <device_id 02>: {
                "controller": <device_id 02>/controller,
                "error": <device_id 02>/error,
                "deviation": <device_id 02>/controller/deviation,
            },
            ...
        }

        dict_mqtt_deviation_to_intern = {
            <device_id 01>/controller/deviation: {
                       "device_id": <device_id 01>,
                       "feature": "deviation"
            },
            <device_id 02>/controller/deviation: {
                       "device_id": <device_id 02>,
                       "feature": "deviation"
            },
            ...
            <device_id N>/controller/deviation: {
                       "device_id": <device_id N>,
                       "feature": "deviation"
            },
        }
        '''

        dict_mqtt_deviation_to_intern = {}

        for device in self.dict_intern_to_mqtt_admin.keys():
            deviation_topic = self.dict_intern_to_mqtt_admin[device]['deviation']
            assert deviation_topic.split('/')[-2:] == ['controller', os.getenv('DEVIATION_MESSAGE_SUBTOPIC')], \
                f"The defined deviation topic is invalid. " \
                f"Defined topic: {deviation_topic}" \
                f"Required structure: <device_key>/controller/{os.getenv('DEVIATION_MESSAGE_SUBTOPIC')} "
            dict_mqtt_deviation_to_intern[deviation_topic] = {'device_id': device,
                                                              'feature': os.getenv('DEVIATION_MESSAGE_SUBTOPIC')}

        return dict_mqtt_deviation_to_intern

    def get_dict_mqtt_error_to_intern(self):
        '''
        Method converts the dictionary dict_intern_to_mqtt so that the
        internal identifiers (device id and feature) can be derived
        from MQTT error topics.

        dict_intern_to_mqtt_admin = {
            <device_id 01>: {
                "controller": <mqtt controller topic 01>,
                "error": <mqtt error topic 01>,
                "deviation": <mqtt deviation topic 01>,
            },
            <device_id 02>: {
                "controller": <mqtt controller topic 02>,
                "error": <mqtt error topic 02>,
                "deviation": <mqtt deviation topic 02>,
            },
            ...
        }

        dict_mqtt_error_to_intern = {
            <mqtt error topic 01>: {
                       "device_id": <device_id 01>,
                       "feature": "error"
            },
            <mqtt error topic 02>: {
                       "device_id": <device_id 02>,
                       "feature": "error"
            },
            ...
            <mqtt error topic N>: {
                       "device_id": <device_id N>,
                       "feature": "error"
            },
        }
        '''

        dict_mqtt_error_to_intern = {}

        for curr_device in self.dict_intern_to_mqtt_admin.keys():
            curr_error_topic = self.dict_intern_to_mqtt_admin[curr_device]['error']

            dict_mqtt_error_to_intern[curr_error_topic] = {}
            dict_mqtt_error_to_intern[curr_error_topic]['device_id'] = curr_device
            dict_mqtt_error_to_intern[curr_error_topic]['feature'] = 'error'

        return dict_mqtt_error_to_intern

    def get_dict_intern_to_controller_topics(self) -> typing.Dict[str, str]:
        """
        This method extracts the controller topics from the dict_intern_to_mqtt_admin map for
        each device to return a simpler map with just the controller topics for each device.

        dict_intern_to_mqtt_admin = {
            <device_id 01>: {
                "controller": <mqtt controller topic 01>,
                "error": <mqtt error topic 01>,
                "deviation": <mqtt deviation topic 01>,
            },
            <device_id 02>: {
                "controller": <mqtt controller topic 02>,
                "error": <mqtt error topic 02>,
                "deviation": <mqtt deviation topic 02>,
            },
            ...
        }

        dict_intern_to_controller_topics = {
            <device_id 01>: <mqtt controller topic 01>,
            <device_id 02>: <mqtt controller topic 02>,
            ...
        }
        """
        dict_intern_to_controller_topics = {}
        for device, topics in self.dict_intern_to_mqtt_admin.items():
            dict_intern_to_controller_topics[device] = topics["controller"]
        return dict_intern_to_controller_topics

    ####################################################################
    #                                                                  #
    #  |||  Methods that support the operation of the connector  |||   #
    #  VVV                                                       VVV   #
    ####################################################################

    def subscribe_mqtt_topics(self, client, userdata, flags, rc):
        logger.info('Connected to MQTT broker.')
        self.subscribe_all_sensor_topics()
        self.subscribe_all_deviation_topics()
        self.mqtt_subscribed = True

    def subscribe_all_sensor_topics(self):
        '''
        The method registers all sensory topics against the MQTT broker.
        Sensor data is received from the devices via the sensory topics in
        the course.

        dict_mqtt_sensor_to_intern = {
            <mqtt sensor topic 01>: {
                "device_id": <device_id 01>,
                "feature": <feature 01>
            },
            <mqtt sensor topic 02>: {
                "device_id": <device_id 01>,
                "feature": <feature 02>
            },
            ...
            <mqtt sensor topic N>: {
                "device_id": <device_id M>,
                "feature": <feature P>
            }
        }

        '''

        logger.debug("MQTT: Subscribe to all sensor topics")
        for curr_sensor_topic in self.dict_mqtt_sensor_to_intern.keys():
            curr_device_id = self.dict_mqtt_sensor_to_intern[curr_sensor_topic]['device_id']
            self.mqtt_client.subscribe(
                topic=curr_sensor_topic,
                qos=1
            )
            logger.debug(f'MQTT: Subscribed to the following topic at device '
                         f'{curr_device_id}: '
                         f'{curr_sensor_topic}')

    def subscribe_all_deviation_topics(self):
        '''
        The method registers all deviation topics against the MQTT broker.
        Deviation messages are received from the controller if deviations from schedule or setpoint values occur.

         dict_mqtt_deviation_to_intern = {
                <device_id 01>/controller/deviation: {
                           "device_id": <device_id 01>,
                           "feature": "deviation"
                },
                <device_id 02>/controller/deviation: {
                           "device_id": <device_id 02>,
                           "feature": "deviation"
                },
                ...
                <device_id N>/controller/deviation: {
                           "device_id": <device_id N>,
                           "feature": "deviation"
                },
            }

        '''

        logger.debug("MQTT: Subscribe to all deviation topics")
        for deviation_topic, info in self.dict_mqtt_deviation_to_intern.items():
            device_id = info['device_id']
            self.mqtt_client.subscribe(
                topic=deviation_topic,
                qos=1
            )
            logger.debug(f'MQTT: Subscribed to the following topic of device '
                         f'{device_id}: '
                         f'{deviation_topic}')

    def subscribe_rabbitmq_topics(self, exchange: str, on_message: typing.Callable):
        """
        Subscribe to topics of the given RabbitMQ topic exchange.
        :param exchange: Name of the topic exchange
        :param on_message: Callback function for received messages
        """
        # Declare queue, bind it to the exchange 'control' and start
        # consuming messages with schedules
        result = self.rabbitmq_channel.queue_declare(queue='',
                                                     exclusive=True)
        # Assign automatically generated queue name
        queue_name = result.method.queue

        schedules_topic = f'*.{os.getenv("SCHEDULE_MESSAGE_SUBTOPIC")}'
        logger.debug(f'RabbitMQ subscribe to {schedules_topic}')
        self.rabbitmq_channel.queue_bind(queue=queue_name,
                                         exchange=exchange,
                                         routing_key=schedules_topic)

        setpoints_topic = f'*.{os.getenv("SETPOINT_MESSAGE_SUBTOPIC")}'
        logger.debug(f'RabbitMQ subscribe to {setpoints_topic}')
        self.rabbitmq_channel.queue_bind(queue=queue_name,
                                         exchange=exchange,
                                         routing_key=setpoints_topic)

        self.rabbitmq_channel.basic_consume(
            queue=queue_name,
            on_message_callback=on_message,
            auto_ack=True)


if __name__ == "__main__":
    # logger = logging.getLogger('bem-connector-interface')
    # sh = logging.StreamHandler()
    # sh.setLevel(level=logging.DEBUG)
    # sh.setFormatter(logging.Formatter('[%(asctime)s - %(name)s - %(module)s - %(levelname)s]:  %(message)s'))
    # logger.addHandler(sh)
    # logging.getLogger("pika").setLevel(logging.WARNING)

    connector_interface = ConnectorInterface()
    connector_interface.run()
