#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import logging
import os
import typing
import asyncio
import json

from dotenv import load_dotenv, find_dotenv
from paho.mqtt.client import Client
from datetime import datetime, timezone

from e3dc import E3DC, rscpFindTag, SendError, rscpUpdateStateCodes
from dispatch.dispatch import DispatchOnce

# Log everything to stdout by default, i.e. to docker container logs.
LOGFORMAT = '%(asctime)s-%(funcName)s-%(levelname)s: %(message)s'
logging.basicConfig(format=LOGFORMAT, level=logging.DEBUG)
logger = logging.getLogger("e3dc-connector")


def timestamp_utc_now():
    """
    Returns the timestamp of the current UTC time in milliseconds.
    Rounded to full microseconds.
    """
    return round(datetime.now(tz=timezone.utc).timestamp() * 1000)


class E3DCConnector(E3DC):

    mode_map = {0: 'auto', 1: 'idle', 2: 'discharge', 3: 'charge'}

    def __init__(self,
                 MQTTClient=Client,
                 **kwargs):
        logger.debug(f'Create E3DC Connector with config from env')

        self._MqttClient = MQTTClient

        logger.debug("Loading settings from environment variables.")
        # dotenv allows us to load env variables from .env files which is
        # convient for developing. If you set override to True tests
        # may fail as the tests assume that the existing environ variables
        # have higher priority over ones defined in the .env file.
        load_dotenv(find_dotenv(), verbose=True, override=False)
        self.E3DC_IP = os.getenv("E3DC_IP")
        self.E3DC_USERNAME = os.getenv("E3DC_USERNAME")
        self.E3DC_PASSWORD = os.getenv("E3DC_PASSWORD")
        self.E3DC_KEY = os.getenv("E3DC_KEY")
        self.POLLING_FREQUENCY = int(os.getenv('POLLING_FREQUENCY'))
        self.SEND_COMMAND_FREQUENCY = int(os.getenv('SEND_COMMAND_FREQUENCY'))
        # Storage only accepts 0 when sending command for auto mode
        self.AUTO_MODE_POWER_VALUE = 0

        self.MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST")
        self.MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT"))
        self.MQTT_FREQUENCY = int(os.getenv("MQTT_FREQUENCY"))

        self.CONNECTOR_NAME = os.getenv("CONNECTOR_NAME")
        self.DEBUG = os.getenv("DEBUG")

        super().__init__(E3DC.CONNECT_LOCAL,
                         username=self.E3DC_USERNAME,
                         password=self.E3DC_PASSWORD,
                         ipAddress=self.E3DC_IP,
                         key=self.E3DC_KEY,
                         **kwargs)

        logger.debug('Set internal data structure')

        self.mode_map_rev = {v: k for k, v in self.mode_map.items()}

        # internal state management
        self.data_to_send = {
            'mode': self.mode_map_rev['auto'],
            'battery_active_power_value': self.AUTO_MODE_POWER_VALUE
        }
        # data structure maps internal sensor IDs to sensor MQTT topics
        #  and actuator MQTT topics to internal actuator IDs

        self.dict_mqtt_topic_internal_map = self.load_and_parse_mqtt_e3dc_map(
            path=os.getenv('MQTT_E3DC_MAP_PATH')
        )

        # data structure stores the latest measurement values
        # and the actuator target value
        self.dict_internal_to_value = {
            'sensor': {},
            'actuator': {
                'mode': self.mode_map_rev['auto'],
                'battery_active_power_value': self.AUTO_MODE_POWER_VALUE
            }
        }

        # Set the appropriate logging level
        if self.DEBUG != "TRUE":
            logger.debug("Changing log level to INFO")
            for logger_name in logging.root.manager.loggerDict:
                logging.getLogger(logger_name).setLevel(logging.INFO)

    def run(self):
        logger.info('Start polling and controlling the E3DC system.')

        # Setup and configure connection with MQTT message broker.
        # Also wire through a reference to the Connector instance (self)
        # as this allows _handle_incoming_mqtt_msg to call Connector methods.
        logger.debug('Configuring MQTT connection')
        self.mqtt_client = self._MqttClient(userdata={"self": self})
        self.mqtt_client.on_message = self._handle_incoming_mqtt_msg
        self.mqtt_client.connect(
            host=self.MQTT_BROKER_HOST,
            port=self.MQTT_BROKER_PORT
        )

        # Subscribe to all MQTT actuator topics
        logger.debug('Subscribe to all known actuator MQTT topics')
        for actuator_topic in self.dict_mqtt_topic_internal_map['actuator'].keys():
            # Ensure actuator messages are delivered once and only once.
            self.mqtt_client.subscribe(topic=actuator_topic,
                                       qos=2)

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

        logger.debug('Starting broker dispatcher with MQTT client loop.')
        broker_dispatcher = DispatchOnce(
            target_func=mqtt_worker,
            target_kwargs={'mqtt_client': self.mqtt_client}
        )
        broker_dispatcher.start()

        # This collects all running dispatchers. These are checked for health
        # in the main loop below.
        dispatchers = [broker_dispatcher]

        # The execution of the two communication methods happens in
        # one thread. Sleep times in both methods allow the
        # scheduler to run both methods within one thread.
        logger.debug('Starting methods for device communication')
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If function is not running in main process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            # Check MQTT dispatcher's status
            asyncio.ensure_future(self.check_dispatcher(dispatchers))

            asyncio.ensure_future(self.send_data_to_device_loop())
            asyncio.ensure_future(self.request_data_from_device_loop())

            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()

    async def request_data_from_device_loop(self):
        """
        Method request a set of data from the device's communicatin
        interface. The requested data is stored within this object.
        Lastly a method is called which publishs all mqtt data
        via MQTT
        """

        while True:
            try:
                # Request virtual SOC and current (dis)charge power separately
                vsoc = self.sendRequest(('EMS_REQ_BAT_SOC', 'None', None), keepAlive=True)[2]

                # Current (dis)charge power; negative -> discharge, positive -> charge
                battery_active_power = self.sendRequest(('EMS_REQ_POWER_BAT', 'None', None), keepAlive=True)[2]

                # Request all available battery information; virtual SOC and current (dis)charge power is not included
                battery_report = self.get_battery_data()

                # Request current PV active power (integrated PV inverter)
                pv_active_power = self.sendRequest(('EMS_REQ_POWER_PV', 'None', None), keepAlive=True)[2]

                # External generator (e.g. external PV inverter; values are negative)
                ext_generator_active_power = self.sendRequest(('EMS_REQ_POWER_ADD', 'None', None), keepAlive=True)[2]

                home_consumption_active_power = self.sendRequest(('EMS_REQ_POWER_HOME', 'None', None), keepAlive=True)[2]

                grid_exchange_active_power = self.sendRequest(('EMS_REQ_POWER_GRID', 'None', None), keepAlive=True)[2]

                ems_used_charge_limit = self.sendRequest(('EMS_REQ_USED_CHARGE_LIMIT', 'None', None), keepAlive=True)[2]
                ems_bat_charge_limit = self.sendRequest(('EMS_REQ_BAT_CHARGE_LIMIT', 'None', None), keepAlive=True)[2]
                ems_dcdc_charge_limit = self.sendRequest(('EMS_REQ_DCDC_CHARGE_LIMIT', 'None', None), keepAlive=True)[2]
                ems_user_charge_limit = self.sendRequest(('EMS_REQ_USER_CHARGE_LIMIT', 'None', None), keepAlive=True)[2]

                ems_used_discharge_limit = \
                    self.sendRequest(('EMS_REQ_USED_DISCHARGE_LIMIT', 'None', None), keepAlive=True)[2]
                ems_bat_discharge_limit = \
                    self.sendRequest(('EMS_REQ_BAT_DISCHARGE_LIMIT', 'None', None), keepAlive=True)[2]
                ems_dcdc_discharge_limit = \
                    self.sendRequest(('EMS_REQ_DCDC_DISCHARGE_LIMIT', 'None', None), keepAlive=True)[2]
                ems_user_discharge_limit = \
                    self.sendRequest(('EMS_REQ_USER_DISCHARGE_LIMIT', 'None', None), keepAlive=True)[2]

                self.dict_internal_to_value['sensor'].update(
                    {'pv_active_power': pv_active_power * -1,
                     'ext_generator_active_power': ext_generator_active_power,
                     'home_consumption_active_power': home_consumption_active_power,
                     'grid_exchange_active_power': grid_exchange_active_power,
                     'battery_active_power': battery_active_power,
                     'vsoc': vsoc * 0.01,
                     'soc': battery_report['rsoc'] * 0.01,
                     'charge_cycles': battery_report['chargeCycles'],
                     'current': battery_report['current'],
                     'status_code': battery_report['statusCode'],
                     'error_code': battery_report['errorCode'],
                     'module_voltage': battery_report['moduleVoltage'],
                     'terminal_voltage': battery_report['terminalVoltage'],
                     'ems_used_charge_limit': ems_used_charge_limit,
                     'ems_bat_charge_limit': ems_bat_charge_limit,
                     'ems_dcdc_charge_limit': ems_dcdc_charge_limit,
                     'ems_user_charge_limit': ems_user_charge_limit,
                     'ems_used_discharge_limit': ems_used_discharge_limit,
                     'ems_bat_discharge_limit': ems_bat_discharge_limit,
                     'ems_dcdc_discharge_limit': ems_dcdc_discharge_limit,
                     'ems_user_discharge_limit': ems_user_discharge_limit})

                logger.debug(f'Collected following data from device: {self.dict_internal_to_value}')

                self.send_mqtt_batch()

                await asyncio.sleep(self.POLLING_FREQUENCY)
            except KeyboardInterrupt:
                break
            except Exception:
                logger.exception(f'Error polling data.')
                await asyncio.sleep(self.POLLING_FREQUENCY)
                # raise

    async def send_data_to_device_loop(self, keepAlive=False):
        """
        Method sends control information and targets concerning
        the discharge or charge power to the device.

        Parameters
        -----------
        keepAlive : bool , optional
            Parameter that sets whether the E3DC connectiont
            should be maintained between sending operations
            (True) or not (False).
        """

        while True:
            mode = self.dict_internal_to_value['actuator']['mode']
            mode_text = self.mode_map[mode]

            if mode_text == 'auto':
                logger.info('Actuator value is set to auto mode. '
                            'Connector will not send any command to allow storage to run in auto mode.')

            else:
                battery_active_power_value = self.dict_internal_to_value['actuator']['battery_active_power_value']

                logger.info(f'Send command: {mode_text} '
                             f'with power: {battery_active_power_value}')

                try:
                    res =self.sendRequest(
                        ('EMS_REQ_SET_POWER',
                         'Container', [
                             ('EMS_REQ_SET_POWER_MODE', 'UChar8', mode),
                             ('EMS_REQ_SET_POWER_VALUE', 'Int32', battery_active_power_value)
                         ]),
                        keepAlive=keepAlive)

                    # Response is a tuple looking like this (with <value> being the set mode or value?):
                    # ('EMS_SET_POWER', 'Uint32', <value>)
                    logger.debug(f'Power request response: {res}')

                    # await asyncio.sleep(self.SEND_COMMAND_FREQUENCY)  # try this otherwise use another thread

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.exception(
                        f'Error when sending power command {self.mode_map[mode]} with '
                        f'power={battery_active_power_value}')

            await asyncio.sleep(self.SEND_COMMAND_FREQUENCY)  # try this otherwise use another thread

    async def check_dispatcher(self, dispatchers: list):
        """
        Method frequently check whether all dispatchers in the
        parameter are still alive or not. If a dispatcher fails,
        the conrector's stop is initiated.

        Parameters
        -----------
        dispatchers : list , required
            A list with objects of the classes
            DispatchOnce or DispatchInInterval.
        """

        # Start the main loop which we spend all the operation time in.
        logger.info("Connector online. Entering main loop.")
        try:
            while True:
                # Check that all dispatchers are alive, and if this is the
                # case assume that the connector operations as expected.
                if not all([d.is_alive() for d in dispatchers]):
                    # If one is not alive, see if we encountered an exception
                    # and raise it, as exceptions in threads are not
                    # automatically forwareded to the main program.
                    for d in dispatchers:
                        if d.exception is not None:
                            raise d.exception
                    # If no exception is found raise a custom on.
                    raise RuntimeError(
                        "At least one dispatcher thread is not alive, but no "
                        "exception was caught."
                    )
                await asyncio.sleep(self.MQTT_FREQUENCY)

        except (KeyboardInterrupt, SystemExit):
            # This is the normal way to exit the Connector. No need to log the
            # exception.
            logger.info(
                "Connector received KeyboardInterrupt or SystemExit"
                ", shuting down."
            )
        except:
            # This is execution when something goes really wrong.
            logger.exception(
                "Connector main loop has caused an unexpected exception. "
                "Shuting down."
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

    #############################################################
    ##              methods for mqtt communication             ##
    #############################################################

    @staticmethod
    def _handle_incoming_mqtt_msg(client, userdata, msg):
        """
        onmessage method for the paho mqtt client class

        method is called when an entity publishes a actuator message
        for this device via the corresponding topics
        """

        self = userdata["self"]
        logger.debug("Handling incoming MQTT message on topic: %s", msg.topic)
        if msg.topic in self.dict_mqtt_topic_internal_map['actuator'].keys():
            self.handle_actuator_msg(
                topic=msg.topic,
                actuator_command=json.loads(msg.payload)
            )
        else:
            logger.debug(f'Received MQTT message has unknown topic {msg.topic}')

    def send_mqtt_batch(self):
        """
        method publishes all stored device data to
        their corresponding mqtt topics
        """

        logger.debug(f'Start MQTT sending process')
        timestamp = timestamp_utc_now()

        for internal_key in self.dict_internal_to_value['sensor'].keys():
            if internal_key not in self.dict_mqtt_topic_internal_map['sensor'].keys():
                logger.info(f'Unknown internal key. Do not know a MQTT topic for key: {internal_key}')
                continue

            value_msg = {
                "value": self.dict_internal_to_value['sensor'][internal_key],
                "timestamp": timestamp,
            }
            topic = self.dict_mqtt_topic_internal_map['sensor'][internal_key]

            logger.debug(f'Publish - MQTT topic: {topic}, Msg:{value_msg}')
            self.mqtt_client.publish(
                payload=json.dumps(value_msg),
                topic=topic,
                retain=True,
            )

    #############################################################
    ##                     auxilliary methods                  ##
    #############################################################

    def handle_actuator_msg(self, topic: str, actuator_command: dict):
        """
        Method processes the actuator messages for the connected
        device. A separate method is called for controlling the
        charging and discharging power. All actuator messages
        for other actuators are saved within the object.

        Parameters
        -----------
        topic : str , required
            MQTT topic via which the message was received
        actuator_command : dict , required
            The parsed MQTT message. Dictionary must contain
             at least the key 'value'.
             e.g.:
             {...
              'value': 123.45,
              OR
              'value': None
              ...}
        """

        if topic not in self.dict_mqtt_topic_internal_map['actuator'].keys():
            logger.warning(f'Receive a MQTT msg with an unkown topic: {topic}')
            return
        logger.debug(f'Receive actuator command {actuator_command},'
                     f' via topic {topic}')

        internal_id = self.dict_mqtt_topic_internal_map['actuator'][topic]
        command_value = actuator_command['value']

        if internal_id == 'battery_active_power':
            # special method for received power commands
            self.set_battery_active_power(battery_active_power=command_value)
        else:
            # normal procedure for all other actuators beside power
            self.dict_internal_to_value['actuator'][internal_id] = command_value

    def set_battery_active_power(self, battery_active_power: typing.Union[int, None]):
        """
        Set (dis)charge power of the battery or set it to
        automatic mode. The set values are send to the system
        every self.E3DC_FREQUENCY seconds.

        Parameters
        -----------
        battery_active_power : [int, None], required
            Signed power value [W] for charging (+) /discharging (-)
            or "None" to activate automatic mode
        """
        if battery_active_power is None:
            # Set to auto mode
            self.dict_internal_to_value['actuator']['mode'] = self.mode_map_rev['auto']
            self.dict_internal_to_value['actuator']['battery_active_power_value'] = self.AUTO_MODE_POWER_VALUE
        else:
            # Set (dis)charge power
            if battery_active_power < 0:  # discharge
                self.dict_internal_to_value['actuator']['mode'] = self.mode_map_rev['discharge']
            elif battery_active_power > 0:  # charge
                self.dict_internal_to_value['actuator']['mode'] = self.mode_map_rev['charge']
            elif battery_active_power == 0:  # idle
                self.dict_internal_to_value['actuator']['mode'] = self.mode_map_rev['idle']

            self.dict_internal_to_value['actuator']['battery_active_power_value'] = abs(battery_active_power)

        logger.debug(f'Convert power_target to '
                     f'mode= {self.dict_internal_to_value["actuator"]["mode"]} and '
                     f'battery_active_power= {self.dict_internal_to_value["actuator"]["battery_active_power_value"]}')

    def load_and_parse_mqtt_e3dc_map(self, path):
        """
        Method imports the MQTT_E3DC_MAP information and
        controls its structure

        Parameters
        -----------
        path : str , required
            Path to the json file's location

        Returns
        -----------
        dict_mqtt_topic_internal_map : dict
            The MQTT_E3DC_MAP parsed as a directory. If there
            was an error in the structure, only an empty directory.
        """
        cn = self.CONNECTOR_NAME

        logger.info("Loading MQTT_E3DC_MAP.")
        with open(path, 'r') as read_file:
            mqtt_e3dc_map = json.load(read_file)

        logger.info("Parsing MQTT_E3DC_MAP.")

        expected_first_keys = [
            "sensor",
            "actuator"
        ]

        actual_first_keys = list(mqtt_e3dc_map.keys())
        actual_first_keys = [single_string.lower() for single_string in actual_first_keys]

        if set(actual_first_keys) != set(expected_first_keys):
            logger.warning(f'The datapoint map has an incorrect structure.'
                           f'Here are its top-level elements: {set(actual_first_keys)}')
            mqtt_e3dc_map = {}
            return mqtt_e3dc_map

        # Prepend connector name to MQTT topics ...
        # ... for sensor topics (topics are dict values)
        for curr_e3dc in mqtt_e3dc_map['sensor'].keys():
            curr_mqtt_ada = cn + mqtt_e3dc_map['sensor'][curr_e3dc]
            mqtt_e3dc_map['sensor'][curr_e3dc] = curr_mqtt_ada

        # ... and for actuator topics (topics are dict keys)
        topic_map_without_cn = mqtt_e3dc_map['actuator']
        topic_map_with_cn = {}
        for topic, parameter in topic_map_without_cn.items():
            topic_map_with_cn[cn + topic] = parameter
        mqtt_e3dc_map['actuator'] = topic_map_with_cn

        return mqtt_e3dc_map


if __name__ == "__main__":
    logger.info('Create E3DC Connector.')

    connector = E3DCConnector()
    connector.run()
