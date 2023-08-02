import datetime
import json
import logging
import os
import typing

import pandas as pd
import pytz

from db import db_helper as db
from devicemanagement import logger
from devicemanagement import utils
from devicemanagement.communication import bess_com, ev_com, hp_com, pv_com, hwt_com
from devicemanagement.management import bess_management, ev_management, heating_management, pv_management
from devicemanagement.models import device_model, bess_model, pv_model, heat_storage_model, evse_model, heat_pump_model


def load_device_config() -> typing.Dict[str, dict]:
    with open(os.path.join(os.getenv('BEM_ROOT_DIR'), 'config', 'device_config.json')) as device_config_file:
        device_config = json.load(device_config_file)
    return device_config


def save_device_config():
    # Save the current device configuration in the settings database for tracking purposes (for later evaluations)
    device_config: dict = load_device_config()
    device_config['timestamp'] = datetime.datetime.now(tz=pytz.utc).isoformat(timespec='seconds')
    db.save_dict_to_db(
        db=os.getenv('MONGO_SETTINGS_DB_NAME'),
        data_category=os.getenv('MONGO_SETTINGS_DEVICE_CONFIG_COLL_NAME'),
        data=device_config
    )
    logger.debug(f"Stored the following device settings: {device_config}")


def get_device_specifications(name: typing.Optional[str] = None) -> dict:
    if name is None:
        return load_device_config()
    else:
        return load_device_config()[name]


def get_device_measurements(source: str, fields: typing.Union[str, list], start_time: datetime.datetime,
                            end_time: typing.Union[datetime.datetime, None] = None) -> pd.DataFrame:
    data: pd.DataFrame = db.get_measurement(
        source=source,
        fields=fields,
        start_time=start_time,
        end_time=end_time,
    )
    return data


def ev_charged_since(timestamp: datetime.datetime, charging_power_measurements: pd.DataFrame = None,
                     return_max_power: bool = False) -> typing.Union[bool, typing.Tuple[bool, float]]:
    if charging_power_measurements is None:
        charging_power_measurements: pd.Series = get_device_measurements(
            source=os.getenv('EVSE_KEY'),
            fields=['active_power'],
            start_time=timestamp
        )['active_power']
    logger.debug(f"Charging power measurements: \n{charging_power_measurements}")
    charged = False
    max_charged_power = charging_power_measurements.max()
    logger.debug(f"Max. charged power = {max_charged_power}")
    if max_charged_power >= 100:
        charged = True

    if return_max_power:
        return charged, max_charged_power

    return charged


def last_time_of_connection() -> datetime.datetime:
    now = datetime.datetime.now(tz=datetime.timezone.utc)

    # Get connection data from the database
    connection_states: pd.Dataframe = get_device_measurements(
        source=os.getenv('EVSE_KEY'),
        fields='connected',
        start_time=now - datetime.timedelta(hours=24),
    )
    logger.debug(f"Connection states: {connection_states}")

    # Get the changes of the connection state
    # (diff=0 -> no change; diff=-1 -> from connected to unconnected; diff=1 from unconnected to connected)
    connection_states_diff = connection_states['connected'].diff()

    # Check if the connection state changed from 0 (unconnected) to 1 (connected) in the considered period
    if connection_states_diff.max(skipna=True) <= 0:
        # Connection state didn't change or vehicle got disconnected
        return None

    # EV got connected during the considered period -> get time of connection
    connected_at = connection_states_diff[connection_states_diff == 1].index[-1].to_pydatetime()
    return connected_at


def evaluate_ev_schedule_deviation(target_value: float, sensor_value: float, timestamp: datetime.datetime,
                                   connected: int = None, charging_state: int = None,
                                   charging_power_measurements: pd.DataFrame = None,
                                   connected_at: datetime.datetime = None
                                   ) -> bool:
    """
    :param target_value: The target value of the schedule in Watt, e.g. 11000
    :param sensor_value: The actual, measured value in Watt
    :param timestamp: Timestamp (datetime) of registered deviation
    :return: Boolean if EV is assumed to be fully charged
    """

    ev_full: bool = None
    evse_config = get_device_specifications(os.getenv('EVSE_KEY'))

    if connected is None:
        connected = utils.get_latest_measurement(os.getenv('EVSE_KEY'), fields='connected')

    if connected == 0:
        logger.info(f'Deviation is caused by EV disconnection.')
        return False

    if charging_state is None:
        charging_state = utils.get_latest_measurement(os.getenv('EVSE_KEY'),
                                                      fields=evse_config["charging_state_parameter_name"])
    if connected_at is None:
        connected_at = last_time_of_connection()

    if target_value > sensor_value:
        charging_state_map_rev = {v: k for k, v in evse_config["charging_state_value_map"].items()}
        if charging_state != evse_config["charging_state_value_map"]["blocked_by_vehicle"]:
            logger.warning(f'Deviation is caused by '
                           f'charging state {charging_state} ("{charging_state_map_rev[charging_state]}").')
            return False

        # It can be observed that the charging state "blocked_by_vehicle" only occurs when charging power changes
        # from >0 to 0, but vehicle stays connected and does not resume charging while connected.
        # To make sure it's not an error of the vehicle, check if it has been charged since it was connected.
        ev_charged, max_power_charged = ev_charged_since(timestamp=connected_at,
                                                         charging_power_measurements=charging_power_measurements,
                                                         return_max_power=True)
        if ev_charged:
            # -> Charging stopped (or power reduced) most likely because EV is already full
            logger.info(f'Based on target_value={target_value}, '
                        f'sensor_value={sensor_value},'
                        f'charging_state={charging_state} ("{charging_state_map_rev[charging_state]}") '
                        f'and max. power charged={max_power_charged}'
                        f'it seems like the charging process stopped '
                        f'because the EV is full.')
            return True
        else:
            logger.info(
                f"EV has not been charged since it was connected at {connected_at} (max. power measured={max_power_charged}W).")
            return False



    else:
        logger.warning(f'The wallbox is charging more than allowed (target={target_value}W, measured={sensor_value}W)!')
        return False



def create_device_management_systems() -> typing.List[object]:
    # TODO 2021-04-19: revisit this whole device management system stuff when usccesfully integrated BEMCom connectors
    """
    Creates device management systems for each device specified in the device configuration file
    :return: List of created device management systems
    """
    logger.info('Set up device management systems.')

    devices = load_device_config()  # Example: category=storage, subcategory=BESS or HWT
    logger.info(f'Integrate the following devices: {devices}')
    device_management_systems = []
    for key, specification in devices.items():

        if specification['subcategory'] == os.getenv('BESS_KEY'):
            bess_connector = bess_com.BESSConnector()
            dev = DeviceManagementSystem(
                model=bess_model.BatteryStorage(key, specification),
                observer=bess_management.BESSObserver(bess_connector),
                controller=bess_management.BESSController(bess_connector),
                connector=bess_connector
            )

        elif specification['subcategory'] == os.getenv('EVSE_KEY'):
            ev_connector = ev_com.WallboxConnector()
            dev = DeviceManagementSystem(
                model=evse_model.EVSE(key, specification),
                observer=ev_management.EVObserver(ev_connector),
                controller=ev_management.EVController(ev_connector),
                connector=ev_connector
            )

        elif specification['subcategory'] == os.getenv('HEAT_PUMP_KEY'):
            heat_pump_connector = hp_com.HeatPumpConnector()
            if specification['inverter']:
                hp_model = heat_pump_model.InverterHeatPump(key, specification)
            else:
                hp_model = heat_pump_model.HeatPump(key, specification)
            dev = DeviceManagementSystem(
                model=hp_model,
                observer=heating_management.HeatPumpObserver(heat_pump_connector),
                controller=heating_management.HeatPumpController(heat_pump_connector),
                connector=heat_pump_connector
            )

        elif specification['subcategory'] == os.getenv('HEAT_STORAGE_KEY'):
            heat_storage_connector = hwt_com.HWTConnector()
            dev = DeviceManagementSystem(
                model=heat_storage_model.HotWaterStorage(key, specification),
                observer=heating_management.HeatStorageObserver(heat_storage_connector),
                controller=heating_management.HeatStorageController(heat_storage_connector),
                connector=heat_storage_connector
            )

        elif specification['subcategory'] == os.getenv('PV_KEY'):
            pv_connector = pv_com.PVInverterConnector()
            model = pv_model.PVPlant(key, specification)
            dev = DeviceManagementSystem(
                model=model,
                observer=pv_management.PVObserver(pv_connector),
                controller=pv_management.PVController(pv_connector),
                connector=pv_connector
            )

        else:
            logger.info(
                f'No specific model for device with key "{key}" of subcategory "{specification["subcategory"]}" '
                f'implemented. Creating object of class "Device".')

            category = specification.pop('category')
            subcategory = specification.pop('subcategory')
            model = device_model.Device(key, category, subcategory, **specification)
            dev = DeviceManagementSystem(
                model=model,
                observer=None,
                controller=None,
                connector=None,
            )

        device_management_systems.append(dev)

    logger.info('Creation finished')

    return device_management_systems


class DeviceManagementSystem:
    """
    A device management systems always consists of (aggregates) a model of the (real) device, an observer, a controller
    and a connector that establishes the connection to the device.
    """
    object_category = os.getenv('MONGO_DEVICE_MGMT_SYSTEM_COLL_NAME')
    class_abbreviation = 'dms'
    instances = []

    def __init__(self, model, observer, controller, connector):
        self.model = model
        self.observer = observer
        self.controller = controller
        self.connector = connector

        # self.observer.attach_to_controller(controller)
        #
        # self.observer.attach_to_model(model)

        self.__db_id = self.__store_in_db()

    def __store_in_db(self):
        returned_id = db.save_objects_to_db(
            db=os.getenv('MONGO_DEVICE_DB_NAME'),
            object_category=self.object_category,
            objects={
                'model': self.model,
                'observer': self.observer,
                'controller': self.controller,
                'connector': self.connector
            },
            device_key=str(self.model),
            device_subcategory=self.model.subcategory
        )
        return returned_id

    @classmethod
    def get_from_db(cls, db_id) -> typing.Dict[str, object]:
        """
        Get the respective instances of all DMS components (model, observer, controller and connector) from the database.
        E.g. {'model': PVPlant instance, 'observer': PVObserver instance, ...}
        :return: dict(component_name, instance)
        """
        instances = db.get_objects_from_db(
            db=os.getenv('MONGO_DEVICE_DB_NAME'),
            object_category=cls.object_category,
            doc_id=db_id
        )
        return instances

    @classmethod
    def update_objects_in_db(cls, db_id, objects: dict):
        db.update_objects_in_db(
            db=os.getenv('MONGO_DEVICE_DB_NAME'),
            object_category=cls.object_category,
            doc_id=db_id,
            objects=objects
        )

    def get_db_id(self):
        return self.__db_id
