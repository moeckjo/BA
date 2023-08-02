import typing

import numpy as np
import os

from db import db_helper as db
from devicemanagement import logger


class Device:
    """
    Generic device model with some common attributes and methods which specific devices can inherit.
    It also serves as generic model for device subcategories without (the need for) a specific model implementation.
    """
    object_category = os.getenv('MONGO_DEVICE_DB_NAME')

    def __init__(self, key: str, category: str, subcategory: str, **kwargs):
        self._key = key  # unique name for this device
        self._category = category  # generator, consumer, storage or converter
        self._subcategory = subcategory  # e.g. BESS, PV, ...

        if self._category == "converter":
            self._energy_carrier_in = kwargs.pop('energy_carrier_in', None)
            self._energy_carrier_out = kwargs.pop('energy_carrier_out', None)
        else:
            self._energy_carrier = kwargs.pop('energy_carrier', None)

        self.actions = np.empty(0)

        for key, value in kwargs.items():
            self.__setattr__(f"_{key}", value)

        if self.__class__ == Device:
            # Subclasses of Device should call this method after setting all specific attributes.
            self._save_specifications()

    def __str__(self):
        return self._key

    def __repr__(self):
        return f"{str(self)}: {self.device_parameters}"

    def _save_specifications(self):
        logger.debug(f'{str(self)} specs: {self.device_parameters}')
        db.save_dict_to_db(
            db=os.getenv('MONGO_DEVICE_DB_NAME'),
            data_category=os.getenv('MONGO_DEVICE_COLL_NAME'),
            data=self.device_parameters
        )

    # def __store_in_db(self):
    #     # returned_id = db.save_object_to_db(object_category=self.object_category, obj=self)
    #     returned_id = db.save_object_to_db(
    #         object_category=self.object_category,
    #         obj=self,
    #         obj_name=self._subcategory,
    #         name=str(self)
    #     )
    #
    #     logger.debug(f'ID returned to model: {returned_id} is of type {type(returned_id)}')
    #     return returned_id
    #
    # def get_db_id(self):
    #     return self.__db_id

    # def get_from_db(self) -> dict:
    #     """
    #     :return: All attributes of the device with values as currently stored in the database
    #     """
    #     attributes = db.get_object_from_db(self.object_category, self.__db_id)
    #     return attributes

    # @classmethod
    # def get_from_db(cls, device_id, device_subcategory):
    #     """
    #     :return: Instance stored in database with provided ID
    #     """
    #     instance = db.get_object_from_db(cls.object_category, device_id, device_subcategory)
    #     return instance

    # def update_model_state_in_db(self, updates: dict):
    #     variable_updates = {}
    #     for attr, value in updates.items():
    #         if not hasattr(self, attr):
    #             raise AttributeError(f'The object has no attribute {attr}.'
    #                                  f' List of modifiable attributes: {list(self.get_device_state_variables().keys())}.')
    #
    #         assert attr in self.get_device_state_variables().keys(), "Changing a device's fixed parameter is not permitted!"
    #         variable_updates[attr] = value
    #
    #     db.update_object_in_db(
    #         object_category=self.object_category,
    #         doc_id=self.get_db_id(),
    #         updates=variable_updates
    #     )

    @staticmethod
    def discretize_power_range(p_min: int, p_max: int, granularity: float,
                               min_power_offset: bool = False) -> np.ndarray:
        """
        :param p_min: Min. power when on
        :param p_max: Max. power
        :param granularity: Step size in Watt
        :param min_power_offset: If the power does not range continuously between "off" state (zero) and max. power
        :return: List (1D-array) of discrete power steps
        """
        n_actions = int(abs(p_max - p_min) / granularity + min_power_offset)  # W/W
        actions = np.zeros(n_actions)
        for i in range(n_actions):
            actions[i] = p_min + i * granularity

        if min_power_offset:
            # Add zero as first element before p_min
            actions = np.append(np.array([0]), actions)

        return actions

    @property
    def device_parameters(self) -> dict:
        """
        :return: The fixed parameters (specifications) of the device (e.g. nominal generation power of a PV plant)
        """
        parameters = {
            attr[1:]: value for attr, value in self.__dict__.items() if
            (str(attr).startswith('_')) and ('id' not in attr)
        }
        return parameters

    @property
    def category(self):
        return self._category

    @property
    def subcategory(self):
        return self._subcategory

    def feasible_actions(self, t_delta: int, *args) -> np.ndarray:
        """
        Should be implemented by a specific device class.
        :param t_delta: Time period duration [s]
        :return: Array of actions that are feasible given the state of the device and the duration
        """
        pass

    def flexibility(self, t_delta, *args, **kwargs) -> typing.Union[np.ndarray, tuple, list]:
        """
        Should be implemented by a specific device class.
        :param t_delta: Time period duration [s]
        :return: List-like object describing the flexibility of the device given it's state and the duration. For example,
        in the form of: (max_generation, min_generation, max_consumption, min_consumption)
        """
        pass
