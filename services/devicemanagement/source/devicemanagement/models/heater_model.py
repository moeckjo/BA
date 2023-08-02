import numpy as np
import typing

from devicemanagement import logger
from devicemanagement.models.device_model import Device
from devicemanagement.utils import get_latest_measurement


class HeatingElement(Device):
    # TODO: Implement correctly and clean
    def __init__(self, key: str, specification: dict):
        super().__init__(
            key=key,
            category=specification["category"],
            subcategory=specification["subcategory"],
            name=specification["product"],
            energy_carrier_in=specification.get("energy_carrier_in"),
            energy_carrier_out=specification.get("energy_carrier_out"),
        )
        # Specifications in W and Ws, respectively
        self._max_active_power = int(specification['max_active_power_kW'] * 1000)

        self._save_specifications()

        # Variables
        self.state = 0  # 0=off, 1=on

        self.actions = np.array([0, 1])  # on or off


    @property
    def power(self):
        return self.state * self._max_active_power

    @property
    def on(self):
        return self.state

    @on.setter
    def on(self, value: bool):
        self.state = value

    def thermal_output(self):
        # TODO: Calculate thermal power from el. power and some efficiency/heat transfer factor (?)
        pass

    def feasible_actions(self, t_delta: int, *args) -> np.ndarray:
        # TODO: Determine feasible action based on HP actvity and storage temperatures
        pass

    def make_state_transition(self, t_delta: float, action: float, interaction: np.ndarray = np.zeros(2)) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Turn the heater on or off
        :param t_delta: Time duration [s]
        :param action: On or off
        :param interaction: Possible interaction from connected system
        :return: Updated state, resulting interaction with connected systems (0: electric, 1: thermal)
        """
        self.on = action

        return self.state, interaction - np.array([self.power, -self.thermal_output()])









