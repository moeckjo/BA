import numpy as np
import typing

from devicemanagement import logger
from devicemanagement.models.device_model import Device
from devicemanagement.utils import get_latest_measurement


class BatteryStorage(Device):
    # TODO: Model self-discharging (relative loss) ?
    #   but rate of today's systems (<4% per month) is negligible -> e.g. 20 kWh capacity => loss of 1 Wh per hour

    def __init__(self, key: str, specification: dict):
        super().__init__(
            key=key,
            category=specification["category"],
            subcategory=specification["subcategory"],
            name=specification["product"],
            energy_carrier=specification.get("energy_carrier"),
        )

        # Specifications in W and Ws, respectively
        self._capacity = int(specification['capacity_kWh'] * 1000 * 3600)
        self._active_power_charge_nominal = int(specification['active_power_charge_nominal_kW'] * 1000)  # positive
        self._active_power_discharge_nominal = int(specification['active_power_discharge_nominal_kW'] * -1000)  # negative
        self._efficiency = specification['efficiency']
        self._soc_min = specification['soc_min']
        self._soc_max = specification['soc_max']

        self._relative_loss_per_second = specification["relative_loss_per_month"] / (
                31 * 24 * 60 * 60)  # Relative to capacity
        self._save_specifications()

        # State variables
        # 0: soc, 1: min. soc, 2: max. soc
        self.state = np.array([0.0, self._soc_min, self._soc_max])

        self._actions = self.discretize_power_range(
            p_min=self._active_power_discharge_nominal,
            p_max=self._active_power_charge_nominal,
            granularity=1.0
        )

    @property
    def soc(self):
        return self.state[0]

    @soc.setter
    def soc(self, value=typing.Union[float, None]):
        if value is None:
            value = get_latest_measurement(source=self._key, fields='soc')
            if value is None:
                value = 0.5
        self.state[0] = value if value <= 1.0 else value / 100

    @property
    def soc_min(self):
        return self.state[1]

    @soc_min.setter
    def soc_min(self, value=typing.Union[float, None]):
        if value is None:
            value = get_latest_measurement(source=self._key, fields='soc_min')
        self.state[1] = value if value <= 1.0 else value / 100

    @property
    def soc_max(self):
        return self.state[2]

    @soc_max.setter
    def soc_max(self, value=typing.Union[float, None]):
        if value is None:
            value = get_latest_measurement(source=self._key, fields='soc_max')
        self.state[2] = value if value <= 1.0 else value / 100

    def max_charge_power(self, t_delta: float) -> float:
        # TODO: Account for reduced max. charging power when close to boundaries
        # print(f'cap: {self._capacity}, eff: {self._efficiency}, t_delta: {t_delta}, soc: {self.soc}')
        max_charge_power = min(
            max(self.soc_max - self.soc, 0) * self._capacity / (t_delta * self._efficiency),
            self._active_power_charge_nominal
        )
        return max_charge_power

    def min_charge_power(self, t_delta: float) -> float:
        # Min. charging power to reach min. SOC again if necessary
        return min(max(0, self.soc_min - self.soc) * self._capacity / (t_delta * self._efficiency),
                   self._active_power_charge_nominal)

    def max_discharge_power(self, t_delta: float):
        # TODO: Account for reduced max. charging power when close to boundaries
        max_discharge_power = max(
            -max(0, self.soc - self.soc_min) * self._capacity * self._efficiency / t_delta,
            self._active_power_discharge_nominal
        )
        return max_discharge_power  # <= 0

    def min_discharge_power(self, t_delta: float) -> float:
        # Min. discharging power to reach max. SOC again if necessary
        # return max(-max(0, self.soc - self.soc_max) * self._capacity * self._efficiency / t_delta,
        #            self._active_power_discharge_nominal)
        # TODO: should there be a min. discharge if it's above the defined max. SOC?
        return 0

    def feasible_actions(self, t_delta: int, *args) -> np.ndarray:
        limits = self.flexibility(t_delta)
        lower_limit = limits[0] if limits[2] == 0 else limits[2]

        actions = np.copy(self._actions)
        return actions[lower_limit <= actions <= limits[1]]

    def flexibility(self, t_delta: int, soc: float = None, *args, **kwargs) -> typing.Tuple[float, float, float, float]:
        """
        Returns the max. constant discharging and charging power, respectively, for the considered duration t_delta
        based on the given or last measured SOC.
        :param t_delta: Time duration [s]
        :param soc: SOC the battery model must be set to
        :return: Max. discharging power (<= 0), max. charging power (>= 0), min. charging power (>= 0)
        """
        self.soc = soc  # if argument soc=None, it gets latest value from the DB

        max_discharge_power = self.max_discharge_power(t_delta)
        min_discharge_power = self.min_discharge_power(t_delta)
        max_charge_power = self.max_charge_power(t_delta)
        min_charge_power = self.min_charge_power(t_delta)
        return max_discharge_power, min_discharge_power, max_charge_power, min_charge_power

    def make_state_transition(self, t_delta: float, action: float, interaction: np.ndarray = np.zeros(2)) -> \
            typing.Tuple[np.ndarray, np.ndarray]:
        # TODO: Account for reduced max. charging power when close to boundaries
        """
        Charge or discharge battery with the given power if feasible.
        :param t_delta: Time duration [s] for which to (dis)charge
        :param action: Target power [W] (positive=charging, negative=discharging)
        :param interaction: Possible interaction from connected system
        :return: Updated state, resulting interaction with connected systems (0: electric, 1: thermal)
        """
        net_energy = feasible_power = 0
        if action > 0:
            feasible_power = min(max(self.min_charge_power(t_delta), action), self.max_charge_power(t_delta))
            net_energy = feasible_power * t_delta * self._efficiency
        elif action < 0:
            feasible_power = max(action, self.max_discharge_power(t_delta))
            net_energy = feasible_power * t_delta / self._efficiency

        self.soc = ((self.soc * self._capacity) + net_energy) / self._capacity
        return self.state, interaction - np.array([feasible_power, 0])
