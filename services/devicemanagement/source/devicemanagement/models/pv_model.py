import numpy as np
import typing

from devicemanagement import logger
from devicemanagement.models.device_model import Device
from devicemanagement.utils import get_latest_measurement




class PVPlant(Device):
    counter = 0

    def __init__(self, key: str, specification: dict):
        super().__init__(
            key=key,
            category=specification["category"],
            subcategory=specification["subcategory"],
            name=specification["product"],
            energy_carrier=specification.get("energy_carrier"),
        )

        # Specifications in W and Ws, respectively
        # Protected attributes to discourage modification
        self._active_power_nominal = int(specification['active_power_nominal_kW'] * -1000)  # < 0
        # TODO: consider inverter efficiency (currently it's not being used)
        self._inverter_efficiency = specification['inverter_eff']
        self._continuous_curtailment = specification['continuous_mod']
        if not self._continuous_curtailment:
            self._curtailment_levels = specification['curtailment_levels']  # list with values in range [0.0, 1.0]
        else:
            discrete_power_values = self.discretize_power_range(p_min=self._active_power_nominal, p_max=0, granularity=1.0)
            self._curtailment_levels = [p / self._active_power_nominal for p in discrete_power_values]

        self._save_specifications()

        # Variables
        # 0: generation power (<= 0) (non-curtailed power currently produced by the PV modules)
        # 1: curtailment level (100% = no curtailment)
        # 2: output power (=generation power if it is below curt._level*nominal_power)
        self.state = np.array([-0.0, 0.0, -0.0])

        self.actions = np.array(self._curtailment_levels)


        # print(f'PV plant (category {self._category}) created with attributes {self.__dict__} (or dev_parameters {self.device_parameters})')
    @property
    def active_power_generation(self):
        return self.state[0]  #

    @active_power_generation.setter
    def active_power_generation(self, value=typing.Union[float, None]):
        if value is None:
            value = get_latest_measurement(source=self._key, fields='active_power')
        try:
            # Make sure value is <= 0
            self.state[0] = -abs(value)
        except TypeError:
            self.state[0] = None

    @property
    def curtailment_level(self):
        return self.state[1]

    @curtailment_level.setter
    def curtailment_level(self, value=typing.Union[float, None]):
        if value is None:
            value = get_latest_measurement(source=self._key, fields='curtailment')
        self.state[1] = value
        try:
            self.state[2] = min(self.active_power_generation, value * self._active_power_nominal)
        except TypeError:
            self.state[2] = None

    @property
    def active_power_output(self):
        return self.state[2]

    @property
    def output_levels(self):
        return [curtailment_level * self._active_power_nominal for curtailment_level in self._curtailment_levels]

    def feasible_actions(self, t_delta: int, *args) -> np.ndarray:
        return np.array(self._curtailment_levels)

    def flexibility(self, t_delta, power: float = None, safety_buffer: float = 0.0, *args, **kwargs) -> typing.List[
        float]:
        """
        Returns possible output power levels based on curtailment levels, with max. output being the current (non-curtailed) power output.
        :param power: Set generation power or, if None, read current power from database (see setter method)
        :param safety_buffer: Possible decrease in irradiance to account for compared to current moment (in [0; 1])
        :return: Possible power output levels
        """
        self.active_power_generation = power  # if power value not provided, it's retrieved from the db
        max_power = (1 - safety_buffer) * self.active_power_generation
        # All output levels that are greater or equal to current generation are possible (>= because of negative values)
        flex = [p for p in self.output_levels if p >= max_power] + [max_power]
        # max. gen, min. gen, 0, 0
        return flex

    def make_state_transition(self, t_delta: float, action: float, interaction: np.ndarray = np.zeros(2), generation_power: float = None) -> typing.Tuple[
        np.ndarray, np.ndarray]:
        """
        Restrict PV output to given level (or closest smaller level, if level cannot be set)
        :param t_delta: Time duration [s] for which to (dis)charge
        :param action: Curtailment level to be set
        :param interaction: Possible interaction from connected system
        :param generation_power: (Optional) Set generation power of PV model
        :return: Updated state, resulting interaction with connected systems (0: electric, 1: thermal)
        """
        if generation_power:
            self.active_power_generation = generation_power
        if action not in self._curtailment_levels:
            for lvl in sorted(self._curtailment_levels, reverse=True):
                if lvl <= action:
                    self.curtailment_level = lvl
                    break
        else:
            self.curtailment_level = action
        return self.state, interaction - np.array([self.active_power_output, 0])


