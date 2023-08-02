import os
import numpy as np
import typing

from devicemanagement import logger
from devicemanagement.models.device_model import Device
from devicemanagement.utils import get_latest_measurement

WATER_DENSITY = 0.99  # kg/l (Density at 45.5° C)


def temperature_unit_conversion(temp: float, return_unit: str = 'K') -> float:
    """
    Convert Celsius to Kelvin or vice versa
    :param temp: Temperature to convert
    :param return_unit: Target unit (Kelvin (K), if Celsius is given or Celsius (C) if Kelvin is given)
    :return: Converted temperature
    """
    if return_unit == 'K':
        return temp + 273.15
    elif return_unit == 'C':
        return temp - 273.15


class HotWaterStorage(Device):
    # Non-stratified storage

    # Storage should always be hot enough to serve the current heat load for the next 15 min
    safe_th_supply_duration = 15 / 60  # hours

    def normalized_loss_from_energy_label(self, ee_class: str,
                                          standing_loss_from_label: typing.Union[None, float] = None,
                                          norm_temp_delta: float = 45.0):
        """
        For the EU energy labels see CELEX_32013R0812
        Provides standing loss [W] for each energy efficiency class based on a storage temperature of 65 (+/- 1) °C and
        ambient temperature of 20 (+/- 3) °C
        -> "norm_temp_delta" = 45 K
        (see: https://www.energiegemeinschaft.com/wp-content/uploads/2017/06/170531_Vortrag_Buderus_Magdeburg_Huch.pdf, slide 9)
        Arguments
        - ee_class ``str`` Energy efficiency class of storage (see label)
        - norm_temp_delta ``float`` difference between tank and ambient temperature used to determine the tank loss (=45 K for multiple DIN norms)
        """
        # a_upper_bound ={'A+': 5.5, 'A': 8.5, 'B': 12.0, 'C': 16.66, 'D': 21.0}
        # b_upper_bound = {'A+': 3.16, 'A': 4.25, 'B': 5.93, 'C': 8.33, 'D': 10.33}

        # Calculate the average standing loss based on energy efficiency class (Standing loss [W] not noted on label)
        if not standing_loss_from_label:
            ee_class_index = {'A+': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
            a_upper_bounds = [5.5, 8.5, 12.0, 16.66, 21.0, 26.0]
            b_upper_bounds = [3.16, 4.25, 5.93, 8.33, 10.33, 13.66]

            # Upper bound of one class is lower bound of the next less efficient class
            # *_bounds: tuple(lower_bound, upper_bound)
            # Except for A+, for which only upper bound is defined in EU directive
            # -> lower_bound(A+) = upper_bound(A+) * (lower_bound(A)/upper_bound(A))
            a_bounds = [(a_upper_bounds[0] * (a_upper_bounds[0]) / a_upper_bounds[1], a_upper_bounds[0])]  # for A+
            a_bounds += [(a_upper_bounds[i], a_upper_bounds[i + 1]) for i in range(len(a_upper_bounds) - 1)]  # for A-E
            b_bounds = [(b_upper_bounds[0] * (b_upper_bounds[0]) / b_upper_bounds[1], b_upper_bounds[0])]  # for A+
            b_bounds += [(b_upper_bounds[i], b_upper_bounds[i + 1]) for i in range(len(b_upper_bounds) - 1)]  # for A-E

            # Take means of upper and lower bounds of the factors of the EE class
            a = (a_bounds[ee_class_index[ee_class]][0] + a_bounds[ee_class_index[ee_class]][1]) / 2
            b = (b_bounds[ee_class_index[ee_class]][0] + b_bounds[ee_class_index[ee_class]][1]) / 2

            standing_loss = (a + b * self._volume ** 0.4)  # W

        # Standing loss [W] is noted on label
        else:
            standing_loss = standing_loss_from_label

        # Normalize by the temp. difference taken determine the standing loss
        standing_loss_normalized = standing_loss / norm_temp_delta  # W/K
        return standing_loss_normalized

    def __init__(self, key: str, specification: dict):
        super().__init__(
            key=key,
            category=specification["category"],
            subcategory=specification["subcategory"],
            name=specification["product"],
            energy_carrier=specification.get("energy_carrier"),
        )
        # Specifications
        self._volume = specification['volume_l']  # l
        self._temp_max = specification['temp_max_C']  # C
        self._temp_min = specification['temp_min_C']  # C
        self._efficiency_charging = specification['efficiency_charging']
        self._efficiency_discharging = specification['efficiency_discharging']

        self._loss_per_temp_delta = self.normalized_loss_from_energy_label(specification['EU_EE_class'])  # W/K

        # self._relative_loss_per_second = specification["relative_loss_per_month"] / (
        #             31 * 24 * 60 * 60)  # Relative to capacity

        self._heat_capacity = 4.19 * WATER_DENSITY * self._volume  # J/K=Ws/K

        self._save_specifications()

        # State variables
        # 0: water temperature [C], 1: ambient temperature [C]
        self.state = np.array([50.0, 20.0])

        self.actions = None


    @property
    def temp_water(self):
        return self.state[0]

    @temp_water.setter
    def temp_water(self, value=typing.Union[float, None]):
        if value is None:
            value = get_latest_measurement(source=self._key, fields='temp_water')
        self.state[0] = value

    @property
    def temp_ambient(self):
        return self.state[1]

    @temp_ambient.setter
    def temp_ambient(self, value: typing.Union[float, None]):
        if value is None:
            value = get_latest_measurement(source=self._key, fields='temp_ambient')
        self.state[1] = value

    # def heat_loss(self, norm_temp_delta: float = 45.):
    #     temp_factor = (self.temperature - self.temp_ambient) / norm_temp_delta
    #     loss = self.static_loss_factor_from_energy_label('a', 'b') * temp_factor  # W
    #     return loss

    @property
    def stored_energy(self):
        """
        :return: Energy [Ws] stored in the water tank (based on current water temperature and ambient temperature)
        """
        return self._heat_capacity * (self.temp_water - self.temp_ambient)

    @property
    def eff_stored_energy(self):
        """
        :return: Effective (usable) energy [Ws] stored in the water tank
                (based on current water temperature and min. temperature)
        """
        return self._heat_capacity * (self.temp_water - self._temp_min)

    @property
    def soc(self):
        """
        :return: SOC related to set min. temp (not ambient temp)
        """
        return (self.temp_water - self._temp_min) / (self._temp_max - self._temp_min)

    def heat_loss(self, start_temp: float, end_temp: float, ambient_temp: float):
        """
        :param start_temp: Storage temperature at beginning of period
        :param end_temp: Storage temperature at end of period
        :param ambient_temp: Ambient temperature (assumed to be constant during period)
        :return: Mean loss [W], assuming linear temperature change
        """
        return self._loss_per_temp_delta * (0.5 * (start_temp + end_temp) - ambient_temp)

    def heat_loss_term(self, t_delta: int):
        """
        Returns value L which has to be integrated as follows when calculating the stored energy in
        period t+1 after (dis)charging with power_th for t_delta seconds:
        E(t+1) = [E(t)*(1 - L) + (t_delta * power_th * efficiency)] / (1 + L)

        The above formula is the result of transforming this equation:
        E(t+1) = E(t) + t_delta * [power_th * efficiency - 0.5*( E(t) + E(t+1) )*(loss_per_temp_delta/heat_capacity)]
        :param t_delta:
        :return:
        """
        return (t_delta/2) * self._loss_per_temp_delta/self._heat_capacity


    def max_charge_power_th(self, t_delta: float, expected_th_load: float = 0):
        """
        Maximum charging power allowed to keep the storage temperature at or below the upper limit, incl. losses
        :param t_delta: Period duration [s]
        :param expected_th_load: Expected thermal load, i.e. expected discharge (efficiency considered)
        :return: max. charge power (>= 0)
        """

        max_stored_energy = (self._temp_max - self.temp_ambient) * self._heat_capacity  # Ws
        loss_term = self.heat_loss_term(t_delta)
        max_th_charge_power = (max_stored_energy * (1 + loss_term) - self.stored_energy * (1 - loss_term)) / (t_delta * self._efficiency_charging)
        # Charging power can be raised by expected discharging power
        max_th_charge_power += (expected_th_load / self._efficiency_discharging) / self._efficiency_charging

        # max_th_charge_energy = (self._temp_max - self.temp_water) * self._heat_capacity  # Ws
        # # Add heat energy loss
        # max_th_charge_energy += self.heat_loss(self.temp_water, self._temp_max, self.temp_ambient) * t_delta
        # max_th_charge_power = (max_th_charge_energy / t_delta) / self._efficiency_charging
        # # Charging power can be raised by expected discharging power
        # max_th_charge_power += expected_th_load / self._efficiency_discharging
        return max_th_charge_power

    def min_charge_power_th(self, t_delta: float, expected_th_load: float = 0):
        """
        Minimum charging power required to keep the storage temperature at or above the lower limit, incl. losses
        :param t_delta: Period duration [s]
        :param expected_th_load: Expected thermal load, i.e. expected discharge (efficiency considered)
        :return: min. charge power (>= 0)
        """
        if expected_th_load > 0:
            expected_th_discharge_energy = -expected_th_load * t_delta / self._efficiency_discharging
            loss_term = self.heat_loss_term(t_delta)
            expected_stored_energy_after = (self.stored_energy*(1-loss_term) + expected_th_discharge_energy)/(1 + loss_term)
            expected_th_discharge_power = (expected_stored_energy_after - self.stored_energy)/t_delta

            # expected_end_temp = self.temp_water - expected_th_discharge_energy / self._heat_capacity
            # expected_loss = self.heat_loss(self.temp_water, expected_end_temp, self.temp_ambient)  # W
            # expected_total_th_discharge_power = expected_th_load / self._efficiency_discharging + expected_loss

            max_discharge_power = self.max_discharge_power_th(t_delta, soft_min=True) # <= 0
            # Recharge difference to maintain the set min. temperature
            # Loss is already considered
            min_th_charge_power = max(0, max_discharge_power - expected_th_discharge_power) / self._efficiency_charging

            return min_th_charge_power

        elif self.temp_water <= self._temp_min:
            # No load, but temperature is already below min. temp -> recharge at least until min.
            min_th_charge_energy = (self._temp_min - self.temp_water) * self._heat_capacity
            min_th_charge_energy += self.heat_loss(self.temp_water, self._temp_min, self.temp_ambient) * t_delta
            return (min_th_charge_energy / t_delta) / self._efficiency_charging


        # safe_max_th_discharge_power = self.stored_energy / self.safe_th_supply_duration  # HWT should be able to supply household for at least 15 min; include conversion loss HWT-pipes
        # if expected_th_load > safe_max_th_discharge_power:
        #     # current heat load would empty storage within 15 min -> charge difference
        #     return (expected_th_load - safe_max_th_discharge_power) / self._charging_efficiency
        # else:
        #     return 0
        return 0.0

    def max_discharge_power_th(self, t_delta: float, soft_min: bool = True):
        """
        Highest possible discharging power without decreasing the water temperature below the lower limit
        :param t_delta: Period duration [s]
        :param soft_min: If the specified min. temperature of storage shall be respected (False -> min := ambient temp.)
        :return: max. discharge power (<= 0)
        """
        min_temp = self._temp_min if soft_min else self.temp_ambient
        min_stored_energy = (min_temp - self.temp_ambient) * self._heat_capacity
        loss_term = self.heat_loss_term(t_delta)
        max_th_discharge_power = (min_stored_energy * (1 + loss_term) - self.stored_energy * (1 - loss_term)) / (t_delta * self._efficiency_discharging)

        # max_th_discharge_energy = self._heat_capacity * (self.temp_water - min_temp)  # Ws
        # # Subtract heat energy loss
        # max_th_discharge_energy -= self.heat_loss(self.temp_water, min_temp, self.temp_ambient) * t_delta
        # max_th_discharge_power = max(0, max_th_discharge_energy) / t_delta * self._efficiency_discharging
        # return -max_th_discharge_power
        return max_th_discharge_power

    def feasible_actions(self, t_delta: int, *args) -> np.ndarray:
        return None

    def flexibility(self, t_delta, heat_load: float = None) -> typing.Tuple[float, float]:
        """
        Calculate min. and max. charging power (thermal), assuming that the given or current heat load
        remains constant for the given duration t_delta.
        :param t_delta: Time duration [s]
        :param temperature: Temperature the storage model shall be set to (if None, last measurement is taken.)
        :param heat_load: Heat load to be assumed
        :return: min. and max. charging power (thermal)
        """
        if not heat_load:
            heat_load = get_latest_measurement(os.getenv('LOAD_TH_KEY'), fields='active_power')
        max_th_charge_power = self.max_charge_power_th(t_delta, heat_load)
        min_th_charge_power = self.min_charge_power_th(t_delta, heat_load)

        return min_th_charge_power, max_th_charge_power

    def make_state_transition(self, t_delta: float, action: float, interaction: np.ndarray = np.zeros(2)) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Charge or discharge hot water storage with the given power if feasible.
        :param t_delta: Time duration [s] for which to (dis)charge
        :param action: None (passive system)
        :param interaction: Interaction from connected systems -> th. charging by heat generator or discharging by heating system (demand)
        :return: Updated state, resulting interaction with connected systems (0: electric, 1: thermal)
        """
        stored_energy_before = self.stored_energy

        th_power = interaction[1]
        if th_power >= 0:
            feasible_power_th = min(th_power, self.max_charge_power_th(t_delta))
            efficiency_factor = self._efficiency_charging
        else:
            feasible_power_th = max(th_power, self.max_discharge_power_th(t_delta))
            efficiency_factor = 1/self._efficiency_discharging
        loss_term = self.heat_loss_term(t_delta)
        stored_energy_after = (stored_energy_before*(1-loss_term) + t_delta * feasible_power_th * efficiency_factor)/(1 + loss_term)
        self.temp_water = stored_energy_after/self._heat_capacity + self.temp_ambient

        return self.state, interaction - np.array([0, feasible_power_th])


