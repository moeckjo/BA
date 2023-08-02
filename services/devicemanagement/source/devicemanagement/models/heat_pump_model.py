import numpy as np
import typing

from devicemanagement import logger
from devicemanagement.models.device_model import Device
from devicemanagement.models.heat_storage_model import HotWaterStorage
from devicemanagement.utils import get_latest_measurement


class HeatPump(Device):
    """
    Heat pump without inverter (either on or off, no modulation)
    """

    min_continuous_runtime = 10 * 60  # seconds (10 minutes)

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
        self._inverter: bool = specification['inverter']
        self._controllable: bool = specification['controllable']
        self._active_power_th_max = int(specification['active_power_th_max_kW'] * -1000)  # generation -> negative sign
        #self._efficiency = specification['quality_grade']

        self._save_specifications()

        # Variables
        # 0: running, 1: heat source temp., 2: heat sink temp., 3. remaining min. runtime
        self.state = np.array([0, 0.0, 35.0, 0])

        if self.controllable:
            self.actions = np.array([0, 1])  # on or off

    @property
    def controllable(self):
        return self._controllable

    @property
    def running(self):
        return self.state[0]

    @running.setter
    def running(self, value: typing.Union[bool, None]):
        """
        Update operation state.
        :param value: If provided, set state to this value; otherwise take the last measurement written to the database
        """
        if value is None:
            value = get_latest_measurement(source=self._key, fields='state')
        self.state[0] = value

    @property
    def temp_heat_source(self):
        return self.state[1]

    @temp_heat_source.setter
    def temp_heat_source(self, value: typing.Union[float, None]):
        """
        Update source temperature.
        :param value: If provided, set source temperature to this value; otherwise take the last measurement written to the database
        """
        if value is None:
            value = get_latest_measurement(source=self._key, fields='source_temp')
        self.state[1] = value

    @property
    def temp_heat_sink(self):
        return self.state[2]

    @temp_heat_sink.setter
    def temp_heat_sink(self, value: float):
        self.state[2] = value

    @property
    def remaining_min_runtime(self):
        return self.state[3]

    @remaining_min_runtime.setter
    def remaining_min_runtime(self, value: int):
        self.state[4] = value

    def cop(self, temp_heat_source, temp_heat_sink: float = 35, quality_grade: float = None) -> float:
        # TODO: Either implement function that approximates the COP or read from some mapping
        # TODO: problem with Dimplex data: source_temp is None when HP is off
        cop = 2.8 + (temp_heat_source - (
            -7)) * 0.1  # Approximated from specs of Dimplex LA9TU heat pump with W35 (used in Lila Walldorf)

        # Alternative calculation: Carnot COP * quality_grade (nu)
        if quality_grade:
            cop_carnot = temp_heat_sink / (temp_heat_sink - temp_heat_source)
            cop = quality_grade * cop_carnot
        return cop

    def feasible_actions(self, t_delta: int, heat_storage: HotWaterStorage, *args) -> np.ndarray:
        limits = self.flexibility(t_delta, heat_storage)
        if limits[1] == 0:
            # Max power = 0 -> don't run
            return np.array([0])
        elif limits[0] != 0:
            # Min not 0 -> must run
            return np.array([1])
        # Both states are feasible
        return np.array([0, 1])

    # def get_flexibility(self, t_delta, heat_storage, run: bool = None, temp_heat_source: float = None, *args,
    #                     **kwargs) -> typing.Tuple[float, float]:
    def flexibility(self, t_delta: int, heat_storage: HotWaterStorage, *args, **kwargs) -> typing.Tuple[
        float, float]:
        """
        Calculate min. and max. power (el.), depending on the current state and flexibility of the heat storage, and the source and target temperature
        :param t_delta: Time duration [s]
        :param heat_storage: Heat storage that is attached to the heat pump
        :return: max. generation (=0), min. generation (=0), min. consumption and max. consumption power (el.)
        """
        # TODO: Account for temporal restrictions and inertia -> or maybe just remove it from instant (= 1min) flexibility, because it cannot react that fast?
        # TODO: Model real behavior of HP and HWT (e.g. COP/el_power dependeing on source temp & target temp, inertia of HP, conversion losses HP-HWT and HWT-pipes, ....)

        # self.running = run
        # self.temp_heat_source = temp_heat_source

        # TODO: What happens if storage min > 0 and storage max < heat pump max? Is it a realistic case for short durations suh as 1 min?
        if not self.running and t_delta < self.min_continuous_runtime:
            # Don't switch it on for short intervals
            return 0, 0, 0, 0

        else:
            storage_min_th_charge_power, storage_max_th_charge_power = heat_storage.flexibility(t_delta)
            storage_parameters = heat_storage.device_parameters

            if self.remaining_min_runtime > t_delta or storage_min_th_charge_power > 0:
                # If has to continue running or switched on
                min_th_power = self._active_power_th_max
                min_el_power = min_th_power / self.cop(self.temp_heat_source,
                                                       self.temp_heat_sink)  # storage_parameters['_min_temp'])
            else:
                min_el_power = 0

            max_el_power = min(storage_max_th_charge_power, self._active_power_th_max) / self.cop(self.temp_heat_source,
                                                                                                  self.temp_heat_sink)  # storage_parameters['_max_temp'])

            return 0, 0,  max_el_power, min_el_power

    def make_state_transition(self, t_delta: int, action: bool, interaction: np.ndarray = np.zeros(2), temp_heat_source: float = None) -> typing.Tuple[np.ndarray, np.ndarray]:
        # self.set_state(to_state)
        previous_state = self.running

        self.running = action
        self.temp_heat_source = temp_heat_source

        power_th = self.running * self._active_power_th_max
        power_el = -power_th / self.cop(self.temp_heat_source)

        # Reset runtime condition if switching mode, else reduce duration
        if self.running != previous_state:
            self.remaining_min_runtime = 0
        else:
            self.remaining_min_runtime -= t_delta

        return self.state, interaction - np.array([power_el, power_th])



class InverterHeatPump(HeatPump):
    """
    Heat pump with inverter -> Power modulation through frequency control
    Variable th. power output and hence el. power need between defined limits.
    """

    # -> Together with heat storage similar concept as a battery storage, but with more constraints and variable parameters (?)

    def __init__(self, key: str, specification: dict):
        self._active_power_th_min = specification['active_power_th_min_kW'] / -1000  # generation -> negative sign

        super().__init__(key, specification)

        self.actions = self.discretize_power_range(
            p_min=self._active_power_th_min,
            p_max=self._active_power_th_max,
            granularity=1.0,
            min_power_offset=True
        )

    def make_state_transition(self, t_delta: int, action: bool, interaction: np.ndarray = np.zeros(2), temp_heat_source: float = None) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Operate heat pump with given el. power
        :param t_delta: Time duration [s]
        :param action: El. power [W]
        :param interaction: Possible interaction from connected system
        :param temp_heat_source: Temperature of heat source [C]
        :return:
        """
        # TODO: What exactly can be set/controlled?
        previous_state = self.running

        self.temp_heat_source = temp_heat_source
        if action == 0:
            self.running = False
            power_th = 0
        else:
            self.running = True
            power_th = -action * self.cop(self.temp_heat_source, self.temp_heat_sink)

        # Reset runtime condition if switching mode, else reduce duration
        if self.running != previous_state:
            self.remaining_min_runtime = 0
        else:
            self.remaining_min_runtime -= t_delta

        return self.state, -np.array([action, power_th])

    def flexibility(self, t_delta: int, heat_storage: HotWaterStorage, *args, **kwargs) -> typing.Tuple[float, float]:
        """
        Calculate min. and max. power (el.), depending on the current state and flexibility of the heat storage, and the source and target temperature
        :param t_delta: Time duration [s]
        :param heat_storage: Heat storage that is attached to the heat pump
        :return: max. generation (=0), min. generation (=0), min. consumption and max. consumption power (el.)
        """
        # TODO: Currentl implmentation: flex is 0 if off
        # TODO: Account for temporal restrictions and inertia -> or maybe just remove it from instant (= 1min) flexibility, because it cannot react that fast?
        # Notes: There might be temporal restriction, e.g. cannot be shut down if has just been started
        # TODO: Model real behavior of HP and HWT (e.g. COP/el_power dependeing on source temp & target temp, inertia of HP, conversion losses HP-HWT and HWT-pipes, ....)

        # self.running = run
        # self.temp_heat_source = temp_heat_source
        # TODO: What exactly can be set/controlled?
        self.running = kwargs.get('state', None)  # Get state from DB if None
        if not self.running and t_delta < self.min_continuous_runtime:
            # Don't switch it on for short intervals
            return 0, 0, 0, 0
        else:
            storage_min_th_charge_power, storage_max_th_charge_power = heat_storage.flexibility(t_delta)
            storage_parameters = heat_storage.device_parameters

            if self.remaining_min_runtime > t_delta or storage_min_th_charge_power > 0:
                # If has to continue running or switched on
                min_th_power = max(min(storage_min_th_charge_power, self._active_power_th_max), self._active_power_th_min)
                min_el_power = -min_th_power / self.cop(self.temp_heat_source,
                                                       temp_heat_sink=self.temp_heat_sink)  # storage_parameters['_max_temp'])
            else:
                min_el_power = 0

            max_el_power = min(storage_max_th_charge_power, -self._active_power_th_max) / self.cop(self.temp_heat_source,
                                                                                                   temp_heat_sink=self.temp_heat_sink)  # target_temp=storage_parameters['_max_temp'])

            return 0, 0,  max_el_power, min_el_power
