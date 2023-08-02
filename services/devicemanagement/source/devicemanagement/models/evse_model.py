import datetime
import numpy as np
import typing

from devicemanagement import logger
from devicemanagement.models.device_model import Device
from devicemanagement.utils import get_latest_measurement, get_latest_user_input_with_future_departure


class EVSE(Device):

    defaults = dict(
        soc=0.5,
        remaining_standing_time=4*3600,  # seconds
        connected=0,  # 0 or 1
        soc_min=0.2,
        soc_max=1.0,
        min_driving_range=30.0  # km
    )

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
        self._active_power_charge_nominal = int(specification['active_power_charge_nominal_kW'] * 1000)
        self._active_power_charge_min = int(specification['active_power_charge_min_kW'] * 1000)  # vehicle-battery specific
        self._fuel_economy = specification['fuel_economy_kWh/km'] * 1000 * 3600  # Ws/km
        self._efficiency = specification['efficiency']

        self._save_specifications()

        # State variables
        # 0: connected, 1: soc, 2: remaining standing time, 3: min. soc, 4: max. soc
        self.state = np.array([
            self.defaults["connected"],
            self.defaults["soc"],
            self.defaults["remaining_standing_time"],
            self.defaults["soc_min"],
            self.defaults["soc_max"],
        ])

        self.actions = self.discretize_power_range(
            p_min=self._active_power_charge_min,
            p_max=self._active_power_charge_nominal,
            granularity=1.0,
            min_power_offset=True
        )


    @property
    def connected(self) -> bool:
        return self.state[0]

    @connected.setter
    def connected(self, value=typing.Union[bool, None]):
        if value is None:
            value = get_latest_measurement(source=self._key, fields='connected')
            if value is None:
                logger.debug(f"Connection state from DB is {value}")
                value = self.defaults["connected"]
        self.state[0] = value

    @property
    def soc(self) -> float:
        return self.state[1]

    @soc.setter
    def soc(self, value=typing.Union[float, None]):
        if value is None:
            value = get_latest_measurement(source=self._key, fields='soc')
            if value is None:
                logger.debug(f"EV SOC from DB is {value}")
                value = self.defaults["soc"]
        self.state[1] = value if value <= 1.0 else value / 100

    @property
    def remaining_standing_time(self) -> int:
        return self.state[2]

    @remaining_standing_time.setter
    def remaining_standing_time(self, value: typing.Union[int, None]):
        if value is None:
            now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
            latest_user_input: typing.Union[dict, None] = get_latest_user_input_with_future_departure(now=now)
            try:
                departure = latest_user_input.get("scheduled_departure")
            except AttributeError:
                departure = None

            if departure is not None:
                time_until_dep = (datetime.datetime.fromisoformat(departure) - now)
                value = int(time_until_dep.total_seconds())
            else:
                value = self.defaults["remaining_standing_time"]
        self.state[2] = value

    @property
    def soc_min(self) -> float:
        return self.state[3]

    @soc_min.setter
    def soc_min(self, value=typing.Union[float, None]):
        self.state[3] = value if value <= 1.0 else value / 100

    @property
    def soc_max(self) -> float:
        return self.state[4]

    @soc_max.setter
    def soc_max(self, value=typing.Union[float, None]):
        self.state[4] = value if value <= 1.0 else value / 100

    @property
    def driving_range(self) -> float:
        return self.soc_to_driving_range(self.soc)

    def soc_to_driving_range(self, soc: float) -> float:
        """
        :return: Driving range [km] estimated from SOC and average fuel economy
        """
        return (soc * self._capacity) / self._fuel_economy

    def driving_range_to_soc(self, driving_range: float) -> float:
        return (driving_range * self._fuel_economy) / self._capacity

    def max_charge_power(self, t_delta: float) -> float:
        """
        :param t_delta: Time duration [s] for which to charge with constant power
        :return: Max. possible charging power [W] (=0 if not connected)
        """

        # TODO: Account for reduced max. charging power when close to boundaries
        if self.connected:
            # min((1.0-0.5)*cap / tdelta -> Watt > 0 / eff -> Watt > 0, 11000
            return min((self.soc_max - self.soc) * self._capacity / t_delta / self._efficiency,
                       self._active_power_charge_nominal)
        else:
            return 0.0

    def min_charge_power(self, t_delta: float, remaining_standing_time: int, min_driving_range: float = None) -> float:
        """
        Calculates the min. charging power [W] needed to achieve the requested driving range
        :param t_delta: Time duration [s] (reference period)
        :param remaining_standing_time: Total remaining time the vehicle is connected and can be charged
        :param min_driving_range: Min. driving range that should be available after the standing time
        :return: Min. charging power [W] (=0 if not connected)
        """
        # self.connected = None  # Get status from DB
        # self.soc = None  # Get status from DB

        if min_driving_range is None:
            min_driving_range = self.defaults["min_driving_range"]

        if self.connected:
            current_driving_range = self.soc_to_driving_range(self.soc)
            range_deficit = max(0, min_driving_range - current_driving_range)
            energy_deficit = range_deficit * self._fuel_economy
            # Amount that must be charged in this period, assuming charging with max. power in the following
            # periods within the remaining standing time
            charge_now = max(
                energy_deficit - (remaining_standing_time - t_delta) * self._active_power_charge_nominal * self._efficiency,
                0)
            return charge_now / t_delta / self._efficiency
        else:
            return 0.0

    def feasible_actions(self, t_delta: int, *args) -> np.ndarray:
        limits = self.flexibility(t_delta)
        actions = np.copy(self.actions)
        return actions[limits[0] <= actions <= limits[1]]

    def flexibility(self, t_delta: float, soc: float = None, connected: bool = None,
                    min_driving_range: float = None, remaining_time: int = None) -> typing.Tuple[float, float]:
        """
        Returns the min. and max. constant charging power, respectively, for the considered duration t_delta
        based on the given or last measured SOC and optionally, a min. driving range.
        :param t_delta: Time duration [s]
        :param soc: SOC the battery model shall be set to
        :param connected: EV connected to Wallbox
        :param min_driving_range: Requested min. driving range
        :param remaining_time: Remaining standing time of vehicle
        :return: min. and max. charging power
        """
        self.soc = soc
        self.connected = connected
        self.remaining_standing_time = remaining_time
        max_charge_power = self.max_charge_power(t_delta)
        min_charge_power = min(
            self.min_charge_power(t_delta, remaining_standing_time=self.remaining_standing_time, min_driving_range=min_driving_range),
            max_charge_power
        )
        return 0, 0, max_charge_power, min_charge_power

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
        charge_power = min(max(self.min_charge_power(t_delta, self.remaining_standing_time), action),
                           self.max_charge_power(t_delta))
        energy = charge_power * t_delta * self._efficiency

        self.soc = ((self.soc * self._capacity) + energy) / self._capacity
        self.remaining_standing_time -= t_delta

        return self.state, interaction - np.array([charge_power, 0])
