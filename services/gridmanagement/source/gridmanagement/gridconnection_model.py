import os
import numpy as np
import typing

from gridmanagement import logger

class GridConnectionPoint():
    """
    Model of a building's grid connection point.

    The power that is exchanged with the grid may be restricted (-> state variable).
    The type of restriction depends on the system that the building participates in (e.g. quota system).
    Here:
        - Limitation is given as a power value [W]
        - Limitation can be set by either providing an absolute power value or a relative value together with a reference value
        - The reference value is either the max. power (e.g. nominal PV power or total conditional consumption)
            or the planned power (result of the schedule optimization)
    """
    def __init__(self, specifications: dict):
        # Specifications
        self._category = os.getenv('GRID_CONNECTION_POINT_KEY')
        self._key = self._category
        self._cluster_id = specifications["cluster_id"]
        self._dso_uuid = specifications["uuid"]
        self._unconditional_consumption = specifications["unconditional_consumption"]  # W
        self._conditional_consumption = specifications["conditional_consumption"]  # W

        if not specifications["dynamic_tariff"]:
            self._tariff_consumption = specifications["grid_consumption_tariff_euro_kwh"]  # EUR/kWh
            self._tariff_feedin = specifications["grid_feedin_tariff_euro_kwh"]  # EUR/kWh

        self._limit_tolerance_active_power_abs = specifications["limit_tolerance_active_power_abs"]
        self._limit_tolerance_active_power_rel = specifications["limit_tolerance_active_power_rel"]
        self._limit_tolerance_active_power_rel_application_threshold = \
            specifications["limit_tolerance_active_power_rel_application_threshold"]

        # State variable = current grid exchange (consumption (+), feedin (-)) [W]
        self.state = 0
        # Power feedin or consumption limit
        # 0: consumption limit (+), 1: feedin limit (-) [W]
        self.limits = np.array([self._unconditional_consumption + self._conditional_consumption, -1000000])

    def __str__(self):
        return self._key

    @property
    def dso_uuid(self) -> str:
        return self._dso_uuid

    @property
    def cluster_id(self) -> str:
        return self._cluster_id

    @property
    def unconditional_consumption(self) -> str:
        return self._unconditional_consumption

    @property
    def conditional_consumption(self) -> str:
        return self._conditional_consumption

    @property
    def consumption(self):
        return max(self.state, 0)

    @property
    def feedin(self):
        return min(self.state, 0)

    @property
    def consumption_limit(self) -> float:
        """
        :return: Maximum grid consumption power [W] (+)
        """
        return self.limits[0]

    @consumption_limit.setter
    def consumption_limit(self, abs_value: int):
        """
        Set maximum grid consumption power by providing to an absolute power value (>= 0)
        :param abs_value: Absolute power value [W] (>= 0)
        """
        # Only conditional consumption may be restricted
        self.limits[0] = max(abs_value, self._unconditional_consumption)

    @property
    def feedin_limit(self) -> float:
        """
        :return: Maximum grid feed-in power [W] (-)
        """
        return self.limits[1]  # negative

    @feedin_limit.setter
    def feedin_limit(self, abs_value: int):
        """
        Set maximum grid feed-in power by providing by providing to an absolute power value (<= 0)
        :param abs_value: Absolute value [W] (<= 0)
        """
        self.limits[1] = abs_value

    def tolerance(self, active_power_limit: float = None) -> tuple:
        """
        Calculate the lower and upper bounds for a given active power limit based on the defined absolute
        and relative tolerances for this grid connection point. Neither tolerance may be violated, hence the
        returned bounds contain the stronger restriction given this limit.
        :param active_power_limit: Limit for the active power exchange with the grid, signed (+=consumption), in Watt
        :return: Resulting bounds such that neither tolerance is violated.
        """
        abs_bounds = (
            active_power_limit - self._limit_tolerance_active_power_abs,
            active_power_limit + self._limit_tolerance_active_power_abs
        )
        rel_bounds = (
            active_power_limit * (1 - self._limit_tolerance_active_power_rel),
            active_power_limit * (1 + self._limit_tolerance_active_power_rel)
        )
        if abs(active_power_limit) <= self._limit_tolerance_active_power_rel_application_threshold:
            # If target power is very small (e.g. below 100W or even zero) set bounds to the absolute tolerance,
            # because the relative tolerance is (nearly) zero.
            bounds = abs_bounds
        else:
            bounds = (
                max(min(abs_bounds), min(rel_bounds)),
                min(max(abs_bounds), max(rel_bounds))
            )
        return bounds

    def flexibility(self, max_generation: typing.List[int], min_generation: typing.List[int], max_load: typing.List[int], min_load: typing.List[int]) -> typing.Tuple[int, int]:
        """
        Flexibility at grid connection resulting from the flexibility of all devices and inflexible load and generation.
        Respects the unconditional consumption power.
        :param max_generation: List of max. power values of all generation units
        :param min_generation: List of min. power values of all generation units
        :param max_load: List of max. power values of all consumption units
        :param min_load: List of min. power values of all consumption units
        :return: Lower and upper bounds of exchangeable power [W] in the given period
        """
        # Lower bound: max. feed-in (<0) or min. consumption (>0)
        lower_bound = sum(max_generation) + sum(min_load)
        # if lower_bound >= 0:
        #     # No feed-in possible. Set max_feedin=min_consumption to unconditional consumption
        #     lower_bound = self._unconditional_consumption
        # Upper bound: max. consumption (>0) or min. feedin (<0)
        upper_bound = sum(max_load) + sum(min_generation)
        return lower_bound, upper_bound


