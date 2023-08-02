import math
import os

import numpy as np
import typing
from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint, \
    PercentFraction, PositiveIntegers, NonNegativeIntegers, Integers
from pyomo.environ import inequality, value

from core.optimization import logger


def get_first_connection_end_period(connection_states):
    # Default: Once connected, EV stays connected until the end of the optimization horizon
    first_connection_block_end_period = len(connection_states) - 1
    if min(connection_states) == 0:
        if sum(connection_states) != 0:
            prev_connection_state = connection_states[0]
            for t, cs in enumerate(connection_states[1:]):
                if prev_connection_state == 1 and cs == 0:
                    first_connection_block_end_period = t
                    break
                else:
                    prev_connection_state = cs
        else:
            first_connection_block_end_period = 0
    return first_connection_block_end_period

def get_quick_charging_periods():
    # TODO: move code below in this function and return arrays for quick_charge and pref_charge
    pass


def factory(model: ConcreteModel, name: str, specification: typing.NamedTuple, init_state: typing.Dict[str, float],
            forecast: np.ndarray,
            **kwargs):
    """
    :param model: The optimization model
    :param name: Unique device key
    :param specification: Technical specification of the EVSE
    :param init_state: Initial SOC at the beginning of the optimization horizon (dict with single pair, key='soc')
    :param forecast: Forecast or plan according to user input, if the EV is connected
    """

    # Array  with 1 for all periods from arrival to expected/planned departure, 0 otherwise
    connection_states = forecast

    soc_init = init_state.get('soc')
    if soc_init is None:
        # Case if either init_states is an empty dict or value of init_states['soc'] = None
        soc_init = float(os.getenv('EV_SOC_DEFAULT'))

    SOC_TARGET = 1.0
    if sum(connection_states) == 0:
        # No opportunity to charge if it's never connected
        SOC_TARGET = soc_init
    SOC_MIN = float(os.getenv('EV_SOC_MIN'))
    SOC_PREF = float(os.getenv('EV_SOC_PREFERRED'))

    # Get period before first disconnection (if ever connected or disconnected)
    first_connection_block_end_period = get_first_connection_end_period(connection_states)

    logger.debug(f'{name} init SOC: {soc_init}; {name} specification: {specification}')
    logger.debug(f'EVSE connection states: {connection_states}')
    logger.debug(f'EVSE connection_states array length={len(connection_states)}')
    logger.debug(f"EVSE gets disconnected for the first time in "
                 f"period {first_connection_block_end_period + 1} (count starts at 1)")

    # Determine the periods in which to charge with max. power to reach the min. SOC asap
    quick_charge_energy = max(0, SOC_MIN - soc_init) * specification.capacity / specification.efficiency
    logger.debug(
        f'quick_charge_energy={quick_charge_energy}, specification.active_power_charge_nominal={specification.active_power_charge_nominal}, model.dt={model.dt}')

    try:
        quick_charge_periods = math.ceil(quick_charge_energy / specification.active_power_charge_nominal / model.dt)
    except TypeError:
        # Sometimes, python claims division by None, although model.dt is definitely not None, but an int
        # -> Workaround: evaluate the parameter to get a "normal" int
        params = model.component_objects(Param, active=True)
        for param in params:
            if str(param) == 'dt':
                dt_value = [value(param[i]) for i in param][0]
                logger.debug(f'(EVSE) Value of dt={dt_value}')
                break
        quick_charge_periods = math.ceil(quick_charge_energy / specification.active_power_charge_nominal / dt_value)

    if quick_charge_periods > 0:
        pref_charge_energy = (SOC_PREF - SOC_MIN) * specification.capacity / specification.efficiency
    else:
        pref_charge_energy = max(0, SOC_PREF - soc_init) * specification.capacity / specification.efficiency

    logger.debug(
        f'pref_charge_energy={pref_charge_energy}, specification.active_power_charge_nominal={specification.active_power_charge_nominal}, model.dt={model.dt}')
    try:
        pref_charge_periods = math.ceil(
            pref_charge_energy / (0.5 * specification.active_power_charge_nominal) / model.dt)
    except TypeError:
        # Sometimes, python claims division by None, although model.dt is definitely not None, but an int
        # -> Workaround: evaluate the parameter to get a "normal" int
        params = model.component_objects(Param, active=True)
        for param in params:
            if str(param) == 'dt':
                dt_value = [value(param[i]) for i in param][0]
                logger.debug(f'(EVSE) Value of dt={dt_value}')
                break
        pref_charge_periods = math.ceil(
            pref_charge_energy / (0.5 * specification.active_power_charge_nominal) / dt_value)

    logger.debug(f'EVSE: Quick charge periods: {quick_charge_periods}; Pref. charge periods: {pref_charge_periods}')

    # Charge with max. power (or 50% of it) for the necessary number of periods as soon as the vehicle is connected
    quick_charge = [0] * len(model.T)
    pref_charge = [0] * len(model.T)

    for i, connected in enumerate(connection_states):
        if sum(quick_charge) == quick_charge_periods:
            if sum(pref_charge) == pref_charge_periods:
                break
            if connected: pref_charge[i] = 1
        else:
            if connected: quick_charge[i] = 1

    logger.debug(f'EVSE: Quick charge array: {quick_charge}; Pref. charge array: {pref_charge}')

    # departure = kwargs.get('departure')

    def s(key, value):
        setattr(model, name + '_' + key, value)

    def g(key):
        return getattr(model, name + '_' + key)

    # Set parameters
    s('b_connected', Param(model.T, initialize={i: int(v) for i, v in enumerate(connection_states, start=1)}))
    s('b_quick_charge', Param(model.T, initialize={i: int(v) for i, v in enumerate(quick_charge, start=1)}))
    s('b_pref_charge', Param(model.T, initialize={i: int(v) for i, v in enumerate(pref_charge, start=1)}))
    s('w_target', Param(within=PositiveIntegers, default=100))  # priority of reaching target SOC (1=lowest)
    # Penalty for prohibiting charging despite a quick/pref charging constraint -> set to high value if charging with
    # little power is preferred over delayed charging with more power
    s('w_charging', Param(within=PositiveIntegers, default=1000000))

    # Variables
    s('b_charging', Var(model.T, within=Binary))  # charging flag
    s('P_el', Var(model.T, within=NonNegativeReals))  # Charging power
    s('SOC', Var(model.T, within=PercentFraction))  # SOC within [0,1]
    s('epsilon_target',
      Var(within=PercentFraction, bounds=(0, SOC_TARGET - soc_init)))  # slack variable to soften target SOC constraint
    s('epsilon_quick_charge', Var(model.T, within=NonNegativeReals, initialize={i: 0 for i in
                                                                                model.T}))  # slack variable to soften quick and preferred charge constraint
    # Slack variable to soften overall charge constraint (otherwise problem might get infeasible due to the
    # min. charging power if there are grid constraints)
    s('epsilon_b_charging', Var(model.T, within=Binary))

    # Charging only possible if connected, but not mandatory
    def connection_constraint(model, t):
        return g('b_charging')[t] <= g('b_connected')[t]

    s('con_connected', Constraint(model.T, rule=connection_constraint))

    # Mandatory charging if below min. or preferred SOC ("quick/pref. charging"), but optional afterwards
    # Except grid constraints even prohibit a charging with the minimum power
    def quick_charging_constraint(model, t):
        return g('b_charging')[t] >= g('b_quick_charge')[t] + g('b_pref_charge')[t] - g('epsilon_b_charging')[t]

    s('con_quick_charge', Constraint(model.T, rule=quick_charging_constraint))

    # charging power lower than max if charging else 0
    def charging_power_constraint_max(model, t):
        return g('P_el')[t] <= specification.active_power_charge_nominal * g('b_charging')[t]

    s('con_charging_max', Constraint(model.T, rule=charging_power_constraint_max))

    # In case of either quick or preferred charging, charging should (soft-constraint) be greater or equal to the
    # defined quick or preferred charging power, respectively.
    def charging_power_constraint_quick(model, t):
        return g('P_el')[t] >= specification.active_power_charge_nominal * (
                    g('b_quick_charge')[t] + 0.5 * g('b_pref_charge')[t]) \
               - g('epsilon_quick_charge')[t] * (g('b_quick_charge')[t] + g('b_pref_charge')[t])

    s('con_charging_quick', Constraint(model.T, rule=charging_power_constraint_quick))

    # If charging, the charging power always needs to be greater or equal than the technical minimum charging power
    def charging_power_constraint_min(model, t):
        return g('P_el')[t] >= specification.active_power_charge_min * g('b_charging')[t]

    s('con_charging_min', Constraint(model.T, rule=charging_power_constraint_min))

    # Equality constraint for state (SOC) computation based on (dis)charging power variables
    def state_equation(model, t):
        dE = model.dt * g('P_el')[t] * specification.efficiency
        previous_soc = soc_init if t == model.T.first() else g('SOC')[t - 1]
        return g('SOC')[t] == (previous_soc * specification.capacity + dE) / specification.capacity

    s('con_state', Constraint(model.T, rule=state_equation))

    # Target SOC must be reached before first departure or, if never connected or disconnected, at the
    # end of optimization horizon
    def soc_constraint(model):
        return g('SOC')[first_connection_block_end_period + 1] == SOC_TARGET - g('epsilon_target')

    if SOC_TARGET != soc_init:
        # Only add constraint if unequal, because otherwise g('epsilon_target') is None due to its bounds=(0,0)
        s('con_state_target', Constraint(rule=soc_constraint))
