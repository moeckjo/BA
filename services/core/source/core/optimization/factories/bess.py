# from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint,
# from pyomo.environ import inequality
import os
from collections import namedtuple

import typing
from pyomo.environ import ConcreteModel, Var, Set, Param, Constraint, Binary, NonNegativeReals, Reals, inequality, \
    Integers, NonNegativeIntegers

from core.optimization import logger


def factory(model: ConcreteModel, name: str, specification: typing.NamedTuple, init_state: typing.Dict[str, float],
            **kwargs):
    """
    :param model: The optimization model
    :param name: Unique device key
    :param specification: Technical specification of the BESS
    :param init_state: Initial SOC at the beginning of the optimization horizon (dict with single pair, key='soc')
    """
    init_soc = init_state['soc']

    if init_soc > 1:
        init_soc = init_soc / 100

    logger.debug(f'{name} init SOC: {init_soc}; {name} specification: {specification}')

    feasible_min_soc = [min(specification.soc_min, init_soc + i * (
            specification.active_power_charge_nominal * model.dt * specification.efficiency) / specification.capacity)
                        for i in model.T]
    logger.debug(f"Feasible min. SOC: {feasible_min_soc} (len={len(feasible_min_soc)})")

    def s(key, value):
        setattr(model, name + '_' + key, value)

    def g(key):
        return getattr(model, name + '_' + key)

    # Set decision variables
    # Var(index, within=domain)
    s('b_charging', Var(model.T, within=Binary))  # charging flag
    s('P_pos', Var(model.T, within=NonNegativeReals))  # charging power
    s('P_neg', Var(model.T, within=NonNegativeReals))  # discharging power (note: positive)
    # s('E', Var(model.T, within=NonNegativeReals))  # energy level
    s('P_el', Var(model.T, within=Reals))  # Either P_pos or P_neg
    s('SOC', Var(model.T, within=NonNegativeReals, bounds=(0, 1)))  # SOC within [0,1]
    s('E_charged_net', Var(within=Reals))  # Total charged energy in the optimization horizon
    s('epsilon_min_soc',Var(model.T, within=NonNegativeReals, bounds=(0, 1), initialize={i: 0 for i in model.T}))  # slack variable to soften min. SOC constraint

    def max_soc_constraint(model, t):
        if init_soc > specification.soc_max:
            return g('SOC')[t] <= init_soc
        return g('SOC')[t] <= specification.soc_max

    def min_soc_constraint(model, t):
        return feasible_min_soc[t - 1] - g('epsilon_min_soc')[t] <= g('SOC')[t]

    s('con_max_soc', Constraint(model.T, rule=max_soc_constraint))
    s('con_min_soc', Constraint(model.T, rule=min_soc_constraint))

    # Incentivize a reduction of the SOC to the defined maximum until the end of the optimization horizon. Only
    # actually applies if the initial SOC is above the defined maximum.
    def soc_reduction(model, t):
        return g('SOC')[model.T.last()] <= specification.soc_max

    s('con_soc_reduction', Constraint(model.T, rule=soc_reduction))

    # charging power <= max charging power if charging else 0
    def charging_power_constraint(model, t):
        return g('P_pos')[t] <= g('b_charging')[t] * specification.active_power_charge_nominal

    s('con_charging', Constraint(model.T, rule=charging_power_constraint))

    # discharging power <= max discharging power if not charging else 0
    def charging_power_constraint(model, t):
        return g('P_neg')[t] <= (1 - g('b_charging')[t]) * abs(specification.active_power_discharge_nominal)

    s('con_discharging', Constraint(model.T, rule=charging_power_constraint))

    # Equality constraint to determine signed power exchange of BESS
    def power_equation(model, t):
        return g('P_el')[t] == g('P_pos')[t] - g('P_neg')[t]

    s('con_power', Constraint(model.T, rule=power_equation))

    # Equality constraint for state (SOC) computation based on (dis)charging power variables
    def state_equation(model, t):
        dE = model.dt * (g('P_pos')[t] * specification.efficiency - g('P_neg')[t] / specification.efficiency)
        relative_loss = model.dt * specification.relative_loss_per_second
        previous_soc = init_soc if t == 1 else g('SOC')[t - 1]
        return g('SOC')[t] == ((previous_soc - relative_loss) * specification.capacity + dE) / specification.capacity

    s('con_state', Constraint(model.T, rule=state_equation))

    # Calculate the total charged energy at the end of the horizon
    # This energy (times efficiency) is available at the start of the next horizon, and must therefore not
    # be drawn from the grid
    def charged_energy_equation(model):
        return g('E_charged_net') == (
                g('SOC')[model.T.last()] - init_soc) * specification.capacity * specification.efficiency

    s('con_charged_energy', Constraint(rule=charged_energy_equation))
