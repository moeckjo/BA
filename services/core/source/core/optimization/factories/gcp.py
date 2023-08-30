# from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint,
# from pyomo.environ import inequality
import os

import numpy as np
import typing
from pyomo.environ import ConcreteModel, Var, Param, Constraint, Binary, NonNegativeReals, Reals, value

from core.optimization import logger


# from device_management.models.bess_model import BatteryStorage  # this should not be necessary


# def factory(model: ConcreteModel, name: str, specification: dict, devices: typing.Dict[str, dict], init_state: float, **kwargs):
def factory(model: ConcreteModel, name: str, specification: dict, devices: typing.List[typing.NamedTuple],
            forecast_load_el: np.ndarray,
            grid_restrictions: typing.Union[np.ndarray, None],
            grid_setpoints: typing.Union[np.ndarray, None],
            **kwargs):
    """
    :param grid_setpoints: Setpoints for grid consumption or feedin (None if there are no setpoints)
    :param grid_restrictions: Absolute power limits for grid consumption or feedin (can be None if there are no restrictions)
    :param forecast_load_el: Forecast of inflexible electric load within the building
    :param model: The optimization model
    :param specification: Technical specification of this building, e.g. grid point characteristics?
    # :param devices: Dict with name (key) and specification and initial state for each flexible device in this building
    :param devices: Dict of device names (key) and specifications. Names must equal the names used in the respective factories.
    """

    logger.debug(f"{name} specifications: {specification}")

    # TODO: get thermal load forecast
    forecast_load_th = []

    grid_consumption_static_limit = specification["unconditional_consumption"] + specification[
        "conditional_consumption"]
    #TESTING JONAS
    #grid_restrictions = [8000] * int((len(model.T)/2))  + [-5000] * int((len(model.T)/2))
    #grid_restrictions = [-5000] * len(model.T) 


    if grid_restrictions is None and grid_setpoints is None:
        logger.debug("No grid limits or grid setpoint.")
        grid_power_limits_cons = [grid_consumption_static_limit] * len(model.T)
        grid_power_limits_feedin = [model.M] * len(model.T)
        consum_limit_active=[0 for t in model.T]
        feedin_limit_active=[0 for t in model.T]
        logger.debug(f'No (actual) grid limits – limits for all periods: '
                     f'consumption={grid_consumption_static_limit} (unconditional + conditional consumption), '
                     f'feedin={value(model.M)} (Big M)')

    elif grid_restrictions is not None and grid_setpoints is None:
        logger.debug(f"There is not grid setpoint, but grid limits: {grid_restrictions}")
        # Replace None values – indicating no limit – with static consumption limit or artifical high feed-in limit
        #Part von Jonas
        logger.debug ("TEST")
        
        consum_limit_active= [(1 if (limit is not None) and ( grid_consumption_static_limit > limit > specification["unconditional_consumption"]) else 0) for limit in grid_restrictions]
        feedin_limit_active = [(1 if (limit is not None) and (limit > -(model.M.value)) and (limit < 0)  else 0) for limit in grid_restrictions]
        logger.debug(f'Grid limits_active: cons={consum_limit_active}, feedin={feedin_limit_active}')
        
        grid_power_limits_cons = [(limit if (limit is not None) and (limit > 0) else grid_consumption_static_limit) for limit in grid_restrictions]
        grid_power_limits_feedin = [(-limit if (limit is not None) and (limit <= 0) else model.M) for limit in grid_restrictions]
        #logger.debug(f'Grid limits: cons={grid_power_limits_cons}, feedin={grid_power_limits_feedin}')

    elif grid_setpoints is not None and grid_restrictions is None:
        logger.debug(f"There are no grid limits, but a grid setpoint: {grid_setpoints}")
        # If there's a consumption setpoint, set feedin limit to 0 and vice versa
        consum_limit_active=[0 for t in model.T]
        feedin_limit_active=[0 for t in model.T]
        grid_power_limits_cons = [(limit if limit > 0 else 0) for limit in grid_setpoints]
        grid_power_limits_feedin = [(-limit if limit <= 0 else 0) for limit in grid_setpoints]
        logger.debug(f'Grid setpoints: cons={grid_power_limits_cons}, feedin={grid_power_limits_feedin}')

    else:
        raise ValueError('Either grid_restrictions or grid_setpoints must be None!')

    #TESTING    
    #logger.debug(f'feedinlimit: {grid_power_limits_feedin}')
    logger.debug (model.M.value)
    logger.debug(f'unconditional: {specification["unconditional_consumption"]}')
    logger.debug(f'consumlimit: {grid_power_limits_cons}')
    logger.debug(f'static limit consum: {grid_consumption_static_limit}')
    logger.debug(f'Grid limits: cons={consum_limit_active}, feedin={feedin_limit_active}')

    generators_el = {}
    consumers_el = {}
    storages_el = {}
    generators_th = {}
    consumers_th = {}
    storages_th = {}

    for device in devices:
        if device.category == 'generator':
            generators_el[device.key] = device
        elif device.category == 'consumer':
            consumers_el[device.key] = device
        elif device.category == 'storage':
            storages_el[device.key] = device
        elif device.category == 'converter':
            # TODO: how to handle these? e.g. heat pump
            if device.controllable: # Otherwise it's consumption is part of the inflexible load
                if device.energy_carrier_in == "el" and device.energy_carrier_out == "th":
                    consumers_el[device.key] = device
                    generators_th[device.key] = device
                elif device.energy_carrier_in == "th" and device.energy_carrier_out == "el":
                    consumers_th[device.key] = device
                    generators_el[device.key] = device
                else:
                    logger.error(
                        f'Handling of converter from {device.energy_carrier_in} to {device.energy_carrier_out} not implemented.')

        else:
            logger.error(f'Unknown device category "{device.category}"')

    logger.debug(f'Generators: {generators_el.keys()}, consumers: {consumers_el.keys()}, storages: {storages_el.keys()}')

    def s(key, value):
        setattr(model, name + '_' + key, value)

    def g(key, name=name):
        return getattr(model, name + '_' + key)

    # Set parameters JONAS
    s('P_el_limit_pos', Param(model.T, initialize={i: v for i, v in enumerate(grid_power_limits_cons, start=1)}))
    s('P_el_limit_neg', Param(model.T, initialize={i: v for i, v in enumerate(grid_power_limits_feedin, start=1)}))
    s('feedin_limit_active', Param(model.T, initialize={i: v for i, v in enumerate(feedin_limit_active, start=1)}))
    s('consum_limit_active', Param(model.T, initialize={i: v for i, v in enumerate(consum_limit_active, start=1)}))
    

    limit_active = 1
    setpoint_active = 0

    s('P_load_el_inflex', Param(model.T, initialize={i: v for i, v in enumerate(forecast_load_el, start=1)}))
    if grid_setpoints is not None:
        limit_active = 0
        setpoint_active = 1
        # Force grid consumption in case of consumption setpoint. Otherwise setting consumption and feed-in
        # both to zero is a valid solution
        s('b_consumption', Param(model.T, within=Binary, initialize={i:(1 if limit > 0 else 0) for i,limit in enumerate(grid_power_limits_cons, start=1)}))
    else:
        # Usual case -> choose to consume or feedin
        s('b_consumption', Var(model.T, within=Binary, initialize={i: 0 for i in model.T}))  # Grid consumption flag



    # Set decision variables JONAS
    # Var(index, within=domain)
    s('P_pos', Var(model.T, within=NonNegativeReals))  # Grid consumption [W]
    s('P_neg', Var(model.T, within=NonNegativeReals))  # Grid feed-in [W]
    s('P_el', Var(model.T, within=Reals))  # Either P_pos or P_neg [W]
    s('epsilon_feedin_limit', Var(model.T, within=NonNegativeReals, initialize={i: 0 for i in model.T})) # Difference (actual_power - setpoint) >= 0 for consumption setpoint
    s('epsilon_pos_above_setpoint', Var(model.T, within=NonNegativeReals, initialize={i: 0 for i in model.T})) # Difference (actual_power - setpoint) >= 0 for consumption setpoint
    s('epsilon_neg_above_setpoint', Var(model.T, within=NonNegativeReals, initialize={i: 0 for i in model.T})) # Difference (actual_power - setpoint) >= 0 for feed-in setpoint
    s('epsilon_pos_below_setpoint', Var(model.T, within=NonNegativeReals, initialize={i: 0 for i in model.T})) # Difference (setpoint - actual_power) >= 0 for consumption setpoint
    s('epsilon_neg_below_setpoint', Var(model.T, within=NonNegativeReals, initialize={i: 0 for i in model.T})) # Difference (setpoint - actual_power) >= 0 for feed-in setpoint
    s('b_pos_above_setpoint', Var(model.T, within=Binary, initialize={i: 0 for i in model.T}))  # Flag if actual power >= setpoint
    s('b_neg_above_setpoint', Var(model.T, within=Binary, initialize={i: 0 for i in model.T}))  # Flag if actual power >= setpoint

    #Part von Jonas
    s('P_market_feedin', Var(model.T, within=Reals)) # Amount that offer/ buy on second market for feedin [W]
    s('P_market_consum', Var(model.T, within=Reals)) # Amount that offer/ buy on second market for consumption [W]

    def difference_limit_feedin (model,t):
        return g('P_market_feedin')[t] == (g('P_neg')[t]-g('P_el_limit_neg')[t])*g('feedin_limit_active')[t]

    def difference_limit_consum (model,t):
        return g('P_market_consum')[t] == (g('P_pos')[t]-g('P_el_limit_pos')[t])*g('consum_limit_active')[t]


    #TESTING
    #s('test_constriant' , Constraint (model.T,rule = test_constraint))
    s('con_difference_limit_feedin', Constraint(model.T, rule=difference_limit_feedin))
    s('con_difference_limit_consum', Constraint(model.T, rule=difference_limit_consum))

    


    def grid_consumption_limit_constraint(model, t):
        if t == model.T.first():
            logger.debug(f'Grid consumption limit: {g("P_el_limit_pos")[t] * g("b_consumption")[t]}')
        return g('P_pos')[t] <= g('P_el_limit_pos')[t] * g('b_consumption')[t]

    def grid_feedin_limit_constraint(model, t):
        return g('P_neg')[t] - g('epsilon_feedin_limit')[t] <= g('P_el_limit_neg')[t] * (1 - g('b_consumption')[t])

    if limit_active:
        s('con_grid_consumption', Constraint(model.T, rule=grid_consumption_limit_constraint))
        s('con_grid_feedin', Constraint(model.T, rule=grid_feedin_limit_constraint))

    def grid_setpoint_cons_constraint(model, t):
        return g('P_pos')[t] - g('epsilon_pos_above_setpoint')[t] + g('epsilon_pos_below_setpoint')[t] == g('P_el_limit_pos')[t] * g('b_consumption')[t] * setpoint_active

    def grid_setpoint_cons_above(model, t):
        return g('epsilon_pos_above_setpoint')[t] <= model.M * g('b_pos_above_setpoint')[t]

    def grid_setpoint_cons_below(model, t):
        return g('epsilon_pos_below_setpoint')[t] <= model.M * (1 - g('b_pos_above_setpoint')[t])

    def grid_setpoint_feedin_constraint(model, t):
        return g('P_neg')[t] - g('epsilon_neg_above_setpoint')[t] + g('epsilon_neg_below_setpoint')[t] == g('P_el_limit_neg')[t] * (1 - g('b_consumption')[t]) * setpoint_active

    def grid_setpoint_feedin_above(model, t):
        return g('epsilon_neg_above_setpoint')[t] <= model.M * g('b_neg_above_setpoint')[t]

    def grid_setpoint_feedin_below(model, t):
        return g('epsilon_neg_below_setpoint')[t] <= model.M * (1 - g('b_neg_above_setpoint')[t])


    if setpoint_active:
        s('con_grid_pos_setpoint', Constraint(model.T, rule=grid_setpoint_cons_constraint))
        s('con_grid_pos_above_setpoint', Constraint(model.T, rule=grid_setpoint_cons_above))
        s('con_grid_pos_below_setpoint', Constraint(model.T, rule=grid_setpoint_cons_below))

        s('con_grid_neg_setpoint', Constraint(model.T, rule=grid_setpoint_feedin_constraint))
        s('con_grid_neg_above_setpoint', Constraint(model.T, rule=grid_setpoint_feedin_above))
        s('con_grid_neg_below_setpoint', Constraint(model.T, rule=grid_setpoint_feedin_below))

    # Only energy from PV plant may be fed into the grid, not from other generators or storages
    def pv_only_feedin_constraint(model, t):
        total_pv_generation = sum(
            [g('P_el', device_key)[t] for device_key, device in generators_el.items() if device.subcategory == os.getenv('PV_KEY')]
        )
        return g('P_neg')[t] <= -total_pv_generation  # P_neg is positive, but P_el of PV plants negative

    if not setpoint_active:
        s('con_pv_feedin', Constraint(model.T, rule=pv_only_feedin_constraint))

    # Equality constraint to determine signed power exchange of building with grid
    def power_equation(model, t):
        return g('P_el')[t] == g('P_pos')[t] - g('P_neg')[t]

    s('con_power', Constraint(model.T, rule=power_equation))

    def power_balance_equation(model, t):

        generation = sum(g('P_el', name)[t] for name in generators_el.keys())  # negative
        consumption = sum(g('P_el', name)[t] for name in consumers_el.keys()) + g('P_load_el_inflex')[t]  # positive
        charge = sum(g('P_el', name)[t] for name in storages_el.keys())  # positive (charge) or negative (discharge)
        if t == model.T.first():
            logger.debug(f'Generation={generation}, consumption={consumption}, charge={charge}')
        return g('P_el')[t] == generation + consumption + charge

    s('con_power_balance', Constraint(model.T, rule=power_balance_equation))

