import typing
from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint, NonNegativeIntegers
from pyomo.environ import inequality

from core.optimization import logger

def factory(model: ConcreteModel, name: str, specification: typing.NamedTuple, init_state: dict, **kwargs):
    """
    :param model: The optimization model
    :param name: Unique device key
    :param specification: Technical specification of the HWT
    :param init_state: Initial state at the beginning of the optimization horizon: dict with 'temp_ambient' and 'temp_water' ([C])
    """

    # HWT_Vitocell_100 - V
    # specs: {'category': 'storage', 'subcategory': 'hwt', 'name': 'Vitocell 100-V', 'energy_carrier': 'heat',
    #         'volume': 200, 'max_temp': 65, 'min_temp': 30, 'charging_efficiency': 0.9, 'discharging_efficiency': 0.9,
    #         'loss_per_temp_delta': 1.16948796126123, 'heat_capacity': 829.6200000000001}


    def s(key, value):
        setattr(model, name+'_'+key, value)

    def g(key):
        return getattr(model, name+'_'+key)

    # TODO: ambient temperature is parameter
    #   -> Either: set same value (latest measurement) for all time steps
    #   -> Or: take measurements from previous day for the optimization horizon
    #   Here: Take option 1, i.e. simply copy intial ambient temp for all time steps
    s('temp_ambient', Param(model.T, initialize=dict(zip(model.T, [init_state['temp_ambient']]*len(model.T)))))
    s('temp_water', Var(model.T, within=NonNegativeReals))
    # s('b_charging', Var(model.T, within=Binary))
    s('P_pos', Var(model.T, within=NonNegativeReals))
    s('P_neg', Var(model.T, within=NonNegativeReals))
    # s('P_th', Var(model.T, within=Reals))
    # TODO: possible to init variables?

    heat_loss_term = (model.dt / 2) * specification.loss_per_temp_delta / specification.heat_capacity

    def temperature_constraint(model, t):
        # return inequality(g('temp_ambient')[t], g('temp_water')[t], specification.temp_max)
        return inequality(specification.temp_min, g('temp_water')[t], specification.temp_max)
    s('con_temp', Constraint(model.T, rule=temperature_constraint))

    # TODO: HWT can be charged and discharged simultaneously, doesn't it?
    # # either charging or discharging
    # def charging_constraint(model, t):
    #     return g('P_pos')[t] <= g('b_charging')[t] * model.M
    #
    # s('con_charging', Constraint(model.T, rule=charging_constraint))
    #
    # # either charging or discharging
    # def discharging_constraint(model, t):
    #     return g('P_neg')[t] <= (1 - g('b_charging')[t]) * model.M
    #
    # s('con_discharging', Constraint(model.T, rule=discharging_constraint))

    # # Equality constraint to determine signed power exchange of HWT
    # def power_equation(model, t):
    #     return g('P_th')[t] == g('P_pos')[t] - g('P_neg')[t]
    # s('con_power', Constraint(model.T, rule=power_equation))

    # Charge if water temperature is already below min. allowed temperature
    # TODO: is this necessary or is it implicitly ensured by temp. constraint?
    def charging_constraint(model, t):
        temp = init_state['temp_water'] if t == 0 else g('temp_water')[t-1]
        if temp < specification.temp_min:
            min_charge_energy = (specification.temp_min - temp) * specification.heat_capacity
            heat_loss = specification.loss_per_temp_delta * (0.5 * (temp + specification.temp_min) - g('temp_ambient')[t])
            min_charge_energy += heat_loss * model.dt
            min_charge_power = min_charge_energy/model.dt/specification.efficiency_charging
        else:
            min_charge_power = 0
        return min_charge_power <= g('P_pos')[t]

    s('con_charging', Constraint(model.T, rule=charging_constraint))

    # Equality constraint for state  computation based on (dis)charging power variables
    def state_equation(model, t):
        dQ = model.dt * (g('P_pos')[t] * specification.efficiency_charging - g('P_neg')[t] / specification.efficiency_discharging)
        previous_temp = init_state['temp_water'] if t == 0 else g('temp_water')[t-1]
        previous_stored_energy = (previous_temp - g('temp_ambient')[t-1]) * specification.heat_capacity
        stored_energy = (previous_stored_energy * (1 - heat_loss_term) + dQ) / (1 + heat_loss_term)
        return g('temp_water')[t] == stored_energy / specification.heat_capacity + g('temp_ambient')[t]

    s('con_state', Constraint(model.T, rule=state_equation))

    '''
    Old
    '''
    # s('theta', Var(model.T, within=NonNegativeReals))
    # s('b_charging', Var(model.T, within=Binary))
    # s('P_pos', Var(model.T, within=NonNegativeReals))
    # s('P_neg', Var(model.T, within=NonNegativeReals))
    # s('P_th', Var(model.T, within=Reals))
    #
    # # minimum and maximum temperature
    # def con_temp(model, i):
    #     return inequality(hwt.ambient_temperature, g('theta')[i], hwt.max_temp)
    # s('con_temp', Constraint(model.T, rule=con_temp))
    
    # # either charging or discharging
    # def con_charging(model, t):
    #     return g('P_pos')[t] <= g('b_charging')[t] * model.M
    # s('con_charging', Constraint(model.T, rule=con_charging))
    #
    # # either charging or discharging
    # def con_discharging(model, t):
    #     return g('P_neg')[t] <= (1 - g('b_charging')[t]) * model.M
    # s('con_discharging', Constraint(model.T, rule=con_discharging))

    # # power
    # def con_power(model, t):
    #     return g('P_th')[t] == g('P_pos')[t] - g('P_neg')[t]
    # s('con_power', Constraint(model.T, rule=con_power))

    # state computation
    # def con_state(model, t):
    #     if t > 0:
    #         stored_energy = (g('theta')[t-1] - hwt.ambient_temperature) * hwt.tank_ws_per_k
    #     else:
    #         stored_energy = hwt.stored_energy
    #
    #     dQ = model.dt * (g('P_pos')[t] * hwt.charging_efficiency - g('P_neg')[t] / hwt.discharging_efficiency)
    #     relative_loss_term = (model.dt / 60. / 60) / 2 * hwt.relative_loss
    #     return g('theta')[t] == (stored_energy * (1 - relative_loss_term) + dQ) / (1 + relative_loss_term) / hwt.tank_ws_per_k + hwt.ambient_temperature
    # s('con_state', Constraint(model.T, rule=con_state))

    

