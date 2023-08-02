# from pyomo.core.base import ConcreteModel, Set, Param, NonNegativeReals, Reals, Var, Binary, Constraint,
# from pyomo.environ import inequality
import os
import numpy as np
import typing

from pyomo.environ import ConcreteModel, Var, Set, Param, Constraint, Binary, NonPositiveReals, Reals, \
    NonPositiveIntegers, PositiveIntegers, PositiveReals, NonNegativeIntegers, PercentFraction, inequality

from core.optimization import logger

def factory(model: ConcreteModel, name: str, specification: typing.NamedTuple, forecast: np.ndarray, **kwargs):
    """
    :param model: The optimization model
    :param name: Unique device key
    :param specification: Technical specification of the PV plant
    :param forecast: Predicted generation in Watt (negative values)
    """

    # PV generation forecast
    generation = forecast
    # TODO: change back to specification when curtailment model works
    logger.warning(f"Setting continuous_curtailment=False and curtailment_levels=[] for {name}.")
    continuous_curtailment: bool = False  # specification.continuous_curtailment
    curtailment_levels: typing.List[float] = []  # specification.curtailment_levels
    if len(specification.curtailment_levels) > 10:
        print_specs = {k: v for k,v in specification._asdict().items() if k != "curtailment_levels"}
        logger.debug(
            f'{name} continuous curtailment: {continuous_curtailment}; {name} no. of curt. levels: '
            f'{len(curtailment_levels)}; {name} specification (without curtailment levels): {print_specs}')
    else:
        logger.debug(f'{name} continuous curtailment: {continuous_curtailment}; {name} no. of curt. levels: '
                     f'{len(curtailment_levels)}; {name} specification: {specification}')
    logger.debug(f'{name} generation forecast: {generation}')

    # Nominal power: specification.active_power_nominal

    # TODO: fix curtailment in this model!!!
    #   Fault here is:

    def s(key, value):
        setattr(model, name + '_' + key, value)

    def g(key):
        return getattr(model, name + '_' + key)

    # Set general parameters and variables
    s('w_curtailment', Param(within=NonNegativeIntegers, default=1000))  # priority of non-curtailment
    s('epsilon_curtailment',
      Var(model.T, within=NonPositiveReals, initialize={i: 0 for i in model.T})
      )  # slack variable to soften non-curtailment constraint and provide basis for penalty

    # Set parameters, sets, variables and constraints depending on the ability of curtailing the PV generation, i.e.
    # controlling the PV inverter output

    # TODO: remove assert statement when curtailment model works
    assert not continuous_curtailment and len(curtailment_levels) <= 1, \
        f"Currently, only non-curtailable PV models are supported. In device_config.json, define continuous_mod=false and curtailment_levels=[]"

    if not continuous_curtailment and len(curtailment_levels) <= 1:
        # TODO: this case is ok and works
        logger.debug("PV is not curtailable")
        # Not curtailable -> inverter output = generation, hence only a parameter and no variable or constraints
        s('P_el', Param(model.T, initialize={i: v for i, v in enumerate(generation, start=1)}))  # [W], <= 0

    if continuous_curtailment or len(curtailment_levels) > 1:
        # Generation (forecast)
        s('P_el_gen', Param(model.T, initialize={i: v for i, v in enumerate(generation, start=1)}))  # [W]

        # Set decision variables
        # Var(index, within=domain)
        s('P_el', Var(model.T, within=NonPositiveReals))  # Inverter output [W]

        if not continuous_curtailment:
            # TODO: this case must correctly developed
            logger.debug(f"PV is curtailable with levels={curtailment_levels}")

            # Curtailment levels (percentage of nominal power) may be discrete like [0,0.3,0.6,1] or
            # quasi-continuous such that each level results in a 1-Watt-step
            s('L', Set(within=PercentFraction, initialize=curtailment_levels))
            s('l', Var(model.T, within=PercentFraction))  # curtailment level
            s('l_selected', Var(model.T, g('L'), within=Binary))  # binary mask for curtailment levels

            # Select a discrete curtailment level
            def select_l(model, t):
                return sum(g('l_selected')[t, l] for l in g('L')) == 1

            s('con_curtailment_level_select', Constraint(model.T, rule=select_l))

            # Set the curtailment level
            def set_l(model, t):
                return g('l')[t] == sum(l * g('l_selected')[t, l] for l in g('L'))

            s('con_curtailment_level_set', Constraint(model.T, rule=set_l))

            # Difference to predicted power
            def curtailment_equation(model, t):
                return g('P_el')[t] + g('epsilon_curtailment')[t] == g('P_el_gen')[t]

            s('con_max_ouput', Constraint(model.T, rule=curtailment_equation))

            # Chosen curtailment level cannot be higher than the current generation

            # TODO: this would result in a curtailment level < 1 at all times where nominal power is not
            #  reached (i.e. always) and then lead to a power limitation command even though it might not be ncessary
            # def output_equation(model, t):
            #     # Negative values! -> below means greater
            #     # return g('P_el_gen')[t] <= g('l')[t] * specification.active_power_nominal
            #     return g('P_el_gen')[t] <= g('P_el')[t]
            #
            # s('con_output', Constraint(model.T, rule=output_equation))

            def curtailment_level_constraint(model, t):
                return g('P_el')[t] == specification.active_power_nominal * g('l')[t]

            s('con_curtailment_level', Constraint(model.T, rule=curtailment_level_constraint))

        else:
            # TODO: in this case, the PV output will be restricted to the prediction
            logger.debug("PV is continuously curtailable")

            def output_constraint(model, t):
                # Negative values! -> below means greater
                return g('P_el')[t] + g('epsilon_curtailment')[t] == g('P_el_gen')[t]

            s('con_output', Constraint(model.T, rule=output_constraint))

