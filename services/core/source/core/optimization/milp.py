import logging
import typing
import os
import csv
import pandas



from pyomo.core.base import ConcreteModel, Objective, Constraint, Set, Param, Var, NonNegativeReals, Reals, RangeSet
from pyomo.environ import minimize, SolverFactory, value
import pyomo.environ as pyo

from core.optimization import logger
from .constraintfactory import ConstraintFactory




class _ZerosList:
    def __getitem__(self, i):
        return 0


class MILP:

    def __init__(self, time_step_duration, time_step_count):

        model = ConcreteModel()
        model.T = Set(initialize=RangeSet(time_step_count))  # 1 to time_step_count, stepsize=1
        model.dt = Param(initialize=time_step_duration)  # duration of a time step [s]
        model.M = Param(default=10000000)
        # model.P_total = Var(model.T, within=Reals)  # total power (see below con_P_total())
        
        

        self.model = model
        self.components = []  # store list of devices by name -> used to calculate total power in each time step
        self.objective_created = False

       
    def add_constraints(self, key: str, specification: typing.Union[dict, typing.NamedTuple], **kwargs):
        if key in self.components:
            raise ValueError('Name \'{}\' already registered'.format(key))
        if self.objective_created is True:
            raise RuntimeError('Objective has already been created')
        ConstraintFactory.add_to_model(pyomo_model=self.model, key=key, specification=specification, **kwargs)
        self.components.append(key)

    def solve(self, solver_type: str = 'glpk', time_limit=60, mipgap=0.05, verbose=True):

        if self.objective_created is False:
            raise RuntimeError('No objective')

        solver = SolverFactory(solver_type)
        if solver_type == 'gurobi':
            solver.options['timelimit'] = time_limit
            solver.options['mipgap'] = mipgap
        elif solver_type == 'glpk':
            solver.options['tmlim'] = time_limit
        
        logger.debug(f'Solver options: {solver.options}')
        result = solver.solve(self.model, tee=True)
        
        
        #print results to csv output file jonas 
        P_market_feedin =getattr(self.model,f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_feedin") # [W]
        P_market_consum =getattr(self.model,f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_consum") # [W]
        Speicher=getattr(self.model, f"{os.getenv('BESS_KEY')}_E_charged_net")
        price_market_feedin = 0.02 # [EURO/W]
        price_market_consum = 0.02 # [EURO/W]
           
        # Get variables defined with GCP
        feedin = getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_neg")  # [W]
        consumption = getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_pos")  # [W]
        # EUR / kWh * (W * kW/W) * (s * h/s) = EUR

        energy_cost_output = sum(
            (self.model.c_grid_cons[t] * consumption[t].value / 1000 - self.model.c_grid_feedin[t] * feedin[t].value / 1000) * (
                    self.model.dt / 3600) + (price_market_consum * P_market_consum[t].value + price_market_feedin* P_market_feedin[t].value)*(1/100) for t in self.model.T)
        if os.getenv('BESS_KEY') in self.components:
            energy_cost_output -= Speicher.value / (1000 * 3600) * sum(
                    self.model.c_grid_cons[t] for t in self.model.T) / len(self.model.T)    
        
        list_help_consum_tarif=[]
        list_help_feedin_tarif=[]
        list_help_consum= []
        list_help_feedin= []
        list_help_limit_active_feedin= []
        list_help_limit_active_consum= []
        list_help_limit_consum= []
        list_help_limit_feedin =[]
        secondmarket_feedin =[]
        secondmarket_con= []
        Konsum_gesammt= 0
        Feedin_gesammt= 0


        for t in self.model.T:
            Konsum_gesammt+= getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_pos")[t].value
            Feedin_gesammt+= getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_neg")[t].value
            list_help_consum_tarif.append (self.model.c_grid_cons[t])
            list_help_feedin_tarif.append (self.model.c_grid_feedin[t])
            list_help_consum.append (getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_pos")[t].value)
            list_help_feedin.append (getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_neg")[t].value)
            list_help_limit_active_feedin.append(getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_feedin_limit_active")[t])
            list_help_limit_active_consum.append(getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_consum_limit_active")[t])
            list_help_limit_consum.append(getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_el_limit_pos")[t])
            list_help_limit_feedin.append(getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_el_limit_neg")[t])
            secondmarket_feedin.append(getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_feedin")[t].value)
            secondmarket_con.append(getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_consum")[t].value)
            




        Zielfunktionswert = self.model.obj ()

        #CSV OUTPUT FILE
        #df = pandas.read_csv("/bem/CSV-Output/Outputfile.csv")
        # Erstellen des DataFrames aus den Listen
        data = {'Konsum': list_help_consum, 'Feedin': list_help_feedin, 'consumlimit_active': list_help_limit_active_consum, 'feedinlimit_active':list_help_limit_active_feedin,'limit_consum':list_help_limit_consum,'limit_feedin':list_help_limit_feedin,'secondmarket_consum':secondmarket_con,'secondmarket_feedin':secondmarket_feedin, 'Zielfunktionswert': Zielfunktionswert, 'Gesammtkostenenergie':energy_cost_output,'Tarif_consum':list_help_consum_tarif,'Tarif_feedin':list_help_feedin_tarif, 'Konsumgesammt':Konsum_gesammt,'feedin_gesammt':Feedin_gesammt}
        df = pandas.DataFrame(data)
        logger.debug (df)
        

        # Schreiben des DataFrames in eine CSV-Datei
        df.to_csv("/bem/CSV-Output/Outputfile.csv",mode='a', index=False)


        return result

class CostMILP(MILP):

    def create_objective(self, tariffs_grid_consumption: typing.List[float], tariffs_grid_feedin: typing.List[float]):

        
        """
        Objective: minimal total energy cost
        :param tariffs_grid_consumption: Price profile for grid consumption [EUR/kWh]
        :param tariffs_grid_feedin: Price profile for grid feed-in [EUR/kWh] / 1000
        """
        if self.objective_created:
            raise RuntimeError('Objective has already been created')
        self.objective_created = True
        if len(self.components) == 0:
            raise RuntimeError('No constraints have been added to the model')

        self.model.c_grid_cons = Param(self.model.T, initialize={i: v for i, v in enumerate(tariffs_grid_consumption,
                                                                                            start=1)})  # [EUR/kWh]
        self.model.c_grid_feedin = Param(self.model.T, initialize={i: v for i, v in enumerate(tariffs_grid_feedin,
                                                                                              start=1)})  # [EUR/kWh]

        def objective(model):
            """
            TODO: Getting the attributes of the device models by their subcategory key (os.getenv('BESS_KEY'),
                os.getenv('PV_KEY') and so on) does only work if the device keys are identical, which may not
                be the case (and never if there are multiple devices of a subcategory). Hence, provide the list of
                device keys per subcategory and loop over this list for each subcategory to add the attributes
                of each device model of this subcategory.
            """
        
            """
            variables= [v for v in model.component_objects (Var,descend_into=True)]
            logger.debug(variables)

            """
            # Erweiterung der Zielfunktion um die Entscheidungsvariablen sowie parameter für die Erweiterung der Zielfunktion. Part von Jonas
            P_market_feedin =getattr(model,f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_feedin") # [W]
            P_market_consum =getattr(model,f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_consum") # [W]
            price_market_feedin = 0.002 # [Euro/W]
            price_market_consum = 0.002 # [Euro/W]
           
            # Get variables defined with GCP
            feedin = getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_neg")  # [W]
            consumption = getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_pos")  # [W]
            # EUR / kWh * (W * kW/W) * (s * h/s) = EUR

            energy_cost = sum(
                (model.c_grid_cons[t] * consumption[t] / 1000 - model.c_grid_feedin[t] * feedin[t] / 1000) * (
                        model.dt / 3600) + (price_market_consum * P_market_consum[t] + price_market_feedin* P_market_feedin[t])*(1/100) for t in model.T)
            

            if os.getenv('BESS_KEY') in self.components:
                # Add value of the net energy charged into the storage: average cost for grid consumption
                energy_cost -= getattr(model, f"{os.getenv('BESS_KEY')}_E_charged_net") / (1000 * 3600) * sum(
                    model.c_grid_cons[t] for t in model.T) / len(model.T)


            # Penalty functions
            penalties = 0

            # Feedin limit softened to keep problem feasible if there's a feedin limit, but PV is not curtailable
            # TODO: this is a workaround to make it work with non-curtailable PV models until curtailment has
            #  been correctly modelled. Review if soft constraint makes sense when curtailment works
            feedin_limit_deviation = getattr(model,
                                             f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_epsilon_feedin_limit")  # [W]
            penalties += sum(feedin_limit_deviation[t] for t in model.T)

            if os.getenv('EVSE_KEY') in self.components:
                # Difference to target SOC, e.g. fully charged
                ev_target_dev = getattr(model, f"{os.getenv('EVSE_KEY')}_epsilon_target")
                w_target = getattr(model, f"{os.getenv('EVSE_KEY')}_w_target")
                penalties += w_target * ev_target_dev

                # Penalty for not charging with max. power (if SOC below min.) because of grid restrictions
                ev_quick_charge_deviation = getattr(model,
                                                    f"{os.getenv('EVSE_KEY')}_epsilon_quick_charge")  # array with Watt values
                penalties += sum(ev_quick_charge_deviation[t] for t in model.T)

                # Penalty for not charging at all despite a quick charging requirement because of grid restrictions
                charging_prohibited = getattr(model, f"{os.getenv('EVSE_KEY')}_epsilon_b_charging")
                penalties += sum(charging_prohibited[t] for t in model.T) * getattr(model,
                                                                                    f"{os.getenv('EVSE_KEY')}_w_charging")

            if os.getenv('PV_KEY') in self.components:
                # Penalty for curtailment
                curtailment = getattr(model,
                                      f"{os.getenv('PV_KEY')}_epsilon_curtailment")  # array with negative Watt values
                w_curtailment = getattr(model, f"{os.getenv('PV_KEY')}_w_curtailment")  # weight 0<w<=1
                penalties += -sum(curtailment[t] for t in model.T) * w_curtailment

            if os.getenv('BESS_KEY') in self.components:
                min_soc_deviation = getattr(model,
                                            f"{os.getenv('BESS_KEY')}_epsilon_min_soc")  # array with values in [0,1]
                penalties += sum(min_soc_deviation[t] for t in model.T)

            total_cost = energy_cost + penalties

            # return sum((model.price[i] * (model.P_total[i]/1000) * (model.dt/60/60)) for i in model.T)
            return total_cost

        self.model.obj = Objective(rule=objective, sense=minimize)


class DeviationMILP(MILP):

    def create_objective(self):
        """
        Objective: minimal deviation of actual power at the grid connection point and the given target.
        Sum of squared deviation is calculated over all time steps (although there might be only one step).
        """
        if self.objective_created:
            raise RuntimeError('Objective has already been created')
        self.objective_created = True
        if len(self.components) == 0:
            raise RuntimeError('No constraints have been added to the model')

        def var(var: str):
            return getattr(self.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_{var}")

        def objective(model):
            # Get target GCP limit
            # gcp_power_setpoint = getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_el_setpoint")  # [W]
            # Get difference of actual power to target power for consumption and feed-in
            # gcp_power_pos_deviation = getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_epsilon_above_setpoint")  # [W]
            # gcp_power_pos_deviation = getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_epsilon_pos_setpoint")  # [W]
            # gcp_power_neg_deviation = getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_epsilon_below_setpoint")  # [W]
            # gcp_power_neg_deviation = getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_epsilon_neg_setpoint")  # [W]
            # Get variables defined with GCP
            # gcp_power = getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_el")  # [W]

            # gcp_deviation = sum(abs(gcp_power_setpoint[t] - gcp_power[t]) for t in model.T)
            # total_gcp_deviation = sum((gcp_power_pos_deviation[t])**2 + (gcp_power_neg_deviation[t])**2 for t in model.T)
            total_gcp_deviation = sum(
                var('epsilon_pos_above_setpoint')[t] + var('epsilon_neg_above_setpoint')[t] +
                var('epsilon_pos_below_setpoint')[t] + var('epsilon_neg_below_setpoint')[t]
                for t in model.T
            )

            return total_gcp_deviation

        self.model.obj = Objective(rule=objective, sense=minimize)
