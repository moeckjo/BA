'''
- Optimizing of schedules based on forecasts
    - Amongst others, use following function input parameters with corresponding defaults: existing_schedule=np.ones(T,1) and quota=1
- Store optimal schedule for each device xx in mongoDB schedule_xx collection
'''
import importlib
import os
import time
import typing
import datetime
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

import core.utils as utils
from core.optimization import logger, db
from .constraintfactory import ConstraintFactory
from pyomo.core.base import Constraint
from .factories import gcp as gcp_factory
from .milp import CostMILP, DeviationMILP


class Scheduler:
    # Device subcategories for which forecasts and/or init state is needed as input for the schedule optimization
    forecast_needed_subcat = {os.getenv('PV_KEY'), os.getenv('LOAD_EL_KEY'), os.getenv('EVSE_KEY')}
    init_state_needed_subcat = {os.getenv('BESS_KEY'), os.getenv('EVSE_KEY'),
                                # os.getenv('HEAT_STORAGE_KEY'),
                                # os.getenv('HEAT_PUMP_KEY')
                                }

    def __init__(self, window_start: datetime.datetime,
                 resolution: int,
                 window_end: datetime.datetime = None,
                 window_size: datetime.timedelta = None,
                 triggering_event: str = None,
                 triggering_source: str = None
                 ):
        """
        :param window_start: Start of scheduling horizon
        :param window_end: End of scheduling horizon
        :param window_size: Length of scheduling horizon [h]
        nested dict. Format: {'pv': {<ISO-timestamp, value>-pairs}, 'load_el': {<ISO-timestamp, value>-pairs}}
        :param triggering_event: (Optional) The event that triggered this scheduling process.
        :param triggering_source: (Optional) The source (e.g. some device) that triggered this scheduling process.
        """

        # Technical specifications of all devices of this system/building and of it's grid connection point
        self.devices, self.grid_connection = utils.get_devices_and_gcp_specifications()

        self.model = None
        self.event = triggering_event
        self.triggering_source = triggering_source
        self.solver_time_limit = None
        self.runtime: float = None
        self.termination_condition = None

        self.__set_temporal_parameters(window_start=window_start, resolution=resolution, window_end=window_end,
                                       window_size=window_size)
        self.meta_data = {'resolution': self.resolution, 'event': self.event}
        if self.event is not None:
            self.meta_data['event_triggering_source'] = self.triggering_source

        self.grid_power_limits = None
        self.grid_power_setpoints = None

        self.forecast_needed: set = self.get_devices_with_forecasts()  # Set of device keys
        self.init_state_needed: typing.List[
            tuple] = self.get_devices_with_init_states()  # List of tuples (device key, device subcategory)
        logger.debug(f'Forecast needed for {self.forecast_needed}')
        logger.debug(f'Init state needed for {self.init_state_needed}')
        self.forecasts = None
        self.init_states: typing.Dict[str, typing.Dict[str, float]] = None

    def __set_temporal_parameters(self, window_start: datetime.datetime, resolution: int,
                                  window_end: datetime.datetime = None, window_size: datetime.timedelta = None):
        self.window_start = window_start
        assert window_end or window_size, 'Scheduler requires either the parameter "window_end" or "window_size". None was provided.'
        if window_end:
            self.window_end = window_end
            self.window_size = window_end - window_start
        else:
            self.window_size = window_size
            self.window_end = self.window_start + window_size

        self.resolution = resolution  # seconds

        # Define temporal parameters
        window_size_seconds = self.window_size.total_seconds()
        assert window_size_seconds % self.resolution == 0, f'Optimization horizon of {window_size_seconds}s and resolution of {self.resolution}s are not compatible.'
        self.time_step_count = int(window_size_seconds / self.resolution)
        self.timesteps = pd.date_range(start=self.window_start, end=self.window_end, freq=f'{self.resolution}S',
                                       closed='left')

    def get_devices_with_forecasts(self) -> set:
        # Get devices (keys) of this system/building that require a forecast as input for schedule optimization
        forecast_needed_subcat = self.forecast_needed_subcat.intersection(
            [device.subcategory for device in self.devices])
        return set(
            [device.key for device in self.devices if device.subcategory in forecast_needed_subcat] + [
                os.getenv('LOAD_EL_KEY')])

    def get_devices_with_init_states(self) -> typing.List[tuple]:
        """
        Get devices (keys and subcategory) of this system/building that require their initial state as
        input for schedule optimization
        :return: List with tuples of (device key, device subcategory)
        """
        init_state_needed_subcat = self.init_state_needed_subcat.intersection(
            [device.subcategory for device in self.devices])
        # return {device.key for device in self.devices if device.subcategory in init_state_needed_subcat}
        return [(device.key, device.subcategory) for device in self.devices if
                device.subcategory in init_state_needed_subcat]

    @staticmethod
    def register_device_factory(device_specifications: typing.NamedTuple) -> bool:
        try:
            factory_module = importlib.import_module(f'core.optimization.factories.{device_specifications.subcategory}')
            # Register factory
            ConstraintFactory.register_converter(device_specifications.key, factory_module.factory)
            return True
        except ModuleNotFoundError:
            logger.warning(f"Problem registering the factory for device {device_specifications.key}: This device of "
                           f"type {device_specifications.subcategory} is defined in this buildings' device "
                           f"configuration, but there is no factory for device type "
                           f"{device_specifications.subcategory} implemented.")
            return False

    def get_device_init_states(self, devices: typing.List[tuple],
                               provided_init_states: typing.Dict[str, typing.Dict[str, float]]) -> \
            typing.Dict[str, typing.Dict[str, float]]:

        def get_initial_state_from_latest_schedule(source: str, state_fields: list) \
                -> typing.Dict[str, typing.Union[None, float]]:
            logger.debug(f'Get init state "{state_fields}" for {source} from schedule.')
            source_init_states = {field: None for field in state_fields}
            latest_schedule: dict = db.get_time_series_data_from_db(
                db=os.getenv('MONGO_SCHEDULE_DB_NAME'),
                collection=source,
                start_time=self.window_start - datetime.timedelta(hours=1),
                end_time=self.window_start,
                grouped_by_date=True
            )
            if latest_schedule is not None:
                # Get the value of the last period of the existing schedule
                sorted_timestamps = sorted(latest_schedule)
                for field in state_fields:
                    try:
                        source_init_states.update({field: latest_schedule[sorted_timestamps[-1]][field]})
                    except KeyError:
                        logger.debug(
                            f'{source} schedule has no field "{field}". '
                            f'Available fields: {list(list(latest_schedule.values())[0])}'
                        )
            return source_init_states

        def get_initial_state_from_latest_measurement(source: str, state_fields: list) -> typing.Dict[
            str, typing.Union[float, None]]:
            logger.debug(f'Get init state "{state_fields}" for {source} from measurements.')
            return utils.get_latest_device_measurement(
                source=source, fields=state_fields,
                search_within_last_minutes=60,
                with_ts=False)

        init_states = provided_init_states if provided_init_states else {}
        subcategories = {device_key: device_subcat for device_key, device_subcat in devices}
        device_keys = set(device_key for device_key, device_subcat in devices)

        for source in device_keys.difference(init_states.keys()):
            state_fields: typing.List[str] = utils.get_state_fields_for_device(subcategories[source])
            last_state: typing.Dict[str, typing.Union[None, float]]

            now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
            next_possible_schedule_window_start = utils.get_next_schedule_window_start(now, datetime.timedelta(seconds=int(os.getenv('SCHEDULER_TEMP_RESOLUTION_SEC'))))
            lead_time = self.window_start - now
            # Initial state should be based on the latest measurement in two cases (which can both apply
            # simultaneously):
            #   1. The schedule starts as soon as possible (depending on schedule resolution and defined computation
            #       buffers), but not later than in 15 minutes.
            #   2. The scheduling process was triggered by a deviation that was caused by the given source.
            #
            #   In any other case, especially when more than 15 minutes will pass until the start of the next schedule,
            #   the initial state will be based on the planned state taken from the latest schedule.
            condition_1: bool = (self.window_start <= next_possible_schedule_window_start) and (lead_time < datetime.timedelta(minutes=15))
            condition_2: bool = self.event == os.getenv('DEVIATION_MESSAGE_SUBTOPIC') and source==self.triggering_source

            if condition_1 or condition_2:
                # Get initial state from the latest measurement (fallback: from schedule)
                if condition_1:
                    logger.debug(f"Try to get initial {source} state(s) from the latest measurement "
                                 f"(fallback: from schedule), because the new schedule will already "
                                 f"start in {lead_time.total_seconds()/60:.2f} minutes.")
                else:
                    logger.debug(f"Try to get initial {source} state(s) from the latest measurement "
                                 f"(fallback: from schedule), because scheduling was triggered by a "
                                 f"deviation of {source}.")

                # Try to get the initial state from the latest measurement
                last_state = get_initial_state_from_latest_measurement(source, state_fields)
                for field, value in last_state.items():
                    if value is None:
                        # Fallback: Try to get the initial state from the latest schedule
                        # If no schedule with this state is found, a dict with value=None is returned.
                        # In this case, the scheduler will take the configured default value for this device.
                        last_state.update(get_initial_state_from_latest_schedule(source, [field]))
            else:
                # Get initial state from the latest schedule (fallback: from measurements)
                logger.debug(
                    f"Try to get initial {source} state(s) from the latest schedule (fallback: from measurements), "
                    f"because the new schedule will start later in {lead_time.total_seconds()/60:.2f} minutes.")

                # Try to get the initial state from the latest schedule
                last_state = get_initial_state_from_latest_schedule(source, state_fields)
                for field, value in last_state.items():
                    if value is None:
                        # Fallback: Try to get the initial state from the latest measurement
                        # If no measurement is found, a dict with value=None is returned.
                        # In this case, the scheduler will take the configured default value for this device.
                        last_state.update(get_initial_state_from_latest_measurement(source, [field]))

            init_states[source] = last_state

        return init_states

    def get_forecasts(self, sources: typing.Set[str],
                      existing_forecasts: typing.Dict[str, typing.Dict[str, int]]) -> typing.Dict[
        str, np.ndarray]:
        """
        Return forecasts for the defined devices as array, covering the scheduling horizon and with the scheduling resolution
        :param sources: Set of device keys for which we need the forecast
        :param existing_forecasts: Nested dict with <device_key, forecast> that were provided to the Schedule. Might
        not cover the entire scheduling horizon.
        :return: Dictionary with forecast as array for each device
        """
        forecasts = {}

        if existing_forecasts:
            # Check provided forecasts: if it spans the scheduling horizon, filter for relevant periods before adding
            # it as array and remove this device key from the set of required devices
            for source, forecast in existing_forecasts.items():
                forecast = {datetime.datetime.fromisoformat(ts): val for ts, val in forecast.items()}
                timestamps = list(forecast.keys())
                if min(timestamps) <= self.window_start and max(timestamps) >= (
                        self.window_end - datetime.timedelta(seconds=self.resolution)):
                    forecast = self.resample(forecast)
                    forecasts[source] = self.ts_dict_to_array(ts_dict=forecast, window_filter=True)
                    sources.remove(source)
                    logger.debug(
                        f'Provided forecast for {source} could be used and has been resampled.')

                else:
                    logger.debug(
                        f'Provided forecast for {source} does not span the entire scheduling horizon from {self.window_start} to {self.window_end - datetime.timedelta(seconds=self.resolution)} (it spans {min(timestamps)} to {max(timestamps)}).')

        if sources: logger.debug(f'Get forecasts from DB for {sources}')
        # Get forecast for remaining devices from the database
        for source in sources:
            logger.debug(f'Get forecast for {source}')
            result = db.get_time_series_data_from_db(
                db=os.getenv('MONGO_FORECAST_DB_NAME'),
                collection=source,
                start_time=self.window_start,
                end_time=self.window_start + self.window_size,
                grouped_by_date=True,
                extra_info=['resolution']
            )
            forecast: typing.Dict[datetime.datetime, object] = result[0]

            if not int(os.getenv('TESTING')):
                assert (
                        min(forecast) <= self.window_start and max(forecast) >= (
                        self.window_end - datetime.timedelta(seconds=self.resolution))
                ), f'Forecast for {source} retrieved from DB does not cover the scheduling horizon!'

            forecast = self.resample(forecast)
            logger.debug(f'Resampled forecast from DB for {source}: {forecast}')

            forecasts[source] = self.ts_dict_to_array(ts_dict=forecast, window_filter=False)
        return forecasts

    def resample(self, time_series: typing.Dict[datetime.datetime, int]):
        """
        Resample time series to scheduling resolution.
        Downsampling: aggregate using the mean
        Upsampling: fill missing period values with previous value ("forward fill")
        :param time_series: original time series
        :return: time series resampled to scheduler resolution
        """
        series = pd.Series(data=time_series.values(), index=pd.to_datetime(list(time_series.keys())))
        resolution = pd.Series(series.index).diff().min(skipna=True)  # datetime.timedelta
        if resolution.total_seconds() < self.resolution:
            # Downsampling
            resampler = series.resample(rule=datetime.timedelta(seconds=self.resolution), label='left',
                                        closed='left')
            if set(series.values) == ({0, 1} or {0} or {1}):
                resampled: pd.Series = resampler.min()
            else:
                resampled: pd.Series = resampler.mean()

        elif resolution.total_seconds() > self.resolution:
            # Upsampling
            resampled: pd.Series = series.resample(rule=datetime.timedelta(seconds=self.resolution)).ffill()
        else:
            # Nothing to resample
            return time_series
        return dict(zip(resampled.index.to_pydatetime(), resampled.values))

    def add_model_constraints(self):
        """
        Add all constraints to the optimization model
        """
        init_state_needed_keys = {device_key for device_key, device_subcat in self.init_state_needed}
        # Register factory for each device and add it to the optimization model together with required parameters
        for device in self.devices:
            logger.debug(f'Register factory for device {device.key}')
            registered: bool = self.register_device_factory(device)

            if registered:
                if device.key in init_state_needed_keys.intersection(self.forecast_needed):
                    add_kwargs = dict(forecast=self.forecasts[device.key], init_state=self.init_states[device.key])
                elif device.key in init_state_needed_keys:
                    add_kwargs = dict(init_state=self.init_states[device.key])
                elif device.key in self.forecast_needed:
                    add_kwargs = dict(forecast=self.forecasts[device.key])
                else:
                    add_kwargs = {}

                self.model.add_constraints(key=device.key, specification=device, **add_kwargs)  # throwing key error

        # Register factory for the GCP, i.e. the entire building, and add it to the optimization model together
        # with required parameters
        ConstraintFactory.register_converter(key=os.getenv('GRID_CONNECTION_POINT_KEY'), func=gcp_factory.factory)
        self.model.add_constraints(key=os.getenv('GRID_CONNECTION_POINT_KEY'), specification=self.grid_connection,
                                   devices=self.devices,
                                   forecast_load_el=self.forecasts[os.getenv('LOAD_EL_KEY')],
                                   grid_restrictions=self.grid_power_limits,
                                   grid_setpoints=self.grid_power_setpoints
                                   )


    def constraint_secondmarket (self,model,t):
        
        feedin_limit_active = getattr(model,f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_feedin_limit_active")
        consum_limit_active = getattr(model,f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_consum_limit_active")
        test_var_consum = getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_consum")
        test_var_feedin = getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_feedin")
        logger.debug (test_var_feedin[t].value)

        #TESTING jonas
        #logger.debug(f'Grid limits_active: cons={consum_limit_active}, feedin={feedin_limit_active}')
        if (consum_limit_active[t] == 1):
            #logger.debug("consumlimit active")
            if getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_consum")[t].value == 0:
                return getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_consum")[t] >= 0
            #kleiner 0 = sell consumlimit
            elif getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_consum")[t].value <= -1:
                return getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_consum")[t] >= (test_var_consum[t].value+1)
            #größer 0 = buy consumlimit
            elif getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_consum")[t].value >= 1:
                return getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_consum")[t] <= (test_var_consum[t].value-1)

        if (feedin_limit_active[t] == 1):
            #logger.debug("feedinlimit active")
            if getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_feedin")[t].value == 0:
                return getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_feedin")[t] >= 0
            #kleiner 0 = sell feedinlimit
            elif getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_feedin")[t].value <= -1:
                return getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_feedin")[t] >= (int(test_var_feedin[t].value+1))
            #größer 0 = buy feedinlimit    
            elif getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_feedin")[t].value >= 1:
                return getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_feedin")[t] <=(test_var_feedin[t].value-1)
        
        elif (consum_limit_active[t] == 0): 
            return Constraint.Skip




    def schedule(self) -> typing.Dict[str, typing.Dict[str, int]]:

        optimization_time_limit = 60 if self.solver_time_limit is None else self.solver_time_limit
        logger.debug(f'Start schedule optimization, time limit = {optimization_time_limit}s')

        self.now: str = datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec='seconds')
        solver_start_time = time.time()
        results = self.model.solve(time_limit=optimization_time_limit, solver_type='glpk')
        #setattr(self.model.model, 'Constraint_secondmarket', Constraint (self.model.model.T, rule= self.constraint_secondmarket))
        self.runtime: float = time.time() - solver_start_time
        self.termination_condition = results.solver.termination_condition
        #setattr(self.model,'consum_secondmarket', Constraint (self.model.model.T, rule=self.constraint_last_solution))
        logger.debug(f'Schedule optimization finished after {self.runtime}s.')
        schedules: typing.Dict[str, typing.Dict[str, int]] = self.get_and_save_solution(results.solver)
        return schedules
    
    #TESTING PART VON JONAS
    """
    def get_list_for_bids(model, bids_feedin,bids_consum):
        help_consum= []
        help_feedin = []
        for t in model.model.T:
            help_consum.append (getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_consum")[t].value)
            help_feedin.append (getattr(model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_feedin")[t].value)
        bids_consum.append (help_consum)
        bids_feedin.append (help_feedin)
        logger.debug(help_consum)
        return {}
    """    
    def resolve (self):
        logger.debug ("Komme bis hier")
        bids_feedin= []
        bids_consum= []
        list_result= []
        #feedin_limit_active = getattr(self.model.model,f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_feedin_limit_active")
        #consum_limit_active = getattr(self.model,f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_consum_limit_active")
        #logger.debug(f'Grid limits_active: cons={consum_limit_active}, feedin={feedin_limit_active}')
        for i in range (4):
            help_consum= []
            help_feedin = []
            result=self.schedule()
            list_result.append(result)
            setattr(self.model.model, 'Constraint_secondmarket', Constraint (self.model.model.T, rule= self.constraint_secondmarket))
            # Get all Constraint objects in the model
            constraints = self.model.model.component_objects(Constraint)

            # Print the names and indices of each Constraint
            for constraint in constraints:
                print("Constraint Name:", constraint.name)
                print("Constraint Index:", constraint.index_set())
            for t in self.model.model.T:    
                help_consum.append (getattr(self.model.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_consum")[t].value)
                help_feedin.append (getattr(self.model.model, f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_P_market_feedin")[t].value)
            bids_feedin.append (help_feedin)
            bids_consum.append(help_consum)
        return result



            

    def get_and_save_solution(self, solver) -> typing.Dict[str, typing.Dict[str, float]]:
        """
        1. Extract the optimization result, i.e. the variable values, from the model.
        2. Save raw solution
        3. Save resulting power schedule for the grid connection point and all devices.
        :return: Power schedules for all devices and the grid connection point (dict keys are device keys, values are dicts with
        key=ISO-format timestamp and value=active_power)
        """
        logger.debug(f'Solver status: {solver.status}; term. condition: {solver.termination_condition}.')
        problem_solved = True

        if solver.termination_condition == TerminationCondition.infeasible:
            logger.warning(f"Optimization problem is {solver.termination_condition}!")
            problem_solved = False
        elif solver.termination_condition == TerminationCondition.maxTimeLimit:
            logger.warning(f"Optimization problem could not be solved in time (time limit: {self.solver_time_limit} s).")
            problem_solved = False

        # Solution is stored in the model
        solved_model = self.model.model
        timestamps = [t.isoformat() for t in self.timesteps]

        # Initialize solution
        objective_value = None
        params_df = pd.DataFrame(index=timestamps)
        scalar_params = {}
        other_params = {}

        variables_df = pd.DataFrame(index=timestamps)
        other_variables = {}

        # Get all input parameters
        for param in solved_model.component_objects(pyo.Param, active=True):
            values = [pyo.value(param[i]) for i in param]
            if len(values) == len(timestamps):
                params_df.loc[:, str(param)] = values
            elif len(values) == 1:
                scalar_params[str(param)] = values[0]
            else:
                other_params[str(param)] = values
        logger.debug(f'All (indexed) parameters: {params_df}')

        if problem_solved:
            # Get objective value and all solution variables' values
            objective_value = pyo.value(solved_model.obj)

            for var in solved_model.component_objects(pyo.Var, active=True):
                values = [pyo.value(var[i]) for i in var]
                if len(values) == len(timestamps):
                    variables_df.loc[:, str(var)] = values
                else:
                    other_variables[str(var)] = values
            logger.debug(f'All (indexed) solution variables: {variables_df}')

        # Always save raw solution to store at least the meta data and parameters
        self.save_raw_solution(objective_value, variables_df, params_df, other_variables,
                               {**other_params, **scalar_params})

        # If the problem was solved, extract the individual schedules from the solution
        if not problem_solved:
            return {}

        # Extract GCP variables, save and return power schedule
        gcp_schedule: typing.Dict[str, int] = self.get_and_save_gcp_schedule(variables_df)

        # Extract and save all device schedules. Only power schedules are returned. Dict keys are device keys.
        device_schedules: typing.Dict[str, typing.Dict[str, int]] = self.get_and_save_device_schedules(
            variables_df)

        power_schedules: typing.Dict[str, typing.Dict[str, int]] = {
            os.getenv("GRID_CONNECTION_POINT_KEY"): gcp_schedule, **device_schedules
        }
        return power_schedules

    def save_raw_solution(self, objective_value: float, indexed_variables: pd.DataFrame,
                          indexed_parameters: pd.DataFrame, other_variables: dict, other_parameters: dict):
        """
        Save all data and meta data of the just solved problem that might later be of interest.

        :param objective_value: Objective value of the corresponding (optimal) solution
        :param indexed_variables: Values of all time-dependent solution variables (e.g. the resulting power schedules)
        :param indexed_parameters: Values of all time-dependent problem parameters
        :param other_variables: Time-independent/scalar variable values
        :param other_parameters: Time-independent/scalar parameter values
        """
        data = {
            'from': self.window_start,
            'to': self.window_end,
            **self.meta_data,
            'created': self.now,
            'time_limit': self.solver_time_limit,
            'runtime': self.runtime,
            'termination_condition': self.termination_condition,
            'objective_value': objective_value,
            'parameters': indexed_parameters.to_dict(orient='index'),
            'other_parameters': other_parameters,
            'variables': indexed_variables.to_dict(orient='index'),
            'other_variables': other_variables
        }
        utils.save_to_schedule_db(os.getenv('MONGO_SCHEDULE_OPTIMIZATION_RAW_SOLUTION_COLL_NAME'), data)

    def get_and_save_gcp_schedule(self, solution: pd.DataFrame) -> typing.Dict[str, int]:
        """
        Extracts the power schedule for the grid connection point, saves it to the database and returns it.
        :param solution: All variables from the solved optimization problem.
        :return: Power schedule for the grid connection point (keys=timestamps, values=power_values; timestamps are ISO-format str)
        """
        gcp_solution = pd.DataFrame.filter(solution, regex=f'^{os.getenv("GRID_CONNECTION_POINT_KEY")}', axis=1)
        signed_power_schedule: dict = gcp_solution[f'{os.getenv("GRID_CONNECTION_POINT_KEY")}_P_el'].to_dict()
        # Cast power values from float to int to get 1 Watt resolution (timestamps are ISO-format strings)
        signed_power_schedule = {t: int(p) for t, p in signed_power_schedule.items()}
        print_n_periods = 5

        logger.debug(
            f'GCP signed power schedule (first {print_n_periods} periods): {({list(signed_power_schedule.keys())[i]: list(signed_power_schedule.values())[i] for i in range(min(print_n_periods, len(signed_power_schedule)))})}')
        utils.save_to_schedule_db(
            source=f'{os.getenv("GRID_CONNECTION_POINT_KEY")}',
            data=signed_power_schedule,
            meta_data={**self.meta_data, 'updated_at': self.now}
        )
        return signed_power_schedule

    def get_and_save_device_schedules(self, solution: pd.DataFrame) -> typing.Dict[str, typing.Dict[str, int]]:
        """
        Extracts the schedule for each device and saves its schedule with all variables to the database.
        Eventually, a dictionary with only the power schedule for each device is returned.
        :param solution: All variables from the solved optimization problem.
        :return: Power schedule for each device (keys=device keys, values=dict(timestamp=power_value); timestamps are ISO-format str)
        """
        power_schedules: typing.Dict[str, typing.Dict[str, float]] = {}
        for device in self.devices:
            device_solution = pd.DataFrame.filter(solution, regex=f'^{device.key}', axis=1).copy()
            logger.debug(f"{device.key} device_solution: {device_solution}")
            if not device_solution.empty:
                device_solution.rename(
                    columns={name: ('active_power' if 'P_el' in name else name.split('_', maxsplit=1)[1].lower()) for name
                             in
                             device_solution.columns},
                    inplace=True
                )

                if 'active_power' in device_solution.columns:
                    device_solution['active_power'] = device_solution['active_power'].astype(int)
                    # Add power schedule to dict of power schedules for all devices
                    power_schedules[device.key] = device_solution['active_power'].to_dict()
                device_schedule = device_solution.to_dict(orient='index')

                print_n_periods = 5
                logger.debug(
                    f'{device.key} schedule (first {print_n_periods} periods): {({list(device_schedule.keys())[i]: list(device_schedule.values())[i] for i in range(min(print_n_periods, len(device_schedule)))})}')
                # Save whole schedule to database
                utils.save_to_schedule_db(
                    source=device.key,
                    data=device_schedule,
                    meta_data={**self.meta_data, 'updated_at': self.now}
                )
        return power_schedules

    def ts_dict_to_array(self, ts_dict: typing.Dict[datetime.datetime, int], window_filter=True,
                         sorting='asc') -> np.ndarray:
        """
        :param ts_dict: dictionary with <timestamp, value>-pairs
        :param window_filter: If True (default), only return entries that fall within the given window
        :param sorting: Sort entries by timestamp in ascending (default) or descending order
        :return: numpy array containing the sorted values
        """
        if not isinstance(list(ts_dict)[0], datetime.datetime):
            # Convert str timestamps to datetime objects
            ts_dict = {datetime.datetime.fromisoformat(key): value for key, value in ts_dict.items()}
        sorted_ts_keys = sorted(list(ts_dict.keys()), reverse=(False if sorting == 'asc' else True))
        if window_filter and len(sorted_ts_keys) > self.time_step_count:
            # Filter for relevant entries
            sorted_ts_keys = [key for key in sorted_ts_keys
                              if self.window_start <= key
                              < self.window_start + self.window_size]
            logger.debug(f'sorted ts keys: {sorted_ts_keys}')
        return np.asarray([ts_dict[key] for key in sorted_ts_keys])


class GeneralScheduler(Scheduler):

    def __init__(self, window_start: datetime.datetime,
                 window_end: datetime.datetime = None,
                 window_size: datetime.timedelta = None,
                 forecasts: typing.Dict[str, typing.Dict[str, int]] = None,
                 init_states: typing.Dict[str, typing.Dict[str, float]] = None,
                 grid_quotas_category: str = None,
                 grid_quotas: typing.Dict[str, int] = None,
                 triggering_event: str = None,
                 triggering_source: str = None
                 ):
        """
        :param window_start: Start of scheduling horizon
        :param window_end: End of scheduling horizon
        :param window_size: Length of scheduling horizon [h]
        :param forecasts: (Optional) Forecasts of generation and load, spanning at least the scheduling horizon, as
        nested dict. Format: {'pv': {<ISO-timestamp, value>-pairs}, 'load_el': {<ISO-timestamp, value>-pairs}}
        :param init_states: (Optional) States of devices at the beginning of the optimization horizon
        :param grid_quotas_category: (Optional) Category of grid restriction, e.g. "primary" (quota)
        :param grid_quotas: (Optional) Dict with quoted power limits for some or all periods
        :param triggering_event: (Optional) The event that triggered this scheduling process.
        :param triggering_source: (Optional) The source (e.g. some device) that triggered this scheduling process.
        """

        super().__init__(
            window_start=window_start,
            window_end=window_end,
            window_size=window_size,
            resolution=int(os.getenv('SCHEDULER_TEMP_RESOLUTION_SEC')),
            triggering_event=triggering_event,
            triggering_source=triggering_source
        )
        logger.debug(f"Time step count: {self.time_step_count}; Timesteps:\n {self.timesteps}")

        # Define optimization model
        self.model = CostMILP(self.resolution, self.time_step_count)
        self.solver_time_limit = int(os.getenv('SCHEDULE_COMPUTATION_TIME_LIMIT'))

        self.forecasts: typing.Dict[str, np.ndarray] = self.get_forecasts(sources=self.forecast_needed.copy(),
                                                                          existing_forecasts=forecasts)
        logger.debug(f'Forecasts: {self.forecasts}')

        self.init_states: typing.Dict[str, typing.Dict[str, float]] = self.get_device_init_states(
            self.init_state_needed,
            provided_init_states=init_states)
        logger.debug(f'Init states: {self.init_states}')

        # Get grid constraints if necessary
        if grid_quotas_category is not None:
            self.grid_power_limits: typing.Union[None, np.ndarray] = self.get_grid_power_limits(
                grid_quotas=grid_quotas, grid_quotas_category=grid_quotas_category
            )

        # Add all constraints to the optimization model
        self.add_model_constraints()

        # This building does not get dynamic tariffs
        self.model.create_objective(
            tariffs_grid_consumption=[self.grid_connection['tariff_consumption']] * self.time_step_count,
            tariffs_grid_feedin=[self.grid_connection['tariff_feedin']] * self.time_step_count
        )

        logger.debug('Scheduler ready!')

    def get_grid_quotas(self, category: str) -> typing.Dict[datetime.datetime, int]:
        quota_period_length_minutes = float(os.getenv('QUOTA_TEMP_RESOLUTION')) * 60
        quota_block_start_minutes = [i * quota_period_length_minutes for i in
                                     range(int(60 / quota_period_length_minutes))]

        if self.window_start.minute not in quota_block_start_minutes:
            search_start = self.window_start - datetime.timedelta(minutes=quota_period_length_minutes)
        else:
            search_start = self.window_start

        result = db.get_time_series_data_from_db(
            db=os.getenv('MONGO_QUOTA_DB_NAME'),
            collection=os.getenv(f'MONGO_{category.upper()}_QUOTAS_COLL_NAME'),
            start_time=search_start,
            end_time=self.window_end,
            grouped_by_date=True
        )

        if result is None:
            # No quotas found
            logger.debug(f"No {category} quotas found for periods between {self.window_start} and "
                         f"{self.window_end} (searched as of {search_start}).")
            return result
        limits = {t: values['abs_power_limit'] for t, values in result.items()}
        return limits

    def get_reference_schedule(self, with_iso_timestamps=False) -> typing.Dict[datetime.datetime, int]:
        schedule: typing.Dict[datetime.datetime, int] = db.get_time_series_data_from_db(
            db=os.getenv('MONGO_SCHEDULE_DB_NAME'),
            collection=f'{os.getenv("GRID_CONNECTION_POINT_KEY")}_reference',
            start_time=self.window_start, end_time=self.window_end,
            grouped_by_date=True,
            return_timestamps_in_isoformat=with_iso_timestamps
        )
        return schedule

    def get_grid_power_limits(self, grid_quotas: typing.Dict[str, int], grid_quotas_category: str) \
            -> typing.Union[None, np.ndarray]:
        """
        Take provided quotas or get them from the database. Resample to the required temporal resolution and fill
        with None to get a dense array with a value for every period of the scheduling horizon.
        :param grid_quotas: Possibly provided quotas for the grid connection point.
        :param grid_quotas_category: Category of grid quotas.
        :return: Dense array of grid power limits if there are any. None, if there are no grid limits
        in this scheduling horizon.
        """
        if grid_quotas:
            logger.debug(f'{grid_quotas_category} quotas were directly provided.')
            # Transform timestamps to datetime objects
            grid_quotas = {datetime.datetime.fromisoformat(t): v for t, v in grid_quotas.items()}
        else:
            # Get quotas from database
            logger.debug(f'Check for {grid_quotas_category} quotas in database.')
            grid_quotas: typing.Dict[datetime.datetime, int] = self.get_grid_quotas(grid_quotas_category)
            logger.debug(f'Quotas from db: {grid_quotas}')

        if grid_quotas is None:
            # No relevant quotas returned from the database -> no limits
            return None

        # Add an additional, actually excluded period with value=None, otherwise the final periods will
        # be missing after resampling
        quota_window_end_excl = max(grid_quotas.keys()) + datetime.timedelta(
            hours=float(os.getenv('QUOTA_TEMP_RESOLUTION')))
        grid_quotas[quota_window_end_excl] = None
        if grid_quotas_category != 'final':
            # Final "quotas" (abs. limits) are already a merge of reference schedule, primary and secondary quotas,
            # whereas primary quotas must be merged here with the reference schedule
            reference_schedule = self.get_reference_schedule(with_iso_timestamps=False)
            logger.debug(f'Reference schedule: {reference_schedule}')
            if reference_schedule:
                # Replace power values from the reference schedule with quoted power value if there is
                # one for this period
                reference_schedule.update(grid_quotas)
                grid_quotas = reference_schedule

        logger.debug(f'(Incomplete) Grid limits before resampling: {grid_quotas}')
        grid_quotas = self.resample(grid_quotas)
        # Remove aux. period again
        grid_quotas.pop(quota_window_end_excl)
        logger.debug(f'(Incomplete) Grid limits after resampling: {grid_quotas}')
        # Fill periods without limit (i.e. gaps between periods within the quota block or periods after the quota
        # block within the optimization window) with None, indicating no limit
        grid_power_limits = {t: grid_quotas.get(t, None) for t in self.timesteps}

        # Convert dict to array with power limit values
        grid_power_limits = self.ts_dict_to_array(grid_power_limits, window_filter=False)

        logger.debug(f'Complete grid power limits array: {grid_power_limits}')

        assert len(grid_power_limits) == self.time_step_count, \
            f"Grid power limits must not be sparse. {self.time_step_count - len(grid_power_limits)} periods are missing."
        return grid_power_limits


# class OperationalScheduler(GeneralScheduler):
#     """
#     Adapts schedules regularly (e.g. every hour) by including latest information such as latest forecasts or
#     in case of "severe" deviations from the current schedule.
#     Example: Reschedule while minimizing the deviation from the originally optimal (daily) schedules and
#     of course respect the grid constraints.
#     """
#
#     def __init__(self, window_start: datetime.datetime, window_end: datetime.datetime = None,
#                  window_size: datetime.timedelta = None, forecasts=None):
#         super(OperationalScheduler, self).__init__(window_start, window_end=window_end, window_size=window_size,
#                                                    forecasts=forecasts)
#         self.gcp_schedule = self.get_existing_gcp_schedule()
#         self.quotas = self.get_quotas(time_window={'from': self.window_start, 'to': self.window_end})
#
#     pass


class GridPowerSetpointScheduler(Scheduler):
    """
    Scheduler needed when a mandatory grid setpoint is sent by the DSO (to resolve a grid congestion).
    Setpoint is signed and therefore corresponds either to a target feed-in or grid consumption.
    """

    def __init__(self, window_start: datetime.datetime, window_size: datetime.datetime, grid_power_setpoint: int):
        """
        :param window_start: Start of scheduling horizon
        :param window_size: Length of the scheduling horizon
        :param grid_power_setpoint: Target power (signed) for the grid connection point [W]
        """

        super().__init__(
            window_start=window_start,
            window_size=window_size,
            resolution=int(os.getenv('GCP_SETPOINT_SCHEDULE_RESOLUTION')),
            triggering_event=f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_setpoint"
        )
        self.meta_data['GCP_power_setpoint'] = grid_power_setpoint
        # Define optimization model
        self.model = DeviationMILP(self.resolution, self.time_step_count)
        self.solver_time_limit = int(os.getenv('GCP_SETPOINT_SCHEDULE_COMPUTATION_TIME_LIMIT'))

        # Get last measurements for each device and provide value(s) as forecast or init state, respectively
        self.forecasts = dict()
        for device_key in self.forecast_needed:
            field = 'active_power' if os.getenv('EVSE_KEY') not in device_key else 'connected'
            latest_measurement: typing.Dict[str, int] = utils.get_latest_device_measurement(
                source=device_key,
                fields=field,
                with_ts=False
            )
            self.forecasts[device_key] = np.ones(self.time_step_count) * int(latest_measurement)
        logger.debug(f'Forecasts: {self.forecasts}')

        self.init_states: typing.Dict[str, typing.Dict[str, float]] = dict()
        for device_key, device_subcat in self.init_state_needed:
            latest_measurement: typing.Dict[str, int] = utils.get_latest_device_measurement(
                source=device_key,
                fields=utils.get_state_fields_for_device(device_subcat),
                with_ts=False
            )
            self.init_states[device_key]: typing.Dict[str, float] = latest_measurement
        logger.debug(f'Init states from DB: {self.init_states}')

        # GCP model expects power limits as nd.array -> convert
        self.grid_power_setpoints = np.ones(self.time_step_count) * grid_power_setpoint
        logger.debug(f'Grid power setpoint(s): {self.grid_power_setpoints}')
        assert len(self.grid_power_setpoints) == self.time_step_count, \
            f"Grid power setpoints must not be sparse. {self.time_step_count - len(self.grid_power_setpoints)} periods are missing."

        # Add all constraints to the optimization model
        self.add_model_constraints()

        # Add objective
        self.model.create_objective()

        logger.debug('Scheduler ready!')


class MockupRandomScheduler:
    """
    Mockup scheduler that creates an operation schedule for the grid-connection point with random
    values between -15 and +16 kW
    """

    def __init__(self, window_start: datetime.datetime):
        self.window_start = window_start
        self.window_size = datetime.timedelta(hours=24)
        self.window_resolution = datetime.timedelta(hours=float(os.getenv('QUOTA_TEMP_RESOLUTION')))
        self.event = None

    def random_gcp_schedule(self, save=True):
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        gcp_schedule = {}
        period_start = self.window_start
        while period_start < self.window_start + self.window_size:
            val = random.randint(-15000, 16000)
            gcp_schedule[period_start.isoformat()] = val

            period_start += self.window_resolution

        logger.debug(f'GCP schedule periods (first 3): {list(gcp_schedule.keys())[:3]}')
        if save:
            meta_data = {'updated_at': now.isoformat(), 'resolution': self.window_resolution.total_seconds(),
                         'event': self.event}
            try:
                self.save_schedule(schedule=gcp_schedule, meta_data=meta_data)
            except Exception as e:
                logger.error(
                    f'MongoDB document with schedule for this day is full (>16 MB). Try without persisting previous schedules of this day.')
                self.save_schedule(gcp_schedule, meta_data, False)

    def save_schedule(self, schedule: typing.Dict[str, float], meta_data: dict = None, persist_previous_schedules=True):
        """
        :param persist_previous_schedules: In case previous schedules exist for the same periods: if False,
        they are replaced by these new scheduled values; if True, they are persisted in the database
        """
        meta_data = meta_data if meta_data else {}
        result = db.save_data_to_db(
            db=os.getenv('MONGO_SCHEDULE_DB_NAME'),
            data_source=os.getenv('GRID_CONNECTION_POINT_KEY'),
            time_series_data=schedule,
            meta_data=meta_data,
            group_by_date=True,
            persist_old_data=persist_previous_schedules
        )
        logger.debug(f'GCP mockup schedule saved to DB.')  #: {result}')
