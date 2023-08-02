# Local imports
import datetime
import os
import typing
import logging
import pandas
import pandas as pd
import pytz

from forecasting.forecaster import Forecaster
from forecasting import *


class EVForecaster(Forecaster):

    def __init__(self, window_start: datetime.datetime, window_size: datetime.timedelta,
                 resolution: datetime.timedelta):
        super(EVForecaster, self).__init__(window_start, window_size, resolution=resolution)
        self.source = os.getenv('EVSE_KEY')

    def get_reference_data(self, parameter: str, ref_start: datetime.datetime,
                           ref_end: datetime.datetime) -> pandas.DataFrame:
        # Query time series database
        reference_data = db.get_measurement(source=self.source, fields=parameter,
                                            start_time=ref_start.astimezone(pytz.utc),
                                            end_time=ref_end.astimezone(pytz.utc))
        # Set index to used timezone
        #TESTING jonas
        logger.debug ("reference_data is:")
        logger.debug (reference_data)
        reference_data.index = reference_data.index.tz_convert(self.forecast_window_start.tzinfo)
        return reference_data

    def calculate_mean_connection_state_duration(self) -> typing.Dict[int, datetime.timedelta]:
        """
        Based on all available connection state (0 or 1) data from the last six months until yesterday,
        calculate the average duration of continuous connection and disconnection, respectively.
        :return: Dict with resulting mean value per connection state
        """
        parameter_name = 'connected'
        # Get the connection status data over the last 6 months
        ref_start = self.forecast_window_start - datetime.timedelta(days=183)  # inclusive
        ref_end = (self.forecast_window_start - datetime.timedelta(days=1))  # exclusive
        connection_data: pandas.DataFrame = self.get_reference_data(parameter=parameter_name, ref_start=ref_start,
                                                                    ref_end=ref_end)
        connection_data[parameter_name] = pandas.to_numeric(connection_data[parameter_name])
        # Remove possibly stored other values than 0 and 1
        connection_data = connection_data.drop(connection_data.loc[~connection_data[parameter_name].isin([0, 1])].index)

        state_durations = {0: [], 1: []}
        current_state = connection_data.iloc[0, 0]
        state_begin = connection_data.index[0]
        state_end = None
        # Get duration of each period with respective constant state
        for i, timestamp in enumerate(connection_data.index, start=1):
            next_state = connection_data.loc[timestamp, parameter_name]
            if next_state != current_state:
                state_end = timestamp
                duration = (state_end - state_begin).total_seconds()
                state_durations[current_state].append(duration)
                current_state = next_state
                state_begin = timestamp

        mean_state_durations: typing.Dict[int, datetime.timedelta] = {}

        # Calculate the mean of period durations for each state
        for state, durations in state_durations.items():
            mean_duration_seconds = sum(durations) / len(durations)
            # Mean values likely don't fit the forecast resolution -> normalize
            mean_timesteps_rounded: int = round(mean_duration_seconds / self.resolution.total_seconds())
            mean_duration: datetime.timedelta = mean_timesteps_rounded * self.resolution
            mean_state_durations[state] = mean_duration

        logger.debug(f"Mean state durations: {mean_state_durations}")
        return mean_state_durations


class ReferenceBasedEVForecaster(EVForecaster):

    def __init__(self, window_start: datetime.datetime, window_size: datetime.timedelta):
        super(ReferenceBasedEVForecaster, self).__init__(window_start, window_size,
                                                         resolution=datetime.timedelta(minutes=5))
        self.forecast_function = self.reference_based_forecast

    def average_state_curve(self, measurements: pandas.DataFrame):
        """
        Calculates the average load for each minute of the day for multiple given days,
        as well as the mean over all given measurements
        :param measurements: a pandas dataframe (containing data of multiple days)
        :return: pandas dataframe containing the average load
        """
        times: pandas.DatetimeIndex = measurements.index
        curve = measurements.groupby([times.hour, times.minute]).mean()
        curve.index = curve.index.rename(['hour', 'minute'])
        return curve

    def filter_for_previous_days_of_same_day_type(self, measurements: pandas.DataFrame,
                                                  time: pandas.Timestamp):
        """
        Calculates the average load for each minute of the day as well as the mean of all measurements
        for a number of previous days that are the same day of the week
        :param measurements: a pandas dataframe (containing data of multiple days)
        :param time: timestamp to identify the current day and time
        :return:
        """
        previous = []
        first_entry = measurements.index[0]
        time = time - pandas.Timedelta("7 D")  # jump to previous week
        while time > first_entry:
            previous.append(measurements[time:time + pandas.Timedelta("1 D")])  # access the next 24 hours
            time = time - pandas.Timedelta("7 D")  # jump to previous week
        previous = pandas.concat(previous)
        return previous

    def shift_curve(self, curve: pandas.DataFrame, starting_time: pandas.Timestamp):
        """
        Transforms a pandas dataframe, that contains a value for each minute of the day, to a numpy array
        and shifts it according to a given timestamp
        :param curve: the predicted curve
        :param starting_time: the curve is shifted so that it starts with this time, and ends 24 hours later
        :return: a numpy array containing the values of the load curve, shifted according to the starting time
        """
        start = curve[(starting_time.hour, starting_time.minute):]
        end = curve[:(starting_time.hour, starting_time.minute)]
        # exclude given timestamp
        end = end.iloc[:-1]
        shifted_curve = np.concatenate([start.values, end.values])
        shifted_curve = shifted_curve.flatten()
        return shifted_curve

    def reference_based_forecast(self) -> typing.Dict[str, float]:
        timestamps = pandas.date_range(self.forecast_window_start, end=self.forecast_window_end, freq=self.resolution,
                                       closed='left').tz_convert(pytz.utc)
        parameter_name = 'connected'
        # Get the reference data
        ref_start = self.forecast_window_start - datetime.timedelta(days=30)  # inclusive
        ref_end = (self.forecast_window_end - datetime.timedelta(days=1))  # exclusive
        month_data: pandas.DataFrame = self.get_reference_data(parameter=parameter_name, ref_start=ref_start,
                                                               ref_end=ref_end)
        month_data[parameter_name] = pandas.to_numeric(month_data[parameter_name])
        logger.debug(f"Raw month_data: {month_data}")
        # Resample to desired forecast resolution
        month_data = month_data.resample(rule=self.resolution, closed='left').median()
        # Larger gaps in the data will produce NaN values -> fill these
        month_data = month_data.fillna(method='ffill')
        logger.debug(f"Resampled (median) and nan-filled month_data: {month_data}")

        data_time_range = max(month_data.index) - min(month_data.index)
        if data_time_range <= datetime.timedelta(days=7):
            # If available data spans less than 7 days the subsequent calculations won't work
            # In this case, simply return the values from the previous day as prediction
            logger.info(f"Available EV connection data does not span 7 days or more. "
                        f"Get values from 1 day ago as connection prediction.")
            ref_start = self.forecast_window_start - datetime.timedelta(days=1)
            ref_end = self.forecast_window_end - datetime.timedelta(days=1)
            previous_day_data = month_data[ref_start:ref_end]
            logger.debug(f"previous_day_data (month_data filtered for time range from {ref_start} "
                         f"to {ref_end}): {previous_day_data}")

            if len(previous_day_data) < len(timestamps):
                # Data is incomplete -> resample and fill missing values
                logger.info(
                    f"Fill up EV previous_day_data, because it only has {len(previous_day_data)} timesteps "
                    f"instead of required {len(timestamps)}")
                if (ref_end - self.resolution) not in previous_day_data.index:
                    logger.debug(f"Values are missing at the end.")
                    # Fill missing trailing values
                    previous_day_data.loc[ref_end] = None
                    previous_day_data = previous_day_data.resample(rule=self.resolution).ffill()
                    previous_day_data.drop(index=ref_end, inplace=True)

                if ref_start not in previous_day_data.index:
                    logger.debug(f"Values are missing at the beginning.")
                    previous_day_data.loc[ref_start - self.resolution] = None
                    previous_day_data = previous_day_data.resample(rule=self.resolution).bfill()
                    previous_day_data.drop(index=ref_start - self.resolution, inplace=True)

                # Values are only missing in between -> fill in between
                previous_day_data = previous_day_data.resample(rule=self.resolution).ffill()

                logger.debug(f"previous_day_data after filling: {previous_day_data}")

            prediction = previous_day_data[parameter_name].values

        else:
            # get data of previous 7 days
            week_data = month_data[self.forecast_window_start - datetime.timedelta(days=7):]
            # Get data for same weekday as forecasted day
            same_weekday_data = self.filter_for_previous_days_of_same_day_type(measurements=month_data,
                                                                               time=self.forecast_window_start)

            # Get votes values
            weekly_average_state_curve = self.average_state_curve(measurements=week_data)
            monthly_average_state_curve = self.average_state_curve(measurements=month_data)
            same_weekday_average_state_curve = self.average_state_curve(measurements=same_weekday_data)

            # Determine state by majority voting
            prediction = round(
                (monthly_average_state_curve + weekly_average_state_curve + same_weekday_average_state_curve) / 3)
            prediction = self.shift_curve(curve=prediction, starting_time=self.forecast_window_start)

        assert len(prediction) >= len(timestamps), f"Insufficient number of predicted connection states. " \
                                                   f"{len(timestamps)} time steps are needed, but only {len(prediction)} time " \
                                                   f"steps predicted. Probably due to too little available data."
        # transform numpy array to dictionary
        forecast = {timestamps[i].isoformat(): int(prediction[i]) for i in range(len(timestamps))}
        return forecast


class UserInputBasedEVForecaster(EVForecaster):

    def __init__(self, window_start: datetime.datetime, window_size: datetime.timedelta,
                 planned_departure: datetime.datetime):
        super(UserInputBasedEVForecaster, self).__init__(window_start, window_size,
                                                         resolution=datetime.timedelta(minutes=5))
        self.forecast_function = self.departure_based_forecast
        self.departure = datetime.datetime.fromisoformat(planned_departure).astimezone(
            pytz.timezone(os.getenv('LOCAL_TIMEZONE')))

    def departure_based_forecast(self):
        timesteps = pandas.date_range(start=self.forecast_window_start, end=self.forecast_window_end,
                                      freq=self.resolution,
                                      closed='left')
        forecast = {ts.astimezone(pytz.utc).isoformat(): (1 if self.departure > ts else 0) for ts in timesteps}
        return forecast


class UserAndMeanBasedConnectedEVForecaster(EVForecaster):

    def __init__(self, window_start: datetime.datetime, window_size: datetime.timedelta):
        super(UserAndMeanBasedConnectedEVForecaster, self).__init__(window_start, window_size,
                                                    resolution=datetime.timedelta(minutes=5))
        self.forecast_function = self.mixed_forecast

    def mixed_forecast(self):
        """
        1. Get user input with a planned departure at or after the forecast window start
        2. If available, predict that the EV stays connected until that time, otherwise it's assumed to have just departed
        3. Get the average duration of continuous connection and disconnection (absence)
        4. Fill the periods after departure for with connected=0 for the mean disconnection duration
        5. Fill the periods remaining periods with connected=1 for the mean connection duration
        6. Back to 4. until end of forecast window is reached

        :return: Forecast as dict with ISO-format str timestamps in UTC as keys and connection states as value

        """
        mean_state_durations: typing.Dict[int, datetime.timedelta] = self.calculate_mean_connection_state_duration()
        timesteps = pandas.date_range(start=self.forecast_window_start, end=self.forecast_window_end,
                                      freq=self.resolution,
                                      closed='left')

        # Get latest EV user input from database
        relevant_entries = self.get_relevant_stored_user_input()
        logger.debug(f"EV user input entries with departure >= {self.forecast_window_start}: {relevant_entries}")

        if relevant_entries:
            if len(relevant_entries) > 1:
                relevant_entries_by_input_timestamp: typing.Dict[str, dict] = {entry['timestamp']: entry for entry in
                                                                               relevant_entries}
                latest_entry: dict = relevant_entries_by_input_timestamp[max(relevant_entries_by_input_timestamp)]
            else:
                latest_entry = relevant_entries[0]

            departure = datetime.datetime.fromisoformat(latest_entry['scheduled_departure']).astimezone(
                self.forecast_window_start.tzinfo)

            # Fill with connected=1 until departure
            forecast = {ts: 1 for ts in timesteps if departure > ts}
            remaining_timesteps = [ts for ts in timesteps if ts >= departure]

        else:
            # No query match, i.e. no departure planned in the forecast window -> set departure to window start
            departure = self.forecast_window_start
            forecast = {}
            remaining_timesteps = timesteps

        current_state = 0
        next_state_switch = departure + mean_state_durations[current_state]
        # Fill forecast alternating with connected=0 or 1 values based on respective mean state duration
        # until the end of the forecast window
        for ts in remaining_timesteps:
            if ts >= next_state_switch:
                current_state = abs(current_state - 1)
                next_state_switch = ts + mean_state_durations[current_state]

            forecast[ts] = current_state

        # Convert timestamps to UTC and ISO-format strings
        forecast = {ts.astimezone(pytz.utc).isoformat(): state for ts, state in forecast.items()}
        return forecast

    def get_relevant_stored_user_input(self) -> typing.List[dict]:
        # Get relevant EV user input entries from database
        # A single entry looks like this:
        #   {
        #   '_id': ObjectId('61dc5468b136d5b118babec4'), 'timestamp': '2022-01-10T15:42:09+00:00',
        #   'soc': 0.15, 'soc_target': 1.0, 'capacity': 35.8, 'scheduled_departure': '2022-01-11T07:00:00+00:00'
        #   }
        forecast_window_utc = self.forecast_window_start.astimezone(pytz.utc)
        departure_filter = {'scheduled_departure': {'$gte': forecast_window_utc.isoformat(timespec='seconds')}}
        results = db.get_data_from_db(
            db=os.getenv('MONGO_USERDATA_DB_NAME'),
            collection=os.getenv('MONGO_EV_CHARGING_INPUT_COLL_NAME'),
            doc_filter=departure_filter,
        )
        # Query returns Cursor object that is emptied when iterating over it -> copy to new list object
        return list(results)


class ConnectedEVForecaster(ReferenceBasedEVForecaster):

    def __init__(self, window_start: datetime.datetime, window_size: datetime.timedelta):
        super(ConnectedEVForecaster, self).__init__(window_start, window_size)
        self.forecast_function = self.mixed_forecast

    def mixed_forecast(self):
        """
        1. Get user input with a planned departure at or after the forecast window start from the database
        2. If available, predict that the EV stays connected until that time, otherwise it's assumed to have just departed
        3. Get the average duration of continuous disconnection (absence)
        4. Make a normal reference-based forecast
        5. Replace the periods until departure with connected=1
        5. Replace the periods after departure with connected=0 for the mean disconnection duration
        :return: Forecast as dict with ISO-format str timestamps in UTC as keys and connection states as value

        """
        # Get latest EV user input from database
        relevant_entries = self.get_relevant_stored_user_input()
        logger.debug(f"EV user input entries with departure >= {self.forecast_window_start}: {relevant_entries}")

        if relevant_entries:
            if len(relevant_entries) > 1:
                relevant_entries_by_input_timestamp: typing.Dict[str, dict] = {entry['timestamp']: entry for entry in
                                                                               relevant_entries}
                latest_entry: dict = relevant_entries_by_input_timestamp[max(relevant_entries_by_input_timestamp)]
            else:
                latest_entry = relevant_entries[0]

            departure = latest_entry['scheduled_departure']

        else:
            # No query match, i.e. no departure planned in the forecast window -> set departure to window start
            departure = self.forecast_window_start.astimezone(pytz.utc).isoformat()

        mean_state_durations: typing.Dict[int, datetime.timedelta] = self.calculate_mean_connection_state_duration()
        disconnected_until = (datetime.datetime.fromisoformat(departure) + mean_state_durations[0]).isoformat()

        # Normal reference based forecast, returned with ISO-format str timestamps in UTC
        forecast: typing.Dict[str, float] = self.reference_based_forecast()
        for ts, state in forecast.items():
            if ts < departure:
                forecast[ts] = 1
            elif ts <= disconnected_until:
                forecast[ts] = 0

        return forecast

    def get_relevant_stored_user_input(self) -> typing.List[dict]:
        # Get relevant EV user input entries from database
        # A single entry looks like this:
        #   {
        #   '_id': ObjectId('61dc5468b136d5b118babec4'), 'timestamp': '2022-01-10T15:42:09+00:00',
        #   'soc': 0.15, 'soc_target': 1.0, 'capacity': 35.8, 'scheduled_departure': '2022-01-11T07:00:00+00:00'
        #   }
        forecast_window_utc = self.forecast_window_start.astimezone(pytz.utc)
        departure_filter = {'scheduled_departure': {'$gte': forecast_window_utc.isoformat(timespec='seconds')}}
        results = db.get_data_from_db(
            db=os.getenv('MONGO_USERDATA_DB_NAME'),
            collection=os.getenv('MONGO_EV_CHARGING_INPUT_COLL_NAME'),
            doc_filter=departure_filter,
        )
        # Query returns Cursor object that is emptied when iterating over it -> copy to new list object
        return list(results)


class DisconnectedEVForecaster(ReferenceBasedEVForecaster):

    def __init__(self, window_start: datetime.datetime, window_size: datetime.timedelta):
        super(DisconnectedEVForecaster, self).__init__(window_start, window_size)
        self.forecast_function = self.mixed_forecast

    def mixed_forecast(self) -> typing.Dict[str, float]:
        """
        1. Make a normal reference-based forecast
        2. Get the average duration of continuous disconnection (absence)
        3. Determine the time until it will probably stay disconnected
        4. Predict connected=0 until that time, afterwards whatever the reference-based forecast predicted
        :return: Forecast as dict with ISO-format str timestamps in UTC as keys and connection states as value
        """

        # Normal reference based forecast, returned with ISO-format str timestamps in UTC
        reference_based_forecast: typing.Dict[str, float] = self.reference_based_forecast()

        mean_state_durations: typing.Dict[int, datetime.timedelta] = self.calculate_mean_connection_state_duration()
        disconnected_until: datetime.datetime = self.forecast_window_start + mean_state_durations[0]
        disconnected_until: str = (disconnected_until.astimezone(pytz.utc)).isoformat()

        forecast = {t: 0 if t <= disconnected_until else v for t, v in reference_based_forecast.items()}

        return forecast
