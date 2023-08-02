# Standard libraries
import math
import warnings
import datetime
import os
import pandas
import numpy as np
import pandas as pd
import pytz
import typing
import logging

from forecasting.forecaster import Forecaster
from forecasting import logger
import db.db_helper as db


class ElectricLoadForecaster(Forecaster):

    def __init__(self, window_start: datetime.datetime, window_size: datetime.timedelta,
                 resolution: datetime.timedelta):
        super(ElectricLoadForecaster, self).__init__(window_start, window_size, resolution=resolution)
        self.source = os.getenv('LOAD_EL_KEY')


class ReferenceBasedLoadForecaster(ElectricLoadForecaster):

    def __init__(self, window_start: datetime.datetime, window_size: datetime.timedelta):
        super(ReferenceBasedLoadForecaster, self).__init__(window_start, window_size,
                                                           resolution=datetime.timedelta(minutes=1))
        self.forecast_function = self.reference_based_forecast

    def average_load_curve_and_mean(self, measurements: pandas.DataFrame):
        """
        Calculates the average load for each minute of the day for multiple given days,
        as well as the mean over all given measurements
        :param measurements: a pandas dataframe (containing data of multiple days)
        :return: pandas dataframe containing the average load
        """
        times: pandas.DatetimeIndex = measurements.index
        load_curve = measurements.groupby([times.hour, times.minute]).mean()
        logger.debug(f"Length of load curve: {len(load_curve)}")
        logger.debug(f"Load curve: {load_curve}")
        mean_measurements = load_curve.mean()
        load_curve.index = load_curve.index.rename(['hour', 'minute'])
        return load_curve, mean_measurements

    def average_curve_and_mean_of_previous_days_of_same_day_type(self, measurements: pandas.DataFrame,
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
        return self.average_load_curve_and_mean(measurements=previous)

    def shift_load_curve(self, load_curve: pandas.DataFrame, starting_time: pandas.Timestamp,
                         ending_time: pandas.Timestamp):
        """
        Transforms a pandas dataframe, that contains a value for each minute of the day, to a numpy array
        and shifts it according to a given timestamp
        :param load_curve: the average daily load curve
        :param starting_time: the curve is shifted so that it starts with this time, and ends 24 hours later
        :return: a numpy array containing the values of the load curve, shifted according to the starting time
        """
        start = load_curve[(starting_time.hour, starting_time.minute):]
        end = load_curve[:(starting_time.hour, starting_time.minute)]

        # exclude given timestamp
        end = end.iloc[:-1]

        shifted_curve = np.concatenate([start.values, end.values])
        shifted_curve = shifted_curve.flatten()
        return shifted_curve

    def fill_values(self, reference_data: pd.DataFrame) -> pd.DataFrame:
        """
        Author: Pascal Bothe (Netze BW)
        :param reference_data: The resampled (to target resolution), but possibly incomplete reference data
        :return: Complete reference data, gaps filled with a method depending on size of the data gap
        """

        def find_losses(df):
            # Liefert einen Array mit den Anfängen und Enden der Ausfälle. Zum Beispiel bedeutet der
            # Array [3,5,16,80], dass es einen Ausfall zwischen dem dritten und fünften Index gibt und einen
            # zweiten Ausfall zwischen dem 16. und 80. Index.
            shifted_nan = np.roll(df.isnull().values, -1)
            return np.where((shifted_nan ^ df.isnull().values) == True)[0]

        logger.debug(f"reference_data before filling (len={len(reference_data)}): {reference_data} ")
        logger.debug(f"number of Nan in reference_data: {len(reference_data[reference_data.active_power.isna()])}")

        last_gap_was_large = True
        # Diese Funktion liefert einen DataFrame, bei dem die Ausfälle kleiner als 60 Minuten interpoliert werden
        # und längere Ausfälle durch Werte des vorherigen Tages aufgefüllt werden.
        drop_rows = None
        loss_start_end_array = find_losses(reference_data)
        logger.debug(f"loss_start_end_array={loss_start_end_array}, i.e. number of gaps={len(loss_start_end_array)/2}")
        for i in range(0, len(loss_start_end_array), 2):
            loss_duration = loss_start_end_array[i + 1] - loss_start_end_array[i]

            if loss_duration <= 60:
                if last_gap_was_large:
                    logger.debug(f"Gap <= 60 minutes")
                    logger.debug(
                        f"Gap between {reference_data.index[loss_start_end_array[i]]} and {reference_data.index[loss_start_end_array[i + 1]]}, that is {loss_duration} minutes")
                    logger.debug(f"Gap: {reference_data.iloc[loss_start_end_array[i]:loss_start_end_array[i + 1] + 2]}")

                reference_data.iloc[loss_start_end_array[i]:loss_start_end_array[i + 1] + 2] = reference_data.iloc[
                                                                                               loss_start_end_array[i]:
                                                                                               loss_start_end_array[
                                                                                                   i + 1] + 2].interpolate()
                filled = reference_data.iloc[loss_start_end_array[i]:loss_start_end_array[i + 1] + 2].copy(deep=True)
                if last_gap_was_large or not filled[filled.active_power.isna()].empty:
                    logger.debug(
                        f"Filled: {filled}")
                    last_gap_was_large = False


            else:
                last_gap_was_large = True
                logger.debug(f"Gap > 60 minutes")
                logger.debug(
                    f"Gap between {reference_data.index[loss_start_end_array[i]]} and {reference_data.index[loss_start_end_array[i + 1]]}, that is {loss_duration} minutes")
                logger.debug(f"Gap: {reference_data.iloc[loss_start_end_array[i]:loss_start_end_array[i + 1] + 2]}")
                # Falls der Ausfall länger als eine Stunde geht, nimm die Daten vom letzten Tag. Falls der Ausfall
                # länger als ein Tag geht, muss man das iterativ wiederholen.
                last_run = math.floor(loss_duration / 1440)
                for j in range(0, last_run + 1, 1):
                    fill_from = loss_start_end_array[i] + j * 1440 + 1

                    if j == last_run:
                        fill_to = loss_start_end_array[i + 1] + 1
                        with_data_from = loss_start_end_array[i] + j * 1440 - 1439
                        with_data_to = loss_start_end_array[i + 1] - 1439
                    else:
                        fill_to = loss_start_end_array[i] + (j + 1) * 1440 + 1
                        with_data_from = loss_start_end_array[i] + (j - 1) * 1440 + 1
                        with_data_to = loss_start_end_array[i] + j * 1440 + 1

                    logger.debug(f"Fill periods from {reference_data.index[fill_from]} to {reference_data.index[fill_to]} "
                                 f"with values from {reference_data.index[with_data_from]} to {reference_data.index[with_data_to]}")

                    logger.debug(f"Fill ...")
                    logger.debug(f"{reference_data.iloc[fill_from:fill_to]}")
                    logger.debug(f"with ...")
                    logger.debug(f"{reference_data.iloc[with_data_from:with_data_to]}")

                    if with_data_from < 0:
                        # Index value is negative, meaning there's not enough old data available
                        logger.debug(f"There's not enough older data to fill the missing periods. Oldest datapoint is {min(reference_data.index)}, but datapoints as of {reference_data.index[fill_from] - datetime.timedelta(minutes=1440)} are required.")
                        drop_rows = reference_data.index[:loss_start_end_array[i + 1] + 1]
                        logger.debug(f"Go to next gap.")
                        break


                    reference_data.iloc[fill_from:fill_to] = reference_data.iloc[with_data_from:with_data_to]
                    filled = reference_data.iloc[fill_from:fill_to].copy(deep=True)
                    logger.debug(f"Filled: {reference_data.iloc[fill_from:fill_to]}")
                    logger.debug(f"Nan in filled: {filled[filled.active_power.isna()]}")

        if drop_rows is not None:
            logger.debug(f"number of Nan in semi-final reference_data: {len(reference_data[reference_data.active_power.isna()])}")
            logger.debug(f"Nan in semi-final reference_data: {reference_data[reference_data.active_power.isna()]}")
            logger.debug(f"Drop all rows until including {drop_rows[-1]} from the reference data.")
            reference_data.drop(index=drop_rows, inplace=True)
        logger.debug(f"Final reference data:")
        logger.debug(reference_data)
        logger.debug(f"Nan in final reference_data: {reference_data[reference_data.active_power.isna()]}")

        return reference_data

    def get_reference_data(self, ref_start: datetime.datetime, ref_end: datetime.datetime, query_start: datetime.datetime) -> pandas.DataFrame:
        """
        Query measurements from the database.
        Clean them by filtering out negative values and interpolating between missing values.
        :param ref_start: Timestamp of first measurement to include in the reference data
        :param ref_end: Timestamp of first measurement that is excluded from the reference data
        :param query_start: Timestamp of first measurement that is queried from the database and used for data
        filling but not as final reference data
        :return: Cleaned reference data
        """
        parameter = 'active_power'
        # Get the reference data from the database
        reference_data: pd.DataFrame = db.get_measurement(source=self.source, fields=parameter,
                                            # start_time=ref_start.astimezone(pytz.utc),
                                            start_time=query_start.astimezone(pytz.utc),
                                            end_time=ref_end.astimezone(pytz.utc))
        # Convert from UTC to timezone used here
        reference_data.index = reference_data.index.tz_convert(self.forecast_window_start.tzinfo)
        # Remove all negative load values (happens if load value is calculated from other sources with
        # (small) temporal offsets
        reference_data = reference_data.loc[reference_data[parameter] >= 0]
        # Resample data to required resolution
        reference_data = reference_data.resample(rule=self.resolution, closed='left').mean()

        # Due to incomplete measurement data, the data may now or still contain many NaN -> fill these gaps
        # to get a complete data set
        reference_data = self.fill_values(reference_data)

        # Remove everything before ref_start
        reference_data = reference_data[ref_start:]
        logger.debug(f"Reference data as of ref_start:")
        logger.debug(f"{reference_data}")
        logger.debug(f"Nan in reference_data: {reference_data[reference_data.active_power.isna()]}")

        return reference_data

    def reference_based_forecast(self) -> typing.Dict[str, float]:
        """
        Make load predictions based on historic values from the last 30 days.
        For each time step of a day (e.g. 15:44 or 06:18) it considers the corresponding mean values over the
        whole 30 days, the last week and all days of the same weekday.
        If there is less than 7 days of data available, the prediction equals the measurements from 1 or 2 days ago,
        depending on forecast window size.
        :return: Load forecast as dict with <ISO-format timestamp, power value>-pairs
        """
        timestamps = pandas.date_range(self.forecast_window_start, end=self.forecast_window_end, freq=self.resolution,
                                       closed='left').tz_convert(pytz.utc)

        # Query and prepare reference data
        #  1. Query last 60 days of data from the database
        #  2. Resample and fill all values with this dataset
        #  3. Drop all data older than 30 days
        #   -> This way, older data is used to fill gaps in the newer data, but the prediction is still mainly
        #   based on the newer data, because older data is only contained in gaps that cannot be filled by the
        #   newer data itself

        month_data = self.get_reference_data(
            ref_start=self.forecast_window_start - datetime.timedelta(days=30),  # inclusive
            ref_end=self.forecast_window_start - datetime.timedelta(hours=1),  # exclusive
            query_start=self.forecast_window_start - datetime.timedelta(days=60)
        )

        data_time_range = max(month_data.index) - min(month_data.index)
        if data_time_range <= datetime.timedelta(days=7):
            # If available data spans less than 7 days the subsequent calculations won't work
            # In this case, simply return the values from the previous day as prediction
            logger.info(f"Available load data does not span 7 days or more. "
                        f"Get values from {2 if self.window_size >= datetime.timedelta(hours=23) else 1} day(s) ago "
                        f"as load prediction.")
            if self.window_size < datetime.timedelta(hours=23):
                previous_day_data = month_data[self.forecast_window_start - datetime.timedelta(
                    days=1):self.forecast_window_end - datetime.timedelta(days=1)]
            else:
                previous_day_data = month_data[self.forecast_window_start - datetime.timedelta(
                    days=2):self.forecast_window_end - datetime.timedelta(days=2)]

            if len(previous_day_data) < len(timestamps):
                # Data is incomplete -> resample and (forward) fill missing values
                logger.debug(
                    f"Fill up load previous_day_data, because it only has {len(previous_day_data)} timesteps instead of required {len(timestamps)}")
                previous_day_data = previous_day_data.resample(rule=self.resolution).ffill()

            logger.debug(f"previous_day_data: {previous_day_data}")

            prediction = previous_day_data["active_power"].values

        else:
            # get data of previous 7 days
            week_data = month_data[self.forecast_window_start - datetime.timedelta(days=7):]

            # get average curves
            weekly_average_curve, weekly_mean = self.average_load_curve_and_mean(measurements=week_data)
            monthly_average_curve, monthly_mean = self.average_load_curve_and_mean(measurements=month_data)
            same_day_type_average_curves, same_day_type_mean = self.average_curve_and_mean_of_previous_days_of_same_day_type(
                measurements=month_data, time=self.forecast_window_start)
            # average individual load curves and shift
            prediction = (weekly_average_curve + monthly_average_curve + same_day_type_average_curves) / 3
            prediction = self.shift_load_curve(load_curve=prediction, starting_time=self.forecast_window_start,
                                               ending_time=self.forecast_window_end)

        assert len(prediction) >= len(timestamps), f"Insufficient number of predicted load values. " \
                                                   f"{len(timestamps)} time steps are needed, but only {len(prediction)} time " \
                                                   f"steps predicted. Probably due to too little available data."
        # transform numpy array to dictionary and filter for forecast window
        forecast = {timestamps[i].isoformat(): prediction[i] for i in range(len(timestamps))}
        return forecast


class PersistenceForecaster(ElectricLoadForecaster):

    def __init__(self, window_start: datetime.datetime, window_size: datetime.timedelta):
        super(PersistenceForecaster, self).__init__(window_start, window_size, resolution=datetime.timedelta(
            hours=float(os.getenv('QUOTA_TEMP_RESOLUTION'))))
        self.forecast_function = self.persistence_forecast

    def persistence_forecast(self) -> typing.Dict[str, float]:
        """
        :return: Forecast as dict with format {'period1': power1, 'period2': power2, ..., 'periodn': powern}, with
        the keys being the start of the respective periods as str in ISO-format
        """
        # Define the reference data
        offset = 0 if self.window_size != datetime.timedelta(
            hours=24) else 1  # Skip antecedent day (=today) if making a day-ahead forecast
        ref_start = self.forecast_window_start - datetime.timedelta(
            weeks=int(os.getenv('NUM_REF_DAYOFWEEK')))  # inclusive
        ref_end = (self.forecast_window_end - datetime.timedelta(days=1 + offset))  # exclusive
        day_of_week = self.forecast_window_start.weekday()  # Monday=0, Sunday=6

        # Get the reference data
        data = db.get_measurement(source=self.source, fields='active_power', start_time=ref_start, end_time=ref_end)
        data_forecast_window = self.filter_for_window(data)

        # Filter for corresponding day of week and preceding days, respectively
        data_ref_day_of_week = data_forecast_window[data_forecast_window.index.dayofweek == day_of_week]
        data_ref_preced_days = data_forecast_window[
            (data_forecast_window.index.date > (
                    ref_end.date() - datetime.timedelta(days=int(os.getenv('NUM_REF_PRECED_DAYS')) + 1)))
            & (data_forecast_window.index.dayofweek != day_of_week)
            ]

        # If data contains less than 3 reference days with that window due to missing measurements,
        # iteratively include older days from (filtered) data
        i = 0
        oldest_day = data_ref_preced_days.index.min().date()
        while len(set(data_ref_preced_days.index.date)) < 3 and i < 5:
            i += 1

            data_ref_preced_days = data_forecast_window[
                (data_forecast_window.index.date >= (oldest_day - datetime.timedelta(days=1)))
                & (data_forecast_window.index.dayofweek != day_of_week)
                ]
            oldest_day -= datetime.timedelta(days=1)

        logging.debug(
            f'Got {len(set(data_ref_day_of_week.index.date))} ref dow and {len(set(data_ref_preced_days.index.date))} ref preceding days .')

        # Calculation of forecast load value for each period within window
        # Forecast value = 15-min-mean
        forecast = {}
        period_start: datetime.datetime = self.forecast_window_start
        while period_start < self.forecast_window_end:
            period_end = period_start + self.resolution

            if period_start.time() < period_end.time():  # Default case
                period_data_ref_day_of_week = data_ref_day_of_week[
                    (data_ref_day_of_week.index.time >= period_start.time())
                    & (data_ref_day_of_week.index.time < period_end.time())
                    ]
                period_data_ref_days = data_ref_preced_days[
                    (data_ref_preced_days.index.time >= period_start.time())
                    & (data_ref_preced_days.index.time < period_end.time())
                    ]
            else:  # Period end is midnight
                period_data_ref_day_of_week = data_ref_day_of_week[
                    (data_ref_day_of_week.index.time >= period_start.time())
                ]
                period_data_ref_days = data_ref_preced_days[
                    (data_ref_preced_days.index.time >= period_start.time())
                ]
            # Catch RuntimeWarning caused by an empty DataFrame when taking the mean of its values
            # Occurs when data is missing for the whole period (not window) of the given ref. day
            # Catching preferred over "if df.emtpy" clause, because it (should) rarely occur
            with warnings.catch_warnings():
                warnings.filterwarnings(action='error', message='Mean of empty slice')
                try:
                    # Take mean over all values within this period (e.g. 15 1-min-values) from same DOW
                    mean_ref_day_of_week = period_data_ref_day_of_week.values.mean()
                except RuntimeWarning:
                    logging.debug('DOW Empty!')
                    # Pass to simply take the mean value from the previous period
                    pass
                try:
                    # Take mean over all values within this period (e.g. 15 1-min-values) from preceding days
                    mean_ref_days = period_data_ref_days.values.mean()
                except RuntimeWarning:
                    logging.debug('Prec Empty!')
                    # Pass to simply take the mean value from the previous period
                    pass
            # Prediction = weighted mean of respective reference days means
            mean_power = 0.7 * mean_ref_day_of_week + 0.3 * mean_ref_days
            # Key must be string (Celery requirement)
            forecast[period_start.isoformat()] = mean_power

            # Next period
            period_start = period_end
        return forecast


class ThermalLoadForecaster(Forecaster):
    # TODO: Implement thermal load forecast
    # TODO: use ref-based forecaster

    def __init__(self, window_start: datetime.datetime, window_size: datetime.timedelta):
        super(ThermalLoadForecaster, self).__init__(window_start, window_size, resolution=datetime.timedelta(minutes=1))
        self.source = os.getenv('LOAD_TH_KEY')
