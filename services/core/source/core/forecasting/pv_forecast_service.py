import copy
import logging
import time
import pytz
import datetime
import os
import json
import requests
import warnings
import typing

import numpy as np
import pandas as pd
import scipy.stats as scipystats
from tensorflow.keras.models import load_model

import db.db_helper as db

# from forecasting import logger, path
from core.forecasting import logger, path
# from forecasting.forecaster import Forecaster
from core.forecasting.forecaster import Forecaster

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def load_weather_forecast_service_config() -> dict:
    with open(os.path.join(os.getenv("BEM_ROOT_DIR"), 'config', 'weather_forecast_service_config.json')) as file:
        weather_forecast_service_config: dict = json.load(file)
    return weather_forecast_service_config


def load_weather_forecast_stations() -> typing.List[dict]:
    with open(os.path.join(path, 'config', 'dwd_stations.json')) as stations_file:
        dwd_stations: typing.List[dict] = json.load(stations_file)
    return dwd_stations


def save_weather_forecast_config():
    # Save the current weather forecast configuration in the settings database for tracking purposes
    # (for later evaluations)
    weather_forecast_service_config = load_weather_forecast_service_config()
    weather_stations = load_weather_forecast_stations()
    weather_forecast_config = {'timestamp': datetime.datetime.now(tz=pytz.utc).isoformat(timespec='seconds'),
                               'stations': weather_stations,
                               'parameters': weather_forecast_service_config.pop('parameters'),
                               'service': weather_forecast_service_config
                               }
    db.save_dict_to_db(
        db=os.getenv('MONGO_SETTINGS_DB_NAME'),
        data_category=os.getenv('MONGO_SETTINGS_WEATHER_FORECAST_CONFIG_COLL_NAME'),
        data=weather_forecast_config
    )
    logger.debug(f"Stored the following weather forecast settings: {weather_forecast_config}")


class PVForecaster(Forecaster):

    def __init__(self, pv_specification: dict, window_start: datetime.datetime, window_size: datetime.timedelta,
                 resolution: datetime.timedelta):
        super(PVForecaster, self).__init__(window_start, window_size, resolution)
        self.source = pv_specification['key']
        self.normalization_value = pv_specification['active_power_nominal']

        # Start and end of hours spanning midnight in which PV generation is always zero and is therefore not predicted
        self.night_hours = (22, 4)

    def request_sun_duration_forecast(self, ref_start: datetime.datetime, ref_end: datetime.datetime,
                                      location_id: str) -> \
            pd.Series:
        """
        Request forecast for the hourly sun duration from a weather forecast service.
        :param ref_start: Start timestamp (incl.), must be tz-aware
        :param ref_end: End timestamp (excl.), must be tz-aware
        :param location_id: ID of the DWD station
        :return: Sunshine duration forecast data. Unit of values is seconds.
        Example:
        timestamp=2021-12-12 10:00:00+00:00, value=300 means 5 minutes of sunshine during the hour from 10 to 11 AM.

        """

        def make_request(url: str, method: str, auth: tuple, **kwargs):
            response = requests.request(method=method.upper(), url=url, auth=auth, **kwargs)
            return response

        weather_forecast_service = load_weather_forecast_service_config()

        base_url = weather_forecast_service["base_url"]
        auth = (weather_forecast_service["username"], weather_forecast_service["password"])
        parameter_info = weather_forecast_service["parameters"]["sun_duration"]
        #
        # Note: the following implementation only supports the request of a single selected MOSMIX parameter
        #
        parameter = parameter_info["key"]
        if parameter_info["reference"] == "back" or parameter_info["reference"] == -1:
            # Value refers to the preceding hour, i.e. SunD1 value for 11 am denotes sunshine
            # duration from 10 am to 11 am -> add offset of 1 hour for request and set data index accordingly later
            offset = datetime.timedelta(seconds=parameter_info["resolution"])
            ref_start += offset
            ref_end += offset
        else:
            offset = datetime.timedelta(seconds=0)

        if ref_start.minute != 0 and (60 - ref_start.minute) * 60 < parameter_info["resolution"]:
            # Given start does not match a possible interval start.
            # Request as of preceding interval start, otherwise only data starting at the next interval is returned.
            # Later, the weighted average of these values is taken.
            diff_to_previous_interval_start = parameter_info["resolution"] - (60 - ref_start.minute) * 60
            ref_start -= datetime.timedelta(seconds=diff_to_previous_interval_start)

        request_body = dict(
            station=location_id,
            start_timestamp=ref_start.isoformat(timespec='seconds'),
            end_timestamp=ref_end.isoformat(timespec='seconds'),
            mosmix_parameters=[parameter],
            updates=weather_forecast_service["updates"]
        )

        # Make initial request to query the data, which returns a request ID
        response = make_request(base_url, 'POST', auth=auth, json=request_body)
        request_id = response.json()['request_ID']

        # Request the status of the query until the results are ready
        i = 0
        status_url = f'{base_url}{request_id}/{weather_forecast_service["status_path"]}'
        for i in range(10):
            try:
                response = make_request(status_url, 'GET', auth=auth, timeout=5)
                status = response.json()
                if status['status_text'] == 'ready':
                    break
            except requests.Timeout:
                pass
            i += 1
            time.sleep(0.5)

        # Get the result of query
        result_url = f'{base_url}{request_id}/{weather_forecast_service["result_path"]}'
        response = make_request(result_url, 'GET', auth=auth, timeout=120)
        data = response.json()
        logger.debug(f'Raw MOSMIX request response json for station {location_id}: {data}')

        data_field_name = weather_forecast_service["data_field"]
        sun_duration = pd.Series(data=data[data_field_name][parameter], dtype=float)
        # Timestamps are UTC but without TZ Info
        sun_duration.index = pd.to_datetime(sun_duration.index, utc=True)
        # Subtract the offset (may be 0)
        sun_duration.index = sun_duration.index - offset
        logger.debug(f'Forecast of {parameter} for station {location_id}: {sun_duration}')

        return sun_duration

    def load_sun_duration_forecast(self, ref_start: datetime.datetime, ref_end: datetime.datetime, location_id) -> \
            pd.DataFrame:
        """
        :param ref_start: Start timestamp (incl.), must be tz-aware
        :param ref_end: End timestamp (excl.), must be tz-aware
        :param location_id: ID of the DWD station
        :return: Processed sun duration forecast. Single value if time filter spans 1 hour, Dataframe otherwise.
        """
        assert ref_start.tzinfo and ref_end.tzinfo, 'Start and end timestamps have to be timezone-aware!'
        ref_window = max(ref_end - ref_start, datetime.timedelta(hours=1))

        sun_duration_forecast: pd.Series = self.request_sun_duration_forecast(ref_start, ref_end,
                                                                              location_id)
        sun_duration_forecast.index = pd.to_datetime(sun_duration_forecast.index).tz_convert(
            tz=self.forecast_window_start.tzinfo)

        if ref_start.minute != 0:
            if ref_window == datetime.timedelta(hours=1):
                if ref_start.hour != (ref_end - datetime.timedelta(seconds=1)).hour:
                    # Get forecast for both hours and take weighted average
                    weights = [1 - ref_start.minute / 60, ref_end.minute / 60]
                    sun_duration_forecast = pd.Series(
                        data={ref_start: sum(weights * sun_duration_forecast[ref_start:ref_end].values)})
                else:
                    logger.debug(
                        f"Nothing to do: ref_start.minute != 0, but ref_start and ref_end are within the same hour, "
                        f"therefore only one sun duration forecast value: \n {sun_duration_forecast} ")
            else:
                logger.warning(f"The forecast window does not start at minute=0 ({ref_start.minute}; "
                               f"window_size={ref_window}). But taking the weighted average of the sun duration "
                               f"forecast hourly values for more than 1 hour is not implemented.")

        # Remove night hours
        if self.night_hours and isinstance(sun_duration_forecast, pd.Series):
            sun_duration_forecast = sun_duration_forecast.between_time(datetime.time(self.night_hours[1]),
                                                                       datetime.time(self.night_hours[0]),
                                                                       include_end=False)
        # Transform duration in seconds to 1-hour-share
        sun_duration_forecast = sun_duration_forecast / 3600
        return sun_duration_forecast

    def get_dwd_data(self, forecast_window_start: datetime.datetime, forecast_window_end: datetime.datetime) -> list:
        dwd_stations = load_weather_forecast_stations()
        dwd_data: typing.List[pd.DataFrame] = []
        for i, station in enumerate(dwd_stations):
            # The stations have to be in a specific order when their values are input to the NN.
            # The order (1,2,3,...) is specified for each station in the file loaded above.
            # (Usually order of list items is persisted, so this is just a precaution.)
            assert station["feature_order"] == i + 1, f'Wrong weather feature order! ' \
                                                      f'Trying to append values of station {station["location_id"]} on place {i}, ' \
                                                      f'but it should be on place {station["feature_order"]}. '

            sund1_forecast: pd.DataFrame = self.load_sun_duration_forecast(forecast_window_start, forecast_window_end,
                                                                           station["location_id"])
            dwd_data.append(sund1_forecast)
        logger.debug(f'Processed DWD data: {dwd_data}')
        return dwd_data

    def fill_missing_leading_data(self, incomplete_data: pd.Series,
                                  requested_start: datetime.datetime) -> pd.Series:
        """
        When data is missing at the beginning of the reference horizon, the values cannot be
        simply interpolated. Therefore, the missing periods are filled using linear regression on succeeding values of
        the provided reference data:
        Given the number of missing periods n_missing_periods until period t, a regression line is fitted
        to the values from t to t+n_missing_periods. This regression line is then used to fill the values of
        the missing periods.

        :param incomplete_data: The reference data, which some periods missing (because they were not recorded)
        :param requested_start: The requested start of the reference data
        :return: Series with filled leading data and original reference data
        """
        first_existing_period = min(incomplete_data.index)
        logger.debug(
            f"Fill missing leading periods from {requested_start} to {first_existing_period} using regression.")
        missing_leading_periods = pd.date_range(requested_start, first_existing_period, freq=self.resolution,
                                                closed='left')

        # Period until which to include values for the regression
        t_cut = first_existing_period + (first_existing_period - requested_start)

        # Prepare x and y parameters as numeric numpy arrays
        cut_index = incomplete_data[:t_cut].index
        x = np.linspace(0, len(cut_index), len(cut_index), endpoint=False)
        y = np.array(incomplete_data[:t_cut].values)

        # Do regression
        slope, intercept, _, _, _ = scipystats.linregress(x, y)

        # Apply regression function to missing periods
        x_hat = np.linspace(-len(missing_leading_periods), 0, len(missing_leading_periods), endpoint=False)
        leading_data = pd.Series(data=(slope * x_hat + intercept), index=missing_leading_periods)

        # Combine provided data with regressed data
        combined = pd.concat([leading_data, incomplete_data], verify_integrity=True)
        # Regression near zero might lead to positive values -> set them to zero
        combined[combined > 0] = 0

        return combined

    def fill_missing_trailing_data(self, incomplete_data: pd.Series,
                                   requested_end: datetime.datetime) -> pd.Series:
        """
        When data is missing at the end of the reference horizon, the values cannot be
        simply interpolated. Therefore, the missing periods are filled using linear regression on preceding values of
        the provided reference data:
        Given the number of missing periods n_missing_periods as of period t, a regression line is fitted
        to the values from t-n_missing_periods to t. This regression line is then used to fill the values of
        the missing periods.

        :param incomplete_data: The reference data, which some periods missing (because they were not recorded)
        :param requested_end: The requested, excluded (!) end of the reference data
        :return: Series with original reference data and filled trailing data

        """
        last_existing_period = max(incomplete_data.index)
        logger.debug(f"Fill missing trailing periods from {last_existing_period} to {requested_end} using regression.")
        missing_trailing_periods = pd.date_range(last_existing_period, requested_end, freq=self.resolution,
                                                 closed='right')[:-1]

        # Period as of which to include values for the regression
        t_cut_end = last_existing_period - (requested_end - last_existing_period - self.resolution)

        # Prepare x and y parameters as numeric numpy arrays
        cut_index = incomplete_data[t_cut_end:].index
        x = np.linspace(0, len(cut_index), len(cut_index), endpoint=False)
        y = np.array(incomplete_data[t_cut_end:].values)

        # Do regression
        slope, intercept, _, _, _ = scipystats.linregress(x, y)

        # Apply regression function to missing periods
        x_hat = np.linspace(len(x), len(x) + len(missing_trailing_periods),
                            len(missing_trailing_periods),
                            endpoint=False)
        trailing_data = pd.Series(data=(slope * x_hat + intercept),
                                  index=missing_trailing_periods)

        # Combine provided data with regressed data
        combined = pd.concat([incomplete_data, trailing_data], verify_integrity=True)
        # Regression near zero might lead to positive values -> set them to zero
        combined[combined > 0] = 0

        return combined


class ShortTermForecaster(PVForecaster):
    # TODO: short term forecast produces very unrealistic (high) values in the early morning hours

    resolution = datetime.timedelta(minutes=1)

    def __init__(self, pv_specification: dict, window_start: datetime.datetime,
                 window_size=datetime.timedelta(hours=1)):
        super(ShortTermForecaster, self).__init__(pv_specification=pv_specification,
                                                  window_start=window_start,
                                                  window_size=window_size,
                                                  resolution=self.resolution)

        self.forecast_function = self.forecast_with_prior_measurements_and_sun_duration

        # load models here
        self.model = load_model(os.path.join(path, 'pv_prediction_models/short_term_prediction/model.hdf5'))

        # time distance between last measurement and first prediction: 5 minutes
        self.tdelta_reference_to_forecast_window = datetime.timedelta(
            seconds=int(os.getenv('PV_FORECAST_SHORTTERM_LEAD_TIME_SEC')))

    def get_feature_measurements(self, measurements: typing.Union[pd.Series, None]):
        """
        Load the 56 latest 1-min-measurements that are needed as input features for the prediction model
        """
        ref_end = self.forecast_window_start - self.tdelta_reference_to_forecast_window  # included
        ref_start = self.forecast_window_start - datetime.timedelta(hours=1)  # included
        measurement_parameter = 'active_power'
        if measurements is None or measurements.empty:
            # Get measurements from the database
            logger.debug(f"Get measurements from database")
            data: pd.DataFrame = db.get_measurement(source=self.source, fields=measurement_parameter,
                                                    start_time=ref_start.astimezone(pytz.utc),
                                                    end_time=ref_end.astimezone(pytz.utc), closed='both')
            data: pd.Series = data[measurement_parameter]

        else:
            logger.debug(f"Measurements provided as argument: {measurements}")
            data = measurements[ref_start:ref_end]
            logger.debug(f"Measurements provided as argument, filtered for {ref_start} and {ref_end}: {data}")

        # Convert timezone to local timezone
        data.index = data.index.tz_convert(self.forecast_window_start.tzinfo)

        ref_resolution = min(pd.Series(data=data.index).diff()[1:])
        if ref_resolution < self.resolution:
            data = data.resample(rule=self.resolution).mean()
        elif ref_resolution >= self.resolution:
            data = data.resample(rule=self.resolution).interpolate(method='linear')

        # Data may contain NaN if measurement series is incomplete -> fill by interpolating
        data = data.interpolate()

        logger.debug(f'Resampled and NaN-filled PV measurements for {self.forecast_function.__name__}: {data}')

        if ref_start < min(data.index):
            # Data is missing at the beginning of the reference horizon.
            logger.debug(
                f"Data of {(min(data.index) - ref_start)} hours is missing at the beginning of the reference horizon .")
            data = self.fill_missing_leading_data(incomplete_data=data, requested_start=ref_start)

        if ref_end > max(data.index):
            # Data is missing at the end of the reference horizon.
            logger.debug(
                f"Data of {ref_end - max(data.index)} hours is missing at the end of the reference horizon.")
            data = self.fill_missing_trailing_data(incomplete_data=data, requested_end=(ref_end + self.resolution))

        logger.debug(
            f"Interpolated and edge-filled data spans from {min(data.index)} to {max(data.index)} and has {len(data)} datapoints.")

        # normalize and make sure that values are <= 0
        data = abs(data) / -self.normalization_value
        logger.debug(f"Return these feature measurements for the PV short term forecaster: {data}")
        return data.to_numpy().flatten()

    def forecast_with_prior_measurements_and_sun_duration(self, measurements: pd.Series = None) -> typing.Dict[
        str, float]:
        """
        Generates a forecast of PV generation based on reference measurements of the preceding 1 hour,
        the target daytime hour and a forecast of the sunshine duration from specific weather stations (data provided by DWD).
        The model is trained to predict the next 1 hour with a 1-minute resolution, independently of the current time.
        The model expects a certain time delta between the last reference value and the forecast window start (e.g. 5 min).
        :param measurements: Optional: Input measurements, spanning at least the required reference horizon.
        :return: Forecast for PV generation.
        """
        timestamps = pd.date_range(self.forecast_window_start, end=self.forecast_window_end, freq=self.resolution,
                                   closed='left').tz_convert(pytz.utc)
        if (self.forecast_window_start.hour >= self.night_hours[0]) or (
                0 <= self.forecast_window_start.hour < self.night_hours[1]):
            # It's night -> power=0
            forecast = {timestamp.isoformat(): 0.0 for timestamp in timestamps}
        else:
            # Load Features
            # A Feature vector consists of 60 Values:
            # The 56 PV measurements with a 1 minute resolution (e.g. measurements from 9:00 until 9:55 (incl.) for the
            # forecast from 10:00 to 11:00 (excl.))
            # 3 values for the sunshine duration (from 3 weather stations) and 1 value for the current hour of the day
            dwd_features = pd.concat(
                self.get_dwd_data(self.forecast_window_start, self.forecast_window_end)).to_numpy()

            features = np.concatenate(
                (self.get_feature_measurements(measurements), dwd_features, [self.forecast_window_start.hour]))

            prediction = self.model.predict(np.expand_dims(features, axis=0))
            prediction = prediction[0]
            prediction = prediction * self.normalization_value

            logger.debug(
                f"{self.forecast_function.__name__} prediction value stats (before cleaning): min={min(prediction)}, max={max(prediction)}, number of positive values={len([v for v in prediction if v > 0])}")

            # transform numpy array to dictionary and set positive values to zero
            forecast = {timestamps[i].isoformat(): min(int(prediction[i]), 0) for i in range(len(timestamps))}

        logger.debug(f'PV short term forecast: {forecast}')
        return forecast


class QuotaBlocksForecaster(PVForecaster):
    resolution = datetime.timedelta(minutes=5)

    def __init__(self, pv_specification: dict, window_start: datetime.datetime, window_size: datetime.timedelta):
        super(QuotaBlocksForecaster, self).__init__(pv_specification=pv_specification,
                                                    window_start=window_start, window_size=window_size,
                                                    resolution=self.resolution)
        self.pv_specification = pv_specification  # Needed below for init of ShortTermForecaster (to complete forecast)
        self.forecast_function = self.nn_ensemble_forecast
        self.fallback_forecaster = PersistenceForecaster
        self.fallback_forecast_function_used: str = None

        self.forecast_window_24h_starts = [int(os.getenv('FIRST_QUOTA_WINDOW_HOUR'))
                                           + (i * int(os.getenv('QUOTA_WINDOW_SIZE')))
                                           for i in range(int(24 / int(os.getenv('QUOTA_WINDOW_SIZE'))))]

        # load models here
        self.ensembles = {
            start: self.load_ensemble(str(start).zfill(2)) for start in self.forecast_window_24h_starts[:-1]
        }

        self.tdelta_reference_to_forecast_window = datetime.timedelta(minutes=80)

    def feasible_nn_ensemble_forecast_start(self):
        """
        The NN ensembles are trained to make a prediction as of a specific hour of the day. This function finds the
        feasible forecast start time that is closest to the requested forecast start.
        :return: Forecast window start and time of the last PV measurement considered as reference (input)
        """
        if self.forecast_window_24h_starts[0] <= self.forecast_window_start.hour < \
                self.forecast_window_24h_starts[-1]:
            # Calculate differences of desired window start to preceding feasible window starts
            start_time_diffs = [self.forecast_window_start.hour - s for s in self.forecast_window_24h_starts if
                                s <= self.forecast_window_start.hour]
            # Get feasible window start with min. difference
            closest_window_start = self.forecast_window_24h_starts[np.argmin(start_time_diffs)]
            feasible_forecast_window_start = self.forecast_window_start.replace(hour=closest_window_start, minute=0)
            # Timestamp of last measurement considered as feature element: shortly before this forecast is made
            ref_end = feasible_forecast_window_start - self.tdelta_reference_to_forecast_window

        else:
            # Desired forecast window starts during night hours, for which no NN model exists
            # In any case, the NN model for the first possible window start is used
            closest_window_start = self.forecast_window_24h_starts[0]
            if 0 <= self.forecast_window_start.hour < closest_window_start:
                # First hours of the day, e.g. 0-4h
                feasible_forecast_window_start = self.forecast_window_start.replace(hour=closest_window_start, minute=0)
            else:
                # Last hours of the day, e.g. 22-24h -> feasible start is next day
                feasible_forecast_window_start = self.forecast_window_end.replace(hour=closest_window_start, minute=0)
            # Timestamp of last measurement considered as feature element is the same for the first and last
            # forecast window start of the day: shortly before the forecast for the last window is made
            ref_end = (
                              feasible_forecast_window_start - datetime.timedelta(days=1)
                      ).replace(hour=self.forecast_window_24h_starts[-1]) - self.tdelta_reference_to_forecast_window

        return feasible_forecast_window_start, ref_end

    def required_measurement_feature_vector_length(self, ref_start: datetime.datetime, ref_end: datetime.datetime):
        """
        Determine the required length of the part of the feature vector containing the reference measurements
        :param ref_start: Timestamp of first reference measurment
        :param ref_end: Timestamp of last reference measurment
        :return: The required length of the part of the feature vector containing the reference measurements
        """
        ref_horizon_duration: datetime.timedelta = ref_end - ref_start
        night_hours_duration: datetime.timedelta = datetime.timedelta(
            hours=(24 + self.night_hours[1] - self.night_hours[0]))

        return int((ref_horizon_duration - night_hours_duration) / self.resolution)

    def complement_prediction(self, prediction_series: pd.Series) -> pd.Series:
        """
        Iteratively fill the given incomplete forecast (i.e. it does not span the entire required time window)
        with the missing periods.
        The ShortTermForecaster is used to make predictions for the next hour based on the data of the
        previous hour. The latter are the already made predictions since no measurements are available.
        :param prediction_series: Existing, incomplete PV prediction
        :return: Complete prediction, spanning the entire time window
        """
        missing_periods = pd.date_range(start=max(prediction_series.index),
                                        end=self.forecast_window_end - self.resolution, freq=self.resolution,
                                        closed='right')
        logger.debug(f"Forecast so far: {prediction_series}")
        logger.debug(f"Periods missing in ensemble forecast: {missing_periods}")

        hour_forecast_start = min(missing_periods.to_pydatetime())
        while not (
                          self.forecast_window_start and self.forecast_window_end - self.resolution) in prediction_series.index:
            hour_forecast_end = hour_forecast_start + datetime.timedelta(hours=1)
            # Use prior prediction as reference data
            ref_data = prediction_series[hour_forecast_start - datetime.timedelta(hours=1):hour_forecast_start]
            logger.debug(
                f"Ref data from ensemble forecast for short term forecast as of {hour_forecast_start}: {ref_data}")

            # Get the forecast for the next hour
            hour_forecaster = ShortTermForecaster(pv_specification=self.pv_specification,
                                                  window_start=hour_forecast_start,
                                                  window_size=datetime.timedelta(hours=1))
            hour_forecast = hour_forecaster.forecast_with_prior_measurements_and_sun_duration(measurements=ref_data)

            # Transform and append to full prediction series
            hour_forecast = pd.Series(data=hour_forecast.values(),
                                      index=pd.to_datetime(list(hour_forecast.keys())).tz_convert(
                                          self.forecast_window_start.tzinfo))
            hour_forecast = hour_forecast.resample(rule=self.resolution).mean()
            prediction_series = pd.concat([prediction_series, hour_forecast])

            hour_forecast_start = hour_forecast_end

        return prediction_series

    def load_ensemble(self, time):
        """
        This method loads all models of an ensemble with specific parameters for a specific time period
        :param time: the start of the prediction time period
        :return: a list of all models of the ensemble
        """
        models = []
        model_dir = os.path.join(path, f"pv_prediction_models/{time}/")
        for filename in os.listdir(model_dir):
            models.append(load_model(os.path.join(model_dir, filename)))
        return models

    def get_ensemble_predictions(self, models, features):
        features = np.expand_dims(features, axis=0)
        predictions = []
        for en, model in enumerate(models):
            predictions.append(model.predict(features))
        # fuse prediction
        ensemble_predictions = np.average(predictions, axis=0)

        # denormalize
        denormalized_predictions = ensemble_predictions[0] * self.normalization_value
        return denormalized_predictions

    def fill_night_hours_of_prediction(self, prediction: np.ndarray, forecast_start_hour: int):
        """
        Split predictions at the start of the night and add 0 for periods in between.
        :param prediction: Predicted values without night hours
        :param forecast_start_hour: The forecast start's hour.
        :return: Predicted values filled with 0 for the night hours.
        """
        split_position = int(
            datetime.timedelta(hours=(self.night_hours[0] - forecast_start_hour)) / self.resolution)
        predictions_split = np.split(prediction, [split_position])
        night_predictions = np.zeros(
            int(datetime.timedelta(hours=(24 - self.night_hours[0] + self.night_hours[1])) / self.resolution))
        return np.concatenate((predictions_split[0], night_predictions, predictions_split[1]))

    def ensure_completeness_of_prediction(self, prediction_series: pd.Series,
                                          feasible_forecast_window_start: datetime.datetime) -> pd.Series:
        """
        Checks for missing periods in given prediction and complements if necessary.
        :param prediction_series: Prediction, possibly missing some periods at the beginning or end
        :param feasible_forecast_window_start: The feasible start of the forecast window w.r.t. to the trained
        ensembles models.
        :return: Complete prediction, i.e. covering the whole requested forecast horizon.
        """
        if self.forecast_window_start not in prediction_series.index:
            # Happens if start should be in the night hours with zero generation, but model predicted as of 4 am
            preceding_night_hours = pd.Series(data=0,
                                              index=pd.date_range(start=self.forecast_window_start,
                                                                  end=feasible_forecast_window_start,
                                                                  freq=self.resolution))
            prediction_series = pd.concat([preceding_night_hours, prediction_series])
        elif not (self.forecast_window_end - self.resolution) in prediction_series.index:
            # Happens if start did not match any of the feasible window starts and the closest preceding
            # feasible start was taken.
            # -> Fill missing hours at the end.
            logger.debug(
                f"Ensemble forecast goes only until {max(prediction_series.index)}. Complement prediction with "
                f"short term forecaster until the end ({self.forecast_window_end})")
            prediction_series = self.complement_prediction(prediction_series)

        return prediction_series

    def get_feature_measurements(self, ref_start: datetime.datetime, ref_end: datetime.datetime,
                                 measurements: typing.Union[pd.Series, None] = None) -> np.ndarray:

        if measurements is None or measurements.empty:
            logger.debug(
                f"Get PV measurements from {ref_start.astimezone(pytz.utc)} to {ref_end.astimezone(pytz.utc)} from DB.")
            measurement_parameter = 'active_power'
            data: pd.DataFrame = db.get_measurement(source=self.source, fields=measurement_parameter,
                                                    start_time=ref_start.astimezone(pytz.utc),
                                                    end_time=ref_end.astimezone(pytz.utc))
            data: pd.Series = data[measurement_parameter]

            logger.debug(
                f"Data returned from DB spans from {min(data.index)} to {max(data.index)} and has {len(data)} datapoints.")

        else:
            data: pd.Series = measurements
            logger.debug(
                f"Measurements provided as argument span from {min(data.index)} to {max(data.index)} "
                f"and have {len(data)} datapoints.")

        # Convert timezone to local timezone
        data.index = data.index.tz_convert(self.forecast_window_start.tzinfo)

        # Resample data to target resolution
        data = data.groupby(pd.Grouper(freq=self.resolution)).mean()
        # Data may contain NaN if measurement series is incomplete -> fill by interpolating
        data = data.interpolate()
        logger.debug(f"TZ-converted, resampled and interpolated data has {len(data)} datapoints.")
        logger.debug(f"Resampled data first and last 5 rows:\n {data.head(5)} \n ... \n {data.tail(5)}.")

        fill_max_timespan = (ref_end - ref_start) * float(os.getenv('LIMIT_PV_REF_EDGE_FILLING'))
        total_missing_timespan = min(data.index) - ref_start + ref_end - max(data.index)
        if total_missing_timespan > fill_max_timespan:
            logger.warning(f"The reference data is still missing {total_missing_timespan} hours of requested "
                           f"data at the edges after interpolation. This exceeds the allowed maximum timespan "
                           f"of {fill_max_timespan} hours that may be filled using regression. "
                           f"Data filling is not performed.")

        else:
            if ref_start < min(data.index):
                # Data is missing at the beginning of the reference horizon.
                logger.debug(
                    f"Data of {(min(data.index) - ref_start)} hours is missing at the beginning of the reference horizon .")
                data = self.fill_missing_leading_data(incomplete_data=data, requested_start=ref_start)

            if ref_end - self.resolution > max(data.index):
                # Data is missing at the end of the reference horizon.
                logger.debug(
                    f"Data of {ref_end - self.resolution - max(data.index)} hours is missing at the end of the reference horizon.")
                data = self.fill_missing_trailing_data(incomplete_data=data, requested_end=ref_end)

        logger.debug(
            f"Interpolated and edge-filled data spans from {min(data.index)} to {max(data.index)} and has {len(data)} datapoints.")

        # delete measurements from the night hours (e.g. between 22 pm and 4 am)
        data = data.between_time(datetime.time(self.night_hours[1]), datetime.time(self.night_hours[0]),
                                 include_end=False)

        logger.debug(
            f'TZ-converted, resampled, NaN-filled, night-hours-removed PV measurements for {self.forecast_function.__name__}: {data}')
        # normalize and make sure that values are <= 0
        data = abs(data) / -self.normalization_value
        data_vector = data.to_numpy().flatten()
        return data_vector

    def get_features(self, ref_start: datetime.datetime, ref_end: datetime.datetime,
                     forecast_window_start: datetime.datetime, forecast_window_end: datetime.datetime,
                     measurements: typing.Union[pd.Series, None] = None):
        """
        Load Features
        A Feature vector consists of 270 Values:
        216 Values for the normalized pv-measurements (18 hours with 5 min resolution) and
        3*18 values for the sunshine duration (18 hours with 1 h resolution, from 3 weather stations)
        """
        measurement_features = self.get_feature_measurements(ref_start=ref_start, ref_end=ref_end,
                                                             measurements=measurements)
        if len(measurement_features) != self.required_measurement_feature_vector_length(ref_start, ref_end):
            logger.warning(f"Measurement feature vector only has {len(measurement_features)} elements instead of the "
                           f"required {self.required_measurement_feature_vector_length(ref_start, ref_end)}. "
                           f"Falling back to a persistence forecast.")
            raise ValueError(
                f"Measurement feature vector only has {len(measurement_features)} elements instead of the required {self.required_measurement_feature_vector_length(ref_start, ref_end)}.")

        # Get weather features
        dwd_features = pd.concat(
            self.get_dwd_data(forecast_window_start, forecast_window_end)).to_numpy()
        # Concat all feature parts to the final feature vector
        features = np.concatenate((measurement_features, dwd_features))
        return features

    def make_fallback_forecast(self):
        fallback_forecaster = self.fallback_forecaster(
            pv_specification=self.pv_specification,
            window_start=self.forecast_window_start,
            window_size=self.window_size,
            resolution=self.resolution
        )

        fallback_forecast = fallback_forecaster.forecast_function()
        logger.debug(f"Fallback forecast returned to QuotaBlocksForecaster: {fallback_forecast}")
        # The used fallback function will be documented as meta data of this forecast
        self.fallback_forecast_function_used = f"{fallback_forecaster.__class__.__name__}.{fallback_forecaster.forecast_function.__name__}"
        return fallback_forecast

    def nn_ensemble_forecast(self, measurements: typing.Union[pd.Series, None] = None) -> typing.Dict[str, float]:
        """
        Generates a forecast of PV generation based on reference measurements of the preceding 24 hours and a forecast
        of the sunshine duration from specific weather stations (data provided by DWD).
        The NN models are trained to predict the next 24 hours starting at one of multiple specific forecast
        window starts, respectively (e.g. 4 am, 10 am, 16 am). The models expect a certain time delta between the last
        reference value and the forecast window start (here: 80 min).
        Final predictions are calculated by taking the mean of an ensemble of the described NNs.
        :return: Forecast for PV generation.
        """

        feasible_forecast_window_start, ref_end = self.feasible_nn_ensemble_forecast_start()
        feasible_forecast_window_end = feasible_forecast_window_start + self.window_size
        logger.debug(
            f'Local feasible window start: {feasible_forecast_window_start}; original: {self.forecast_window_start}')
        ref_start = ref_end - datetime.timedelta(days=1)
        # Load matching ensemble
        ensemble = self.ensembles[feasible_forecast_window_start.hour]

        # Load Features
        try:
            features = self.get_features(ref_start=ref_start, ref_end=ref_end,
                                         forecast_window_start=feasible_forecast_window_start,
                                         forecast_window_end=feasible_forecast_window_end,
                                         measurements=measurements)
        except (KeyError, ValueError, requests.exceptions.ConnectionError) as e:
            # except (ValueError, Exception) as e:
            # Raised in above function if the measurement feature vector is too short or if there were no
            # active_power measurements found in the database for the requested period.
            logger.warning(f"{type(e)}:{str(e)}")
            logger.warning(
                f"Falling back to making a prediction with the fallback forecaster {self.fallback_forecaster.__name__}.")
            fallback_forecast = self.make_fallback_forecast()
            return fallback_forecast

        # Use the correct ensemble to make a Prediction on the given features
        # Returned prediction is de-normalized, i.e. contains power values in Watt
        prediction = self.get_ensemble_predictions(ensemble, features)
        # Add night hours, which were initially excluded from the prediction
        prediction = self.fill_night_hours_of_prediction(prediction=prediction,
                                                         forecast_start_hour=feasible_forecast_window_start.hour)

        # Transform numpy array to pd.Series with timestamps
        prediction_series = pd.Series(
            data=prediction,
            index=pd.date_range(
                feasible_forecast_window_start, end=feasible_forecast_window_end,
                freq=self.resolution, closed='left', tz=self.forecast_window_start.tzinfo
            )
        )
        logger.debug(
            f"{self.forecast_function.__name__} prediction value stats (before cleaning): min={min(prediction_series)}, max={max(prediction_series)}, number of positive values={prediction_series[prediction_series > 0].count()}")

        # Convert all positive predicted values to zero. This usually happens only for predictions close to zero and
        # otherwise will cause trouble.
        prediction_series[prediction_series > 0] = 0

        # Make sure the prediction covers the whole requested forecast horizon and complement if necessary
        prediction_series = self.ensure_completeness_of_prediction(prediction_series, feasible_forecast_window_start)

        logger.debug(
            f"{self.forecast_function.__name__} prediction value stats (final): min={min(prediction_series)}, max={max(prediction_series)}, number of positive values={prediction_series[prediction_series > 0].count()}")

        # Filter for the originally requested forecast window
        forecast = prediction_series[self.forecast_window_start:self.forecast_window_end - self.resolution]
        # Convert timestamps to UTC, then to dict with ISO-format str keys
        forecast.index = forecast.index.tz_convert(datetime.timezone.utc)

        forecast = {ts.isoformat(): int(val) for ts, val in forecast.iteritems()}
        logger.debug(f'Long-term PV forecast as of {self.forecast_window_start}: {forecast}')
        return forecast


class PersistenceForecaster(PVForecaster):

    def __init__(self, pv_specification: dict, window_start: datetime.datetime, window_size: datetime.timedelta,
                 resolution: datetime.timedelta):
        super(PersistenceForecaster, self).__init__(pv_specification=pv_specification, window_start=window_start,
                                                    window_size=window_size, resolution=resolution)

        self.forecast_function = self.persistence_forecast

    def get_reference_measurements(self, ref_start: datetime.datetime, ref_end: datetime.datetime,
                                   measurements: typing.Union[pd.Series, None] = None) -> pd.Series:

        measurement_parameter = 'active_power'
        if measurements is None:
            # Get the reference data from the database
            measurements = db.get_measurement(source=self.source, fields=measurement_parameter,
                                              start_time=ref_start.astimezone(pytz.utc),
                                              end_time=ref_end.astimezone(pytz.utc))
            try:
                measurements: pd.Series = measurements[measurement_parameter]
            except KeyError:
                # No data found for measurement parameter
                measurements = pd.DataFrame()
                measurements.index = pd.to_datetime(measurements.index)
                return measurements
        else:
            # Filter for reference horizon
            measurements = measurements[ref_start:ref_end]

        # Convert timezone to local timezone
        measurements.index = measurements.index.tz_convert(self.forecast_window_start.tzinfo)
        return measurements

    def persistence_forecast(self, data: typing.Union[pd.Series, None] = None) -> typing.Dict[str, float]:
        """
        :return: Forecast as dict with format {'period1': power1, 'period2': power2, ..., 'periodn': powern}, with the
        keys being the start of the respective periods as str in ISO-format
        """
        # Define the reference data
        offset = 0 if self.window_size != datetime.timedelta(
            hours=24) else 1  # Skip antecedent day (=today) if making a day-ahead forecast
        ref_start = self.forecast_window_start - datetime.timedelta(
            days=int(os.getenv('NUM_REF_PRECED_DAYS')) + offset)  # inclusive
        ref_end = (self.forecast_window_start - datetime.timedelta(days=1))  # exclusive

        data = self.get_reference_measurements(ref_start=ref_start, ref_end=ref_end, measurements=data)
        logger.debug(f"Ref data (len={len(data)}): {data}")

        # Make sure that values are <= 0
        data = -abs(data)

        # Split data by days
        days_data: typing.List[pd.DataFrame] = []
        dates = set(sorted(set(data.index.date), reverse=True))
        for day in dates:
            days_data.append(data[data.index.date == day])

        # If data contains less than 3 reference days with that window due to missing measurements,
        # iteratively include data in corresponding window from older days, but go back 14 days max
        min_num_dates = int(os.getenv('NUM_REF_PRECED_DAYS')) if self.forecast_window_start.time() == datetime.time(
            0) else int(os.getenv('NUM_REF_PRECED_DAYS')) + 1
        if len(dates) < min_num_dates:
            i = 1
            ref_end = copy.deepcopy(ref_start)
            while len(dates) < min_num_dates and i < 14:
                logger.debug(f"number of unique dates in ref data={len(dates)} (dates: {dates})")
                logger.debug(f'Try additional day no. {i}')
                ref_start = ref_start - datetime.timedelta(days=1)
                add_data = self.get_reference_measurements(ref_start=ref_start, ref_end=ref_end)
                logger.debug(f'Data from Influx: {add_data}')

                if not add_data.empty:
                    add_dates = set(add_data.index.date)
                    dates = dates.union(add_dates)
                else:
                    add_dates = set()
                    logger.debug('Again, window not contained.')

                i += 1

            # Make sure that values are <= 0
            add_data = -abs(add_data)

            # Append the additional data, again split by days in reverse order (oldest last)
            for date in sorted(add_dates, reverse=True):
                days_data.append(add_data[add_data.index.date == date])

            logger.debug(f"number of unique dates in ref data={len(dates)} (dates: {dates})")

        # Calculation of forecast pv value for each period within window
        forecast = {}
        period_start: datetime.datetime = self.forecast_window_start

        while period_start < self.forecast_window_end:

            period_end = period_start + self.resolution
            logger.debug(f"Period {period_start.time()} - {period_end.time()}")
            # Get mean over values that fall in this period for each reference day
            period_mean_values = []
            for i, day_data in enumerate(days_data):
                # Catch RuntimeWarning caused by an empty DataFrame when taking the mean of its values
                # Occurs when data is missing for the whole period (not window) of the given ref. day
                # Catching preferred over "if df.emtpy" clause, because otherwise we more or less do the same
                # check twice
                with warnings.catch_warnings():
                    warnings.filterwarnings(action='error', message='Mean of empty slice')

                    try:
                        if period_start.time() < period_end.time():  # Default case
                            mean = day_data[(day_data.index.time >= period_start.time())
                                            & (day_data.index.time < period_end.time())].values.mean()
                        else:  # Period end is midnight
                            mean = day_data[(day_data.index.time >= period_start.time())].values.mean()

                        period_mean_values.append(mean)

                    except RuntimeWarning:
                        logger.debug(
                            f'Data from {min(day_data.index)} to {max(day_data.index)} does not include period {period_start.time()} to {period_end.time()}.')

            # Calculate exponentially weighted mean, i.e. the more recent a day, the higher its weight
            num_ref_days = len(period_mean_values)

            if num_ref_days == 0:
                logger.debug(f"No data for period from {period_start.time()} to {period_end.time()} on any day. "
                             f"Period will be filled by interpolation later.")

                # Next period
                period_start = period_end
                continue

            # Note: num_ref_days might can be zero if none of the days data contains the given period; This would
            # throw a ZeroDivisionError in the following
            weights = [2 ** (-(i + 1)) for i in range(num_ref_days)]
            period_mean = sum([period_mean_values[i] * weights[i] for i in range(num_ref_days)]) / sum(weights)
            # Key must be string (Celery requirement)
            forecast[period_start.isoformat()] = period_mean

            # Next period
            period_start = period_end

        forecast_series = pd.Series(data=forecast)
        forecast_series.index = pd.to_datetime(forecast_series.index)
        forecast_series = forecast_series.resample(rule=self.resolution).interpolate(method='linear')

        # Convert timestamps to UTC, then to dict with ISO-format str keys
        forecast_series.index = forecast_series.index.tz_convert(datetime.timezone.utc)
        forecast_dict = {ts.isoformat(): int(val) for ts, val in forecast_series.iteritems()}

        return forecast_dict
