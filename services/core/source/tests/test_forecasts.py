import datetime
import os
import random

import numpy as np
import pandas as pd
import pytest
import pytz
from dotenv import load_dotenv, find_dotenv

os.environ["DEBUG"] = "False"
os.environ["DEBUG_FORECASTING"] = "True"
os.environ["DEBUG_OPTIMIZATION"] = "False"
# os.environ["INFLUX_HOSTNAME"] = "localhost"

load_dotenv(find_dotenv(), verbose=True, override=False)

from core.forecasting.pv_forecast_service import QuotaBlocksForecaster, ShortTermForecaster, PersistenceForecaster


@pytest.fixture(scope='class')
def window_start():
    return datetime.datetime(2022, 5, 1, 8, 0, tzinfo=datetime.timezone.utc)


@pytest.fixture(scope='module')
def pv_specification():
    return {'key': 'pv', 'category': 'generator', 'subcategory': 'pv', 'energy_carrier': 'el',
            'active_power_nominal': -10000,
            'inverter_efficiency': 0.97, 'continuous_curtailment': False,
            'curtailment_levels': [0, 0.3, 0.6, 1]}


@pytest.fixture
def pv_measurements(reference_horizon, pv_specification):
    def bell_curve(x):
        mean = np.mean(x)
        std = np.std(x)
        return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))

    # ref_start and ref_end are datetime.datetime in local timezone; resolution: datetime.timedelta
    ref_start, ref_end, resolution = reference_horizon

    sunrise = 6
    sunset = 20
    sunshine_hours = sunset - sunrise
    x = np.arange(0, sunshine_hours * 60 * 2)
    y = bell_curve(x)

    n_days = max(2, (ref_end - ref_start).days + 1)
    # Create data series with bell curve-shaped PV generation spanning n_days
    # Values are 0 from sunset to sunrise, otherwise a "perfect" PV generation curve with some noise
    values = [0] * sunrise * 60 * 2 \
             + [v * (pv_specification['active_power_nominal'] / max(y)) + random.gauss(0, 100) for v in y] \
             + [0] * (24 - sunset) * 60 * 2
    index = pd.date_range(start=ref_start.date(), freq=resolution, periods=n_days * len(values), tz=pytz.utc)
    data = pd.Series(index=index, data=n_days * values)
    reference_data = data[ref_start:ref_end - resolution]
    return reference_data


def incomplete_measurements(drop_n_first_rows, drop_n_last_rows, measurements):
    if drop_n_last_rows > 0:
        return measurements.iloc[drop_n_first_rows:-drop_n_last_rows]
    else:
        return measurements.iloc[drop_n_first_rows:]


def check_returned_forecast(forecast: dict, source: str, required_window_start: datetime.datetime,
                            required_length: int):
    # Returned forecast is a nested dict with single key=source and a dict with the forecast as value
    assert list(forecast.keys())[0] == source
    # Check if forecast starts at requested datetime and has the requested number of periods
    assert datetime.datetime.fromisoformat(min(forecast[source])) == required_window_start
    assert len(forecast[source]) == required_length


class TestPVQuotaBlocksForecast:
    source = os.getenv('PV_KEY')

    @pytest.fixture(scope='class')
    def window_size(self):
        return datetime.timedelta(hours=24)

    @pytest.fixture(scope='class')
    def forecaster(self, pv_specification, window_start, window_size):
        return QuotaBlocksForecaster(
            pv_specification=pv_specification,
            window_start=window_start,
            window_size=window_size
        )

    @pytest.fixture(scope='function', params=[30, 60])
    def reference_horizon(self, request, window_start, forecaster):
        end = forecaster.forecast_window_start - forecaster.tdelta_reference_to_forecast_window
        start = end - datetime.timedelta(hours=24)
        resolution = datetime.timedelta(seconds=request.param)
        return start, end, resolution

    @pytest.fixture(scope='function')
    def required_feature_vector_length(self, reference_horizon, forecaster):
        ref_horizon_duration: datetime.timedelta = reference_horizon[1] - reference_horizon[0]
        night_hours_duration: datetime.timedelta = datetime.timedelta(
            hours=(24 + forecaster.night_hours[1] - forecaster.night_hours[0]))

        return int((ref_horizon_duration - night_hours_duration) / forecaster.resolution)

    def test_forecast(self, forecaster, window_start, window_size):
        forecast = forecaster.make_forecast()
        check_returned_forecast(forecast=forecast,
                                source=self.source,
                                required_window_start=window_start,
                                required_length=int(window_size / forecaster.resolution))
        assert forecaster.fallback_forecast_function_used is None
        positive_values = [v for v in forecast[self.source].values() if v > 0]
        assert len(positive_values) == 0

    @pytest.mark.parametrize("n_leading_rows, n_trailing_rows", [(30, 50), (0, 30), (30, 0), (0, 0)])
    def test_feature_measurements_completion_missing_at_edges(
            self, n_leading_rows, n_trailing_rows, reference_horizon,
            forecaster, pv_measurements, required_feature_vector_length):

        ref_start, ref_end, _ = reference_horizon
        ref_data = incomplete_measurements(n_leading_rows, n_trailing_rows, pv_measurements)
        feature_measurements = forecaster.get_feature_measurements(ref_start, ref_end, ref_data)
        assert len(feature_measurements) == required_feature_vector_length

    def test_feature_measurements_interpolation(self, reference_horizon, forecaster,
                                                pv_measurements, required_feature_vector_length):
        ref_start, ref_end, _ = reference_horizon

        idcs_to_delete = random.sample(list(pv_measurements.index), k=max(50, int(0.5*len(pv_measurements))))
        print(f'Delete these {len(idcs_to_delete)} periods from the reference measurements.')
        ref_data = pv_measurements.drop(index=idcs_to_delete)
        print(f'Sparse reference measurements:\n {ref_data}')
        feature_measurements = forecaster.get_feature_measurements(ref_start, ref_end, ref_data)
        print(f'Feature measurements:\n {feature_measurements}')
        assert len(feature_measurements) == required_feature_vector_length

    @pytest.mark.parametrize("leading, trailing", [(True, False), (False, True), (True, True)])
    def test_abort_feature_measurement_completion_if_too_much_missing(self, leading, trailing, reference_horizon,
                                                                      forecaster, pv_measurements,
                                                                      required_feature_vector_length):
        ref_start, ref_end, resolution = reference_horizon
        fill_max_timespan = (ref_end - ref_start) * float(os.getenv('LIMIT_PV_REF_EDGE_FILLING'))
        highly_incomplete_ref_data = None
        if leading and trailing:
            highly_incomplete_ref_data = pv_measurements[
                                         ref_start + fill_max_timespan / 2 + forecaster.resolution:
                                         ref_end - fill_max_timespan / 2]
        elif leading:
            highly_incomplete_ref_data = pv_measurements[ref_start + fill_max_timespan + forecaster.resolution:]
        elif trailing:
            highly_incomplete_ref_data = pv_measurements[:ref_end - fill_max_timespan - forecaster.resolution]

        feature_measurements = forecaster.get_feature_measurements(ref_start, ref_end, highly_incomplete_ref_data)
        # Measurements should not be filled
        assert len(feature_measurements) < required_feature_vector_length

    @pytest.mark.parametrize("leading, trailing", [(True, False), (False, True), (True, True)])
    def test_fallback_forecast(self, leading, trailing, forecaster, pv_measurements, window_start, window_size):
        fill_max_periods = len(pv_measurements) * float(os.getenv('LIMIT_PV_REF_EDGE_FILLING'))
        ref_data = None
        # Manipulate the reference measurements such that more than fill_max_periods periods of data are missing
        if leading and trailing:
            # Data missing at both, start and end
            ref_data = incomplete_measurements(drop_n_first_rows=int(fill_max_periods / 2) + 1,
                                               drop_n_last_rows=int(fill_max_periods / 2) + 1,
                                               measurements=pv_measurements)
        elif leading:
            # Data only missing at the beginning
            ref_data = incomplete_measurements(drop_n_first_rows=int(fill_max_periods) + 5, drop_n_last_rows=0,
                                               measurements=pv_measurements)
        elif trailing:
            # Data only missing at the end
            ref_data = incomplete_measurements(drop_n_first_rows=0, drop_n_last_rows=int(fill_max_periods) + 5,
                                               measurements=pv_measurements)
        forecast = forecaster.make_forecast(ref_data)
        # Check if the fallback forecaster was used. If so, the attribute has the format "class_name.function_name".
        # Since we don't have access to the forecast function of the fallback forecaster (because it's an
        # instance method), simply check the class name.
        assert forecaster.fallback_forecast_function_used.split('.')[0] == forecaster.fallback_forecaster.__name__
        check_returned_forecast(forecast=forecast,
                                source=self.source,
                                required_window_start=window_start,
                                required_length=int(window_size / forecaster.resolution))


class TestPVShortTermForecast:
    source = os.getenv('PV_KEY')

    @pytest.fixture(scope='function')
    def forecaster(self, pv_specification, window_start, window_size):
        return ShortTermForecaster(
            pv_specification=pv_specification,
            window_start=window_start,
            window_size=window_size
        )

    @pytest.fixture(scope='function', params=[0, 35])
    def window_start(self, request):
        return datetime.datetime(2022, 4, 1, 8, int(request.param), tzinfo=datetime.timezone.utc)

    @pytest.fixture(scope='class')
    def window_size_default(self):
        return datetime.timedelta(hours=1)

    @pytest.fixture(scope='function', params=[datetime.timedelta(minutes=60), datetime.timedelta(minutes=25)])
    def window_size(self, request):
        return request.param

    @pytest.fixture(scope='function', params=[30, 60])
    def reference_horizon(self, request, window_start, forecaster):
        end = forecaster.forecast_window_start - forecaster.tdelta_reference_to_forecast_window + forecaster.resolution
        start = forecaster.forecast_window_start - datetime.timedelta(hours=1)
        resolution = datetime.timedelta(seconds=request.param)
        return start, end, resolution

    def test_forecast(self, forecaster, window_start, window_size):
        forecast = forecaster.make_forecast()
        print(f"Forecast: {forecast}")
        check_returned_forecast(forecast=forecast,
                                source=self.source,
                                required_window_start=window_start,
                                required_length=int(window_size/ forecaster.resolution))
        assert forecaster.fallback_forecast_function_used is None
        positive_values = [v for v in forecast[self.source].values() if v > 0]
        assert len(positive_values) == 0

    @pytest.mark.parametrize("n_leading_rows, n_trailing_rows", [(0, 5), (5, 0), (3, 5), (20, 20), (0, 0)])
    def test_feature_measurements_completion_missing_at_edges(
            self, n_leading_rows, n_trailing_rows,
            forecaster, pv_measurements):
        ref_data = incomplete_measurements(n_leading_rows, n_trailing_rows, pv_measurements)
        feature_measurements = forecaster.get_feature_measurements(ref_data)
        assert len(feature_measurements) == 56

    def test_feature_measurements_interpolation(self, forecaster, pv_measurements):
        idcs_to_delete = random.sample(list(pv_measurements.index), k=max(20, int(0.5*len(pv_measurements))))
        print(f'Delete these periods from the reference measurements:\n {idcs_to_delete}')
        ref_data = pv_measurements.drop(index=idcs_to_delete)
        print(f'Sparse reference measurements:\n {ref_data}')
        feature_measurements = forecaster.get_feature_measurements(ref_data)
        print(f'Feature measurements:\n {feature_measurements}')
        assert len(feature_measurements) == 56


class TestPersistenceForecast:
    source = os.getenv('PV_KEY')

    @pytest.fixture(scope='class')
    def window_size(self):
        return datetime.timedelta(hours=24)

    @pytest.fixture(scope='class')
    def forecaster(self, pv_specification, window_start, window_size):
        return PersistenceForecaster(
            pv_specification=pv_specification,
            window_start=window_start,
            window_size=window_size,
            resolution=datetime.timedelta(minutes=5)
        )

    @pytest.fixture(scope='function', params=[int(os.getenv('NUM_REF_PRECED_DAYS')), 1])
    def reference_horizon(self, request, window_start, forecaster):
        end = forecaster.forecast_window_start - datetime.timedelta(days=1)
        start = end - datetime.timedelta(days=request.param)
        resolution = datetime.timedelta(seconds=30)
        return start, end, resolution

    @pytest.fixture(scope='function')
    def required_feature_vector_length(self, reference_horizon, forecaster):
        ref_horizon_duration: datetime.timedelta = reference_horizon[1] - reference_horizon[0]
        night_hours_duration: datetime.timedelta = datetime.timedelta(
            hours=(24 + forecaster.night_hours[1] - forecaster.night_hours[0]))

        return int((ref_horizon_duration - night_hours_duration) / forecaster.resolution)

    def test_forecast(self, forecaster, window_start, window_size, pv_measurements):
        forecast = forecaster.make_forecast(pv_measurements)
        print(f"Forecast: {forecast}")
        check_returned_forecast(forecast=forecast,
                                source=self.source,
                                required_window_start=window_start,
                                required_length=int(window_size / forecaster.resolution))
        assert forecaster.fallback_forecast_function_used is None
        positive_values = [v for v in forecast[self.source].values() if v > 0]
        assert len(positive_values) == 0

    def test_forecast_incomplete_measurements(self, forecaster, window_start, window_size, pv_measurements):
        # Drop 3 days
        ref_data = incomplete_measurements(drop_n_first_rows=int(3 * 24 * 60 * 60 / 30),
                                           drop_n_last_rows=0,
                                           measurements=pv_measurements)
        print(f"Incomplete ref data: {ref_data}")
        forecast = forecaster.make_forecast(ref_data)
        print(f"Forecast: {forecast}")
        check_returned_forecast(forecast=forecast,
                                source=self.source,
                                required_window_start=window_start,
                                required_length=int(window_size / forecaster.resolution))
        assert forecaster.fallback_forecast_function_used is None
        positive_values = [v for v in forecast[self.source].values() if v > 0]
        assert len(positive_values) == 0
