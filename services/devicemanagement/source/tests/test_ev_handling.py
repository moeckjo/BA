import datetime
import logging
import os
import random

import pandas as pd
import pytest
from dotenv import load_dotenv, find_dotenv

os.environ["DEBUG"] = "True"
load_dotenv(find_dotenv(), verbose=True, override=False)

# import utils
from devicemanagement import device_manager, logger

logger.addHandler(logging.StreamHandler())


@pytest.fixture(scope='function', params=[0, 1, 2, 3, 5])
def charging_state(request):
    return request.param


@pytest.fixture(scope='function', params=[0, 1])
def connected(request):
    return request.param


@pytest.fixture(scope='module')
def deviation_timestamp():
    return datetime.datetime(2022, 5, 5, 20, 5, 30)


@pytest.fixture(scope='function', params=[5, 120])
def connected_at(request, deviation_timestamp):
    return deviation_timestamp - datetime.timedelta(minutes=request.param)


@pytest.fixture(scope='function')
def measurements_charged(connected_at, deviation_timestamp):
    unconnected_duration = datetime.timedelta(hours=1)
    index = pd.date_range(start=connected_at - unconnected_duration, end=deviation_timestamp, freq='30S')
    data = pd.Series(index=index, data=random.choices(range(4000, 11000), k=len(index)))
    data.loc[data.index <= connected_at] = 0
    print(f"EV measurements: {data}")
    return data


@pytest.fixture(scope='function')
def measurements_uncharged(connected_at, deviation_timestamp):
    unconnected_duration = datetime.timedelta(hours=1)
    index = pd.date_range(start=connected_at - unconnected_duration, end=deviation_timestamp, freq='30S')
    data = pd.Series(index=index, data=[0] * len(index))
    # Add small power value to also test threshold
    data.iloc[-1] = 50
    print(f"EV measurements: {data}")
    return data


@pytest.mark.parametrize("target_value, sensor_value", [(5000, 0), (0, 0)])
def test_schedule_deviation_evaluation_unconnected(target_value, sensor_value, charging_state):
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc)
    ev_full = device_manager.evaluate_ev_schedule_deviation(target_value=target_value,
                                                            sensor_value=sensor_value,
                                                            timestamp=timestamp,
                                                            connected=0,
                                                            charging_state=charging_state,
                                                            )
    assert ev_full is False


@pytest.mark.parametrize("target_value, sensor_value", [(5000, 0), (5000, 4000), (5000, 10000), (0, 0)])
def test_schedule_deviation_evaluation_connected_charged(target_value, sensor_value, charging_state,
                                                         measurements_charged, connected_at):
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc)
    ev_full = device_manager.evaluate_ev_schedule_deviation(target_value=target_value,
                                                            sensor_value=sensor_value,
                                                            timestamp=timestamp,
                                                            connected=1,
                                                            connected_at=connected_at,
                                                            charging_state=charging_state,
                                                            charging_power_measurements=measurements_charged
                                                            )
    evse_config = device_manager.get_device_specifications(os.getenv('EVSE_KEY'))

    if target_value > sensor_value:
        if charging_state == evse_config["charging_state_value_map"]["blocked_by_vehicle"]:
            assert ev_full is True
        else:
            assert ev_full is False
    else:
        assert ev_full is False


@pytest.mark.parametrize("target_value, sensor_value", [(5000, 0), (0, 0)])
def test_schedule_deviation_evaluation_connected_uncharged(target_value, sensor_value, charging_state,
                                                           measurements_uncharged, connected_at):
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc)
    ev_full = device_manager.evaluate_ev_schedule_deviation(target_value=target_value,
                                                            sensor_value=sensor_value,
                                                            timestamp=timestamp,
                                                            connected=1,
                                                            connected_at=connected_at,
                                                            charging_state=charging_state,
                                                            charging_power_measurements=measurements_uncharged
                                                            )
    assert ev_full is False


def test_ev_charged_since_uncharged(measurements_uncharged, connected_at):
    charged = device_manager.ev_charged_since(timestamp=connected_at,
                                              charging_power_measurements=measurements_uncharged)
    assert not charged


def test_ev_charged_since_charged(measurements_charged, connected_at):
    charged = device_manager.ev_charged_since(timestamp=connected_at, charging_power_measurements=measurements_charged)
    assert charged
