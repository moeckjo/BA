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
import gridmanagement.utils as utils


def create_time_series(resolution_sec: int, n_periods: int, second: int = 0) -> pd.Series:
    values = random.choices(range(0, 10000), k=n_periods)
    dt_index = pd.date_range(
        start=datetime.datetime(2022, 4, 29, 9, 30, second, tzinfo=datetime.timezone.utc),
        freq=f'{resolution_sec}S',
        periods=n_periods
    )
    s = pd.Series(index=dt_index, data=values)
    print(f"Original time series:\n {s}")
    return s


def test_auto_downsampling():
    target_resolution_sec = 60
    time_series = create_time_series(resolution_sec=30, n_periods=3)
    resampled_series = utils.resample_time_series(
        time_series=time_series,
        target_resolution=target_resolution_sec,
        conversion='auto'
    )
    print(f"Resampled time series:\n {resampled_series}")

    assert pd.Series(resampled_series.index).diff().min().total_seconds() == target_resolution_sec
    assert pd.Series(resampled_series.index).diff().mean().total_seconds() == target_resolution_sec
    assert sum(resampled_series.isnull()) == 0


def test_auto_upsampling():
    target_resolution_sec = 10
    time_series = create_time_series(resolution_sec=60, n_periods=2)
    resampled_series = utils.resample_time_series(
        time_series=time_series,
        target_resolution=target_resolution_sec,
        conversion='auto'
    )
    print(f"Resampled time series:\n {resampled_series}")

    assert pd.Series(resampled_series.index).diff().min().total_seconds() == target_resolution_sec
    assert pd.Series(resampled_series.index).diff().mean().total_seconds() == target_resolution_sec
    assert sum(resampled_series.isnull()) == 0


@pytest.mark.parametrize("timestamp_seconds, target_resolution_sec",
                         [(0, 60), (30, 60), (0, 30), (30, 30), (30, 15), (15, 30), (30, 300)])
def test_auto_resampling_with_single_datapoint_closed_left(timestamp_seconds, target_resolution_sec):
    time_series = create_time_series(resolution_sec=60, n_periods=1, second=timestamp_seconds)
    resampled_series = utils.resample_time_series(
        time_series=time_series,
        target_resolution=target_resolution_sec,
        conversion='auto',
        resample_kwargs=dict(closed='left', label='left')
    )
    print(f"Resampled time series:\n {resampled_series}")

    assert len(resampled_series) == 1
    assert sum(resampled_series.isnull()) == 0
    # Target resolution matched?
    assert resampled_series.index[0].timestamp() % target_resolution_sec == 0
    # New timestamp must not be greater than original timestamp when closing & labelling left
    assert resampled_series.index[0] <= time_series.index[0]


@pytest.mark.parametrize("timestamp_seconds, target_resolution_sec",
                         [(0, 60), (30, 60), (0, 30), (30, 30), (30, 15), (15, 30), (30, 300)])
def test_auto_resampling_with_single_datapoint_closed_right(timestamp_seconds, target_resolution_sec):
    time_series = create_time_series(resolution_sec=60, n_periods=1, second=timestamp_seconds)
    resampled_series = utils.resample_time_series(
        time_series=time_series,
        target_resolution=target_resolution_sec,
        conversion='auto',
        resample_kwargs=dict(closed='right', label='right')
    )
    print(f"Resampled time series:\n {resampled_series}")

    assert len(resampled_series) == 1
    assert sum(resampled_series.isnull()) == 0
    # Target resolution matched?
    assert resampled_series.index[0].timestamp() % target_resolution_sec == 0
    # New timestamp has to be greater or equal than original timestamp when closing & labelling right
    assert resampled_series.index[0] >= time_series.index[0]


def test_resampling_ignore_nan():
    target_resolution_sec = 60
    time_series = create_time_series(resolution_sec=60, n_periods=15)
    # Drop a row in between and the last one
    time_series.drop(index=[time_series.index[3], time_series.index[-1]], inplace=True)
    print(f"Series with missing data:\n {time_series}")

    resampled_series = utils.resample_time_series(
        time_series=time_series,
        target_resolution=target_resolution_sec,
        conversion='auto',
        downsample_method=os.getenv(
            'OUTGOING_MEASUREMENTS_DOWNSAMPLING_METHOD'),
        upsample_method=os.getenv(
            'OUTGOING_MEASUREMENTS_UPSAMPLING_METHOD'),
        resample_kwargs=dict(closed='right', label='right'),
        ignore_nan=True
    )
    print(f"Resampled time series:\n {resampled_series}")
    assert len(resampled_series) == 14


def test_resampling_not_ignore_nan():
    target_resolution_sec = 60
    time_series = create_time_series(resolution_sec=60, n_periods=15)
    # Drop a row in between and the last one
    time_series.drop(index=[time_series.index[3], time_series.index[-1]], inplace=True)
    print(f"Series with missing data:\n {time_series}")

    try:
        resampled_series = utils.resample_time_series(
            time_series=time_series,
            target_resolution=target_resolution_sec,
            conversion='auto',
            downsample_method=os.getenv(
                'OUTGOING_MEASUREMENTS_DOWNSAMPLING_METHOD'),
            upsample_method=os.getenv(
                'OUTGOING_MEASUREMENTS_UPSAMPLING_METHOD'),
            resample_kwargs=dict(closed='right', label='right'),
            ignore_nan=False
        )
        print(f"Resampled time series:\n {resampled_series}")
    except Exception as e:
        assert isinstance(e, AssertionError)