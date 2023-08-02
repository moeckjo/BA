import datetime
import typing

import numpy as np
import pandas

from gridmanagement import logger


def resample_time_series(time_series: typing.Union[typing.Dict[datetime.datetime, int], pandas.DataFrame],
                         target_resolution: int, return_type: str = None,
                         conversion: str = 'auto',
                         downsample_method: str = 'mean',
                         upsample_method: str = 'interpolate',
                         resample_kwargs: dict = None,
                         ignore_nan: bool = False
                         ) -> typing.Union[typing.Dict[datetime.datetime, int], pandas.DataFrame]:
    """
    Resample a time series to a target resolution using a specific aggregation method.
    :param time_series: Original time series (dict or DataFrame)
    :param target_resolution: Resolution of the new time series in seconds
    :param return_type: Desired type of returned time series ('df': DataFrame, 'dict': dict). If not specified, return
    type equals input type.

    :param conversion: 'up' for upsampling to higher resolution,'down' for downsampling to lower resolution, or 'auto'.
    Default: 'auto': the resolution of the original series is checked based on the difference between the
    first and second timestamp. Comparison to the target resolution then determines the direction and the
    corresponding resampling method.

    :param downsample_method: Aggregation method, e.g. 'mean' (default), "sum", "prod","min", "max", "first", "last", "mean", "median"
    :param upsample_method: Filling method, e.g. 'interpolate' (default), 'pad', 'bfill'
    :param resample_kwargs: Additional settings for resampling, e.g. (default) closed='left', label='left'. See
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html?highlight=resample#pandas.DataFrame.resample
    for and options.
    :param ignore_nan: If NaN values in the resampled data are acceptable. Otherwise an AssertionError is raised in that case.

    :return: resampled time series
    """
    if isinstance(time_series, dict):
        return_type = return_type if return_type else 'dict'
        series_elem_type = type(list(time_series.values())[0])
        if not isinstance(series_elem_type, (dict, list)):
            time_series_df = pandas.Series(data=time_series)
        else:
            time_series_df = pandas.DataFrame.from_dict(data=time_series, orient='index')
    else:
        return_type = return_type if return_type else 'df'
        time_series_df = time_series

    # Ensure index is a DateTimeIndex
    time_series_df.index = pandas.to_datetime(time_series_df.index)
    # resolution = min(pandas.Series(time_series_df.index).diff()[1:]).total_seconds()

    if len(time_series_df.index) == 1:
        resolution = time_series_df.index[0].second
        conversion = 'down'
        logger.debug(
            f'Index of time series with single datapoint and timestamp {time_series_df.index[0]} will be set '
            f'to {target_resolution}s using conversion "{conversion}".')
    else:
        # Get resolution of this series.
        resolution = pandas.Series(time_series_df.index).diff().min(skipna=True).total_seconds()

        if conversion == 'auto':
            # If resolution is nan, conversion is 'up', because np.nan <= target_resolution is always False
            conversion = 'down' if resolution <= target_resolution else 'up'
        else:
            assert conversion == 'up' or conversion == 'down', f'Unknown conversion "{conversion}".'

    # Use default pandas options if not defined
    resample_kwargs = {} if resample_kwargs is None else resample_kwargs

    method = downsample_method if conversion == 'down' else upsample_method

    logger.debug(
        f'Data will be {conversion}sampled from {resolution}s '
        f'to {target_resolution}s, using method "{method}".')

    resampler = time_series_df.resample(rule=datetime.timedelta(seconds=target_resolution), **resample_kwargs)

    try:
        # Resample with given method, e.g. 'mean', 'max', 'interpolate', ...
        resampled_time_series = resampler.aggregate(method)

    except AttributeError as e:
        logger.error(
                f'Resampling method {method} not supported. Supported methods include: '
                f'for downsampling: ["sum", "prod","min", "max", "first", "last", "mean", "median"],'
                f'for upsampling: ["pad/ffill", "bfill", "interpolate", "nearest"]. '
                f'See https://pandas.pydata.org/docs/reference/resampling.html for all options.'
        )
        raise e

    if not ignore_nan:
        # Check for NaN values: happens for example when upsampling with 'mean'
        if isinstance(resampled_time_series, pandas.DataFrame):
            assert sum(resampled_time_series.isnull().any()) == 0, f"Resampled time series contains NaN. Review original ({resolution}s) and target resolution ({target_resolution}s) and the chosen method {method}."
        else:
            assert sum(resampled_time_series.isnull()) == 0, f"Resampled time series contains NaN. Review original ({resolution}s) and target resolution ({target_resolution}s) and the chosen method {method}."

    if return_type == 'df':
        return resampled_time_series
    elif return_type == 'dict':
        resampled_time_series.index = resampled_time_series.index.map(lambda t: t.to_pydatetime().isoformat())
        return resampled_time_series.to_dict()
