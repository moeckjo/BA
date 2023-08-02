"""
Base class for specific forecasting classes
"""
import pytz
import datetime
import os
import typing

import pandas as pd

import db.db_helper as db
from core.forecasting import logger, path


class Forecaster:

    def __init__(self, window_start: datetime.datetime, window_size: datetime.timedelta,
                 resolution: datetime.timedelta):
        """
        :param  window_start: Start of forecast horizon
        :param window_size: Length of forecast horizon
        """
        self.window_size = window_size
        self.forecast_window_start = window_start.astimezone(pytz.timezone(os.getenv('LOCAL_TIMEZONE')))
        self.forecast_window_end = (self.forecast_window_start + window_size).astimezone(
            pytz.timezone(os.getenv('LOCAL_TIMEZONE')))
        self.resolution = resolution
        self.source = None
        self.forecast_function: typing.Callable = None
        self.fallback_forecaster: Forecaster = None
        self.fallback_forecast_function_used: str = None

    def make_forecast(self, *args, **kwargs) -> typing.Dict[str, typing.Dict[str, float]]:
        """
        Calls forecast method of class and returns forecast.
        Temporal parameters such as forecast window start & end are provided to the object at init.
        :return: Forecast as dict with format (for PV)
                {'pv': {'period1': 'power1', 'period2': 'power2', ..., 'periodn': 'powern'}},
                where the period/timestamp keys are ISO-format strings and the values float.
        """
        now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
        logger.info(
            f'Make {self.forecast_function.__name__} of {self.source} for forecast window from {self.forecast_window_start} to {self.forecast_window_end}')
        forecast = self.forecast_function(*args, **kwargs)
        meta_data = {'updated_at': now.isoformat(timespec='seconds'),
                     'resolution': self.resolution.total_seconds(),
                     'function': f"{self.__class__.__name__}.{self.forecast_function.__name__}",
                     'fallback_forecast_function_used': self.fallback_forecast_function_used
                     }

        self.save_forecast(forecast=forecast,
                           meta_data=meta_data
                           )
        logger.info(f'Successfully executed and saved {meta_data["function"]} for '
                    f'{self.source} with meta data {meta_data}.')
        named_forecast = {self.source: forecast}
        return named_forecast

    def save_forecast(self, forecast: typing.Dict[str, float], meta_data=None):
        meta_data = meta_data if meta_data else {}
        db.save_data_to_db(
            db=os.getenv('MONGO_FORECAST_DB_NAME'),
            data_source=self.source,
            time_series_data=forecast,
            meta_data=meta_data,
            group_by_date=True,
            persist_old_data=True
        )

    def filter_for_window(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.window_size == datetime.timedelta(hours=24):
            data_forecast_window = data.copy()
        else:
            # Filter for forecast window hours
            if self.forecast_window_end.hour < self.forecast_window_start.hour:  # Window spans midnight
                data_forecast_window = data[
                    (data.index.hour >= self.forecast_window_start.hour) | (
                                data.index.hour < self.forecast_window_end.hour)
                    ]

            else:  # Start and end on same day
                data_forecast_window = data[
                    (data.index.hour >= self.forecast_window_start.hour) & (
                                data.index.hour < self.forecast_window_end.hour)
                    ]
        return data_forecast_window
