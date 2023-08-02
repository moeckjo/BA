"""
- Establishes connection to the market platform
- Handles order placement
- Handles market result notification
- ...
"""
import datetime
import json
import os

import typing
from flexqgrid_python_sdk import FQG, OrderSet
from pathlib import Path
import inspect


class BlockchainConnector(FQG):
    trading_period_length = datetime.timedelta(hours=float(os.getenv('QUOTA_TEMP_RESOLUTION')))
    trading_block_length = datetime.timedelta(hours=float(os.getenv('QUOTA_WINDOW_SIZE')))
    number_of_trading_periods = int(trading_block_length / trading_period_length)

    def __init__(self, specifications):
        super().__init__(
            dict(save_path=specifications['local_path']),
        )
        self.plant_id = specifications["plant_id"]

    @classmethod
    def map_slot_numbers_to_time_periods(cls, block_start: datetime.datetime) -> typing.Dict[datetime.datetime, int]:
        """
        Maps slot numbers to period start timestamps based on the total duration and the temporal
        resolution (period length) of a given temporal horizon.
        :param block_start: Timestamp of first period (inclusive)
        :return: Dictionary with <period_start_timestamp, slot_no>-pairs
        """
        mapping = {block_start + cls.trading_period_length * i: i for i in range(cls.number_of_trading_periods)}
        return mapping

    @classmethod
    def map_time_periods_to_slot_numbers(cls, block_start: datetime.datetime) -> typing.Dict[int, datetime.datetime]:
        """
        Maps period start timestamps to slot numbers based on the total duration and the temporal
        resolution (period length) of a given temporal horizon.
        :param block_start: Timestamp of first period (inclusive)
        :return: Dictionary with <slot_no, period_start_timestamp>-pairs
        """
        map = cls.map_slot_numbers_to_time_periods(block_start)
        reverse = {slot: time for time, slot in map.items()}
        return reverse
