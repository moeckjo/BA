import datetime
from influxdb import DataFrameClient
import pandas as pd
import os
import typing

from db import logger


def make_select_clause(fields):
    if fields == '*':
        # Return all fields
        return fields

    # Multiple fields
    if isinstance(fields, list):
        select_string = ''
        for field in fields:
            select_string += f'"{field}",'
        # Remove trailing comma of returned string
        return select_string[:-1]
    # Single field
    else:
        return f'"{fields}"'


def make_where_clause(start_time, end_time, closed):
    start_time_iso = start_time.replace(tzinfo=None).isoformat(timespec="seconds")
    if end_time is None:
        return f'time {">=" if closed in ("left", "both") else ">"} \'{start_time_iso}Z\''
    else:
        end_time_iso = end_time.replace(tzinfo=None).isoformat(timespec="seconds")
        return f'time {">=" if closed in ("left", "both") else ">"} \'{start_time_iso}Z\' AND time {"<=" if closed in ("right", "both") else "<"} \'{end_time_iso}Z\''


def timestamp_unit(timestamp):
    """
    Infer type and unit of the timestamp by trial and error.
    Types can be ISO-format string or UNIX timestamp. The latter can be given in different units from seconds to nanoseconds.
    :param timestamp: Timestamp in undetermined format.
    :return: None, if it's ISO-format, or the unit of the UNIX timestamp (s,ms,us,ns) as expected by the
    arg "unit" of pd.to_datetime()
    """
    if isinstance(timestamp, str):
        try:
            datetime.datetime.fromisoformat(timestamp)
            return None
        except ValueError as ve:
            if "Invalid isoformat string" in str(ve):
                try:
                    # Maybe it's a UNIX timestamp in a string
                    timestamp = float(timestamp)
                except ValueError:
                    # Doesn't seem like it -> raise the original ValueError for an invalid isoformat str
                    raise ve
            else:
                # Raise this unexpected error
                raise ve

    # Seems like a UNIX timestamp -> determine unit
    unit_map = {'s': 1, 'ms': 10 ** 3, 'us': 10 ** 6, 'ns': 10 ** 9}
    for unit, scale in unit_map.items():
        try:
            dt = datetime.datetime.fromtimestamp(timestamp / scale)
            logger.debug(f'Resulting datetime from timestamp {timestamp} [{unit}]: {dt}')
            return unit
        except ValueError as ve:
            # If trying to parse a ms-timestamp as s-timestamp an error like "year 53826 is out of range" is thrown
            if "is out of range" in str(ve):
                # Try next smaller unit
                continue
            else:
                raise ve
        except OSError as oe:
            if "[Errno 22] Invalid argument" in str(oe):
                # Thrown if timestamp in nanoseconds -> continue iterating through units
                continue
            else:
                raise oe


class InfluxConnector(DataFrameClient):

    def __init__(self):
        super(DataFrameClient, self).__init__(
            host=os.getenv('INFLUX_HOSTNAME'),
            port=os.getenv('INFLUX_PORT'),
            username=os.getenv('ADMIN_NAME'),
            password=os.getenv('ADMIN_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        # logger.debug(self.get_list_measurements())
        # for m in self.get_list_measurements():
        #     if m['name'] !=  'vZP52.2':
        #         self.drop_measurement(m['name'])
        # logger.debug(self.get_list_measurements())

    def query_data(self, source: str, fields: typing.Union[str, list], start_time: datetime.datetime,
                   end_time: typing.Optional[datetime.datetime] = None, limit: typing.Optional[int] = None,
                   closed: str = 'left',
                   order: typing.Optional[str] = 'ASC') -> pd.DataFrame:
        # start_time_iso = start_time.replace(tzinfo=None).isoformat(timespec="seconds")
        # end_time_iso = end_time.replace(tzinfo=None).isoformat(timespec="seconds")
        select_clause = make_select_clause(fields)
        from_clause = f'"{source}"'
        where_clause = make_where_clause(start_time=start_time, end_time=end_time, closed=closed)
        limit_clause = '' if limit is None else f'LIMIT {limit}'
        q = f'SELECT {select_clause} FROM {from_clause} WHERE {where_clause} ORDER BY time {order.upper()} {limit_clause};'
        logger.debug(f'Query: {q}')
        query_result = self.query(q)  # collections.defaultdict
        try:
            data_df = dict(query_result)[source]  # Dataframe
            return data_df
        except KeyError:
            logger.debug(f'Empty result: No data matching this query ({q}).')
            return pd.DataFrame()

    def write_datapoints(self, source: str, data: typing.Union[dict, pd.DataFrame, pd.Series]):
        """
        :param data: Time series, either as dict with keys=timestamps, values=dicts or Dataframe or Series
        :param source: measurement/device name
        :return: True if write operation was successful, otherwise False
        """

        # Convert input data to a dataframe if necessary
        if isinstance(data, dict):
            assert isinstance(list(data.values())[0],
                              dict), f'Values of the time series dict of measurement {source} must be of type dict!'
            data = pd.DataFrame.from_dict(data=data, orient='index')
        elif isinstance(data, pd.Series):
            assert data.name is not None, f'Series of measurement {source} needs a name!'
            data = pd.DataFrame(data=data)
        assert isinstance(data, pd.DataFrame)

        # Determine unit of the timestamps (=None, if it's a ISO-format string)
        # Assumption: timestamps in the dataframe index all have the same format. If not, there will be an
        # error when trying to convert the index to a DatetimeIndex
        ts_unit: typing.Union[None, str] = timestamp_unit(data.index[0])
        data.index = pd.to_datetime(data.index, unit=ts_unit, utc=True)
        successful = self.write_points(dataframe=data, measurement=source)
        return successful
