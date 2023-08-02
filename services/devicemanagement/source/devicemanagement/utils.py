import datetime
import typing
import pandas
import os
import logging

from devicemanagement import logger
from db import db_helper as db

# TODO: move these functions to device manager

def get_latest_measurement(source: str, fields: typing.Union[str, list], with_ts=False) -> typing.Union[
    typing.Tuple[datetime.datetime, dict], dict]:
    """
    Get the latest measurement for a given source. The past 10 minutes are considered.
    :param source: Time series name
    :param fields: Parameters of time series (single or list)
    :param with_ts: If True, tuple with (timestamp, value) is returned instead of value only
    :return: Value or (timestamp, value)
    """
    # Get last value from influxdb
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    data: pandas.DataFrame = db.get_measurement(
        source=source,
        fields=fields,
        start_time=now - datetime.timedelta(minutes=10),
        end_time=now + datetime.timedelta(seconds=int(os.getenv('DEVICE_RECORDING_FREQ'))),
        limit=1,
        order='DESC'
    )
    try:
        values = data.iloc[0, :].to_dict()
        if isinstance(fields, str):  # Single field -> single value
            values = values[fields]

        if with_ts:
            timestamp = data.index[0]
            return timestamp, values
        else:
            return values
    except (AttributeError, IndexError):
        # No match for query
        logger.debug(f'No measurement returned for {source}, field(s)={fields}: {data} (type({type(data)})')
        return None


def get_latest_data(db_name: str, source: str, at_time: datetime.datetime, grouped_by_date: bool) -> float:
    # TODO: reveiew if resolution of quote period makes sense
    data_dict: typing.Dict[datetime.datetime, float] = db.get_time_series_data_from_db(
        db=db_name, collection=source,
        start_time=at_time - datetime.timedelta(hours=float(os.getenv('QUOTA_TEMP_RESOLUTION'))),
        end_time=at_time + datetime.timedelta(hours=float(os.getenv('QUOTA_TEMP_RESOLUTION'))),
        grouped_by_date=grouped_by_date
    )

    if len(data_dict) > 1:
        # Make dataframe from dict with timestamps as index (DatetimeIndex)
        data_df = pandas.DataFrame.from_dict(data_dict, orient='index', columns=['value'])

        resolution = pandas.Series(data_df.index).diff()[1]  # datetime.timedelta
        # Filter for time period in which the provided timestamp falls and return the corresponding value
        period_value = data_df[
            (data_df.index.time > (at_time - resolution).time()) & (data_df.index.time <= at_time.time())
            ].iat[0, 0]
    else:
        # Only a single period
        period_value = list(data_dict.values())[0]
    return period_value


def get_latest_forecast(source: str, at_time: datetime.datetime) -> float:
    db_name = os.getenv('MONGO_FORECAST_DB_NAME'),
    return get_latest_data(db_name, source, at_time, grouped_by_date=True)


def get_latest_schedule(source: str, at_time: datetime.datetime) -> float:
    db_name = os.getenv('MONGO_SCHEDULE_DB_NAME'),
    return get_latest_data(db_name, source, at_time, grouped_by_date=True)


def get_latest_user_input_with_future_departure(now: datetime.datetime) -> typing.Union[None, typing.Dict[str, typing.Union[str, float]]]:
    # Get relevant EV user input entries from database, i.e. with a departure in the future
    # A single entry looks like this:
    #   {
    #   '_id': ObjectId('61dc5468b136d5b118babec4'), 'timestamp': '2022-01-10T15:42:09+00:00',
    #   'soc': 0.15, 'soc_target': 1.0, 'capacity': 35.8, 'scheduled_departure': '2022-01-11T07:00:00+00:00'
    #   }
    departure_filter = {'scheduled_departure': {'$gte': now.isoformat(timespec='seconds')}}
    results = db.get_data_from_db(
        db=os.getenv('MONGO_USERDATA_DB_NAME'),
        collection=os.getenv('MONGO_EV_CHARGING_INPUT_COLL_NAME'),
        doc_filter=departure_filter,
    )
    # Query returns Cursor object that is emptied when iterating over it -> copy to new list object
    entries = list(results)
    if entries:
        if len(entries) > 1:
            relevant_entries_by_input_timestamp: typing.Dict[str, dict] = {entry['timestamp']: entry for entry in
                                                                           entries}
            latest_entry: dict = relevant_entries_by_input_timestamp[max(relevant_entries_by_input_timestamp)]
        else:
            latest_entry = entries[0]
        latest_entry.pop("_id")
        return latest_entry
    return None
