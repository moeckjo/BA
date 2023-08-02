import datetime
import traceback
import typing
from collections import namedtuple

import pandas

import os

from core import logger
from db import db_helper as db


def get_latest_device_measurement(source: str, fields: typing.Union[str, list], search_within_last_minutes: int = 10,
                                  with_ts=False) -> typing.Union[
    typing.Tuple[datetime.datetime, dict], dict]:
    """

    :param source: Time series name
    :param fields: Parameters of time series (single or list)
    :param with_ts: If True, tuple with (timestamp, value) is returned instead of value(s) only
    :return: Single value, values as dict or (timestamp, values as dict)-Tuple
    """

    def get_data(field):
        data: pandas.DataFrame = db.get_measurement(
            source=source,
            fields=field,
            start_time=now - datetime.timedelta(minutes=search_within_last_minutes),
            end_time=now + datetime.timedelta(seconds=5),
            limit=1,
            order='DESC'
        )
        return data

    now = datetime.datetime.now(tz=datetime.timezone.utc)

    if isinstance(fields, list):
        values = dict()
        timestamps = dict()
        for field in fields:
            # Get last value from influxdb
            data = get_data(field)
            try:
                values[field] = data.iloc[0, 0]
                timestamps[field] = data.index[0]
            except (AttributeError, IndexError):
                # No match for query
                logger.debug(f'Measurement returned for {source}: {data}')
                values[field] = None
        try:
            timestamp = max(timestamps.values())
        except ValueError:
            # No data for any field -> timestamp dict is empty
            timestamp = None
    else:
        # fields is single string
        data = get_data(fields)
        try:
            values = data.iloc[0, 0]
            timestamp = data.index[0]
        except IndexError:
            # No match for query
            logger.debug(f'Measurement returned for {source}: {data}')
            values = None
            timestamp = None

    if with_ts:
        return timestamp, values
    else:
        return values


def get_devices_of_subcategory(subcategory: str) -> typing.List[dict]:
    result = db.get_data_from_db(
        db=os.getenv('MONGO_DEVICE_DB_NAME'),
        collection=os.getenv('MONGO_DEVICE_COLL_NAME'),
        doc_filter={'subcategory': subcategory},
        return_fields={'_id': False, 'model': False}
    )
    # Query returns Cursor objects that is emptied when iterating over it -> copy to new list object
    specifications_of_devices: typing.List[dict] = list(result)
    return specifications_of_devices


def are_devices_of_subcategory_installed(subcategory: str) -> bool:
    devices = get_devices_of_subcategory(subcategory)
    if len(devices) >= 1:
        return True
    return False


def get_devices_and_gcp_specifications() -> typing.Tuple[typing.List[typing.NamedTuple], dict]:
    result = db.get_data_from_db(
        db=os.getenv('MONGO_DEVICE_DB_NAME'),
        collection=os.getenv('MONGO_DEVICE_COLL_NAME'),
        doc_filter={},
        return_fields={'_id': False, 'model': False}
    )
    # Query returns Cursor objects that is emptied when iterating over it -> copy to new list object
    specifications_of_devices: typing.List[dict] = list(result)
    # List all possible device parameters (keys, not values) in a set
    all_device_parameters = set()
    gcp_specifications_idx = None
    for i, specs in enumerate(specifications_of_devices):
        if specs['category'] != os.getenv('GRID_CONNECTION_POINT_KEY'):
            # Update the set of device parameters with this device's parameter keys
            all_device_parameters.update(specs.keys())
        else:
            gcp_specifications_idx = i
    gcp_specifications: dict = specifications_of_devices.pop(gcp_specifications_idx)
    Device = namedtuple('Device', field_names=all_device_parameters, defaults=[None] * len(all_device_parameters))
    devices = [Device(**specs) for specs in specifications_of_devices]
    return devices, gcp_specifications


def get_state_fields_for_device(device_subcategory: str) -> list:
    # TODO: this should be configured in config not hardcoded
    states = {
        os.getenv('BESS_KEY'): ['soc'],
        os.getenv('EVSE_KEY'): ['soc'],
        os.getenv('HEAT_STORAGE_KEY'): ["temp_water", "temp_ambient"],
        os.getenv('HEAT_PUMP_KEY'): ["state"],
    }
    assert device_subcategory in states, f'State fields for {device_subcategory} not defined.'
    return states[device_subcategory]


def get_next_schedule_window_start(now: datetime.datetime,
                                   schedule_resolution: datetime.timedelta,
                                   buffer_sec: int = None) -> datetime.datetime:
    if not buffer_sec:
        buffer_sec = int(os.getenv('SCHEDULE_COMPUTATION_BUFFER_SEC'))

    schedule_resolution_minutes = int(schedule_resolution.total_seconds() / 60)
    start_minutes = list(range(schedule_resolution_minutes, 61, schedule_resolution_minutes))
    if max(start_minutes) < now.minute + buffer_sec / 60:
        # Happens towards the end of an hour, e.g. at minute=59 and buffer=2
        # Calculation of future_start_diffs would return an empty list
        next_start = now.replace(hour=now.hour + 1, minute=min(start_minutes), second=0, microsecond=0)
    else:
        future_start_diffs = [m - now.minute for m in start_minutes if (m >= now.minute + buffer_sec / 60)]
        next_start = (now + datetime.timedelta(minutes=min(future_start_diffs))).replace(second=0)

    return next_start


def save_to_schedule_db(source: str, data: dict, meta_data: dict = None):
    """
    Save raw optimization results "as is", but GCP and device schedules as time series.
    """
    schedule_db = os.getenv('MONGO_SCHEDULE_DB_NAME')

    if source == os.getenv('MONGO_SCHEDULE_OPTIMIZATION_RAW_SOLUTION_COLL_NAME'):
        db.save_dict_to_db(db=schedule_db, data_category=source, data=data)
        return

    # Get the previous schedule's meta data to decide if it should be persisted
    previous_schedule_meta_data = latest_schedule_meta_data(
        source=source,
        query_start_time=datetime.datetime.fromisoformat(min(data)),
        query_end_time=datetime.datetime.fromisoformat(max(data)) + datetime.timedelta(minutes=1)
    )

    try:
        events = set(previous_schedule_meta_data['event'].values())
    except AttributeError:
        # Single string -> create set with single item for easier handling of both cases below
        events = {previous_schedule_meta_data['event']}
    except KeyError:
        # Empty dict, because no schedule was found
        events = []

    if (len(events) == 1) and (
            os.getenv('DEVIATION_MESSAGE_SUBTOPIC') in events or
            f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_setpoint" in events
    ):
        # Don't persist schedules triggered by a schedule deviation or a GCP setpoint. This happens every other minute
        # and may cause the mongoDB document to blow up and exceed the max. document size. If this happens, no other
        # schedule can be stored for this day.
        persist_previous_schedule = False
    else:
        # Persist schedules triggered by infrequent events
        persist_previous_schedule = True
    logger.debug(f"Persist previous schedule triggered by event(s) {events}?: {persist_previous_schedule}")

    if meta_data['event'] == f"{os.getenv('GRID_CONNECTION_POINT_KEY')}_setpoint":
        db.save_data_to_db(
            db=schedule_db,
            data_source=source,
            time_series_data=data,
            meta_data=meta_data,
            persist_old_data=persist_previous_schedule
        )
    else:
        # Only group by date "normal" schedules over multiple hours (i.e. contrary to
        # 1-min schedule in case of GCP setpoint)
        db.save_data_to_db(
            db=schedule_db,
            data_source=source,
            time_series_data=data,
            group_by_date=True,
            meta_data=meta_data,
            persist_old_data=persist_previous_schedule
        )


def latest_schedule_meta_data(source: str, query_start_time: datetime.datetime, query_end_time: datetime.datetime, ) \
        -> typing.Dict[str, typing.Any]:
    try:
        latest_schedule, meta_data = db.get_time_series_data_from_db(
            db=os.getenv('MONGO_SCHEDULE_DB_NAME'), collection=source,
            start_time=query_start_time, end_time=query_end_time,
            grouped_by_date=True,
            return_timestamps_in_isoformat=False,
            extra_info=['event']
        )
        logger.debug(
            f"Last period of latest schedule of {source}: {max(latest_schedule.keys())}: {latest_schedule[max(latest_schedule.keys())]}")

        return {'start': min(latest_schedule.keys()), 'end': max(latest_schedule.keys()), **meta_data}
    except TypeError:
        # No schedule found
        return {}


def forecast_update_necessary(window_start: datetime.datetime, window_end: datetime.datetime, sources: set,
                              scheduling_trigger: dict, update_every_seconds: int = 3600) -> typing.List[str]:
    """
    For each source, decide if the forecast needs to be updated for the upcoming scheduling run.
    Necessity is given if the latest forecast was made more than update_every_seconds seconds ago.
    Furthermore, for some triggering events or sources, an update of the forecast of certain sources might
    be necessary independent of the time of the last forecast.
    :param window_start: Start of the required forecast
    :param window_end: End of the required forecast
    :param sources: Sources (e.g. devices or load) for which a forecast is needed
    :param scheduling_trigger: Dict with "event" and "source"
    :param update_every_seconds:
    :return: List of source/device subcategories whose forecast should be updated prio to optimization.
    """

    update_needed_for_sources = []

    # If the scheduling process is triggered by an event related to the EV, update its forecast.
    if (os.getenv('EVSE_KEY') in sources) and (scheduling_trigger["source"] == os.getenv('EVSE_KEY')):
        update_needed_for_sources.append(os.getenv('EVSE_KEY'))
        sources.remove(os.getenv('EVSE_KEY'))

    for source in sources:
        # Get latest forecast with its time of update for the selected source
        result: typing.Tuple[dict, dict] = db.get_time_series_data_from_db(
            db=os.getenv('MONGO_FORECAST_DB_NAME'), collection=source,
            start_time=window_start, end_time=window_end,
            grouped_by_date=True,
            return_timestamps_in_isoformat=False,
            extra_info=['updated_at']
        )
        updated_info: typing.Union[str, typing.Dict[datetime.date, str]] = result[1]['updated_at']
        if isinstance(updated_info, dict):
            # Dict with different values per date, e.g.
            # {datetime.date(2022, 5, 23): '2022-05-23T09:09:00+00:00',
            # datetime.date(2022, 5, 24): '2022-05-23T08:44:06+00:00'}
            last_update = datetime.datetime.fromisoformat(max(updated_info.values()))
        else:
            last_update = datetime.datetime.fromisoformat(updated_info)
        logger.debug(f"Last update of {source} forecast: {last_update} (raw update info: {result[1]})")

        now = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0, second=0)
        if last_update <= now - datetime.timedelta(seconds=update_every_seconds):
            # Last update was made more than update_every_seconds seconds ago -> make new forecast
            update_needed_for_sources.append(source)

    return update_needed_for_sources
