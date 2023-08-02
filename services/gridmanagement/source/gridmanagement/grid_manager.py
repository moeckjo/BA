import datetime
import json
import logging
import os
import typing
from functools import reduce

import numpy as np
import pandas as pd
import pika
import pytz

from db import db_helper as db
from gridmanagement import logger

from gridconnection_model import GridConnectionPoint
import utils

logging.getLogger("pika").setLevel(logging.WARNING)

'''
Grid point administration
'''

db_object_category = os.getenv('MONGO_DEVICE_COLL_NAME')


def load_grid_config() -> dict:
    with open(os.path.join(os.getenv('BEM_ROOT_DIR'), 'config', 'gridconnection_config.json')) as grid_config_file:
        grid_config = json.load(grid_config_file)
    return grid_config


def save_gridconnection_config():
    # Save the current grid connection configuration in the settings database for tracking purposes
    # (for later evaluations)
    gridconnection_config = load_grid_config()
    gridconnection_config['timestamp'] = datetime.datetime.now(tz=pytz.utc).isoformat(timespec='seconds')
    db.save_dict_to_db(
        db=os.getenv('MONGO_SETTINGS_DB_NAME'),
        data_category=os.getenv('MONGO_SETTINGS_GRID_CONNECTION_CONFIG_COLL_NAME'),
        data=gridconnection_config
    )
    logger.debug(f"Stored the following grid connection settings: {gridconnection_config}")


def create_grid_connection() -> str:
    """
    Creates grid connection models from specifications file and stores it in the DB
    :return: ID of model in DB
    """

    logger.info('Set up grid connection.')
    grid_connection = GridConnectionPoint(specifications=load_grid_config()['specifications'])
    parameters = {
        attr[1:]: value for attr, value in grid_connection.__dict__.items() if (str(attr).startswith('_'))
    }
    # Store model and parameters of GCP in database
    gcp_db_id = db.save_objects_to_db(
        db=os.getenv('MONGO_DEVICE_DB_NAME'),
        object_category=db_object_category,
        objects={'model': grid_connection},
        **parameters
    )

    return gcp_db_id, grid_connection.dso_uuid


def get_model_from_db(gcp_db_id: str = None, dso_uuid: str = None, return_model_only=True) -> typing.Union[
    GridConnectionPoint, dict]:
    """
    Get the respective instance of the grid connection model from the database, optionally with further info.
    :param gcp_db_id: ID of entry in DB
    :param dso_uuid: UUID of the grid connection point defined by the local DSO
    :return: Either the instance or dict with instance (key 'model') and further info
    """
    assert gcp_db_id or dso_uuid, 'DB ID or DSO UUID must be provided.'

    entry = db.get_objects_from_db(
        db=os.getenv('MONGO_DEVICE_DB_NAME'),
        object_category=db_object_category,
        doc_id=gcp_db_id,
        doc_filter={'dso_uuid': dso_uuid} if dso_uuid else None
    )
    if return_model_only:
        return entry['model']
    return entry


def update_model_in_db(gcp_db_id: str, model: GridConnectionPoint):
    """
    Update the respective instance of the grid connection model in the database.
    :param gcp_db_id: ID of model in DB
    :param model: Instance of the grid connection model
    """
    db.update_objects_in_db(
        db=os.getenv('MONGO_DEVICE_DB_NAME'),
        object_category=db_object_category,
        doc_id=gcp_db_id,
        objects={'model': model}
    )


'''
Data querying and calculations
'''


def get_latest_gcp_schedule(start_time: datetime.datetime = None, end_time: datetime.datetime = None,
                            at_time: datetime.datetime = None, return_period_start: bool = False,
                            return_timestamps_in_isoformat: bool = True,
                            with_meta_data: typing.List[str] = None, with_doc_id: bool = False) -> \
        typing.Union[float, typing.Dict[str, float], tuple, None]:
    """
    Returns the latest update of the schedule at the grid connection point from the database.
    Either all data between the given start and end time or only the value for a specific timestamp/period (at_time).
    :param start_time: Returns schedule data after this timestamp (inclusive)
    :param end_time: Returns schedule data until this timestamp (exclusive)
    :param at_time: Returns single scheduled value at this timestamp
    :param return_period_start: Only effective with at_time=True; Return the start of the period that includes at_time.
    :param return_timestamps_in_isoformat: If true, timestamps are returned as string in
    ISO-format (e.g. 2021-01-01T01:00:23+01:00), else as datetime object
    :param with_meta_data: List of additional fields that should be returned
    :param with_doc_id: If the ID(s) of the matching document(s) should be returned additionally.
    :return: Schedule for time filter (dict) or single value if at_time=True. If there's no schedule for
    the time filter, the return value is None.
    """
    assert (start_time and end_time) or at_time, \
        'Neither start and end time (i.e. both) nor specific timestamp (at_time) provided.'

    if not start_time:
        start_time = at_time - datetime.timedelta(seconds=float(os.getenv('SCHEDULER_TEMP_RESOLUTION_SEC')))
        end_time = at_time + datetime.timedelta(seconds=float(os.getenv('SCHEDULER_TEMP_RESOLUTION_SEC')))

    result = db.get_time_series_data_from_db(
        db=os.getenv('MONGO_SCHEDULE_DB_NAME'), collection=os.getenv('GRID_CONNECTION_POINT_KEY'),
        start_time=start_time, end_time=end_time, grouped_by_date=True,
        return_timestamps_in_isoformat=return_timestamps_in_isoformat, extra_info=with_meta_data,
        return_doc_id=with_doc_id)

    if result is None:
        # No schedule found
        if at_time:
            logger.debug(f'No GCP schedule for time {at_time.isoformat()} available.')
        else:
            logger.debug(f'No GCP schedule for periods between {start_time} and {end_time} available.')
        return None

    if with_meta_data or with_doc_id:
        data_dict, meta_data = result
    else:
        data_dict: typing.Dict[datetime.datetime, float] = result
        meta_data = {}

    if at_time:
        # Convert timestamp string keys to datetime
        data_dict = {datetime.datetime.fromisoformat(t): v for t, v in data_dict.items()}

        period_starts: typing.List[datetime.datetime] = list(data_dict.keys())
        period_value = None
        period_start = None
        for i, start in enumerate(period_starts):
            if start <= at_time:
                period_value = data_dict[start]
                period_start = start.isoformat() if return_timestamps_in_isoformat else start

        if period_value is None:
            return None

        if with_meta_data or with_doc_id:
            relevant_meta_data = meta_data.copy()
            for field, content in meta_data.items():
                if isinstance(content, dict):
                    # The following is only relevant if the start and end time are on different days.
                    # Their might be meta data that differs for different days and was hence added as
                    # dict with key=date. Check if that's the case or if it's some other dict.
                    if return_timestamps_in_isoformat:
                        try:
                            # Try converting str to datetime.date objects – fails if keys are not str of dates
                            content = {datetime.date.fromisoformat(date): value for date, value in content.items()}
                            keys_are_dates = True
                        except ValueError:
                            keys_are_dates = False
                    else:
                        # If the keys are dates, they are datetime.date objects
                        keys_are_dates = isinstance(list(content)[0], datetime.date)

                    if keys_are_dates:
                        # Yes, the keys are dates -> get the value of the date corresponding to the
                        # requested period (at_time)
                        relevant_value = content.get(at_time.date())
                        relevant_meta_data[field] = relevant_value
            logger.debug(
                f'Meta data returned with latest GCP schedule request: all: {meta_data}; relevant: {relevant_meta_data}')

            if return_period_start:
                return period_value, relevant_meta_data, period_start
            else:
                return period_value, relevant_meta_data

        if return_period_start:
            return period_value, period_start
        else:
            return period_value

    return data_dict


def get_latest_gcp_setpoint() -> int:
    # Get last value from influxdb
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    data = db.get_measurement(
        source=os.getenv('GRID_CONNECTION_POINT_KEY'),
        fields='setpoint',
        start_time=now - datetime.timedelta(hours=1),
        limit=1,
        order='DESC'
    )
    try:
        value = data.iloc[0, 0]
    except IndexError:
        # No match, empty DataFrame was returned
        return None
    return int(value)


def get_list_of_devices_with_categorization(include_load: bool) -> typing.List[dict]:
    # Get list of devices from the database
    results = db.get_data_from_db(
        db=os.getenv('MONGO_DEVICE_DB_NAME'),
        collection=os.getenv('MONGO_DEVICE_COLL_NAME'),
        doc_filter={'key': {'$ne': os.getenv('GRID_CONNECTION_POINT_KEY')}},
        return_fields={'_id': False, **{f: True for f in ['key', 'subcategory', 'category']}}
    )
    docs = list(results)
    if include_load:
        docs.append(
            dict(key=os.getenv('LOAD_EL_KEY'), subcategory=os.getenv('LOAD_EL_KEY').split('_')[0], category='consumer'))
    return docs


def group_devices_by_subcategory(devices_with_category: typing.List[dict]) -> typing.Dict[str, typing.List[dict]]:
    devices_per_subcategories = {}

    # Sort per subcategory
    for device in devices_with_category:
        if not device['subcategory'] in devices_per_subcategories.keys():
            devices_per_subcategories[device['subcategory']] = [device]
        else:
            devices_per_subcategories[device['subcategory']].append(device)

    return devices_per_subcategories


def filter_device_measurements(data: pd.DataFrame, field_filter: typing.List[str], device: dict):
    filtered = data.filter(items=field_filter, axis='columns')  # requires exact match

    if device['subcategory'] == os.getenv('BESS_KEY') and set(filtered.columns) != field_filter:
        # Assuming that problems occur only with possibly diverse naming of charging and
        # discharging power, not SOC
        charge_power_label = "^c?(harge)?_?power_?c?(harge)?$"  # matches: c_power, charge_power, power_c, power_charge (or any without _)
        discharge_power_label = "^d?(ischarge)?_?power_?d?(ischarge)?$"  # matches: d_power, discharge_power, power_d, power_discharge (or any without _)
        charge_power = data.filter(regex=charge_power_label, axis=1)
        discharge_power = data.filter(regex=discharge_power_label, axis=1)
        assert (len(charge_power.columns) == 1) and (len(
            discharge_power.columns) == 1), f'No unique match of column filter for charging ({charge_power.columns}) or discharging power ({discharge_power.columns}).'
        filtered['active_power'] = charge_power[charge_power.columns[0]] - abs(
            discharge_power[discharge_power.columns[0]])

    assert set(
        filtered.columns) == field_filter, f'Required fields {field_filter} not available for {device["key"]} (available fields: {list(filtered.columns)}).'

    return filtered


def add_up_device_data(data_per_device: typing.List[pd.DataFrame]) -> pd.DataFrame:
    """
    Column-wise addition of values from (possibly) multiple devices. Column labels are used for matching.
    Note: this only makes sense for power, energy and so, but not for state values such as SOC. This function does
    not perform any corresponding check or whatsoever, thus, use at your own risk.

    :param data_per_device: List of dataframes, each dataframe containing the data of a single device
    :return: Single dataframe containing the sum of the values of all devices for each shared parameter (column)
    """
    if len(data_per_device) > 1:
        return reduce(lambda df1, df2: df1.add(df2), data_per_device)
    else:
        return data_per_device[0]


def get_device_measurements_aggregated_per_subcategory(
        window_start: datetime.datetime,
        window_end: datetime.datetime,
        resolution: int,
        include_load: bool = False,
        filter_fields: typing.Dict[str, typing.Set[str]] = None) -> typing.Dict[str, pd.DataFrame]:
    """
   Get a list of all connected devices in this building and query their measurements.
   Resample this data to the target resolution.
   Aggregate measurements from multiple devices of the same subcategory to obtain a single measurement per
   subcategory (e.g. multiple PV inverters are aggregated to result in a single PV generation measurement).

   :param resolution: Requested resolution of the data in seconds. Data will be resampled if necessary.
   :param window_start: Start of requested interval (included).
   :param window_end: End of requested interval (excluded).
   :param include_load: If True, inflexible load measurements are also queried and returned.
   :param filter_fields: Only return specified measurement fields (as Set), given a subcategory. Example:
   filter_fields = {'pv': {'active_power'}, 'evse': {'active_power'}, 'bess': {'active_power', 'soc'}}
   To apply the same fields filter to all subcategories, use key "all", e.g. filter_fields = {'all': {'active_power'}}.
   If filter_fields is undefined (None), all fields are queried.

   :return: Measurements of all devices and (if requested) inflexible load, aggregated by subcategory
   """
    devices_with_category: typing.List[dict] = get_list_of_devices_with_categorization(include_load=include_load)
    devices_per_subcategories: typing.Dict[str, typing.List[dict]] = group_devices_by_subcategory(devices_with_category)

    data_per_subcategory = {}
    logger.debug(f'Filter fields: {filter_fields}')
    for subcategory, devices in devices_per_subcategories.items():
        data_per_device = []

        fields = '*'
        if filter_fields:
            # Filter queried data for desired fields, e.g. only power and soc
            try:
                fields = list(filter_fields[subcategory])
            except KeyError:
                fields = list(filter_fields['all'])

        # Query measurements
        for device in devices:
            device_data: pd.DataFrame = db.get_measurement(
                source=device['key'],
                fields=fields,
                start_time=window_start,
                end_time=window_end,
                closed='left'
            )
            logger.debug(f'Raw data for {device["key"]} (queried fields: {fields}): {device_data}')

            if not device_data.empty:
                # Round timestamps to 1s – otherwise upsampling will fail if the precision is higher (-> ms)
                device_data.index = device_data.index.round('S')
                # Resample data to target resolution
                resampled_device_data = utils.resample_time_series(
                    time_series=device_data,
                    target_resolution=resolution,
                    conversion='auto',
                    downsample_method='mean',
                    upsample_method='interpolate',
                    resample_kwargs=dict(closed='left', label='left')
                )
                data_per_device.append(resampled_device_data)

        if len(data_per_device) >= 1:
            # Add up the values of all devices of this subcategory
            data_per_subcategory[subcategory] = add_up_device_data(data_per_device)
        else:
            data_per_subcategory[subcategory] = pd.DataFrame()

    return data_per_subcategory


def get_device_measurements_aggregated_per_subcategory_formatted(
        resolution: int, window_size: datetime.timedelta,
        ignore_nan_in_resampled_data: bool = False,
        window_end: datetime.datetime = None,
        include_load: bool = False,
        filter_fields: typing.Dict[str, typing.Set[str]] = None,
        return_readable_value_format: bool = False,
        iso_timestamps: bool = False,
) -> typing.Dict[str, pd.DataFrame]:
    """
    Get a list of all connected devices in this building and query their latest measurements. If necessary,
    aggregate measurements from multiple devices of the same subcategory to obtain a single measurement per
    subcategory (e.g. multiple PV inverters are aggregated to result in a single PV generation measurement)
    Further resample the measurement time series to the given target resolution and format the values.

    :param resolution: Requested resolution of the data in seconds. Data will be resampled if necessary.
    :param ignore_nan_in_resampled_data: If NaN values in the resampled data are acceptable.
    :param window_end: End of requested interval. If not defined, it is set to now.
    :param window_size: Requested interval of the recent past. Determines time range = now - window_size
    :param include_load: If True, inflexible load measurements are also queried and returned.
    :param filter_fields: Only return specified measurement fields (as Set), given a subcategory. Example:
    filter_fields = {'pv': {'active_power'}, 'evse': {'active_power'}, 'bess': {'active_power', 'soc'}}
    To apply the same fields filter to all subcategories, use key "all", e.g. filter_fields = {'all': {'active_power'}}
    :param return_readable_value_format: Due to resampling, values can have lots of decimal places. If true, Watt
    values are converted to int and SOC rounded to 2 decimal places.
    :param iso_timestamps: If timestamps should be returned as ISO-formatted strings
    :return: Measurements of all devices and general load, sorted by subcategory
    """
    # Get list of devices as dict with key, category and subcategory
    devices_with_category: typing.List[dict] = get_list_of_devices_with_categorization(include_load=include_load)
    devices_per_subcategories: typing.Dict[str, typing.List[dict]] = group_devices_by_subcategory(devices_with_category)

    window_end = window_end if window_end else datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
    data_per_subcategory = {}
    logger.debug(f'Filter fields: {filter_fields}')
    for subcategory, devices in devices_per_subcategories.items():
        data_per_device = []

        # Query measurements
        for device in devices:
            data: pd.DataFrame = db.get_measurement(
                source=device['key'],
                fields='*',
                start_time=window_end - window_size,
                end_time=window_end,
                closed='right'
            )
            logger.debug(f'Raw data for {device["key"]}: {data}')

            if not data.empty:
                if filter_fields:
                    # Filter queried data for desired fields, e.g. only power and soc
                    try:
                        filter_for = filter_fields[subcategory]
                    except KeyError:
                        filter_for = filter_fields['all']

                    filtered = filter_device_measurements(data=data, field_filter=filter_for, device=device)

                else:
                    filtered = data

                if len(data) > 1:
                    # Resample to target resolution
                    resampled_data = utils.resample_time_series(time_series=filtered,
                                                                target_resolution=resolution,
                                                                conversion='auto',
                                                                downsample_method=os.getenv(
                                                                    'OUTGOING_MEASUREMENTS_DOWNSAMPLING_METHOD'),
                                                                upsample_method=os.getenv(
                                                                    'OUTGOING_MEASUREMENTS_UPSAMPLING_METHOD'),
                                                                resample_kwargs=dict(closed='right', label='right'),
                                                                ignore_nan=ignore_nan_in_resampled_data
                                                                )
                else:
                    resampled_data = filtered

                if return_readable_value_format:
                    # Remove NaN values, because astype() cannot handle them
                    resampled_data.dropna(inplace=True)
                    # Convert Watt values to int and round SOC to 2 decimal places
                    for col in resampled_data.columns:
                        if 'active_power' in col:
                            resampled_data[col] = resampled_data[col].astype('int')
                        elif 'soc' in col:
                            resampled_data[col] = resampled_data[col].round(decimals=2)

                if iso_timestamps:
                    resampled_data.index = resampled_data.index.strftime('%Y-%m-%dT%H:%M:%S%z')
            else:
                resampled_data = data

            data_per_device.append(resampled_data)

        data_per_subcategory[subcategory] = add_up_device_data(data_per_device)

    return data_per_subcategory


def process_meter_values(meter_active_power: typing.Dict[str, float]) \
        -> typing.Tuple[typing.Dict[str, typing.Dict[str, float]], typing.Dict[str, typing.Dict[str, float]]]:
    """
    Calculate ...
    - Active power at the grid connection point
    - Total load
    - inflexible load
    ... based on the meter value, the type of meter and all device power outputs or consumption
    :param meter_active_power: Active power measurement of the energy meter
    :return: Active power at GCP and of the inflexible load
    """

    def prepare_device_data() -> typing.Dict[str, pd.Series]:
        data_per_subcategory: typing.Dict[str, pd.DataFrame] = get_device_measurements_aggregated_per_subcategory(
            # Search for measurements with meter value timestamp +/- it's resolution
            window_start=min(meter_values.index).to_pydatetime() - datetime.timedelta(seconds=resolution),
            window_end=max(meter_values.index).to_pydatetime() + datetime.timedelta(seconds=resolution),
            # Resample measurements to 1s (interpolating) to always get a value matching the meter value's timestamp
            resolution=1,
            include_load=False,
            filter_fields={'all': {'active_power'}},
        )

        no_data_for = []
        for subcategory, df in data_per_subcategory.items():
            if df.empty:
                # Collect subcategories for which no data was returned (-> empty dataframe) to remove them afterwards
                no_data_for.append(subcategory)
            else:
                # Convert pandas dataframe with single column to pandas series
                series = pd.Series(df['active_power'])
                if len(series) == 1:
                    # Problem with single datapoint: The resampling done above does not have any effect. Hence, if
                    # the timestamp does not match the the meter value's timestamp exactly, the arithmetic
                    # operations below practically fail.
                    logger.debug(f"{subcategory} data has only 1 datapoint: {series}")

                    # Add the same value again with timestamps - and + the meter values' resolution
                    ser_enhanced = pd.Series(index=[series.index[0] - datetime.timedelta(seconds=resolution),
                                                    series.index[0],
                                                    series.index[0] + datetime.timedelta(seconds=resolution)],
                                             data=list(series.values) * 3,
                                             name=series.name)
                    # Resample to the target resolution of 1s and fill up the values in between
                    # As a result, the meter value's timestamp will be included
                    series = ser_enhanced.resample(rule=datetime.timedelta(seconds=1)).ffill()
                    logger.debug(f"Enhanced and resampled single datapoint of {subcategory} data: {series}")

                data_per_subcategory[subcategory] = series

        # Remove the empty dataframes
        for subcategory in no_data_for:
            logger.info(f"No data returned from DB for device subcategory {subcategory}. Removing this empty "
                        f"dataframe from the data_per_subcategory dict.")
            data_per_subcategory.pop(subcategory)

        if os.getenv('BESS_KEY') not in data_per_subcategory:
            # No battery storage data -> Add series with power values = 0 to make calculation easier in the following
            data_per_subcategory[os.getenv('BESS_KEY')] = pd.Series(data={idx: 0 for idx in meter_values.index},
                                                                    name='active_power')

        try:
            pv_values = data_per_subcategory[os.getenv('PV_KEY')]
            # Make sure pv power values are <= 0
            data_per_subcategory[os.getenv('PV_KEY')] = -abs(pv_values)
        except KeyError:
            pass

        return data_per_subcategory

    # Prepare meter data
    meter_config = load_grid_config()["metering"]
    assert meter_config["type"] in ("consumption", "balance"), \
        f"Meter of type {meter_config['type']} is not supported. Supported types: 'balance' and 'consumption'"

    # Convert to series to not worry about number of values (which can or (mostly) will be 1)
    meter_values = pd.Series(data=meter_active_power, name='active_power')
    meter_values.index = pd.to_datetime(meter_values.index)
    timestamp_index_iso: typing.List[str] = [ts.isoformat() for ts in meter_values.index]

    resolution = int(os.getenv('METER_DATA_RESOLUTION'))

    # Get the resampled measurements from the devices of this building/site
    device_data_per_subcategory: typing.Dict[str, pd.Series] = prepare_device_data()

    if meter_config["type"] == "balance":

        # Calculate power exchanged at the grid connection point of this site
        # Although it's a balancing meter, not all consumers (e.g. heat pumps) or generators (e.g. additional PV plants)
        # might be included in the balancing meters' measurement, because they have separate meters.
        gcp_values = meter_values.copy()
        for subcategory in (meter_config["consumers_excluded"] + meter_config["generators_excluded"]):
            try:
                gcp_values = (gcp_values + device_data_per_subcategory[subcategory]).dropna()
            except KeyError:
                logger.warning(f"Trying to add {subcategory} power to GCP power, but there's no data. Unable "
                               f"to calculate GCP power without all data of balancing-meter-excluded generators "
                               f"and consumers, hence setting GCP power to None.")
                gcp_values = None

        # Calculate the total load (flexible + inflexible) from the balancing meter's value, generation power and
        # possible additional consumers with a separate meter.
        total_load_values = meter_values.copy()
        for subcategory in (meter_config["generators_included"] + meter_config["consumers_excluded"]):
            try:
                total_load_values = (total_load_values + abs(device_data_per_subcategory[subcategory])).dropna()
            except KeyError:
                logger.warning(f"Trying to subtract {subcategory} power of meter value to get total load, but there's "
                               f"no data. Unable to calculate total load without all data of meter-included generators "
                               f"and meter-excluded inflexible consumers, hence setting total load to None.")
                total_load_values = None

        # Account for charged or discharged power of the storage (power=0 if no storage installed)
        try:
            total_load_values = (total_load_values - device_data_per_subcategory[os.getenv('BESS_KEY')]).dropna()
        except TypeError:
            # Total load is None
            pass

    else:  # type = consumption
        assert len(meter_config["generators_included"]) == 0, \
            "The list 'generators_included' in the metering config must be empty! " \
            "A consumption meter cannot account for generated energy."

        total_load_values = meter_values.copy()

        for subcategory in meter_config["consumers_excluded"]:
            try:
                total_load_values = (total_load_values + device_data_per_subcategory[subcategory]).dropna()
            except KeyError:
                logger.warning(f"Trying to add {subcategory} power to total load, but there's no data. Unable "
                               f"to calculate GCP power or inflexible load without total load, hence "
                               f"returning (None, None).")
                return None, None

        # Account for charged or discharged power of the storage (power=0 if no storage installed)
        total_load_values = (total_load_values - device_data_per_subcategory[os.getenv('BESS_KEY')]).dropna()

        gcp_values = total_load_values.copy()
        for subcategory in meter_config["generators_excluded"]:
            try:
                gcp_values = (gcp_values + device_data_per_subcategory[subcategory]).dropna()
            except KeyError:
                logger.warning(f"Trying to add {subcategory} power to GCP power, but there's no data. Unable to "
                               f"calculate the power at the GCP without all generation data, hence setting GCP "
                               f"power to None.")
                gcp_values = None

    # Add timestamp(s) (ISO format) again and convert to dict
    if gcp_values is not None:
        gcp_values = {ts: {gcp_values.name: value} for ts, value in zip(timestamp_index_iso, gcp_values.values)}

    # Calculate the inflexible load by subtracting the power of all flexible, controlled consumers from the
    # meter's measurement, if the meter accounts for them.
    try:
        inflexible_load_values = total_load_values.copy()

        for subcategory in meter_config["flexible_consumers_included"]:
            inflexible_load_values = (inflexible_load_values - device_data_per_subcategory[subcategory]).dropna()

        # Add timestamp(s) (ISO format) again and convert to dict
        inflexible_load_values = {ts: {inflexible_load_values.name: value} for ts, value in
                                  zip(timestamp_index_iso, inflexible_load_values.values)}

    except AttributeError:
        logger.warning(f"Total load is None. Unable to calculate inflexible load, hence setting it to None.")
        inflexible_load_values = None
    except KeyError:
        logger.warning(f"Trying to subtract flexible {subcategory} power of total load, but there's no data. Unable "
                       f"to calculate inflexible load without data of all flexible consumers, hence setting "
                       f"inflexible load to None.")
        inflexible_load_values = None

    return gcp_values, inflexible_load_values


def check_for_tolerated_deviation(gcp_active_power: typing.Dict[str, typing.Dict[str, float]]):
    """
    Check if the current measured grid power exchange is still within the tolerated bounds
    given the current limitation (setpoint or schedule).
    If so, and a deviation has not been reported yet for the current schedule, publish a corresponding message.
    :param gcp_active_power: measured active power exchange with the grid with timestamp, e.g.
        {'2021-12-22T08:16:00+00:00': {'active_power': 785.0}}
    """
    timestamp: str = max(gcp_active_power)  # ISO-format timestamp
    # Take the latest one if multiple measurements
    gcp_active_power_value: float = gcp_active_power[timestamp]["active_power"]

    target_since = None
    # Get latest setpoint within the last 60 minutes
    current_setpoint: typing.Union[int, None] = get_latest_gcp_setpoint()
    if not (current_setpoint is None or current_setpoint == int(os.getenv("GRID_SETPOINT_CLEAR_VALUE"))):
        setpoint_active = True
        target_value = current_setpoint
        deviation_reported = False
    else:
        setpoint_active = False
        # There is no setpoint -> get the target power from the currently valid schedule
        # Retrieve some meta data of this schedule, too, and the start of the matching schedule period
        result: typing.Tuple[float, dict, str] = get_latest_gcp_schedule(
            at_time=datetime.datetime.fromisoformat(timestamp),
            return_timestamps_in_isoformat=True,
            with_doc_id=True,
            with_meta_data=["deviation_reported", "updated_at"],
            return_period_start=True
        )
        if result is None:
            return

        target_value, meta_data, target_since = result

        # Get boolean entry that specifies if a deviation from this very schedule has already been reported.
        # This field does not exist for new/updated schedules, but is set in this function after a deviation
        # has been reported (see below).
        deviation_reported: typing.Union[None, bool] = meta_data.pop("deviation_reported")
        # Get of last update of this schedule
        updated_at: str = meta_data.pop("updated_at")  # ISO-format timestamp

        if (updated_at >= timestamp) or (updated_at > target_since):
            logger.info(f"The GCP schedule has already been updated (at {updated_at}) after the measurement was "
                        f"recorded (at {timestamp}, target since {target_since}). Skipping GCP limit violation check "
                        f"until the next (possibly updated) period starts.")
            return

    gcp = GridConnectionPoint(specifications=load_grid_config()["specifications"])
    bounds = gcp.tolerance(active_power_limit=target_value)
    if not (min(bounds) <= gcp_active_power_value <= max(bounds)):
        logger.info(f'Deviation of grid power from {"schedule" if not setpoint_active else "setpoint"} detected! '
                    f'(target={target_value}, measured={gcp_active_power_value}, tolerance={bounds})')

        if deviation_reported:
            logger.info(f"Deviation from GCP schedule at {timestamp} has already been reported.")
            # Don't report it again. Scheduling has already been triggered by the first deviation message.
            return

        # Deviation detected, but not yet reported -> publish the corresponding message
        publish_deviation_to_ems(
            schedule_value=target_value,
            measured_value=gcp_active_power_value,
            scheduled_since=target_since,
            timestamp=datetime.datetime.now(tz=datetime.timezone.utc),
            setpoint_active=setpoint_active,
        )
        # Add the field "deviation_reported": True to the current schedule
        if not setpoint_active:
            successful = db.update_data_in_db(
                db=os.getenv('MONGO_SCHEDULE_DB_NAME'),
                data_source=os.getenv('GRID_CONNECTION_POINT_KEY'),
                filter_fields=meta_data,  # meta_data only contains the document's ID now
                updates={"deviation_reported": True}
            )
    else:
        logger.info(
            f"GCP power={gcp_active_power_value} is within bounds=({min(bounds)}, {max(bounds)}) of target={target_value}")


'''
All about quotas and reference schedules
'''


def store_reference_schedule(schedule: typing.Dict[str, int], updated_at: datetime.datetime):
    db.save_data_to_db(
        db=os.getenv('MONGO_SCHEDULE_DB_NAME'),
        data_source=f'{os.getenv("GRID_CONNECTION_POINT_KEY")}_reference',
        time_series_data=schedule,
        group_by_date=True,
        meta_data={'updated_at': updated_at.isoformat(timespec='seconds')},
        persist_old_data=True
    )


def get_reference_schedule_from_db(period_block_start: datetime.datetime, period_block_end: datetime.datetime) -> \
        typing.Dict[datetime.datetime, int]:
    schedule: typing.Dict[datetime.datetime, int] = db.get_time_series_data_from_db(
        db=os.getenv('MONGO_SCHEDULE_DB_NAME'),
        collection=f'{os.getenv("GRID_CONNECTION_POINT_KEY")}_reference',
        start_time=period_block_start, end_time=period_block_end,
        grouped_by_date=True
    )
    return schedule


def publish_reference_schedule_to_ems(schedule: typing.Dict[str, float]):
    """
    Publish the reference schedule to the message exchange to be received by other BEM services
    :param schedule: Reference schedule, as dict with ISO-formatted timestamps as keys
    """
    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()
    try:
        payload = json.dumps(schedule)
        logger.debug(f'Publish reference schedule: {payload}.')
        channel.basic_publish(exchange=os.getenv('RABBITMQ_BEM_CONTROL'),
                              routing_key=f'{os.getenv("GRID_CONNECTION_POINT_KEY")}.schedule.reference',
                              body=payload)
    finally:
        connection.close()


def save_quotas_to_db(quota_information: typing.Dict[str, dict], quota_category: str, calculation_method: str):
    db.save_data_to_db(
        db=os.getenv('MONGO_QUOTA_DB_NAME'),
        data_source=os.getenv(f'MONGO_{quota_category.upper()}_QUOTAS_COLL_NAME'),
        time_series_data=quota_information,
        group_by_date=True,
        meta_data={'updated_at': datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec='seconds'),
                   "calculation_method": calculation_method},
        persist_old_data=True
    )


def get_primary_quotas_from_db(period_block_start: datetime.datetime, period_block_end: datetime.datetime,
                               with_calculation_method: bool = False) -> \
        typing.Dict[datetime.datetime, dict]:
    """
    The data is returned as a dictionary with
        keys: timestamps (datetime, with tzinfo=UTC)
        values: dict with keys = 'quota', 'reference_power', 'type' and 'abs_power_limit'
    Example:
    {
    datetime(2021,1,1,16,0,0, tzinfo=datetime.timezone.utc): {'quota': 0.8, 'reference_power': -8650, 'type': 'feedin', 'abs_power_limit': -6920},
        ....
    }
    :return: Quota information for each period where quota < 1 (format see above)
    """
    # Database query returns quota data as Dict[datetime.datetime, dict] or, if with_calculation_method, tuple with
    # quota data and the calculation method (str)
    result = db.get_time_series_data_from_db(
        db=os.getenv('MONGO_QUOTA_DB_NAME'),
        collection=os.getenv('MONGO_PRIMARY_QUOTAS_COLL_NAME'),
        start_time=period_block_start, end_time=period_block_end,
        grouped_by_date=True,
        extra_info=['calculation_method'] if with_calculation_method else None
    )
    if with_calculation_method:
        primary_quota_data = result[0]
        calculation_method = result[1]['calculation_method']
        assert not isinstance(calculation_method,
                              datetime.date), f'There are different calculation methods across the relevant dates: {calculation_method}'
        return primary_quota_data, calculation_method
    return result


def determine_final_quotas(period_block_start: datetime.datetime, secondary_power_limits: typing.Dict[str, int],
                           primary_power_limits: typing.Union[None, typing.Dict[str, int]] = None,
                           calculation_method: str = None):
    """
    TODO: are values received from secondary market the total limit or the flexible limit? (also adjust docstring)
    Calculate, store and and publish the final quotas and limits.

    Logic is as follows for each period with a primary quota:
    If there is a power limit in secondary quotas (signed; resulting from successful trading at the quota
    secondary market), replace the power limit that resulted from the primary quota with it.
    Otherwise the primary limit remains the same.

    Then, for each period without a quota (unrestricted periods):
    Take the reference schedule for these periods and add quota=1.0, type=None and power limit (=reference power).

    Combine quoted and unrestricted periods to obtain the final quotas and limits.

    :param period_block_start: Start of the relevant quota period block
    :param secondary_power_limits: Dict with power values denoting flexible power limits for each timestamp after trading
    :param primary_power_limits: Dict with power values denoting flexible power limits for each timestamp as received form the DSO
    :param calculation_method: The method used by the DSO to calculate the quotas
    """

    def get_data(start, end):
        # Convert dict to Series with datetime index
        secondary_limits = pd.Series(
            data=secondary_power_limits.values(),
            index=pd.to_datetime(list(secondary_power_limits.keys())),
            name='active_power'
        )
        if primary_power_limits is None:
            # Get the primary quotas (possibly sparse w.r.t. to all periods of block) for the upcoming period block
            primary_quotas_dict, calc_method = get_primary_quotas_from_db(start, end, with_calculation_method=True)
            primary_quotas = pd.DataFrame.from_dict(primary_quotas_dict, orient='index')
        else:
            primary_quotas = pd.DataFrame.from_dict(primary_power_limits, orient='index')
            primary_quotas.index = pd.to_datetime(primary_quotas.index)
            calc_method = calculation_method
        # Get the reference schedule (complete) for the upcoming period block
        ref_schedule = pd.Series(get_reference_schedule_from_db(start, end), name='reference_power')

        return primary_quotas, secondary_limits, ref_schedule, calc_method

    # Get all necessary data from the database and return them as Dataframes or Series
    primary_quota_data, secondary_limits, reference_schedule, calculation_method = get_data(
        period_block_start, period_block_start + datetime.timedelta(hours=int(os.getenv('QUOTA_WINDOW_SIZE')))
    )

    logger.debug(f'Primary quota data: {primary_quota_data}')
    logger.debug(f'Power limits resulting from trading: {secondary_limits}')
    logger.debug(f'Reference schedule: {reference_schedule}')

    # Initialize with primary quota data and overwrite all relevant periods in the following
    quotas_and_limits = primary_quota_data.copy()

    # """
    # Assumption: secondary quotas are the total limits
    # TODO: uncomment if applicable
    #
    # """
    # # Apply limit from the secondary market if there is one for the period
    # quotas_and_limits.loc[secondary_limits.index, 'abs_power_limit'] = secondary_limits
    #
    # # Calculate final quota (the relative value)
    # quotas_and_limits.loc[
    #     secondary_limits.index, 'quota'
    # ] = (quotas_and_limits['abs_power_limit'] - quotas_and_limits['inflexible_power']) / quotas_and_limits['reference_power']

    """
    Alternative: secondary quotas are the flexible limits
    """
    # Apply limit from the secondary market if there is one for the period
    quotas_and_limits.loc[secondary_limits.index, 'flexible_abs_power_limit'] = secondary_limits
    quotas_and_limits.loc[
        secondary_limits.index, 'abs_power_limit'
    ] = quotas_and_limits['flexible_abs_power_limit'] + quotas_and_limits['inflexible_power']

    # Calculate final quota for cases where reference_power != 0 (others not possible due to division by zero)
    non_zero_reference_power_quotas = quotas_and_limits.loc[quotas_and_limits["reference_power"] != 0]
    non_zero_reference_power_quotas.loc[
        secondary_limits.index, 'quota'
    ] = non_zero_reference_power_quotas['flexible_abs_power_limit'] / non_zero_reference_power_quotas['reference_power']
    quotas_and_limits.loc[non_zero_reference_power_quotas.index, 'quota'] = non_zero_reference_power_quotas['quota']

    logger.debug(f'Secondary quota data (merge of primary with traded quotas): {quotas_and_limits}')

    # Get reference schedule for periods without any quota (neither primary nor secondary)
    # Transform to dataframe with single column "reference_power"
    unrestricted_schedule = pd.DataFrame(
        reference_schedule[~reference_schedule.index.isin(quotas_and_limits.index)]
    )
    logger.debug(f'Schedule for unrestricted periods: {unrestricted_schedule}')
    if not unrestricted_schedule.empty:
        # Add and fill remaining columns to obtain same format as primary quota data
        non_quoted_period_data = unrestricted_schedule.assign(quota=1.0, type=np.nan,
                                                              abs_power_limit=unrestricted_schedule.active_power)
        # If scheduled consumption is below unconditional consumption, set to unconditional consumption
        non_quoted_period_data.loc[
            non_quoted_period_data["abs_power_limit"] >= 0, "abs_power_limit"
        ] = max(non_quoted_period_data["abs_power_limit"],
                load_grid_config()["specifications"]["unconditional_consumption"])

        # Combine quoted periods with non-quoted periods; throw ValueError in case of duplicate index
        final_quotas_and_limits = quotas_and_limits.append(non_quoted_period_data, verify_integrity=True)
    else:
        final_quotas_and_limits = quotas_and_limits

    # Save to database and publish to EMS
    final_quotas_and_limits.index = [t.isoformat() for t in final_quotas_and_limits.index]
    final_quotas_and_limits_dict = final_quotas_and_limits.to_dict(orient='index')
    logger.info(f'Final quotas and limits: {final_quotas_and_limits_dict}')

    save_quotas_to_db(final_quotas_and_limits_dict, quota_category='final', calculation_method=calculation_method)

    publish_quotas_to_ems(
        quota_information={**final_quotas_and_limits_dict, 'start': period_block_start.isoformat()},
        quota_category='final'
    )


'''
Construction of outgoing messages
'''


# TODO: Move this to communication module


def construct_gcp_schedule_message(window_start: datetime.datetime, gcp_schedule: typing.Dict[str, int],
                                   uuid: str) -> dict:
    schedule_list = []
    for period, value in gcp_schedule.items():
        if value >= 0:
            valtype = "consume"
            flexible_value = max(0, value - load_grid_config()["specifications"]["unconditional_consumption"])
        else:
            valtype = "feedin"
            flexible_value = value
        schedule_list.append({
            "time": period,
            "type": valtype,
            "total_value": value,
            "flexible_value": flexible_value
        })
    message = {
        "uuid": uuid,
        "start_time": window_start.isoformat(timespec='seconds'),
        "schedule": schedule_list
    }
    return message


def construct_flexibility_message(lower_bound: float, upper_bound: float,
                                  gcp_attributes: dict, timestamp: datetime.datetime) -> dict:
    message = {
        # "cluster_id": gcp_attributes["cluster_id"],
        # "uuid": gcp_attributes["uuid"],
        # "unconditional_consumption": gcp_attributes["unconditional_consumption"],
        **gcp_attributes,
        "time": timestamp.replace(microsecond=0).isoformat(),
        "min": lower_bound,
        "max": upper_bound,
        "scheduled": get_latest_gcp_schedule(at_time=timestamp)
    }
    return message


def construct_ack_setpoint_message(timestamp: str, feedin_setpoint: int, consumption_setpoint: int,
                                   clear: bool) -> dict:
    grid_config = load_grid_config()
    namespace = grid_config['namespace']
    specifications = grid_config['specifications']
    ack_message = {
        "cluster_id": specifications['cluster_id'],
        "uuid": specifications['uuid'],
        "time": timestamp,
        namespace["consumption_setpoint_received"]: consumption_setpoint,
        namespace["feedin_setpoint_received"]: feedin_setpoint,
        namespace["setpoint_active_bool"]: str(not clear),

    }
    return ack_message


def construct_realized_setpoint_message(power: int):
    # Get corresponding setpoint from the database
    setpoint = get_latest_gcp_setpoint()
    namespace = load_grid_config()['namespace']
    message = {
        **construct_ack_setpoint_message(
            timestamp=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec='seconds'),
            feedin_setpoint=abs(setpoint) if setpoint <= 0 else None,
            consumption_setpoint=setpoint if setpoint > 0 else None,
            clear=False
        ),
        namespace['consumption_schedule_applied']: max(0, power),
        namespace['feedin_schedule_applied']: abs(min(0, power)),
    }
    return message


def construct_market_orders_message(order_period_start: datetime.datetime, order_sets: typing.List[typing.List[dict]],
                                    uuid: str) -> dict:
    # Assert that given order set has the right format (list of lists with dict)
    assert isinstance(order_sets, list), f'order_sets is of type {type(order_sets)}'
    for order_set in order_sets:
        assert isinstance(order_set, list) and isinstance(order_set[0], dict)

    message = {
        "uuid": uuid,
        "start_time": order_period_start.isoformat(),
        "order_sets": order_sets
    }
    return message


def construct_measurements_message(measurements: typing.Dict[str, pd.DataFrame], uuid: str):
    """
    :param measurements: Dict of dataframes, where each dataframe contains the data of a single device  or other source
     (->keys=device keys)
    :param uuid: DSO UUID of this grid connection point
    :return: Message with measurements, ready for sending as json body
    """
    all_data = {}
    for source, df in measurements.items():
        df.index.rename("timestamp", inplace=True)
        df.reset_index(inplace=True, drop=False)
        data_list = df.to_dict(
            'records')  # -> produces [{'timestamp': ..., 'active_power': 0.5}, {'timestamp': ..., 'active_power': 0.75}, ...]
        if source in os.getenv('LOAD_EL_KEY'):
            all_data['other_consumption'] = data_list  # key agreed on with project partners
        else:
            all_data[source] = data_list

    message = {
        "uuid": uuid,
        "values": all_data
    }
    return message


'''
Processing of incoming and outgoing messages
'''


def publish_deviation_to_ems(schedule_value: int, measured_value: int, scheduled_since: str, setpoint_active: bool,
                             timestamp: datetime.datetime):
    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()
    try:
        payload = json.dumps({
            "feature": "active_power",
            'sensor_value': measured_value,
            'target_value': schedule_value,
            'target_since': scheduled_since,
            'timestamp': timestamp.isoformat(timespec='seconds'),
            'setpoint_active': setpoint_active
        })
        logger.info(f'Publish deviation message from {os.getenv("GRID_CONNECTION_POINT_KEY")}: {payload}.')
        channel.basic_publish(exchange=os.getenv('RABBITMQ_BEM_CONTROL'),
                              routing_key=f'{os.getenv("GRID_CONNECTION_POINT_KEY")}.{os.getenv("DEVIATION_MESSAGE_SUBTOPIC")}',
                              body=payload)
    finally:
        connection.close()


def publish_quotas_to_ems(quota_information: typing.Dict[str, dict], quota_category: str):
    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()

    try:
        quotas_payload = json.dumps(quota_information)
        logger.debug(
            f'Publish {quota_category} quotas. First period={list(quota_information.items())[0]}, '
            f'last period={list(quota_information.items())[-2]}.')
        channel.basic_publish(exchange=os.getenv('RABBITMQ_BEM_INBOX'), routing_key=f'quotas.{quota_category}',
                              body=quotas_payload)
    finally:
        connection.close()


def publish_ev_charging_user_input_to_ems(input):
    logger.debug(f"Publish EV user input data to EMS: {input}")
    payload = json.dumps(input)
    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()
    try:
        channel.basic_publish(exchange=os.getenv('RABBITMQ_BEM_INBOX'), routing_key=f'user.input.ev',
                              body=payload)
    finally:
        connection.close()

# TODO: Move this to communication module


def process_quota_message(message: dict, uuid: str) -> typing.Tuple[
    typing.Dict[str, typing.Dict[str, dict]], str, str]:
    """
    Extracts the time series with quotas (and further values) for each period from the message and
    calculates the resulting absolute, signed power limits. Also distinguishes between mandatory primary quotas (upcoming period block)
    and preliminary quotas (later period blocks).
    The data is returned as a nested dictionary: for each quota category ('primary' and 'preliminary') it contains...
        keys: timestamps (period start times, ISO format)
        values: dict with keys 'quota', 'reference_power', 'type' and 'abs_power_limit'
    Example for quotas_and_limits_categorized['primary']:
    {
     '2021-01-01T16:00:00+00:00': {'quota': 0.8, 'reference_power': -8650, 'type': 'feedin', 'abs_power_limit': -6920},
        ....
     '2021-01-01T18:15:00+00:00': {'quota': 0.45, 'reference_power': 9585, 'type': 'consume', 'abs_power_limit': 4313.25},
        ...
    }
    """

    def calculate_absolute_signed_power_limit(quotas: pd.DataFrame):
        # For consumption, the reference values denotes only the flexible consumption.
        # Hence, total_limit = unconditional_consumption + quota*reference_power. Except if it's a feedin quota.
        # quotas.loc[(quotas["reference_power"] >= 0) & ((quotas[ns["quota_type_key"]].isna()) | (quotas[ns["quota_type_key"]] == ns["quota_type_value_consumption"])), 'abs_power_limit'] = unconditional_consumption + quotas["reference_power"] * quotas[ns["quota_value_key"]]
        # quotas.loc[
        #     (quotas["reference_power"] >= 0) & ~(quotas[ns["quota_type_key"]] == ns["quota_type_value_feedin"]),
        #     'abs_power_limit'
        # ] = unconditional_consumption + quotas["reference_power"] * quotas[ns["quota_value_key"]]
        # # If it's a feedin quota, the quota does not apply to the planned flexible consumption
        # quotas.loc[
        #     (quotas["reference_power"] >= 0) & (quotas[ns["quota_type_key"]] == ns["quota_type_value_feedin"]),
        #     'abs_power_limit'
        # ] = unconditional_consumption + quotas["reference_power"]
        #
        # # The total generation is defined as flexible. Hence, total_limit = quota*reference_power,
        # # except if it's a consumption quota.
        # # quotas.loc[(quotas["reference_power"] < 0) & ((quotas[ns["quota_type_key"]].isna()) | (quotas[ns["quota_type_key"]] == ns["quota_type_value_feedin"])), 'abs_power_limit'] = quotas["reference_power"] * quotas[ns["quota_value_key"]]
        # quotas.loc[
        #     (quotas["reference_power"] < 0) & ~(quotas[ns["quota_type_key"]] == ns["quota_type_value_consumption"]),
        #     'abs_power_limit'
        # ] = quotas["reference_power"] * quotas[ns["quota_value_key"]]
        # # If it's a consumption quota, the quota does not apply to the planned feedin
        # quotas.loc[
        #     (quotas["reference_power"] < 0) & (quotas[ns["quota_type_key"]] == ns["quota_type_value_consumption"]),
        #     'abs_power_limit'
        # ] = quotas["reference_power"]

        # Calculate flexible power limits
        # Consumption
        quotas.loc[
            (quotas["reference_power"] >= 0) & ~(quotas[ns["quota_type_key"]] == ns["quota_type_value_feedin"]),
            "flexible_abs_power_limit"
        ] = quotas["reference_power"] * quotas[ns["quota_value_key"]]
        # Feed-in
        quotas.loc[
            (quotas["reference_power"] < 0) & ~(quotas[ns["quota_type_key"]] == ns["quota_type_value_consumption"]),
            "flexible_abs_power_limit"
        ] = quotas["reference_power"] * quotas[ns["quota_value_key"]]
        # Fill those rows where the quota does not apply to us, i.e. if there's a consumption quota,
        # but feedin is planned, and vice versa. In these cases, the reference power marks the limit.
        quotas["flexible_abs_power_limit"].fillna(quotas["reference_power"], inplace=True)

        # Add the inflexible power to the flexible limits to get the total limits
        quotas["abs_power_limit"] = quotas["flexible_abs_power_limit"] + quotas["inflexible_power"]
        return quotas

    def categorize_quotas(quotas: pd.DataFrame) -> typing.Dict[str, typing.Dict[str, dict]]:
        mandatory = (quotas.index >= block_start) & (quotas.index < block_end)
        quotas_categorized = {
            'primary': quotas.loc[mandatory].to_dict(orient='index'),
            'preliminary': quotas.loc[~mandatory].to_dict(orient='index')
        }
        return quotas_categorized

    # Load the quota message namespace from the configuration
    ns = load_grid_config()["namespace"]["quota_message"]

    # Get the actual content (omit message info fields)
    message = message[ns["content_key"]]

    assert message[ns["uuid_key"]] == uuid, \
        f"UUID in quota message ({message[ns['uuid_key']]}) does not match this grid point's UUID ({uuid})."

    block_start = message[ns["block_start_key"]]
    block_end = (datetime.datetime.fromisoformat(block_start) + datetime.timedelta(
        hours=int(os.getenv('QUOTA_WINDOW_SIZE')))).isoformat()

    # Preprocessing
    quota_data = pd.DataFrame.from_records(data=message[ns["time_series_key"]], index=ns["timestamp_key"])
    quota_data.rename(columns={ns["reference_power_value_key"]: "reference_power"}, inplace=True)
    quota_data[ns["quota_type_key"]].replace(
        to_replace=dict(zip(["none", "None", "NONE", "null", "Null"], [np.nan] * 5)),
        inplace=True)
    unconditional_consumption = load_grid_config()["specifications"]["unconditional_consumption"]
    # Whereas the total generation is defined as flexible, there's an unconditional consumption that
    # is defined as inflexible
    quota_data["inflexible_power"] = np.where(quota_data["reference_power"] >= 0, unconditional_consumption, 0)

    quotas_and_limits: pd.DataFrame = calculate_absolute_signed_power_limit(quota_data)
    logger.debug(f'Processed quotas: {quotas_and_limits}')

    # Distinguish between mandatory primary and preliminary quotas
    quotas_and_limits_categorized: typing.Dict[str, typing.Dict[str, dict]] = categorize_quotas(quotas_and_limits)

    return quotas_and_limits_categorized, message[ns["calculation_method_key"]], block_start


def process_meter_data_message(message: dict) -> typing.Dict[str, float]:
    """
    Processing of message containing meter data as received from the DSO via ESB.
    The following measurements are extracted: active power (3-phase), voltage (per phase)
    All extracted values are saved to the database. Only active power measurement(s) is returned.
    :return: Active power measurement(s), as dict with (timestamp,value)-pairs; timestamp as ISO-formatted str,
    value as float in Watt
    """

    def get_value(raw_value: typing.Any, expected_types: tuple, expected_unit: str, conversion_factor: int):
        assert isinstance(raw_value, expected_types), f"Unexpected type {type(raw_value)} of the 'values' field " \
                                                      f"in the meter data message: measurement={measurement}. "

        assert unit == expected_unit, f'Unexpected unit {unit} for measurement {measurement_name}!'
        if isinstance(raw_value, list):
            assert len(
                raw_value) == 1, f'Unexpected number of values ({len(raw_value)}) for measurement {measurement_name}!'

            processed_value = float(raw_value[0])
        else:
            processed_value = float(raw_value)
        return processed_value * conversion_factor

    config = load_grid_config()
    ns = config["namespace"]["meter_data_message"]

    # Get list of measurements by accessing the corresponding level of the (nested) message dict
    all_measurements: typing.List[dict] = reduce(lambda m, k: m[k], ns["time_series_key"], message)

    if all_measurements is None:
        logger.info(f'Meter data message is empty: {message}.')
        return None

    measurements_by_timestamp = {}
    # Store active power values additionally,  to return it for subsequent calculation of the inflexible
    # load in the building
    active_power_measurements = {}
    try:
        for measurement in all_measurements:
            assert measurement[ns["uuid_key"]] == load_grid_config()["specifications"][
                "uuid"], "UUID does not match this GCP's UUID!"
            measurement_name = measurement[ns["name_key"]]
            timestamp: str = measurement[ns["timestamp_key"]].replace("Z", "+00:00")
            # Values field has once been an array of floats, then suddenly only a float. Handle both.
            values: typing.Union[list, float] = measurement[ns["values_key"]]
            # Unit value is also "hidden" behind more than one key
            unit: str = reduce(lambda d, k: d[k], ns["unit_key"], measurement)

            if measurement_name == ns["active_power_name"]["three_phase"]:
                value = get_value(values, expected_types=(list, float, int, str), expected_unit="MW",
                                  conversion_factor=10 ** 6)
                name = 'active_power'
                active_power_measurements[timestamp] = value

            elif measurement_name == ns["voltage_name"]:
                phase_letter = measurement[ns["phase_key"]]
                phase = ns["phase_map"][phase_letter]
                value = get_value(values, expected_types=(list, float, int, str), expected_unit="kV",
                                  conversion_factor=10 ** 3)
                name = f'voltage_{phase}'
            else:
                continue
            # Append this measurement to other measurements with same timestamp (no problem if it's the
            # first for this timestamp)
            # Result will look like this:
            # {'2021-03-22T10:19:00+00:00': {'voltage_1': 233.22, 'voltage_2': 232.8,
            # 'voltage_3': 233.7, 'active_power': 63400.0}, ...}
            measurements_by_timestamp[timestamp] = {**measurements_by_timestamp.get(timestamp, {}), **{name: value}}
    except Exception as e:
        logger.error(f"Something went wrong, raising {type(e)}. Complete message: {message}")
        raise e

    logger.debug(f'Measurements by timestamp: {measurements_by_timestamp}')

    # Save all values
    db.save_measurement(
        source="meter",
        data=measurements_by_timestamp
    )

    return active_power_measurements


def process_ev_charging_user_input_message(message: dict, dso_uuid: str):
    ns = load_grid_config()["namespace"]["ev_charging_user_input_message"]

    try:
        input: dict = reduce(lambda m, k: m[k], ns["data_key"], message)[0]
    except TypeError:
        # Happens if message contains no actual data and looks like this: {'data': {'car_charge': None}}
        return

    assert input['uuid'] == dso_uuid, "UUID does not match this GCP's UUID!"
    logger.debug(f'EV user input: {input}')
    timestamp_of_input = input[ns["timestamp_key"]]

    # Query for previous input with same timestamp -> if not existing, query result is None
    existing_input = db.get_data_from_db(
        db=os.getenv('MONGO_USERDATA_DB_NAME'),
        collection=os.getenv('MONGO_EV_CHARGING_INPUT_COLL_NAME'),
        doc_filter={'timestamp': timestamp_of_input},
        limit=1
    )
    if existing_input is not None:
        logger.debug(f'Existing EV input: {existing_input}')
        # Input is not new -> ignore it
        return

    # Input is new -> process input
    logger.debug('EV input is new. Process and store it.')
    soc = input[ns["soc_key"]] / (100 if input[ns["soc_key"]] > 1 else 1)
    data = dict(
        timestamp=timestamp_of_input,
        soc=soc,
        soc_target=input[ns["soc_target_key"]] / (100 if input[ns["soc_target_key"]] > 1 else 1),
        capacity=input[ns["capacity_key"]],
        scheduled_departure=input[ns["scheduled_departure_key"]]
    )

    # Save input to database
    db.save_dict_to_db(
        db=os.getenv('MONGO_USERDATA_DB_NAME'),
        data_category=os.getenv('MONGO_EV_CHARGING_INPUT_COLL_NAME'),
        data=data
    )
    try:
        # A mongoDB ObjectID object gets added to this dict when saving it.
        # Remove it, because it's not JSON serializable.
        data.pop('_id')
    except KeyError:
        pass

    # Save SOC as measurement and schedules departure in time series database
    db.save_measurement(os.getenv('EVSE_KEY'),
                        data={timestamp_of_input: {'soc': soc, 'departure': data['scheduled_departure']}}
                        )

    # Publish input to EMS to trigger scheduling
    publish_ev_charging_user_input_to_ems(data)
