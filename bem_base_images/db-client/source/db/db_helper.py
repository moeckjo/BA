"""
Wrapper functions for writing and querying an InfluxDB and MongoDB database, respectively.
InfluxDB: Stores all time series (e.g. measurements from devices)
MongoDB: Stores everything that is not an immutable time series: forecasts, schedules, objects, etc.
When replacing a database, the corresponding function bodies should be adapted without changing the function name and arguments.
"""
# from core.forecasting import *

# Local imports
import datetime

import bson

import typing

import pandas as pd

from db import logger, mongodb_handler
from db.influxdb_handler import InfluxConnector
from db.mongodb_handler import MongoConnector

# window_size_to_mongo_coll_suffix_map = {24: 'daily', int(os.getenv('QUOTA_WINDOW_SIZE')): 'intraday',
#                                         1: 'intrawindow'}

''' Wrapper functions for MongoDB'''


def create_doc_id(seed=None):
    if isinstance(seed, datetime.datetime):
        return mongodb_handler.doc_id_from_datetime(seed)
    else:
        return bson.objectid.ObjectId()


def save_dict_to_db(db: str, data_category: str, data: dict):
    """
    Simply stores the provided data in the given mongoDB collection without any transformation.
    :param db: Name of MongoDB database
    :param data_category: Category of data = collection name
    :param data: The dict to store
    :return: document id
    """
    assert isinstance(data, dict)
    mongo_connector = MongoConnector(db)
    doc_id = mongo_connector.db[data_category].insert_one(data)
    return doc_id


def update_data_in_db(db: str, data_source: str, filter_fields: typing.Dict[str, typing.Any],
                      updates: typing.Dict[str, typing.Any]):
    """
    Insert new fields or update values of existing fields in an existing document of a mongoDB collection.
    :param db: Name of MongoDB database
    :param data_source: source of data, e.g. for devices: bess, hp -> determines collection
    :param updates: Dict with <(new_)field_name, new_field_value>-pairs
    :param filter_fields: Filter based on existing fields and values of the document, including the special "_id" field.
    :return Boolean indicating if the modification has been successful
    """
    mongo_connector = MongoConnector(db)
    successful = mongo_connector.update_document(
        collection_name=data_source,
        updates=updates,
        doc_filter=filter_fields,
    )
    return successful


def save_data_to_db(db: str, data_source: str, time_series_data: typing.Dict[str, float], group_by_date=False,
                    meta_data: dict = None, doc_id: bson.objectid.ObjectId = None, persist_old_data: bool = False):
    # TODO: rename to save_time_series_data_to_db; and overall delete the _to/from_db in all functions here
    """
    Saves any time series data with meta data (optional) to a corresponding MongoDB collection.
    The fields 'first_timestamp' and 'last_timestamp' (or 'date', if group_by_date=True) are added to the document
    to enable easy querying later.
    :param db: Name of MongoDB database
    :param data_source: source of data, e.g. for devices: bess, hp -> determines collection
    :param time_series_data: dict with periods (ISO-format str timestamp) and values pairs,
                                e.g. {period1: value1, ...,periodn: valuen}. Data can span multiple days.
    :param group_by_date: If true, the data is first filtered by date, then stored in a separate document for each date.
                            The field "date" is added to the document to allow easy querying by date later.
    :param meta_data: Additional info about time series data that can be stored (optional)
    :param doc_id: document ID, if available
    :param persist_old_data: If true, existing data for given timestamps is persisted. Otherwise it is overwritten.
    """
    result_docs = {}
    meta_data = meta_data if meta_data else {}
    mongo_connector = MongoConnector(db)

    def write(data_header, data):
        # Connect and write to MongoDB
        result_doc = mongo_connector.write_data(
            collection_name=data_source,
            data_header=data_header,
            data=data,
            meta_data=meta_data,
            doc_id=doc_id,
            persist_old_data=persist_old_data
        )

    if group_by_date:
        # Convert ISO-format string timestamps to datetime timestamps to allow filtering by date
        time_series_data = {datetime.datetime.fromisoformat(timestamp): value for timestamp, value in
                            time_series_data.items()}
        # logger.info(f'TS data with datetime ts: {time_series_data}')
        dates: typing.Set[datetime.date] = set([ts.date() for ts in time_series_data.keys()])
        for date in dates:
            # logger.info(f'Date: {date} (all dates: {dates})')
            date_filtered_data = {timestamp: value for timestamp, value in time_series_data.items() if
                                  timestamp.date() == date}
            # logger.info(f'Filtered TS data with datetime ts: {date_filtered_data}')
            write(
                data_header={'date': date.isoformat()},
                data=date_filtered_data
            )
    else:
        timestamps = list(time_series_data.keys())
        write(
            data_header={'first_timestamp': min(timestamps), 'last_timestamp': max(timestamps)},
            data=time_series_data
        )
    # return result_docs


def get_time_series_data_from_db(db: str, collection: str, start_time: datetime.datetime, end_time: datetime.datetime,
                                 grouped_by_date: bool = False, extra_info: typing.List[str] = None,
                                 return_timestamps_in_isoformat: bool = False, return_doc_id: bool = False) -> typing.Dict[
    datetime.datetime, typing.Union[float, dict]]:
    """
    Queries a MongoDB collection for documents by time filter and processes the query results to return a
    time series with the last-added values for each period, respectively.
    It allows to retrieve a time series for the requested time span where the partial time series are spread
    across multiple documents.
    Arbitrary additional information (-> extra_info) can be requested as well.

    :param db: MongoDB database
    :param collection: name of the collection which contains the document
    :param start_time: first timestamp of requested time series (inclusive)
    :param end_time: last timestamp of requested time series (exclusive)
    :param grouped_by_date: If the stored data is grouped by date, i.e. one document contains all and exclusively data for a specific date
    :param extra_info: List of additional information that should be returned -> Elements must equal field names
    :param return_timestamps_in_isoformat: If the timestamp keys should remain strings in ISO format (if False (default): datetime objects)
    :param return_doc_id: If the document ID (key "_id") should be returned besides the data
    :return: Requested document/data, with timestamps as datetime objects (or str), or None if no data matches time filter.
            If grouped_by_date=True and the values of a field in extra_info differs across different dates, the value for this key in meta_data will be a dict with keys=dates.
    """

    def query(qfilter: dict, limit: int) -> typing.Dict[str, float]:
        fields = {'data': True, '_id': False}
        if extra_info:
            fields.update({name: True for name in extra_info})

        result = mongo_connector.query_db(
            collection=collection,
            doc_filter=qfilter,
            return_fields=fields,
            limit=limit
        )
        if limit == 1:
            try:
                returned_data = result.get('data', {})
                if extra_info:
                    meta_data = {name: result.get(name) for name in extra_info}
                    return returned_data, meta_data
                return returned_data
            except AttributeError:
                # Query returned None, i.e. no match for this filter -> return empty dict for better handling
                return {} if not extra_info else ({}, {})
        else:
            return result  # Cursor object

    mongo_connector = MongoConnector(db)
    if return_doc_id:
        if extra_info and "_id" not in extra_info:
            extra_info.append("_id")
        elif not extra_info:
            extra_info = ["_id"]

    if grouped_by_date:
        # Get list of all dates included in the period from start_time to end_time
        dates = [start_time.date()]
        while dates[-1] < end_time.date():
            next_day = dates[-1] + datetime.timedelta(days=1)
            dates.append(next_day)

        all_data = {}
        meta_data = {name: {} for name in extra_info} if extra_info else {}
        for date in dates:
            query_result: typing.Tuple[dict, dict] = query(qfilter={'date': date.isoformat()}, limit=1)
            if extra_info:
                data, this_datas_meta_data = query_result

                for name, value in this_datas_meta_data.items():
                    date_key = date.isoformat() if return_timestamps_in_isoformat else date
                    meta_data[name].update({date_key: value})

            else:
                data = query_result

            all_data.update(data)

        # If meta data values are the same for each date for a given field, just store this value without dates.
        for field, date_value_dict in meta_data.items():
            distinct_values = list(set(date_value_dict.values()))
            if len(distinct_values) == 1:
                meta_data[field] = distinct_values[0]

    else:
        if extra_info is None or 'updated_at' not in extra_info:
            # Parameter is needed for sorting of query results (see below)
            extra_info = extra_info + ['updated_at'] if extra_info else ['updated_at']
            remove_update_field = True
        else:
            remove_update_field = False

        # Try first to find docs that span at least the requested time range in total
        query_results = query(qfilter={
            'first_timestamp': {'$lte': start_time.isoformat(timespec='seconds')},
            'last_timestamp': {'$gte': end_time.isoformat(timespec='seconds')}
        }, limit=0)

        if len(query_results) == 0:
            # Not successful -> Search for multiple docs which cover parts of the requested time range
            query_results = query(qfilter={
                '$or': [
                    {'first_timestamp': {'$and': [{'$gte': start_time.isoformat(timespec='seconds')},
                                                  {'$lte': end_time.isoformat(timespec='seconds')}]}},
                    {'last_timestamp': {'$and': [{'$gte': start_time.isoformat(timespec='seconds')},
                                                 {'$lte': end_time.isoformat(timespec='seconds')}]}}
                ]
            }, limit=0)
        all_data = {}
        all_meta_data = {name: {} for name in extra_info}
        docs: typing.List[dict] = list(query_results)
        sorted_by_update_timestamp = sorted(docs, key=lambda k: k['updated_at'],
                                            reverse=True)  # descending, newest first
        i = 0
        # Iteratively update with older data until spanning the requested time range
        while not (start_time.isoformat(timespec='seconds') >= min(all_data) and end_time.isoformat(
                timespec='seconds') <= max(all_data)):
            doc = sorted_by_update_timestamp[i]
            data: dict = doc.pop('data')
            meta_data = doc
            all_data = data.update(all_data)  # keep newer data for overlapping timestamps
            all_meta_data = meta_data.update(all_meta_data)
            i += 1

        if remove_update_field:
            all_meta_data.pop('updated_at')
        meta_data = all_meta_data

    # Filter data by start and end time
    data = {t: v for t, v in all_data.items() if start_time.isoformat() <= t < end_time.isoformat()}

    if not return_timestamps_in_isoformat:
        # Convert string keys to datetime
        data = {datetime.datetime.fromisoformat(t): v for t, v in data.items()}

    if not data:
        # Return None if dict is empty (because no data matches time filter)
        return None

    if extra_info:
        return data, meta_data

    return data


def get_data_from_db(db: str, collection: str, doc_filter: dict = None,
                     doc_id: bson.objectid.ObjectId = None, return_fields: typing.Union[list, dict] = None,
                     limit: int = None) -> dict:
    """
    Queries a MongoDB collection for documents, either by ID or other filter
    :param db: MongoDB database
    :param collection: name of the collection which contains the document
    :param doc_id: ID of the requested document
    :param doc_filter: Dictionary with <field: value>-pair(s) contained in the requested document
    :param return_fields: Fields (keys) of document to return. Either as positive list or as dict with <field, True/False>. If not provided, all fields are returned.
    :param limit: The maximum number of results to return (default None -> return all)
    :return: Requested document/data
    """
    if doc_filter is None:
        doc_filter = {'_id': doc_id}

    if limit is None:
        # For MongoDB, limit=0 means no limit
        limit = 0

    mongo_connector = MongoConnector(db)
    # if return_fields:
    doc = mongo_connector.query_db(collection, doc_filter=doc_filter, return_fields=return_fields, limit=limit)
    # doc = mongo_connector.db[collection].find_one(filter=doc_filter, projection=return_fields)
    # else:  # Return entire document
    #     doc = mongo_connector.db[collection].find_one(filter=doc_filter)
    return doc


def save_objects_to_db(db: str, object_category: str, objects: dict, **kwargs) -> bson.objectid.ObjectId:
    """
    Stores the provided object in a MongoDB document in an object's type-specific collection
    :param db: MongoDB database name (e.g. 'devices')
    :param object_category: Category of the object (e.g. 'device_management_systems'); determines collection
    :param objects: dict(object_name, object) of objects to be stored
    :param kwargs: Further information about the objects that shall be stored
    :return: ID of the document that stores the objects and possible additional info
    """
    logger.debug(f'add info to object in DB: {kwargs}')
    mongo_connector = MongoConnector(db)
    object_id = mongo_connector.write_objects(
        collection_name=object_category,
        objects=objects,
        further_fields=kwargs if kwargs else None
    ).inserted_id
    return str(object_id)


def get_objects_from_db(db: str, object_category: str, doc_id: str = None, doc_filter: dict = None,
                        objects_only: bool = True) -> dict:
    mongo_connector = MongoConnector(db)
    result = mongo_connector.get_objects(
        collection=object_category,
        doc_id=doc_id,
        doc_filter=doc_filter,
        objects_only=objects_only
    )
    return result


def update_objects_in_db(db: str, object_category: str, doc_id: str, objects: dict):
    mongo_connector = MongoConnector(db)
    mongo_connector.update_objects(
        collection_name=object_category,
        doc_id=doc_id,
        objects=objects,
    )


''' Wrapper functions for InfluxDB'''


def save_measurement(source: str, data: typing.Union[dict, pd.DataFrame, pd.Series]):
    """
    Save measurements to the InfluxDB.
    :param source: Source/name of the measurement, e.g. 'pv'
    :param data: The time series. Either as pandas Dataframe or Series (named!) or dict with
        - keys: timestamps
        - values: dict with <field, value>-pairs
    """
    influx_connector = InfluxConnector()
    influx_connector.write_datapoints(source, data)
    influx_connector.close()


def get_measurement(source: str,
                    fields: typing.Union[str, list],
                    start_time: datetime.datetime,
                    end_time: typing.Optional[datetime.datetime] = None,
                    closed: str = 'left',
                    limit: typing.Optional[int] = None,
                    order: typing.Optional[str] = 'ASC') -> pd.DataFrame:
    """
    Queries an InfluxDB for time series of device measurements
    :param source: device or system
    :param start_time: first timestamp of requested time series
    :param end_time: last timestamp of requested time series (exclusive per default)
    :param closed: Boundaries of interval; can be "left" (default, only includes start time), "right" (only includes end time) or "both"
    :param fields: Parameter(s) of source for which the measurements are requested or "*" to query all fields.
    :param limit: Limit the number of returned values
    :param order: Sorting of returned data (by timestamp): ASC (default) or DESC
    :return: time series as pd.DataFrame
    """
    influx_connector = InfluxConnector()
    logger.debug(f'Reference data: from {start_time} to {end_time}')
    data = influx_connector.query_data(source=source, fields=fields, start_time=start_time, end_time=end_time,
                                       closed=closed, limit=limit, order=order)
    influx_connector.close()
    return data
