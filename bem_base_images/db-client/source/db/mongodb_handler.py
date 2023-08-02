import datetime
import os
import pickle

import typing

from bson import objectid
from pymongo import MongoClient, ReturnDocument
from pymongo.cursor import Cursor
from pymongo.errors import ConnectionFailure
from pymongo.results import InsertOneResult, UpdateResult

from db import logger


def doc_id_from_datetime(dt):
    # Convert datetime to UNIX timestamp and then to hex string (without '0x' prefix)
    hex_ts = format(int(dt.timestamp()), 'x')
    # Generate BSON ObjectID based with custom timestamp instead of generation time (first 4 bytes = 8 characters)
    custom_did = objectid.ObjectId(hex_ts + str(objectid.ObjectId())[8:])
    return custom_did


def datetime_from_doc_id(self, did):
    return did.generation_time.replace(tzinfo=datetime.timezone.utc)


def keys_to_string(data_dict):
    try:
        if isinstance(list(data_dict.keys())[0], datetime.datetime):
            data = {dtime.isoformat(): val for dtime, val in data_dict.items()}
        else:
            data = {str(key): val for key, val in data_dict.items()}
        return data

    except AttributeError:  # Thrown if data_dict is None
        return data_dict


class MongoConnector:

    def __init__(self, db: str):
        logger.debug('Hello mongo!')
        try:
            self.client = MongoClient(
                host=os.getenv('MONGO_HOSTNAME'),
                port=int(os.getenv('MONGO_PORT')),
                username=os.getenv('ADMIN_NAME'),
                password=os.getenv('ADMIN_PASSWORD')
            )
        except ConnectionFailure as C:
            '''
            From the docs: The client object is thread-safe and has connection-pooling built in.
            If an operation fails because of a network error, ConnectionFailure is raised and the client reconnects
            in the background. Application code should handle this exception (recognizing that the operation failed)
            and then continue to execute.
            '''
            # TODO: handling needed?
            logger.debug(C.with_traceback())
            pass

        self.db = self.client[db]
        logger.debug(f'MongoDB client connected to {self.db}')

    def __del__(self):
        self.client.close()

    def query_db(self, collection: str, doc_id: str = None, doc_filter: dict = None,
                 return_fields: typing.Union[list, dict] = None, limit: int = 0) -> typing.Union[dict, Cursor]:
        """
        Query database and return result. If limit=1, result is a single doc (dict) or None if there's no match. Otherwise (limit != 1) result is a
        Cursor object containing the docs (dicts).
        """
        if doc_filter is None:
            assert doc_id is not None, 'Either doc_id or doc_filter must be provided. You provided none.'
            # if doc_id is None:
            #     raise ValueError('Either doc_id or doc_filter must be provided. You provided none.')
            doc_filter = {'_id': objectid.ObjectId(doc_id)}
        if limit == 1:
            doc = self.db[collection].find_one(filter=doc_filter, projection=return_fields)
            return doc
        docs = self.db[collection].find(filter=doc_filter, projection=return_fields, limit=limit)
        return docs

    def write_data(self, collection_name: str, data: dict, data_header: typing.Dict[str, object],
                   meta_data: None, doc_id=None, persist_old_data: bool = False):
        # Convert dictionary keys to string (BSON requirement) -> datetime keys become ISOformat str timestamps
        # logger.info(f'(mongo handler) Filtered TS data with datetime ts: {data}')

        data = keys_to_string(data)
        meta_data = {} if not meta_data else meta_data
        # logger.info(f'(mongo handler) Filtered TS data with isoformat ts: {data}')

        logger.debug(self.db[collection_name])
        # TODO: Move all the logic to db_helper!
        # Update matching document (or create document if no match) with the data
        if doc_id is None:
            match_filter = {'$and': [{key: val} for key, val in data_header.items()]} if len(
                data_header) > 1 else data_header
        else:
            match_filter = {'_id': doc_id}

        doc: dict = self.db[collection_name].find_one(filter=match_filter)
        if doc is not None:
            doc_id = doc.pop('_id')
            existing_data = doc.pop('data')

            not_data_fields = list(data_header.keys()) + list(meta_data.keys())
            old_data_field = 'old_data'

            # Fields that don't exist for new data
            incongruent_doc_fields = [field for field in list(doc.keys()) if
                                      field not in not_data_fields and field != old_data_field]

            if persist_old_data:
                existing_meta_data = {key: val for key, val in doc.items() if
                                      key not in list(data_header.keys()) and key != old_data_field}
                data_to_persist = {**existing_meta_data, "data": existing_data}

                # Append data_to_persist to list of field "old_data"
                self.db[collection_name].update_one(
                    filter={'_id': doc_id},
                    # Append dict with data to persist to the list of even older data
                    # If the field does not exist yet, it is created with this old data as first element
                    update={'$push': {'old_data': data_to_persist}}
                )

            # Update existing data period-wise with new data
            latest_data = existing_data.copy()
            # Apply values from data update for each key=timestamp contained in the existing data.
            # Values for other timestamps are not changed. New key-value-pairs are added.
            latest_data.update(data)

            set_fields = {'$set': {
                'data': latest_data,
                **data_header,
                **meta_data
            }}
            # Clear fields that don't exist for new data
            unset_fields = {'$unset': {field: "" for field in incongruent_doc_fields}}

            doc_update = {**set_fields, **unset_fields} if len(incongruent_doc_fields) != 0 else set_fields
            # Insert updated data and corresponding further information
            doc_after = self.db[collection_name].find_one_and_update(
                filter={'_id': doc_id},
                # Replace fields in doc with latest data and further information
                update=doc_update,
                return_document=ReturnDocument.AFTER
            )
            return doc_after
        else:  # Document doesn't exist yet
            result = self.db[collection_name].insert_one(
                document={**data_header, 'data': data, **meta_data}
            )
            return {'_id': result.inserted_id, **data_header, 'data': data, **meta_data}

    def update_document(self, collection_name: str, updates: typing.Dict[str, typing.Any], doc_filter: typing.Dict[str, typing.Any]):
        """
        Insert new fields or update values of existing fields in an existing document.
        :param collection_name: Name of the mongoDB collection
        :param updates: Dict with <(new_)field_name, new_field_value>-pairs
        :param doc_filter: Filter based on existing fields and values of the document, including the special "_id" field.
        :return Boolean indicating if the modification has been successful
        """
        if "_id" in doc_filter:
            # Note: ObjectId(id) works on strings as well as ObjectId instances
            doc_filter = {'_id': objectid.ObjectId(doc_filter["_id"])}

        assert doc_filter is not None and len(doc_filter) > 0, f"Cannot update a document without document filter. " \
                                                               f"Provided filter: {doc_filter}"

        result: UpdateResult = self.db[collection_name].update_one(filter=doc_filter, update={'$set': updates})
        if result.modified_count == 1:
            return True
        else:
            return False


    def write_objects(self, collection_name: str, objects: dict, further_fields: dict = None) -> InsertOneResult:
        pickled_objects = {}
        for key, obj in objects.items():
            pickled_objects[key] = pickle.dumps(obj)
        doc = {**pickled_objects, **further_fields} if further_fields else pickled_objects
        # print(f'doc to be inserted: {doc}')
        result = self.db[collection_name].insert_one(
            document=doc
        )
        return result

    def get_objects(self, collection, doc_id: str = None, doc_filter: dict = None,
                    objects_only: bool = True) -> typing.Union[dict, typing.List[dict]]:

        def extract(doc):
            objects = {}
            info = {}
            for field, entry in doc.items():
                if isinstance(entry, bytes):
                    # Unpickle bytes to python objects
                    objects[field] = pickle.loads(entry)
                else:
                    info[field] = entry
            return objects if objects_only else {**objects, **info}

        docs = list(self.query_db(collection, doc_id=doc_id, doc_filter=doc_filter))
        if len(docs) == 1:
            doc = docs[0]
            result = extract(doc)
        else:
            result = []
            for doc in docs:
                result.append(extract(doc))
        return result


    def update_objects(self, collection_name: str, doc_id: str, objects: dict):
        assert doc_id is not None, 'The mongoDB document ID (doc_id) cannot be None.'
        doc_filter = {'_id': objectid.ObjectId(doc_id)}
        updates = {}
        for key, obj in objects.items():
            updates[key] = pickle.dumps(obj)
        self.db[collection_name].update_one(filter=doc_filter, update={'$set': updates})

    def clear_collection(self, collection: str):
        result = self.db[collection].delete_many(filter={})
        logger.info(f'Deleted all {result} documents in collection {collection}.')

    def update_embedded_doc(self, collection_name: str, data_source: str,
                            data_update: typing.Dict[datetime.datetime, int]):
        # TODO: This process makes no sense if data_update spans multiple days because it stores everything under the first day
        # TODO: delete if definitely not used anymore (03/2021: not used for months now..)
        """
        Retrieves a document based on a date filter, then updates the entry for the field given by data_source.
        In this case, the field entry is again a dict, hence an 'embedded document'.
        Only the fields of the embedded doc that matches the keys in the data_update are updated.

        If there is not match for the date filter, an entire new doc with this date and the data_update is inserted.

        :param collection_name: Name of mongoDB collection
        :param data_source: Field in document whose embedded doc shall be updated with data_update
        :param data_update: Dict to be inserted as embedded doc of field data_source
        """
        # Get corresponding date (needed for filtering) by taking the minimum of the timestamps found in the data
        # date = datetime.datetime.fromisoformat(min(data_update)).date()
        date = min(data_update).date()

        # Convert dictionary keys to string (BSON requirement)
        data_update = keys_to_string(data_update)

        # BSON does not support datetime.date objects -> use datetime.datetime object with date and midnight
        datetime_normalized = datetime.datetime.combine(date, datetime.time(0))

        logger.debug(f'Data update for {data_source}: {data_update}')
        match_filter = {'date': datetime_normalized}
        # new_doc =  match_filter.update({data_source: data})

        """
        TODO: remove delete operation after debugging
        """
        # self.db[collection_name].find_one_and_delete(filter=match_filter)
        #
        # logger.debug(f'Get whole doc (for {data_source} field)...')
        # whole_doc = self.db[collection_name].find_one(
        #     filter=match_filter
        # )
        # logger.debug(whole_doc)
        # logger.debug(f'Get embedded doc for {data_source}...')

        # Get doc or insert a new doc if there is no match
        doc = self.db[collection_name].find_one_and_update(
            filter=match_filter,
            # Fields to return:
            projection={data_source: True, '_id': False},
            # Insert complete doc (incl. date field) if none exists
            update={'$setOnInsert': {**match_filter, **{data_source: data_update}}},
            upsert=True,  # Insert new doc if no match
            # Return doc before any insertions (= None if it didn't exist before)
            return_document=ReturnDocument.BEFORE
        )
        if doc is not None:  # Update embedded doc
            embedded_doc = doc.get(data_source)
            try:
                for key, val in data_update.items():
                    # Apply values from data_update for each key=timestamp contained in data_update.
                    # Values for other timestamps are not changed.
                    embedded_doc[key] = val
            except TypeError:
                # Field 'data source' doesn't exist yet (therefore embedded_doc=None)
                embedded_doc = data_update

            # Insert updated embedded document
            result = self.db[collection_name].update_one(
                filter=match_filter,
                update={'$set': {data_source: embedded_doc}}
            )
        else:  # New doc was inserted, together with data_update as first entry
            pass
