# Load device models from DB
# Register converter (factory functions) for each device
# Set state of device models
# Add all constraints to MILP with milp.add_constraints -> this takes the instance of the device model as input
# Create objective
# Get and process solution (or maybe not here or renmae this module)
import json
import os
import importlib
import datetime
import random
import time

import pandas as pd
import pika
import pytz
import pyomo.environ as pyo
import typing

import db.db_helper as db
from db.mongodb_handler import MongoConnector

from optimization.scheduler import Scheduler, GeneralScheduler
from core import utils
from core import tasks


def publish_quotas_to_ems(quota_information: typing.Dict[str, dict], quota_category: str):
    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()

    quotas_payload = json.dumps(quota_information)
    print(f'Publish {quota_category} quotas. payload={quotas_payload}.')
    channel.basic_publish(exchange=os.getenv('RABBITMQ_BEM_INBOX'), routing_key=f'quotas.{quota_category}',
                          body=quotas_payload)
    connection.close()


def get_gcp_schedule(schedules: dict, save_as_ref: bool = False):
    gcp_schedule = schedules[os.getenv('GRID_CONNECTION_POINT_KEY')]
    # print(f'GCP schedule: {gcp_schedule}')

    if save_as_ref:
        print(f'Save GCP schedule as reference schedule.')
        db.save_data_to_db(
            db=os.getenv('MONGO_SCHEDULE_DB_NAME'),
            data_source=f'{os.getenv("GRID_CONNECTION_POINT_KEY")}_reference',
            time_series_data=gcp_schedule,
            group_by_date=True,
            persist_old_data=False
        )
    return gcp_schedule


def create_quotas(qcategory, window_start, grid_limits_by_hour, publish=True):
    time.sleep(5)

    quotas = {'start': window_start.isoformat()}
    for t, value in grid_limits_by_hour.items():
        if t < window_start + datetime.timedelta(hours=int(os.getenv('QUOTA_WINDOW_SIZE'))):
            # if random.random() < (0.6 if qcategory == 'primary' else 1.0):
            if value is not None:
                qtype = 'feedin' if value <= 0 else 'consume'
            else:
                qtype = None
            quotas[t.isoformat()] = dict(type=qtype, abs_power_limit=value)
        else:
            break
    if publish:
        print(f'Now publish {qcategory} quotas to trigger re-scheduling.')
        time.sleep(5)
        publish_quotas_to_ems(quotas, qcategory)
        # print('Publish again')

        # time.sleep(10)
        # publish_quotas_to_ems(quotas, qcat)

    if qcategory == 'final':
        quotas.pop('start')
        db.save_data_to_db(
            db=os.getenv('MONGO_QUOTA_DB_NAME'),
            data_source=os.getenv(f'MONGO_{qcategory.upper()}_QUOTAS_COLL_NAME'),
            time_series_data=quotas,
            group_by_date=True,
            persist_old_data=False
        )
    return quotas


def resample_limits_to_timestamps(grid_limits_by_hour: dict, timestamps: pd.DatetimeIndex):
    value = None
    for ts in timestamps:
        if ts.hour in grid_limits_by_hour.keys():
            value = grid_limits_by_hour[ts.hour]
            grid_limits_by_hour[ts] = value
            grid_limits_by_hour.pop(ts.hour)
        else:
            grid_limits_by_hour[ts] = value
    return grid_limits_by_hour


def run(call_task=True, test_opt_with_primary_quotas: bool = False):

    init_states = None
    forecasts = None
    user_input = None
    grid_quotas_category = None
    grid_quotas = None
    grid_limits_by_hour = None

    """
    Define setting here
    
    #Define window start
    window_start = datetime.datetime(2022, 9, 5, 10, 0, 0)
    utc_timezone = pytz.UTC
    localized_timestamp = utc_timezone.localize(window_start)
    """

    date = datetime.date (2023,7,20)
    window_start = datetime.datetime.combine(date, datetime.time(10), tzinfo=pytz.utc)
    #window_start = datetime.datetime.combine(datetime.date.today(), datetime.time(10), tzinfo=pytz.utc)
    #date = datetime.date (2022,11,13) 
    #window_start = datetime.datetime.combine(datetime.date (2023,3,28), datetime.time(10), tzinfo=pytz.utc)

    # Define grid limits starting at specific hours and being valid until the next defined hour
    # Example:
    #grid_limits_by_hour = {0: 1000, 4: 8000, 8: 5000}
    #grid_limits_by_hour = {window_start.hour: -1000, 4:2000, 8: 8000}
    #   -> This means that there's no limit from 0 to 4 a.m., then a limit of 8000 W from 4 to 8 a.m. and
    #       afterwards a limit of -10000W until the end of the optimization horizon.
    #
    # Note: If setting grid limits to use them as quotas, it's sufficient to define limits until 6 hours
    #       after window start. Any limits after that time are not considered as quotas anyways.

    grid_limits_by_hour = {window_start.hour: -3000, 14: 7000, 15: 6000}
    #grid_limits_by_hour = {window_start.hour: -5000}
    

    """
    Don't change anything below here
    """

    window_size = datetime.timedelta(hours=24)
    resolution = int(os.getenv('SCHEDULER_TEMP_RESOLUTION_SEC'))
    timestamps = pd.date_range(start=window_start, end=window_start + window_size, freq=f'{resolution}S',
                               closed='left', tz=pytz.utc)

    print(f'Timestamps: {timestamps}')

    if test_opt_with_primary_quotas:
        quota_resolution = int(datetime.timedelta(hours=float(os.getenv('QUOTA_TEMP_RESOLUTION'))).total_seconds())
        quota_timestamps = pd.date_range(start=window_start,
                                         end=window_start + datetime.timedelta(
                                             hours=int(os.getenv('QUOTA_WINDOW_SIZE'))),
                                         freq=f'{quota_resolution}S',
                                         closed='left', tz=pytz.utc)

        grid_quotas_category = 'primary'
        grid_quotas: dict = resample_limits_to_timestamps(grid_limits_by_hour, quota_timestamps)
        grid_quotas = {t.isoformat(): v for t,v in grid_quotas.items()}

        # grid_quotas = create_quotas(qcategory=grid_quotas_category,
        #                             window_start=window_start,
        #                             grid_limits_by_hour=grid_limits,
        #                             publish=True)


    if call_task:
        # Scheduling task call
        tasks.scheduling(
            window_start=window_start,
            window_size=window_size,
            update_forecasts='all',
            grid_quotas_category=grid_quotas_category,
            grid_quotas=grid_quotas,
            user_input=user_input,
            init_states=init_states,
        )

    else:
        # Scheduler object method call

        start_timer = time.time()
        scheduler = GeneralScheduler(
            window_start=window_start,
            window_size=window_size,
            forecasts=forecasts,
            init_states=init_states,
            grid_quotas_category=grid_quotas_category,
            grid_quotas=grid_quotas,
        )

        schedules: typing.Dict[str, typing.Dict[str, int]] = scheduler.schedule()
        print(f'Scheduling with resolution {resolution}s took {time.time() - start_timer}s')
        # print(f'Power schedules: {schedules}')

        gcp_schedule = get_gcp_schedule(schedules, save_as_ref=True)