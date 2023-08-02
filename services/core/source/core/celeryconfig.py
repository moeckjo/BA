import os
from celery.schedules import crontab
from datetime import timedelta, datetime


# Task settings
task_serializer = 'pickle'
accept_content = ['json', 'pickle']
task_routes = {
    'core.*': {'queue': 'core'},
    # TODO: Separate queues for reading meter/sensor data and sending power values to devices, respectively
}

worker_send_task_events = True
