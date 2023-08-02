
import datetime
import os
import random

import pandas

from db import db_helper as db
#from device_management.celery_setup import celery_app
from devicemanagement.utils import get_latest_schedule, get_latest_measurement
from devicemanagement.models.heat_storage_model import HotWaterStorage
from devicemanagement.models.heat_pump_model import HeatPump
from devicemanagement.communication.hwt_com import HWTConnector
from devicemanagement.communication.hp_com import HeatPumpConnector

HP_NAME = os.getenv('HEAT_PUMP_KEY')
HWT_NAME = os.getenv('HEAT_STORAGE_KEY')

TOLERANCE_P = 5
TOLERANCE_TEMP = 0.05


class HeatPumpController:
    """
    Based on input from the observer and scheduler, determines and executes control actions.
    """

    def __init__(self, connector):
        self.connector = connector

    # @celery_app.task(name='devmgmt.control_hp')
    def action_in_case_of_deviation(self, target, actual):
        print('Do something quickly!!!!')
        pass

    def react(self, check_result, model):
        if check_result == 'OK':
            s = f'Everything ok with {str(model)}'
            adjust_schedules = False
        else:
            s = f'{str(model)} not ok'
            # TODO: Determine periods/window
            adjust_schedules_from = 'start time'
            adjust_schedules_to = 'end time'
            adjust_schedules = {'from': adjust_schedules_from, 'to': adjust_schedules_to}
        return s, adjust_schedules


class HeatPumpObserver:
    """
    Regularly requests current power from device, saves it to InfluxDB and compares it to scheduled value.
    Triggers actions to be executed by the controller in case of "large" deviations (tbd) or errors.
    """

    def __init__(self, connector):
        self.model = None
        self.controller = None
        self.connector = connector

    def attach_to_controller(self, controller):
        self.controller = controller

    def attach_to_connector(self, connector):
        self.connector = connector

    def attach_to_model(self, model):
        self.model = model


    # #@celery_app.task(name='devmgmt.observe_hp')
    # def observe(self):
    #     print('Lets observe!')
    #     measurements = self.connector.request_measurements() # dictionary
    #     timestamp = datetime.datetime.now()
    #     #celery_app.send_task(name='devmgmt.record_hp', args=(self, measurements, timestamp))
    #
    #     current_el_power = measurements['active_power']
    #     #current_th_power = el_to_th_power(get_cop(tu, to))
    #     #celery_app.send_task(name='devmgmt.compare_hp', args=(self, current_el_power))
    #
    # #@celery_app.task(name='devmgmt.record_hp')
    # def record(self, measurements, timestamp):
    #     print('Save' + measurements + 'from' + str(timestamp))
    #     db.save_measurement(HP_NAME, measurements, timestamp)

    def observe(self, connector: HeatPumpConnector):
        ts, last_values = get_latest_measurement(source=HP_NAME, fields='active_power', with_ts=True)
        return ts, last_values

    def record(self, timestamp: datetime.datetime, measurements: dict):
        print(f'Save {measurements} from {timestamp}')
        db.save_measurement(HP_NAME, measurements, timestamp)

    def check(self, timestamp: datetime.datetime, measurements: dict, model: HeatPump):
        print(f'Compare measurements of {str(model)} with forecast.')
        results = ['OK']*3 + ['WARNING']
        result = results[random.randint(0,len(results)-1)]
        return result

    #@celery_app.task(name='devmgmt.compare_hp')
    def compare(self, current_el_power):
        # TODO: Determine what to compare, i.e., decisive factors, and call controller task
        print('Lets compare!')
        now = datetime.datetime.now()
        schedule_whole_day = get_latest_schedule(HP_NAME)
        resolution = pandas.Series(schedule_whole_day.index).diff()[1]  # datetime.timedelta
        scheduled_power = schedule_whole_day[
            (schedule_whole_day.index.time > (now - resolution).time()) & (schedule_whole_day.index.time <= now.time())
            ]
        if (scheduled_power - TOLERANCE_P <= current_el_power) and (current_el_power >= scheduled_power + TOLERANCE_P) :
            # Within bounds -> Perfect!
            print('Wihitn bounds!:', 'scheduled=', scheduled_power, 'current=', current_el_power)
            pass
        else:
            # Actions to take if current output deviates from prediction by more than x W (TODO: x tbd)
            #celery_app.send_task(name='devmgmt.control_hp', args=(self.controller, scheduled_power, current_el_power))
            pass


class HeatStorageController:
    """
    Based on input from the observer and scheduler, determines and executes control actions.
    """

    def __init__(self, connector):
        self.connector = connector

    def react(self, check_result, model):
        if check_result == 'OK':
            s = f'Everything ok with {str(model)}'
            adjust_schedules = False
        else:
            s = f'{str(model)} not ok'
            # TODO: Determine periods/window
            adjust_schedules_from = 'start time'
            adjust_schedules_to = 'end time'
            adjust_schedules = {'from': adjust_schedules_from, 'to': adjust_schedules_to}
        return s, adjust_schedules


class HeatStorageObserver:
    """
    Regularly requests current power from device, saves it to InfluxDB and compares it to scheduled value.
    Triggers actions to be executed by the controller in case of "large" deviations (tbd) or errors.
    """

    def __init__(self, connector):
        self.model = None
        self.controller = None
        self.connector = connector

    def attach_to_controller(self, controller):
        self.controller = controller

    def attach_to_connector(self, connector):
        self.connector = connector

    def attach_to_model(self, model):
        self.model = model

    #@celery_app.task(name='devmgmt.observe_hwt')
    # def observe(self):
    #     print('Lets observe!')
    #     measurements = self.connector.request_measurements() # dictionary
    #     timestamp = datetime.datetime.now()
    #     #celery_app.send_task(name='devmgmt.record_hwt', args=(self, measurements, timestamp))
    #
    #     current_temp = measurements['temp']
    #     #celery_app.send_task(name='devmgmt.compare_hwt', args=(self, current_temp))

    def observe(self, connector: HWTConnector):
        ts, last_values = get_latest_measurement(source=HWT_NAME, fields='active_power', with_ts=True)
        return ts, last_values

    def record(self, timestamp: datetime.datetime, measurements: dict):
        print(f'Save {measurements} from {timestamp}')
        db.save_measurement(HWT_NAME, measurements, timestamp)

    def check(self, timestamp: datetime.datetime, measurements: dict, model: HotWaterStorage):
        print(f'Compare measurements of {str(model)} with forecast.')
        results = ['OK']*3 + ['WARNING']
        result = results[random.randint(0,len(results)-1)]
        return result

    # # @celery_app.task(name='devmgmt.record_hwt')
    # def record(self, measurements, timestamp):
    #     print('Save' + measurements + 'from' + str(timestamp))
    #     db.save_measurement(HP_NAME, measurements, timestamp)

    #@celery_app.task(name='devmgmt.compare_hwt')
    def compare(self, current_temp):
        # TODO: Determine what to compare, i.e., decisive factors, and call controller task
        print('Lets compare!')
        now = datetime.datetime.now()
        schedule_whole_day = get_latest_schedule(HP_NAME)
        resolution = pandas.Series(schedule_whole_day.index).diff()[1]  # datetime.timedelta
        scheduled_temp = schedule_whole_day[
            (schedule_whole_day.index.time <= now) & (schedule_whole_day.index.time + resolution > now)
            ]
        if (scheduled_temp - TOLERANCE_TEMP <= current_temp) and (current_temp >= scheduled_temp + TOLERANCE_TEMP) :
            # Within bounds -> Perfect!
            print('Wihitn bounds!:', 'scheduled=', scheduled_temp, 'current=', current_temp)
            pass
        else:
            # Actions to take if current output deviates from prediction by more than x W (TODO: x tbd)
            #celery_app.send_task(name='devmgmt.control_hp', args=(self.controller, scheduled_temp, current_temp))
            pass
