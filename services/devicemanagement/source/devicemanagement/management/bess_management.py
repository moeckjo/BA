
import datetime
import os

#from device_management.celery_setup import celery_app
import logging
import random

logging.basicConfig(level=logging.INFO)
logging.debug('Before importing from utils (in BESSMAN')

from db import db_helper as db

from devicemanagement.utils import get_latest_measurement
from devicemanagement.models.bess_model import BatteryStorage
from devicemanagement.communication.bess_com import BESSConnector

BESS_KEY = os.getenv('BESS_KEY')
TOLERANCE_P = 5
TOLERANCE_SOC = 0.05

class BESSController:
    """
    Based on input from the observer and scheduler, determines and executes control actions.
    """

    # def __init__(self, observer):
    #     self.observer = observer

    def __init__(self, connector):
        self.connector = connector

    def attach_to_connector(self, connector):
        self.connector = connector

    #@celery_app.task(name='devmgmt.control_bess')
    def action_in_case_of_deviation(self, target, actual):
        print('Do something quickly for BESS!!!!')
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

class BESSObserver:
    # TODO: Include requesting of measurements from device (comm. via driver) and storing in InfluxDB
    """
    Regularly requests current power from device, saves it to InfluxDB and compares it to scheduled value.
    Triggers actions to be executed by the controller in case of "large" deviations (tbd) or errors.
    """

    def __init__(self, connector):
        self.model = None
        self.controller = None #BESSController(self)
        self.connector = connector #BESSConnector()

    def attach_to_controller(self, controller):
        self.controller = controller

    def attach_to_connector(self, connector):
        self.connector = connector

    def attach_to_model(self, model):
        print('attach BESS model.')

        self.model = model

    #@celery_app.task(name='devmgmt.observe_bess')
    # def observe(self):
    #     print('Lets observe BESS!')
    #     print(f'My {self.model.category} (=BESS?) (observer) ID: {id(self)}')
    #
    #     # TESTING
    #     measurements = 'BESS measurement' #self.connector.request_measurements() # dictionary
    #     timestamp = datetime.datetime.now()
    #     #celery_app.send_task(name='devmgmt.record_bess', args=(self, measurements, timestamp))
    #
    #     # current_charge_power = measurements['c_power']
    #     # current_discharge_power = measurements['d_power']
    #     # current_soc = measurements['soc']
    #
    #     # TESTING
    #     current_charge_power = get_latest_measurement(str(self.model), 'c_power')
    #     current_discharge_power = get_latest_measurement(str(self.model), 'd_power')
    #     current_soc = get_latest_measurement(str(self.model), 'soc')
    #     #celery_app.send_task(name='devmgmt.compare_bess', args=(self, current_charge_power, current_discharge_power, current_soc))


    def observe(self, connector: BESSConnector):
        print(f'Observe operation of BESS')
        timestamp = datetime.datetime.now()
        ts, last_values = get_latest_measurement(source=str(self.model), fields='active_power', with_ts=True)
        #ts, last_value = get_latest_measurement(source=str(self.model), fields=['c_power', 'd_power', 'soc'], with_ts=True)
        return ts, last_values

    def record(self, timestamp: datetime.datetime, measurements: dict):
        print(f'Save {measurements} from {timestamp}')
        db.save_measurement(str(self.model), measurements, timestamp)

    def check(self, timestamp: datetime.datetime, measurements: dict, model: BatteryStorage):
        print(f'Compare measurements of {str(model)} with forecast.')
        results = ['OK']*3 + ['WARNING']
        result = results[random.randint(0,len(results)-1)]
        return result

    #@celery_app.task(name='devmgmt.compare_bess')
    def compare(self, current_charge_power, current_discharge_power, current_soc):
        # TODO: Determine what to compare, i.e., decisive factors, and call controller task
        print('Lets compare BESS!')
        now = datetime.datetime.now()
        # TESTING
        # schedule_whole_day = get_latest_schedule(str(self.model))
        # resolution = pandas.Series(schedule_whole_day.index).diff()[1]  # datetime.timedelta
        # scheduled_power = schedule_whole_day[
        #     (schedule_whole_day.index.time > (now - resolution).time()) & (schedule_whole_day.index.time <= now.time())
        #     ]
        scheduled_power = 3289
        if scheduled_power >= 0 and (scheduled_power - TOLERANCE_P <= current_charge_power <= scheduled_power + TOLERANCE_P):
            print('Charging within bounds.')
        elif scheduled_power < 0 and (scheduled_power - TOLERANCE_P <= current_discharge_power <= scheduled_power + TOLERANCE_P):
            print('Discharging within bounds.')
        else:
            #celery_app.send_task(name='devmgmt.control_bess', args=(self.controller, scheduled_power, current_charge_power))
            pass



