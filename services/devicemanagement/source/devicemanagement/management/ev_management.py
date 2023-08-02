
import datetime
import os
import random

import db.db_helper as db
#from device_management.celery_setup import celery_app
from devicemanagement.utils import get_latest_forecast, get_latest_schedule, get_latest_measurement
from devicemanagement.models.evse_model import EVSE
from devicemanagement.communication.ev_com import WallboxConnector

EVSE_KEY = os.getenv('EVSE_KEY')
TOLERANCE_P = 5
TOLERANCE_SOC = 0.05


class EVController:
    """
    Based on input from the observer and scheduler, determines and executes control actions.
    """

    def __init__(self, connector):
        self.connector = connector

    # @celery_app.task(name='devmgmt.reschedule_ev')
    def action_if_unexpectedly_unconnected(self):
        print('Reschedule quickly!!!!')
        pass

    # @celery_app.task(name='devmgmt.control_ev')
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


class EVObserver:
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

    # #@celery_app.task(name='devmgmt.observe_ev')
    # def observe(self):
    #     print('Lets observe!')
    #     measurements = self.connector.request_measurements() # dictionary
    #     timestamp = datetime.datetime.now()
    #     #celery_app.send_task(name='devmgmt.record_ev', args=(self, measurements, timestamp))
    #
    #     connection_status = measurements['connected']
    #     if connection_status is True:
    #         current_charge_power = measurements['c_power']
    #         current_discharge_power = measurements['d_power']
    #         current_soc = measurements['soc']
    #         current_driving_range = measurements['driving_range']
    #
    #         #celery_app.send_task(name='devmgmt.compare_ev', args=(self, connection_status, current_charge_power, current_discharge_power, current_soc))
    #     else:
    #         #celery_app.send_task(name='devmgmt.compare_ev', args=(self, connection_status))
    #         pass
    #
    # #@celery_app.task(name='devmgmt.record_ev')
    # def record(self, measurements, timestamp):
    #     print('Save' + measurements + 'from' + str(timestamp))
    #     db.save_measurement(EVSE_KEY, measurements, timestamp)

    def observe(self, connector: WallboxConnector):
        ts, last_values = get_latest_measurement(source=str(self.model), fields='active_power', with_ts=True)
        return ts, last_values #'measurements'

    def record(self, timestamp: datetime.datetime, measurements: dict):
        print(f'Save {measurements} from {timestamp}')
        db.save_measurement(str(self.model), measurements, timestamp)

    def check(self, timestamp: datetime.datetime, measurements: dict, model: EVSE):
        print(f'Compare measurements of {str(model)} with forecast.')
        results = ['OK'] * 3 + ['WARNING']
        result = results[random.randint(0, len(results) - 1)]
        return result

    #@celery_app.task(name='devmgmt.compare_ev')
    def compare(self, connected: bool, current_charge_power=None, current_discharge_power=None, current_soc=None):
        # TODO: Determine what to compare, i.e., decisive factors, and call controller task
        print('Lets compare!')
        now = datetime.datetime.now(tz=datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
        if not connected:
            predicted_connection_status = get_latest_forecast(str(self.model), at_time=now)
            if predicted_connection_status != connected:
                #celery_app.send_task(name='devmgmt.reschedule_ev', args=(self.controller,))
                pass
        else:
            scheduled_power = get_latest_schedule(str(self.model), at_time=now)
            if True:
                #celery_app.send_task(name='devmgmt.control_ev', args=(self.controller, scheduled_power, current_charge_power))
                pass


