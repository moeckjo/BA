import datetime
import logging
import os
import random

logging.basicConfig(level=logging.DEBUG)
logging.debug('Before importing from utils (in PVMAN')

from db import db_helper as db

from devicemanagement.utils import get_latest_forecast, get_latest_measurement, get_latest_schedule
from devicemanagement.models.pv_model import PVPlant
from devicemanagement.communication.pv_com import PVInverterConnector

PV_KEY = os.getenv('PV_KEY')
TOLERANCE = 5


# class PVObserverTask(Task):
#     # TODO Set name dynamically, to creat emultiple tasks in case there are multiple PV plants? Or even general ManagementTask that can be used with individual name for each device type?
#     autoregister = False # Must be registered manually
#     observer = None
#     name = None
#
#     @classmethod
#     def task_setup(cls, name, observer):
#         cls.name = name
#         cls.observer = observer
#
#     def run(self, *args, **kwargs):
#         print(f'I am the observer task {self.name} for observer {self.observer}')

class PVController:
    # TODO: If managament cycle time < scheduling computation time, then prevent initiating the rescheduling again (e.g. set flag (attribute)
    """
    Based on input from the observer and scheduler, determines and executes control actions.
    """


    def __init__(self, connector):
        self.connector = connector

    def action_in_case_of_deviation(self, target, actual):
        logging.debug('Do something quickly for PV!!!!')
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

    def control(self, connector):
        value = self.get_control_value()
        self.send_control_message(connector, value)

    def get_control_value(self):
        now = datetime.datetime.now(tz=datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
        value = get_latest_schedule(str(self.model), at_time=now)
        return value

    def send_control_message(self, connector, message):
        # connector.send(message)
        pass



class PVObserver:
    """
    Regularly requests current power from device, saves it to InfluxDB and compares it to scheduled value.
    Triggers actions to be executed by the controller in case of "large" deviations (tbd) or errors.
    """

    def __init__(self, connector):
        self.controller = None
        self.model = None
        self.connector = connector
        #
        # print(f'PVObserverTask observer before setting it: {PVObserverTask.observer}')
        # PVObserverTask.task_setup('observetask1', self)
        # print(f'PVObserverTask observer after setting it (still inside PVObserver init): {PVObserverTask.observer}')
        # PVObserverTask.task_setup('observetask2', self)

    def attach_to_controller(self, controller):
        self.controller = controller

    def attach_to_connector(self, connector):
        self.connector = connector

    def attach_to_model(self, model):
        logging.debug('attach PV model.')
        self.model = model

    def observe(self, connector: PVInverterConnector):
        ts, last_values = get_latest_measurement(source=str(self.model), fields='active_power', with_ts=True)
        return ts, last_values #'measurements'

    def record(self, timestamp: datetime.datetime, measurements: dict):
        print(f'Save {measurements} from {timestamp}')
        db.save_measurement(str(self.model), measurements, timestamp)

    def check(self, timestamp: datetime.datetime, measurements: dict, model: PVPlant):
        # TODO: Implement check
        print(f'Compare measurements of {str(model)} with forecast.')
        results = ['OK'] * 3 + ['WARNING']
        result = results[random.randint(0, len(results) - 1)]
        return result

    # @celery_app.task(name='devmgmt.compare_pv')
    def compare(self, current_power):
        logging.debug(f'Lets compare PV with observer ID: {id(self)}  address: {self.__str__}.')
        # Compare current output to predicted output
        # current_power = get_latest_measurement(source=str(self.model), fields='active_power')
        now = datetime.datetime.now(tz=datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
        logging.debug(f'Get forecast for now={now}')
        predicted_power = get_latest_forecast(str(self.model), at_time=now)
        logging.debug(f'Predicted power: {predicted_power}')

        if predicted_power - TOLERANCE <= current_power <= predicted_power + TOLERANCE:
            # Within bounds -> Perfect!
            logging.debug(f'Wihitn bounds!: predicted={predicted_power}, current={current_power}')
            pass
        else:
            # Actions to take if current output deviates from prediction by more than x W (TODO: x tbd)
            logging.debug(f'Out of bounds!: predicted={predicted_power}, current={current_power}')
            # celery_app.send_task(name='devmgmt.control_pv',
            # args=(self.controller, predicted_power, current_power))
