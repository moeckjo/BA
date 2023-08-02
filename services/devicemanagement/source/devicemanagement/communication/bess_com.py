"""
Driver for the following battery storages:
    - ...
"""
import datetime

import typing


class BESSConnector:

    def __init__(self):
        #self.client = pymodbus.ModbusClient(ip, port)
        self.client = object
        pass

    def connect(self, ip, port):
        # Connect to device
        pass

    def request_measurements(self) -> typing.Tuple[datetime.datetime, dict]:
        # data = self.client.read()

        ### For testing -> TODO: remove when real data is available
        timestamp = datetime.datetime.now()
        measurements = {'active_power': -3000.0}#, 'soc': 0.5}
        data = (timestamp, measurements)
        ### End of testing code
        print(f'Return data received from BESS: {measurements}')
        return data
