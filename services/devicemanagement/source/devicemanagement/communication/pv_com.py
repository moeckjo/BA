"""
Driver for the following PV inverters:
    - ...
"""
import datetime
import typing


class PVInverterConnector:
    # TODO: [for all devices] Create connector task to keep connection open instead of connecting and disconnecting every (few) seconds

    def __init__(self):
        # self.client = pymodbus.ModbusClient(ip, port)
        self.client = object
        pass

    def connect(self, ip, port):
        # Connect to device
        pass

    def request_measurements(self) -> typing.Tuple[datetime.datetime, dict]:
        # data = self.client.read()

        ### For testing -> TODO: remove when real data is available
        timestamp = datetime.datetime.now()
        measurements = {'active_power': -9999.0}
        data = (timestamp, measurements)
        ### End of testing code
        print(f'Return data received from PV: {measurements}')

        return data

    def send(self, message):
        # self.client.write(message)
        pass
