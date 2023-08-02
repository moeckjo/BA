#!/usr/bin/env python

import datetime
import json
import logging
import os
import sys
from queue import Queue, Empty
from threading import Thread

import pika
import typing
from pymodbus.server.asynchronous import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSparseDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder, Endian

import db.db_helper as db
from db.influxdb_handler import InfluxConnector

logger = logging.getLogger('bem-modbus-server')
logformat = "[%(asctime)s - %(name)s - %(funcName)s - %(levelname)s]:  %(message)s"
loglevel = logging.DEBUG if os.getenv("MODBUS_SERVER_DEBUG").lower() == "true" else logging.INFO
logging.basicConfig(stream=sys.stdout, level=loglevel, format=logformat)
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("pymodbus").setLevel(logging.INFO)


def hello():
    logger.info('Hello from modbus server!')

# ----------------------------------------------------------------------- #
# Server information
# ----------------------------------------------------------------------- #

PORT = int(os.getenv('MODBUS_SERVER_PORT'))

identity = ModbusDeviceIdentification()
identity.VendorName = 'FZI'
identity.ProductCode = 'FZI-GEMS-MS'
identity.VendorUrl = 'https://fzi.de'
identity.ProductName = 'GEMS Modbus Server'
identity.ModelName = 'gems-modbus-server-01'
identity.MajorMinorRevision = '1.0.0'

# ----------------------------------------------------------------------- #
# Register specification
# ----------------------------------------------------------------------- #

# Coils
# Discrete Inputs
# Input Registers
# Holding Registers
holding_registers = {
    'active power setpoint': {'address': 100, 'bytes': 4, 'signed': True, 'init': 30000},
}

# Byte order (for values >16bit spanning multiple registers)
BYTE_ORDER = Endian.Little


# ----------------------------------------------------------------------- #
# Helper functions
# ----------------------------------------------------------------------- #


def publish_setpoint_to_ems(value: int, timestamp: datetime.datetime):
    """
    Publish the GCP setpoint to the message exchange to be received by other BEM services
    :param value: The signed power value setpoint [W]
    :param timestamp: Timestamp of reception of the setpoint
    """
    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()

    payload = json.dumps({
        'timestamp': timestamp.isoformat(),
        'value': value
    })
    logger.debug(f'Publish GCP setpoint: {payload}.')
    try:
        channel.basic_publish(exchange=os.getenv('RABBITMQ_GRID_SETPOINT'),
                              routing_key="",
                              body=payload)
    finally:
        connection.close()


def save_and_publish_setpoint(value: int, timestamp: datetime.datetime):
    """
    1. Save setpoint to database
    2. Check if the value corresponds to the defined value that clears the red phase (and does not set a new setpoint),
        if yes, set to None
    3. Call publish function with setpoint value
    :param value: Signed setpoint value
    :param timestamp: Timestamp of setpoint reception
    """
    data = {timestamp.isoformat(): {'setpoint': value}}
    if os.getenv('MODBUS_SERVER_DEBUG').lower() == 'true':
        # Get success status returned
        influx_connector = InfluxConnector()
        success = influx_connector.write_datapoints(os.getenv('GRID_CONNECTION_POINT_KEY'), data)
        influx_connector.close()
        logger.debug(f'Successful writing to DB? -> {success}')
        # writtendata = db.get_measurement(source=os.getenv('GRID_CONNECTION_POINT_KEY'), fields='setpoint',
        #                                     start_time=datetime.datetime.now(tz=datetime.timezone.utc)-datetime.timedelta(minutes=5),
        #                                     )
        # print(f'From db: {writtendata}')
    else:
        db.save_measurement(
            source=os.getenv('GRID_CONNECTION_POINT_KEY'),
            data=data
        )
    logger.debug(f'Setpoint data written to DB: {data}')

    if value == int(os.getenv('GRID_SETPOINT_CLEAR_VALUE')):
        # Red phase is being revoked -> no new setpoint
        value = None

    logger.debug(f'Setpoint is {"cleared" if value is None else f"set to {value}"}.')
    publish_setpoint_to_ems(value, timestamp)


def encode_signed_int(value: int, nbytes: int, byte_order=BYTE_ORDER) -> list:
    """
    Transform decimal integer to signed 16bit oder 32bit register entry/entries
    :param value: Decimal value (positive or negative)
    :param nbytes: Number of bytes determines representation -> 2 for 16bit, 4 for 32bit
    :param byte_order: Endianness (Endian.Big or Endian.Little)
    :return: Register entry (list with one (16bit) or two (32bit) elements)
    """
    builder = BinaryPayloadBuilder(byteorder=byte_order)
    if nbytes == 2:
        builder.add_16bit_int(value)
    elif nbytes == 4:
        builder.add_32bit_int(value)
    return builder.to_registers()


def decode_signed_int(register_content: list, nbytes: int, byte_order: Endian = BYTE_ORDER) -> int:
    """
    Transform signed 16bit oder 32bit register entry/entries to decimal integer
    :param register_content: List with one (16bit int) or two (32bit int) elements
    :param nbytes: Number of bytes determines representation -> 2 for 16bit, 4 for 32bit
    :param byte_order: Endianness (Endian.Big or Endian.Little)
    :return: Decimal value of register entry
    """
    decoder = BinaryPayloadDecoder.fromRegisters(register_content, byteorder=byte_order)
    if nbytes == 2:
        value: int = decoder.decode_16bit_int()
    elif nbytes == 4:
        value: int = decoder.decode_32bit_int()
    else:
        logger.warning('Only decoding of 16 bit or 32 bit signed integer implemented. Returned value might incorrect.')
        value: int = register_content[0]
    return value


# --------------------------------------------------------------------------- #
# Storing and callback process
# --------------------------------------------------------------------------- #

class CallbackDataBlock(ModbusSparseDataBlock):
    """
    A datablock that calls a function for further processing of values written to registers by a Modbus client.
    """

    def __init__(self, init_values: dict):
        """
        :param init_values: Initial register values at datablock initialization; dict with <address, register entry>-pair
        """
        logger.debug(f'Init values: {init_values}')
        store = {}
        for start_address, value in init_values.items():
            # Allocate values > 16bit to subsequent addresses, starting with the given address
            for i in range(len(value)):
                store[start_address + i] = value[i]

        super(CallbackDataBlock, self).__init__(store)
        self.callback = process_values

    def setValues(self, address: int, values: list):
        """
        Sets the received values in the datastore, then calls a function for further processing
        :param address: The starting address
        :param values: The value (register entry) to be set
        """
        super(CallbackDataBlock, self).setValues(address, values)
        logger.debug(f'Values written: {values} to {address} (starting address)')
        self.callback(address, values)


def process_values(address: int, values: list):
    """
    Decode values written to register and call save and publish function
    :param address: Address of register written to
    :param values: (Encoded) Values written to this register
    """
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)
    decoded_value = decode_signed_int(values, nbytes=len(values) * 2)
    logger.info(f'Received value={decoded_value} ({values}) for register {address}')

    assert address == holding_registers['active power setpoint']['address']
    save_and_publish_setpoint(decoded_value, timestamp)


def run_modbus_server():
    """
    Initialize Modbus server with slave and datastore,
    and start the server and further handling processes.
    """
    # Initialize slave with store and the single holding register
    logger.info('Set up server with single slave and datastore...')
    register = holding_registers['active power setpoint']
    hr_block = CallbackDataBlock(
        init_values={register['address']: encode_signed_int(register['init'], nbytes=register['bytes'])},
    )
    # Create single-register block for the unused object types to save space (default creates block with max. address
    # space)
    unused_block = ModbusSparseDataBlock([0])
    slave_context = ModbusSlaveContext(
        hr=hr_block, di=unused_block, co=unused_block, ir=unused_block,
        zero_mode=True  # start address counting at 0
    )
    server_context = ModbusServerContext(slaves=slave_context, single=True)

    StartTcpServer(server_context, identity=identity, address=(os.getenv('MODBUS_SERVER_HOSTNAME'), PORT))


if __name__ == "__main__":
    hello()
    run_modbus_server()  # Runs until interrupted

    logger.info('Server stopped.')
