#!/usr/bin/env python
import os
import sys

from pymodbus.client.sync import ModbusTcpClient
from pymodbus.exceptions import ConnectionException, ModbusIOException
from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder, Endian

import time

HOST = os.getenv('MODBUS_SERVER_HOSTNAME')
PORT = os.getenv('MODBUS_SERVER_PORT')
COUNT = 2  # Number of registers: 1 for 16 Bit and 2 for 32 Bit
register_address = 100
# Byte order (for values >16bit spanning multiple registers)
byte_order = Endian.Little

# try:
#     time.sleep(120)
# except KeyboardInterrupt:
#     sys.exit()

def encode_signed_int(value: int, nbytes: int, byte_order=byte_order) -> list:
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


def decode_signed_int(register_content: list, nbytes: int, byte_order=byte_order) -> int:
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
        print('Only decoding of 16 bit or 32 bit signed integer implemented. Returned value might incorrect.')
        value: int = register_content[0]
    return value

def connect(retry=False):
    if retry:
        print(f'Connection lost. Trying to reconnect.')
        attempts = 0
        while not client.is_socket_open() and attempts < 10:
            client.connect()
            attempts += 1
            time.sleep(10)
        print(
            'Server down for more than 60 seconds. Stopping trying to reconnect.' if not client.is_socket_open()
            else "Server is up again.")
    else:
        print(f'Connecting to {HOST}...')
        connected = client.connect()
    print(f'Connection {"" if client.is_socket_open() else "could not be"} established.')
    if not client.is_socket_open():
        connect(True)


def close():
    print(f'Close connection to server {HOST}.')
    client.close()


client = ModbusTcpClient(host=HOST, port=PORT)
connect()

try:
    print(f'Setpoint clear value={int(os.getenv("GRID_SETPOINT_CLEAR_VALUE"))}')
    values = [11000, 599, 0, -100, -67006, 11000, -600, 0, 5000, int(os.getenv('GRID_SETPOINT_CLEAR_VALUE'))]
    # values = [11000, 0, -67006, -600, int(os.getenv('GRID_SETPOINT_CLEAR_VALUE'))]
    for val in values:
        value_encoded = encode_signed_int(val, nbytes=4)
        print(f'Write {val} W (encoded 32bit -> {value_encoded}) to 2 registers.')
        w = client.write_registers(register_address, value_encoded)
        time.sleep(0.5)
        print('Read same registers and assert equality.')
        r = client.read_holding_registers(register_address, count=COUNT)
        r_value = decode_signed_int(r.registers, COUNT * 2)
        assert val == r_value
        time.sleep(10)

except (ConnectionException, ModbusIOException, AttributeError) as e:
    print(f'Exception: {e}')
    connect(retry=True)
finally:
    close()
