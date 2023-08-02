from pymodbus.client.sync import ModbusSerialClient

# Config from Delta Protocol Definition V1.18, first pag (General information)
# Port is probably /dev/ttyUSB0, but need to check.
client = ModbusSerialClient(method='rtu', port='/dev/ttyUSB0', baudrate=19200, parity='N', bytesize=8, stopbits=1)

connected = client.connect()
print(f'connected to {client}?: {connected}')

read = client.read_holding_registers(address=1, count=10, unit=1)
data = read.registers
print(data)
