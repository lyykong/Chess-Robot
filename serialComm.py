import serial
import time

# Setting up arduino serial connection
arduino = serial.Serial(port='COM8', baudrate=9600, timeout=1)
time.sleep(2)

arduino.write(b'Test Message\n')
response = arduino.readline().decode('utf-8').strip()
print("Received from Arduino:", response)

arduino.close()