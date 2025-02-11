import serial
import time

arduino = serial.Serial('COM8', 9600)
time.sleep(2)

print("Listening for data from Arduino...")

while True:
    if arduino.in_waiting > 0:
        data = arduino.readline().decode('utf-8').strip()
        print(f"Received: {data}")
    
    user_input = input("Press 'q' to quit or any other key to continue: ")
    if user_input.lower() == 'q':
        print("Exiting program.")
        break

arduino.close()  # Close the serial connection