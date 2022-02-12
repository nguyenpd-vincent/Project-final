import RPi.GPIO as GPIO
import time

def Distance():
 

    try:
        GPIO.setmode(GPIO.BOARD)

        PIN_TRIGGER = 8
        PIN_ECHO = 7

        GPIO.setup(PIN_TRIGGER, GPIO.OUT)
        GPIO.setup(PIN_ECHO, GPIO.IN)

        GPIO.output(PIN_TRIGGER, GPIO.LOW)

        print ("Waiting for sensor to settle")

        time.sleep(2)

        print ("Calculating distance")
          
        GPIO.output(PIN_TRIGGER, GPIO.HIGH)

        time.sleep(0.00001)

        GPIO.output(PIN_TRIGGER, GPIO.LOW)

        while GPIO.input(PIN_ECHO)==0:
            pulse_start_time = time.time()
        while GPIO.input(PIN_ECHO)==1:
            pulse_end_time = time.time()

        pulse_duration = pulse_end_time - pulse_start_time
        distance = round(pulse_duration * 17150, 2)
        print ("Distance:",distance,"cm")

    finally:
        GPIO.cleanup()
        return Distance

if __name__ == '__main__':
    try:
        while True:
            dist = Distance()
            #print ("Measured Distance = %.1f cm" % dist)
            #print ("Distance:",distance,"cm")
            time.sleep(1)
 
        # Reset by pressing CTRL + C
    except KeyboardInterrupt:
        print("Measurement stopped by User")
    
