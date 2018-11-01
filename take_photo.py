import time
import datetime
import picamera
import RPi.GPIO as GPIO

LIGHT_PIN = 7

GPIO.setmode(GPIO.BOARD)
GPIO.setup(LIGHT_PIN, GPIO.OUT)

GPIO.output(LIGHT_PIN, True)

filename = 'img/image-{}.jpg'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

with picamera.PiCamera() as camera:
    camera.resolution = (1024, 768)
    camera.start_preview()
    # Camera warm-up time
    time.sleep(2)
    camera.capture(filename)

GPIO.output(LIGHT_PIN, False)
 
