import serial
import time
# import pygame
# from main import key
def control_motor():
    serialPort = "COM3"  # serial
    baudRate = 9600  
    ser = serial.Serial(serialPort, baudRate)#, timeout=0.5)
    # pygame.init()
    # screen = pygame.display.set_mode((640, 480))
    # pygame.display.set_caption("Hello World")

    demo1=b"w"
    demo2=b"s"
    demo3=b'a'
    demo4=b'd'
    demo5=b'p'
    demo6=b'u'
    demo7=b'i'

    while 1:
        # keys_pressed = pygame.key.get_pressed()
        # if keys_pressed[pygame.K_w]:
        #     print('w')
        #     ser.write(demo1)
        # elif keys_pressed[pygame.K_s]:
        #     print('s')
        #     ser.write(demo2)
        # elif keys_pressed[pygame.K_a]:
        #     print('a')
        #     ser.write(demo3)
        # elif keys_pressed[pygame.K_d]:
        #     print('d')
        #     ser.write(demo4)
        # elif keys_pressed[pygame.K_u]:

        #     ser.write(demo6)
        #     time.sleep(0.7)
        #     ser.write(demo7)
        #     time.sleep(0.7)
        #     ser.write(demo5)
        # else:
        #     print('p')
        #     ser.write(demo5)
        c=input('order:')
        if c == "":
            continue
        if c == "q":
            break
        c=ord(c)#transfer c to ASCII
        if(c==119):
            ser.write(demo1)#ser.write write data
            time.sleep(0.7)
            ser.write(demo5)
        if(c==115):
            ser.write(demo2)
            time.sleep(0.7)
            ser.write(demo5)
        if(c==97):
            ser.write(demo3)
            time.sleep(0.7)
            ser.write(demo5)
        if(c==100):
            ser.write(demo4)
            time.sleep(0.7)
            ser.write(demo5)
        if(c==117):
            ser.write(demo6)
            time.sleep(1)
            ser.write(demo7)
            time.sleep(0.7)
            ser.write(demo5)