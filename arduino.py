import serial
import time
# from main import key
def control_motor():
    serialPort = "COM3"  # serial
    baudRate = 9600  
    ser = serial.Serial(serialPort, baudRate)#, timeout=0.5)

    demo1=b"w"
    demo2=b"s"
    demo3=b'a'
    demo4=b'd'
    demo5=b'p'
    demo6=b'u'
    demo7=b'i'

    while 1:
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
            time.sleep(0.7)
            ser.write(demo7)
            time.sleep(0.7)
            ser.write(demo5)