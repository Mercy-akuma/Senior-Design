# python3 run.py

import time
import subprocess
from numpy.random import random
import os


                
def main():
    dir_name = "./py/sd/image_process/" # change to global dir to run this code
    file_name = 'image_seg.py'
    
    os.chdir(dir_name)   #change dir
    os.getcwd()    #look up current dir
    
    subprocess.run(['python3',file_name])
    # p=subprocess.Popen(['python3',file_name])
    # time.sleep(20)
    # p.terminate()
    

if __name__ == "__main__":
    main()
    
    
