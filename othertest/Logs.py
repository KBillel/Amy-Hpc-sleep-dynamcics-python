import logging as log
import os 
FORMAT = "%(asctime)-15s %(levelname)s %(message)s"
log.basicConfig(filename='blablabla.log',format=FORMAT,level=log.INFO,datefmt='%Y-%m-%d %H:%M:%S')
log.warning('this is just a fucking test')
print(os.getcwd())