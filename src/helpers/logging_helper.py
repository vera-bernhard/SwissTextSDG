import logging
import sys



def setup_logging():
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s][%(levelname)s]\t%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='logfile.log',
                        filemode='a') 
    logging.getLogger('SwissTextSDG').setLevel(logging.DEBUG)