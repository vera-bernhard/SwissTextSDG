import logging
import sys


def setup_logging():
    logging.basicConfig(level=logging.WARN,
                        format='[%(asctime)s][%(levelname)s]\t%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
    logging.getLogger('SwissTextSDG')