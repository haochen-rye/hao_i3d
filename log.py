import logging
logging.basicConfig(filename='test.log',filemode='w',level=logging.INFO, format='%(message)s')
logging.info('hello')
logging.info('bye')
