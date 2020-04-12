"""
Sets up the logger
@author: Umesh.Menon
"""
import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def setup_logger(config=None, log_level=None):
    """
    Sets up a logger to use in the file
    :param config:
    :param log_level:
    :return:
    """
    LOG_KEY = 'log'

    if log_level is None:
        log_level = 'INFO'

    if config is None:
        out_type = 'console'
    else:
        out_type = config.get(LOG_KEY, 'out_type')
        log_file = config.get(LOG_KEY, 'log.file.name')
        log_level = config.get(LOG_KEY, 'log.level')

    log_level = log_level.upper()

    log_level = getattr(logging, log_level)

    log_format = '%(asctime)s:%(name)s:%(levelname)s:process %(process)d:process name %(processName)s:%(funcName)s():line %(lineno)d:%(message)s'
    if out_type == "file":
        logging.basicConfig(filename=log_file, format=log_format, level=log_level)
    else:
        logging.basicConfig(format=log_format, level=log_level)
