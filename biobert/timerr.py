#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Umesh.Menon

A decorator timer class
"""
#import logging
import time


class Timer(object):
    """
    A decorator class to print the execution time of the function
    """
    def __init__(self, logger):
    # def __init__(self, f):
        """
        If there are no decorator arguments, the function
        to be decorated is passed to the constructor.
        """
        # self.f = f
        self.logger = logger

    def __call__(self, f,  *args, **kwargs):
    # def __call__(self, *args):
        """
        The __call__ method is not called until the
        decorated function is called.
        """
        start_time = time.time()
        fn_name = f.__name__
        self.logger.info("Started executing {}".format(fn_name))

        def wrapper(*args, **kwargs):
            ret = f(*args, **kwargs)
            seconds_to_complete = time.time() - start_time
            self.logger.info("{} execution completed in {} minutes".format(fn_name, seconds_to_complete / 3600.0))
            return ret
        return wrapper
