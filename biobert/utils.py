#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Umesh.Menon

Contains all the utility functions
"""
import os, sys
import os.path
import errno
from numbers import Number
from pathlib import Path


def get_stop_words(stop_file_path):
    """load stop words """

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def get_project_root_path():
    return Path(os.path.dirname(os.path.abspath(__file__)))


def get_full_path(folder, file):
    if is_python3():
        if is_str(folder):
            folder = Path(folder)
        file_name = folder / file
        file_name = str(file_name)
    else:
        file_name = os.path.join(folder, file)
    return file_name


def get_fq_path(folder, *args):
    file_name = folder
    if is_python3():
        if is_str(file_name):
            file_name = Path(file_name)
        for arg in args:
            file_name = file_name / arg
        file_name = str(file_name)
    else:
        for arg in args:
            file_name = os.path.join(file_name, arg)
    return file_name


def construct_fq_path(*args):
    file_name = args[0]
    if is_python3():
        if is_str(file_name):
            file_name = Path(file_name)
        for arg in args[1:]:
            file_name = file_name / arg
        file_name = str(file_name)
    else:
        for arg in args[1:]:
            file_name = os.path.join(file_name, arg)

    # we can also do something like this
    # for arg in args[1:]:
    #    file_name = file_name + os.sep + arg
    return file_name


def get_fq_project_file_path(filename):
    """
    Gets fully qualified path of a file within the project directory
    :param file:
    :return:
    """
    if is_python3():
        file_name = get_project_root_path() / filename
        file_name = str(file_name)
    else:
        file_name = os.path.join(get_project_root(), filename)

    return file_name


def mkpath(path):
    """Alias for mkdir -p."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def mkpath2(path):
    """Alias for mkdir -p."""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            return False
    return True


def idxmax(lst):
    """
    Max index of a list
    :param lst:
    :return:
    """
    m = max(lst)
    idx = [i for i, j in enumerate(lst) if j == m]
    return idx


def flatten_list(lst):
    """
    Flatten a given array to a single array
    :param lst:
    :return:
    """
    #flat_list = [item for sublist in lst for item in sublist]

    flat_list = []
    for sublist in lst:
        if sublist:
            for item in sublist:
                flat_list.append(item)
    return flat_list


def flatten_json(doc, prefix=""):
    '''
    Flattens a json document.
    Behavior on list flattening might not be what you are looking for
    so please, be sure that you understand code before using
    '''

    def flatten(lis):
        """Given a list, possibly nested to any level, return it flattened."""
        new_lis = []
        for item in lis:
            if type(item) == type([]):
                new_lis.extend(flatten(item))
            else:
                new_lis.append(item)
        return new_lis

    def map_update(target_m, source_m):
        '''
        Adds data from target_m to source_m
        '''
        for key, val in source_m.items():
            if key in target_m:
                target_m[key] = [target_m[key]]
                target_m[key].append(val)
                target_m[key] = flatten(target_m[key])
            else:
                target_m[key] = val
        return target_m

    flattened_fields = {}

    if type(doc) is list:
        for a in doc:
            fields = flatten_json(a, prefix)
            map_update(flattened_fields, fields)
    elif type(doc) is dict:
        for key, val in doc.items():
            if type(val) is list:
                new_prefix = prefix + "." + key if len(prefix) > 0 else key
                for a in val:
                    if type(a) is dict:
                        fields = flatten_json(a, new_prefix)
                        map_update(flattened_fields, fields)
                    else:
                        map_update(flattened_fields, {new_prefix: a})
            elif type(val) is dict:
                new_prefix = prefix + "." + key if len(prefix) > 0 else key
                fields = flatten_json(val, new_prefix)
                map_update(flattened_fields, fields)
            else:
                field_name = prefix + "." + key if len(prefix) > 0 else key
                map_update(flattened_fields, {field_name: val})
    else:
        raise ValueError("json doc is not list or dictionary")
    return flattened_fields


def list_to_file(list, file_path):
    """
    Write a list to file
    :param list:
    :param file_path:
    :return:
    """
    # save as text file
    with open(file_path, 'w') as filehandle: #, encoding="utf-8"
        filehandle.writelines("%s\n" % item for item in list)


def topn_list_val(lst, thresh):
    """
    Returns the top n items from a numeric list. Checks if the value in the list is greater than or equal to threshold
    if given
    :param lst:
    :param thresh:
    :return:
    """
    #sorted(range(len(lst)), key=lambda k: lst[k])
    #sorted(range(len(lst)), key=lst.__getitem__)
    import numpy
    vals = numpy.array(lst)
    sort_index = numpy.argsort(vals)
    topn = [vals[idx] for idx in sort_index if vals[idx] >= thresh]
    return topn


def topn_list_idx(lst, thresh=0.0, topn=2):
    """
    Returns the top n items from a numeric list. Checks if the value in the list is greater than or equal to threshold
    if given
    :param lst:
    :param thresh:
    :return:
    """
    #sorted(range(len(lst)), key=lambda k: lst[k])
    #sorted(range(len(lst)), key=lst.__getitem__)
    import numpy
    vals = numpy.array(lst)
    sort_index = vals.argsort()[::-1]#numpy.argsort(vals) # to sort in descending order
    all_sorted = [idx for idx in sort_index if vals[idx] >= thresh]
    topn_lst = all_sorted
    if len(all_sorted) >= topn + 1:
        topn_lst = all_sorted[0:topn]
    return topn_lst


def is_number(val):
    """
    Checks if a given value is numeric or not
    :param val:
    :return:
    """
    return isinstance(val, Number)


def is_python3():
    """
    Checks if the python version is 3
    :return:
    """
    return sys.version_info.major == 3


def flatten2(list_of_lists):
    '''
    flatten()
    Purpose: Given a list of lists, flatten one level deep
    @param list_of_lists. <list-of-lists> of objects.
    @return               <list> of objects (AKA flattened one level)
    >>> flatten([['a','b','c'],['d','e'],['f','g','h']])
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    '''
    return sum(list_of_lists, [])


def file_contents_to_list(file_path):
    """
    Reads a file line by line and stores to a list and returns the list
    :param file_path:
    :return:
    """
    lst = [line.rstrip('\n') for line in open(file_path)]
    return lst


def is_str(obj):
    """
    Checks if the given object is type of string or not
    :param str:
    :return:
    """
    return isinstance(obj, str) #basestring in Python 2.x


def intersection(lst1, lst2):
    """
    Rerurns the intersection of two lists
    :param lst1:
    :param lst2:
    :return:
    """
    return list(set(lst1) & set(lst2))


def dedup_lst(lst):
    """
    Dedupes a list
    :param lst:
    :return:
    """
    return list(set(lst))