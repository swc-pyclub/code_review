__author__ = 'blota'

"""
Function not directly usefull for the analysis, but for the code writing itself
"""

import re


def cmp_versions(version1, version2):
    """Compare two dot-separated list of numbers

    From: http://stackoverflow.com/questions/1714027/version-number-comparison

    :param version1: str
    :param version2: str
    :return: int
    """

    def normalize(v):
        return [int(x) for x in re.sub(r'(\.0+)*$', '', v).split(".")]

    return normalize(version1) >= normalize(version2)


def make_list(param, check_func=None):
    """Make sure param is iterable and convert it to a list

    If check_func is not None, will also map check func on the param list (to check input type and force conversion)

    :param param:
    :return: list
    """
    if isinstance(param, str) or (not hasattr(param, '__iter__')):
        param = [param]
    param = list(param)
    if check_func is not None:
        param = list(map(check_func, param))
    return param
