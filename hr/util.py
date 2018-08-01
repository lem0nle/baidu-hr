# -*- coding: UTF-8 -*-
from __future__ import unicode_literals, print_function
import pkg_resources


def get_resource(resource, mode=None):
    s = pkg_resources.resource_string(__package__, resource)
    if mode == 'b':
        return s
    return s.decode('utf-8')


def get_resource_filename(resource):
    return pkg_resources.resource_filename(__package__, resource)
