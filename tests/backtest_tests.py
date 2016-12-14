# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:36:43 2016

@author: Richard
"""

from nose.tools import *
import NAME

def setup():
    print "SETUP!"

def teardown():
    print "TEAR DOWN!"

def test_basic():
    print "I RAN!"