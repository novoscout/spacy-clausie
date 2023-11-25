#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:21:51 2018

@author: Emmanouil Theofanis Chourdakis
"""

import unittest
from setuptools import setup, find_packages

packages = find_packages()
install_requires = [
    "spacy>=3.0.0",
    "lemminflect>=0.2.1",
    "fastenum @ git+https://gitlab.com/austinjp/fastenum.git@bugfix/mypy-plugin-nodes-issues"
]


setup(
    name="claucy",
    version="0.0.3",
    packages=packages,
    install_requires=install_requires,
    test_suite="tests.test_suite",
    author="Emmanouil Theofanis Chourdakis",
    author_email="etchourdakis@gmail.com",
    description="A reimplementation of ClausIE Information Extraction System in python",
    url="https://github.com/novoscout/spacy-clausie",
    keywords="openie clausie information extraction spacy",
    include_package_data=True,
)
