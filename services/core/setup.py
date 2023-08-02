"""
Script to create package from  (to solve (relative) import issues)
See https://stackoverflow.com/a/50193944/4458173 for instructions
"""
from setuptools import setup, find_packages

setup(name='core', version='1.0', packages=find_packages(where='.'))
