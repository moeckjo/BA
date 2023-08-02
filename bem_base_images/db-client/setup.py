from setuptools import setup, find_packages

setup(name='db',
      version='1.0',
      packages=find_packages(where='.'),
      install_requires=['pandas==1.2.5', 'influxdb==5.2.3', 'pymongo==3.11.4']
      )
