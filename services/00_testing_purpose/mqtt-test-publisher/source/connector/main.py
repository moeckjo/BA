#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import logging
import os
import json
import time

from dotenv import load_dotenv, find_dotenv
import paho.mqtt.client as paho
from datetime import datetime, timezone

# Log everything to stdout by default, i.e. to docker container logs.
LOGFORMAT = '%(asctime)s-%(funcName)s-%(levelname)s: %(message)s'
logging.basicConfig(format=LOGFORMAT, level=logging.DEBUG)
logger = logging.getLogger(os.getenv('NAME'))


class MQTTPublisher():

    def __init__(self):

        # dotenv allows us to load env variables from .env files which is
        # convient for developing. If you set override to True tests
        # may fail as the tests assume that the existing environ variables
        # have higher priority over ones defined in the .env file.
        load_dotenv(find_dotenv(), verbose=True, override=False)
        self.DEBUG = os.getenv("DEBUG")

        self.MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST")
        self.MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT"))
        self.MQTT_TOPIC = os.getenv("MQTT_TOPIC")
        self.ACTUATOR_FEATURE = os.getenv("ACTUATOR_FEATURE")
        self.PUBLISH_INTERVAL = int(os.getenv('MQTT_PUBLISH_INTERVAL'))
        self.FROM_FILE = os.getenv('FROM_FILE')

    def run(self):
        # Create a client instance
        client = paho.Client()

        # Register callbacks
        client.on_connect = self._on_connect
        client.on_log = self._on_log
        client.on_publish = self._on_publish
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message

        client.connect(host=self.MQTT_BROKER_HOST,
                       port=self.MQTT_BROKER_PORT)

        if self.FROM_FILE.lower() == "true":
            with open(f'/source/connector/config/{os.getenv("COMMAND_FILE_NAME")}') as file:
                topics_and_values = json.load(file)

            while True:
                for topic, value in topics_and_values.items():
                    payload = {
                        'value': value,
                        'timestamp': int(time.time() * 1000)
                    }
                    client.publish(topic=topic, payload=json.dumps(payload), qos=0)
                    logger.info(f'Message published to topic {topic}: {payload}')

                time.sleep(self.PUBLISH_INTERVAL)

        else:
            if os.getenv('DEVIATION_MESSAGE_SUBTOPIC') in self.MQTT_TOPIC.split('/'):
                while True:
                    payload = {
                        "feature": self.ACTUATOR_FEATURE,
                        'sensor_value': os.getenv('SENSOR_VALUE'),
                        'target_value': str(int(os.getenv('SENSOR_VALUE')) - 1000),
                        'timestamp': int(time.time()*1000)
                    }
                    client.publish(topic=self.MQTT_TOPIC, payload=json.dumps(payload), qos=0)
                    logger.info(f'Message published to topic {self.MQTT_TOPIC}: {payload}')
                    time.sleep(self.PUBLISH_INTERVAL)

            if "sensor" in self.MQTT_TOPIC.split('/'):
                while True:
                    payload = {
                     'value': os.getenv('SENSOR_VALUE'),
                    'timestamp': int(time.time()*1000)
                    }
                    client.publish(topic=self.MQTT_TOPIC, payload=json.dumps(payload), qos=0)
                    logger.info(f'Message published to topic {self.MQTT_TOPIC}: {payload}')
                    time.sleep(self.PUBLISH_INTERVAL)


    @staticmethod
    def _on_connect(pahoClient, obj, rc, properties=None):
        # Once connected, publish message
        logger.info(f"\nConnected Code = {rc}")
        pahoClient.publish('initial_msg/tist', 'Hello World MQTT', 0)

    @staticmethod
    def _on_log(pahoClient, obj, level, string):
        # print("\n" + string)
        pass

    @staticmethod
    def _on_publish(pahoClient, packet, mid):
        logger.info("\nPublished")

    @staticmethod
    def _on_disconnect(pahoClient, obj, rc):
        # print("\nPublished")
        pass

    @staticmethod
    def _on_message(pahoClient, userdata, message):
        logger.info(f'\nNEW MESSAGE')
        logger.info(f'message topic= {message.topic}')
        logger.info(f'message received= {message.payload} \n')


if __name__ == "__main__":
    logger.info('Create MQTT Test Publisher')

    connector = MQTTPublisher()
    connector.run()
