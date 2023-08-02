### For testing purpose only ###
import datetime
import typing
import pika
import json
import time
import os

class RabbitProducer():

    def __init__(self):
        self.RABBITMQ_HOST = os.getenv('RABBITMQ_HOSTNAME')
        self.RABBITMQ_EXCHANGE = os.getenv('RABBITMQ_BEM_CONTROL')

    def publish_schedule(self,
                         device_key: str,
                         schedule: typing.Dict[str, dict]):
        """
        Publish the device schedule to the message exchange to be received by the controller-template
        :param device_key: Name/key of the device (unique for this BEMS)
        :param schedule: The schedule for this device, as dict with ISO-formatted timestamps as keys
        """
        # Establish connection to RabbitMQ server
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.RABBITMQ_HOST))
        channel = connection.channel()

        payload = json.dumps(schedule)
        print(f'Publish schedule for {device_key}: {payload}.')
        # channel.exchange_declare(exchange=self.RABBITMQ_EXCHANGE, exchange_type='topic')
        channel.basic_publish(exchange=self.RABBITMQ_EXCHANGE,
                              routing_key=f'{device_key}.schedule',
                              body=payload)
        connection.close()

    def publish_setpoint(self,
                         device_key: str,
                         setpoint: typing.Dict[str, float]):
        """
        Publish the device schedule to the message exchange to be received by the controller-template
        :param device_key: Name/key of the device (unique for this BEMS)
        :param schedule: The schedule for this device, as dict with ISO-formatted timestamps as keys
        """
        # Establish connection to RabbitMQ server
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.RABBITMQ_HOST))
        channel = connection.channel()

        payload = json.dumps(setpoint)
        print(f'Publish setpoint for {device_key}: {payload}.')
        # channel.exchange_declare(exchange=self.RABBITMQ_EXCHANGE, exchange_type='topic')
        channel.basic_publish(exchange=self.RABBITMQ_EXCHANGE,
                              routing_key=f'{device_key}.setpoint',
                              body=payload)
        connection.close()


if __name__ == "__main__":
    print('Lets go')
    r_producer = RabbitProducer()
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    value = os.getenv('SEND_VALUE')
    if value == "None":
        value = None
    else:
        value = int(value)

    if os.getenv('SEND_TYPE') == 'schedule':
        if bool(os.getenv('SCHEDULE_FROM_FILE')):
            # Load schedule
            print('Read schedule(s) from file.')
            with open("/source/producer/config/schedule.json") as file:
                schedules = json.load(file)

            for device, schedule in schedules.items():
                r_producer.publish_schedule(device_key=device,
                                            schedule=schedule
                                            )
        else:
            now = datetime.datetime.now(tz=datetime.timezone.utc)
            timestamps = [now] + [
                (now + datetime.timedelta(seconds=int(os.getenv('SCHEDULE_PERIOD_LENGTH'))*n))
                for n in range(int(os.getenv('SCHEDULE_PERIOD_NUM')))
            ]
            if value:
                schedule = {ts.isoformat(timespec='seconds'): value+j for j,ts in enumerate(timestamps)}
            else:
                schedule = {ts.isoformat(timespec='seconds'): value for j,ts in enumerate(timestamps)}
            r_producer.publish_schedule(device_key=os.getenv('DEVICE_KEY'),
                                        schedule={'active_power': schedule}
                                        )

            next_schedule_in = (timestamps[-1] - now).total_seconds()
            print(f'Send next schedule in {next_schedule_in} seconds')
            time.sleep(next_schedule_in)
    elif os.getenv('SEND_TYPE') == 'setpoint':
        r_producer.publish_setpoint(device_key=os.getenv('DEVICE_KEY'),
                                    setpoint={'active_power': value}
                                    )
        time.sleep(int(os.getenv('SETPOINT_PERIOD_LENGTH')))