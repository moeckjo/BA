import datetime
import json
import logging
import os
import random
import time
import typing

import pandas as pd
import pika
import pytz

from celery import Celery, chain, group
from celery.signals import after_setup_task_logger
from celery.app.log import TaskFormatter
from celery.utils.log import get_task_logger

from db import db_helper as db

from flexqgrid_python_sdk import OrderSet

from trading import logger, logformat
from communication import BlockchainConnector


# Rest is defined in environment in docker-compose file
celery_app = Celery()
celery_app.config_from_object('celeryconfig')

logger = get_task_logger(logger.name)


@celery_app.on_after_finalize.connect
def hello(sender, **kwargs):
    logger.info('Hello from trading!')


'''
Initial setup
'''


@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    """
    Configure format for all log messages by setting the format of all handlers of the celery
    task logger (which has been initialized above).
    Example log message:
    [2022-02-22 17:36:41,545 - bem-trading - task:trading.setup - setup_trading - INFO]:  Setup trading service.
    """
    for handler in logger.handlers:
        handler.setFormatter(TaskFormatter(logformat))
    logger.info(f'Logger is configured.')


@celery_app.task(name='trading.setup')
def setup_trading() -> dict:
    """
    Function that is called by the orchestration service (celery beat) when starting the whole BEM.
    It sets up the trading service and initiates necessary processes.
    """
    logger.info('Setup trading service.')

    # Register this BEMS at the quota market
    bcc = BlockchainConnector(load_connector_configuration())
    logger.debug(f'My BCC: {bcc.__dict__}')
    logger.debug(
        f'Block start timestamp without tz info: {bcc.get_next_period_block_start()} -> {datetime.datetime.fromtimestamp(bcc.get_next_period_block_start())}')
    logger.debug(
        f'Block start timestamp with tz info: {bcc.get_next_period_block_start()} -> {datetime.datetime.fromtimestamp(bcc.get_next_period_block_start(), tz=pytz.utc)}')

    # Register this actor ("plant") at the market platform
    result = bcc.register_plant(bcc.plant_id)
    logger.info(result)

    # Start waiting for primary quotas and reference schedules (i.e. subscribe to the respective message topics)
    wait_for_primary_quotas.delay()
    wait_for_reference_schedule.delay()

    time.sleep(1)
    return 'Trading setup finished.'


def load_connector_configuration():
    with open(os.path.join(os.getenv('BEM_ROOT_DIR'), 'config', 'blockchain_config.json')) as config_file:
        specifications = json.load(config_file)
    return specifications


'''
Receiving and publishing of BEM-internal messages
'''


@celery_app.task(name='trading.await_reference_schedule')
def wait_for_reference_schedule():
    def trigger_sending_process(ch, method, properties, body):
        """
        Executed when receiving message for topics '*.schedule'.
        Deserializes payload to dict and provides it to the trading task.
        """
        schedule = json.loads(body)
        logger.debug(f'Message received for topic {method.routing_key}: {body} -> data: {schedule}')
        place_reference_schedule.apply_async((schedule,), expires=60*20)

    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()

    # Declare queue, bind it to the exchange 'control' and start consuming messages with schedules
    result = channel.queue_declare(queue='', exclusive=True)
    # Assign automatically generated queue name
    queue_name = result.method.queue
    channel.queue_bind(queue=queue_name, exchange=os.getenv('RABBITMQ_BEM_CONTROL'),
                       routing_key=f'{os.getenv("GRID_CONNECTION_POINT_KEY")}.schedule.reference')
    channel.basic_consume(queue=queue_name, on_message_callback=trigger_sending_process, auto_ack=True)
    logger.info('Queue declared for topic "gcp.schedule.reference" -> start consuming...')
    channel.start_consuming()


@celery_app.task(name='trading.await_primary_quotas')
def wait_for_primary_quotas():
    """
    Sets up a queue bound to the message exchange "inbox" to subscribe to the topic "quotas.primary".
    When quota information is received, the callback function to trigger the trading process is called.
    :return:
    """

    def trigger_trading_process(ch, method, properties, body):
        """
        Executed when receiving message for topic 'quotas.primary'.
        Deserializes payload to dict and provides it to the trading task.
        """
        quotas = json.loads(body)
        quota_block_start = datetime.datetime.fromisoformat(quotas['start'])
        market_closure_time = quota_block_start - datetime.timedelta(minutes=int(os.getenv('MARKET_CLOSURE_LEAD_TIME_MINUTES')))
        logger.info(f'Message received for topic {method.routing_key}: {quotas}')
        mockup_trade_quotas.apply_async((quotas,market_closure_time, ), expires=market_closure_time)

    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()

    # Declare queue, bind it to the exchange 'inbox' and start consuming messages of the topic 'quotas.primary'
    result = channel.queue_declare(queue='', exclusive=True)
    queue_name = result.method.queue
    channel.queue_bind(queue=queue_name, exchange=os.getenv('RABBITMQ_BEM_INBOX'), routing_key='quotas.primary')
    channel.basic_consume(queue=queue_name, on_message_callback=trigger_trading_process, auto_ack=True)
    logger.info('Queue declared for topic "quotas.primary" -> start consuming...')
    channel.start_consuming()


def publish_final_quotas_to_ems(power_time_series: typing.Dict[str, int]):
    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()

    payload = json.dumps(power_time_series)
    logger.info(f'Publish final quotas. payload={payload}.')
    channel.basic_publish(exchange=os.getenv('RABBITMQ_BEM_INBOX'), routing_key='quotas.final.active_power',
                          body=payload)
    connection.close()


def publish_secondary_quotas_to_ems(power_time_series: typing.Dict[str, int]):
    # Establish connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOSTNAME')))
    channel = connection.channel()

    payload = json.dumps(power_time_series)
    logger.info(f'Publish secondary quotas. payload={payload}.')
    channel.basic_publish(exchange=os.getenv('RABBITMQ_BEM_INBOX'), routing_key='quotas.secondary.active_power',
                          body=payload)
    connection.close()


'''
Helper functions
'''


def save_market_result_to_db(power_limits: typing.Dict[str, int], total_cost: float = None):
    db.save_data_to_db(
        db=os.getenv('MONGO_QUOTA_DB_NAME'),
        data_source=os.getenv('MONGO_SECONDARY_QUOTAS_COLL_NAME'),
        time_series_data=power_limits,
        group_by_date=False,
        meta_data={'updated_at': datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec='seconds'),
                   "total_cost_euro": total_cost * 100 if total_cost is not None else total_cost
                   },
        # Unlikely that another market result will be stored for the exact same periods, but just in case -> persist
        persist_old_data=True
    )


@celery_app.task(name='trading.save_market_orders')
def save_market_orders_to_db(order_sets: typing.List[typing.List[dict]], trading_block_start: datetime.datetime):
    for order_set in order_sets:
        doc = {
            'period_block_start': trading_block_start.isoformat(),
            'period_block_end': (trading_block_start + datetime.timedelta(
                hours=int(os.getenv('QUOTA_WINDOW_SIZE')))).isoformat(),
            'orders': order_set
        }
        db.save_dict_to_db(
            db=os.getenv('MONGO_QUOTA_DB_NAME'),
            data_category=os.getenv('MONGO_QUOTA_MARKET_ORDERS_COLL_NAME'),
            data=doc
        )


def determine_tradeable_quotas(quotas: pd.DataFrame, trading_block_start: datetime.datetime):
    last_period_end = trading_block_start + datetime.timedelta(hours=int(os.getenv('QUOTA_WINDOW_SIZE')))

    tradeable_quotas = quotas.loc[
        (quotas.index >= trading_block_start) & (quotas.index < last_period_end)]
    return tradeable_quotas


'''
Communication and message processing
'''


@celery_app.task(name='trading.forward_market_orders')
def forward_market_orders(order_sets: typing.List[typing.List[dict]], trading_block_start: datetime.datetime):
    # TODO: publish orders instead of directly sending task

    """
    Send orders placed on the blockchain secondary market additionally to the ESB.
    :param trading_block_start: Start of the trading horizon
    :param order_sets: List of order sets placed at the secondary market. Each order set is again a list of orders.
                        An order is a dict specifying the period (timestamp), order type, quantity and price.
    """
    celery_app.send_task(
        name='gridmgmt.send_market_orders',
        kwargs={'order_period_start': trading_block_start, 'order_sets': order_sets}
    )


@celery_app.task(name='trading.process_market_result')
def process_market_result(market_result: list, trading_block_start: datetime.datetime) -> typing.Dict[str, int]:
    """
    From the list of power values (the market result) create a dictionary with corresponding period start timestamps as keys.
    Publish the resulting dict (with additional info of the block start) and store it in the database.
    :param market_result: List of absolute signed power values or None if there
    :param trading_block_start: Start of the trading/quota period block
    :return: Absolute power limits per period as dict with keys=(period start timestamps) (ISO-format string)
    """
    slot_period_time_map = BlockchainConnector.map_time_periods_to_slot_numbers(block_start=trading_block_start)
    # final_power_limits = {slot_period_time_map[i].isoformat(): power for i, power in enumerate(market_result)}
    # publish_final_quotas_to_ems(final_power_limits)
    secondary_power_limits = {slot_period_time_map[i].isoformat(): power for i, power in enumerate(market_result) if
                              power is not None}
    secondary_power_limits['period_block_start'] = trading_block_start.isoformat()
    publish_secondary_quotas_to_ems(secondary_power_limits)
    # TODO (optional): calculate total cost
    save_market_result_to_db(secondary_power_limits)
    return secondary_power_limits


@celery_app.task(name='trading.market_result')
def get_market_result() -> list:
    """
    Request effective power limits from the market platform.
    FYI: If no orders were matched, the limits equal the ones resulting from the primary quotas
    Limits are returned as list with a signed power value [W] for each period (="slot").

    Note: this task is scheduled to be executed at the known time at which the results are published

    :return: List of power limits (n=quota_block_length/quota_period_length elements)
    """
    # TODO: task timeout and retry necessary?
    bcc = BlockchainConnector(load_connector_configuration())
    next_period_block_start_utc = datetime.datetime.fromtimestamp(bcc.get_next_period_block_start(),
                                                                  tz=datetime.timezone.utc)
    logger.info(f'Request market result for period block of {next_period_block_start_utc}.')
    # Returned values are absolute power limits [W], as merge of primary and secondary quotas and reference schedule (?)
    # effective_power_limits = bcc.get_effective_plant_schedule(plant_id=bcc.plant_id, balance_period=int(trading_block_start.timestamp()))
    # Returned values are absolute power limits [W], as merge of primary and secondary quotas
    market_result = bcc.get_plant_schedule_by_secondary_quotas(plant_id=bcc.plant_id,
                                                               balance_period=int(next_period_block_start_utc.timestamp()))

    # Process the returned message, store and publish results
    process_market_result.apply_async(kwargs=dict(market_result=market_result, trading_block_start=next_period_block_start_utc))

    # Effective power limits after trading
    return market_result


@celery_app.task(name='trading.place_reference_schedule')
def place_reference_schedule(schedule: typing.Dict[str, int]):
    """
    Write reference schedule to the blockchain. Before writing, filter the schedule for relevant periods and the
    required create array of schedule values in the correct order.
    :param schedule: Reference schedule, starting with the quota period block, but can span more than the block's periods
    :return:
    """
    bcc = BlockchainConnector(load_connector_configuration())
    schedule = {datetime.datetime.fromisoformat(t): v for t, v in schedule.items()}
    block_start: datetime.datetime = min(schedule)
    slot_map: typing.Dict[int, datetime.datetime] = bcc.map_time_periods_to_slot_numbers(block_start)
    schedule_values = []
    # Create array of values from schedule dict, but only for upcoming block periods and ensure correct order
    for i in range(bcc.number_of_trading_periods):
        period_timestamp: datetime.datetime = slot_map[i]
        schedule_values.append(schedule[period_timestamp])
    assert len(
        schedule_values) == bcc.number_of_trading_periods, f'Schedule array needs to have exactly {bcc.number_of_trading_periods} periods (has {len(schedule)}.'
    success = bcc.publish_plant_schedule(bcc.plant_id, schedule_values=schedule_values,
                                         balance_period=int(block_start.timestamp()))
    assert success, f'Reference schedule for block starting at {block_start} could not be published on market platform.'


@celery_app.task(name='trading.place_order_set')
def place_order_set(order_set: OrderSet, no: int):
    logger.debug(f'Place {len(order_set.orders)} orders with order set no. {no}.')
    start_timer = time.time()
    bcc = BlockchainConnector(load_connector_configuration())
    success = bcc.place_order_set(order_set)
    assert success, f'Error placing order set: {order_set}'
    return f'Order set no. {no} successfully placed (runtime: {time.time() - start_timer}s.'


'''
Mockup functions
'''


@celery_app.task(name='trading.mockup_place_orders')
def mockup_place_orders(tradeable_quotas: pd.DataFrame, block_start: datetime.datetime) -> typing.List[
    typing.List[dict]]:
    bcc = BlockchainConnector(load_connector_configuration())
    published_schedule = bcc.get_effective_plant_schedule(plant_id=bcc.plant_id,
                                                               balance_period=int(block_start.timestamp()))
    period_map: typing.Dict[datetime.datetime, int] = bcc.map_slot_numbers_to_time_periods()
    order_sets: typing.List[OrderSet] = []  # List of order sets
    order_sets_internal: typing.List[
        typing.List[dict]] = []  # List of order sets that stores order parameters for further internal usage
    for s in range(2):  # Place three order sets, that each can be matched only as a whole
        order_set = OrderSet()  # OrderSet instance that is placed at the market
        order_sets_internal.append([])
        logger.debug(f'Calculate order set no. {s}...')

        for period_start, values in tradeable_quotas.iterrows():
            if random.random() > 0.6:  # Place orders for x% of the periods on average
                reference_power = values["reference_power"]
                quota = values["quota"]
                if reference_power is None:
                    reference_power = published_schedule[period_map[period_start]]
                if quota == 1:
                    offer_share = round(random.random(),2)
                else:
                    offer_share = 1-quota

                order = dict(
                    slot=period_map[period_start],
                    order_type='BID' if random.random() > 0.5 else 'ASK',
                    quantity=int(offer_share * reference_power),
                    price=random.randint(1, 50)
                )
                try:
                    order_set.add_order(
                        plant_id=bcc.plant_id,
                        **order
                    )
                except KeyError:
                    logging.debug(f'Period {period_start} cannot be traded yet.')
                order['time'] = period_start.isoformat()
                order_sets_internal[s].append(order)
        order_sets.append(order_set)
        # logger.debug(f'Complete order set no. {s}: {order_set.orders}')
        logger.debug(f'Complete order set no. {s} for internal use: {order_sets_internal[s]}')

    # TODO: this is non-mockup
    group(place_order_set.s(order_set, i) for i, order_set in enumerate(order_sets))()

    return order_sets_internal


@celery_app.task(name='trading.mockup_trade_quotas')
def mockup_trade_quotas(quotas: typing.Dict[str, dict], market_closure_time: datetime.datetime):
    """
    :param quotas: Dict with keys=period starts (ISO str) and values=dict with keys=['quota', 'type', 'reference_power']
    """
    bcc = BlockchainConnector(load_connector_configuration())
    next_trading_block_start_utc = datetime.datetime.fromtimestamp(bcc.get_next_period_block_start(),
                                                                   tz=datetime.timezone.utc)
    quota_block_start = quotas.pop('start')
    logger.debug(
        f'Trading block: from {next_trading_block_start_utc} to {next_trading_block_start_utc + datetime.timedelta(hours=int(os.getenv("QUOTA_WINDOW_SIZE")))} (excl.)')

    logger.debug(f'Time info: {bcc.get_time_info()}')
    quotas_df = pd.DataFrame.from_dict(quotas, orient='index')
    quotas_df.index = pd.to_datetime(quotas_df.index)
    tradeable_quotas = determine_tradeable_quotas(quotas_df, next_trading_block_start_utc)
    logger.debug(f'Tradeable quotas (df): {tradeable_quotas}')

    # TODO: delete this workaround when getting up-to-date quotas
    ### Only as long as old quotas are returned from ESB: change index to start at the upcoming block ####
    if tradeable_quotas.empty:
        quotas_df.index = pd.date_range(
            start=next_trading_block_start_utc,
            periods=len(quotas_df.index),
            # int(int(os.getenv('QUOTA_WINDOW_SIZE'))/float(os.getenv('QUOTA_TEMP_RESOLUTION'))),
            freq='15min'
        )
        tradeable_quotas = determine_tradeable_quotas(quotas_df, next_trading_block_start_utc)
        logger.debug(f'Fake tradeable quotas (df): {tradeable_quotas}')

    # TODO: all below is non-mockup (exceptions are marked)
    order_placement = chain(
        # TODO: replace with real market computation function
        mockup_place_orders.s(tradeable_quotas,next_trading_block_start_utc).set(expires=market_closure_time),
        # Order sets are first argument to both functions
        group(
            forward_market_orders.s(next_trading_block_start_utc),
            save_market_orders_to_db.s(next_trading_block_start_utc)
        )
    )

    order_placement()
