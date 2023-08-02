import datetime
from wsgiref.handlers import format_date_time

import requests
from requests.exceptions import Timeout
import json
import logging
import time
import pytz
import os

import typing

from gridmanagement import logger


def load_esb_config():
    with open(os.path.join(os.getenv('BEM_ROOT_DIR'), 'config', 'esb_config.json')) as esb_config_file:
        esb_config = json.load(esb_config_file)
    return esb_config


class ESBConnector():
    REQUEST_TIMEOUT = 120
    RETRY_REQUEST_AFTER = 2  # seconds to wait until making same request again if unsuccessful
    STOP_RETRY_AFTER = 30  # Total seconds to retry a request before giving up

    def __init__(self, specifications):
        self.base_url = specifications["base_url"]
        self.endpoints = specifications["endpoints"]
        self.endpoints_with_uuid = specifications["endpoints_with_uuid"]
        self.authentication = specifications["auth"]

    def make_request(self, url: str, method: str, timeout: int = REQUEST_TIMEOUT,
                     assert_status: typing.Union[list, int] = 200,
                     **kwargs) -> requests.Response:
        """
        Performs HTTP request to specified URL with the given method and parameters
        :param url: Full request URL
        :param method: HTTP method (get, post. put ...)
        :param timeout: Request timeout
        :param assert_status: Accepted HTTP response code(s) (single code or list)
        :param kwargs: Further parameters
        :return:
        """
        # Extract additional headers from kwargs dict
        headers = kwargs.pop('headers', dict())
        if headers:
            for field_name, field_val in headers.items():
                if isinstance(field_val, datetime.datetime):
                    # Convert to HTTP date format (e.g.'Wed, 22 Oct 2008 10:52:40 GMT')
                    field_val = format_date_time(time.mktime(field_val.timetuple()))
                    headers[field_name] = field_val

        if headers: logger.info(f'Additional headers: {headers}')
        logger.info(f'{method.upper()} request to {url}.')

        response = requests.request(
            method=method.upper(),
            url=url,
            # auth=tuple(self.authentication.values()),
            headers={'content-type': 'application/json', **headers},
            timeout=timeout,
            verify=True,  # Verify the server's certificate
            cert=(os.path.join(os.getenv('BEM_ROOT_DIR'), 'certificates', 'client.flexqgrid.crt'),
                  os.path.join(os.getenv('BEM_ROOT_DIR'), 'certificates', 'client.flexqgrid.key')),
            **kwargs
        )

        if assert_status:
            # If single accepted status code, make single-element list with it
            assert_status = [assert_status] if isinstance(assert_status, int) else assert_status
            # Check if status code is acceptable
            assert response.status_code in assert_status, f'{method.upper()} request to {url} not successful' \
                                                          f' (HTTP code: {response.status_code})!'
        return response

    def send_message(self, message_type: str, message: dict, uuid: str):
        """
        Prepares POST request of json-encoded message, provided as dict.
        :param message_type: Type of message, required for URL
        :param message: Message to be sent
        :param uuid: DSO-specified UUID of this system
        """
        url = f'{self.base_url}/{self.endpoints[message_type]}'

        # Make sure message contains this system's UUID
        assert message['uuid'] == uuid

        logger.info(f'Send {message_type}.')

        self.make_request(url, 'post', json=message, assert_status=[200, 201, 202])

    def get_message(self, message_type: str, initial_delay: int = 0, timeout: int = REQUEST_TIMEOUT,
                    retry_after: int = RETRY_REQUEST_AFTER, stop_retry_after: int = STOP_RETRY_AFTER,
                    catch_exceptions: tuple = (), uuid: str = None, **kwargs) -> dict:
        """
        Make a HTTP GET request with the specified parameters to obtain a message (json) in the response
        body and save it as json file.
        :param message_type: Type of message, equals file name when saved
        :param timeout: Request timeout [s]
        :param retry_after: Seconds to wait until making the same request again if unsuccessful
        :param stop_retry_after: Total retry period. If passed, no more retries of a request.
        :param initial_delay: Seconds to wait until making the first request (default: 0s)
        :param catch_exceptions: Exceptions that shall not break the program
        :param uuid: UUID of client
        :param kwargs: Further request options, e.g. a custom request header
        """
        if message_type in self.endpoints_with_uuid:
            assert uuid is not None, f'UUID required to request {message_type}.'
            url = f'{self.base_url}/{self.endpoints_with_uuid[message_type]}/{uuid}'
        else:
            url = f'{self.base_url}/{self.endpoints[message_type]}'

        logger.info(f'Request {message_type} for {uuid}.')
        logger.info(f'Wait {initial_delay} sec before executing the first request.')
        time.sleep(initial_delay)

        request_status = 0
        response = None
        initial_request_at = time.time()
        # Repeat request until successful (e.g. quota received) or max. retries is reached
        while not (200 <= request_status < 300):
            try:
                # TODO: different handling of 404 and 304, after FIT fixed it
                response = self.make_request(url, 'get', timeout=timeout, assert_status=[200, 201, 202], **kwargs)
                assert response is not None
                request_status = response.status_code

            except catch_exceptions as e:
                if time.time() <= initial_request_at + stop_retry_after:
                    logger.warning(f'Caught exception: {e}. Retrying in {retry_after} seconds.')
                    time.sleep(retry_after)

                else:
                    logger.warning(f'Caught exception: {e}. Retry duration has passed ({stop_retry_after}s). '
                                   f'Raising exception.')
                    raise e

        # Retrieve json body from response
        msg = response.json()
        logger.info(f'{message_type.capitalize()} message  for {uuid} received (status code {request_status}).')

        return msg
