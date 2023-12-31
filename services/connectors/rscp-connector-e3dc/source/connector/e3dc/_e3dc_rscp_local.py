#!/usr/bin/env python
# Python class to connect to an E3/DC system through the internet portal
#
# Copyright 2017 Francesco Santini <francesco.santini@gmail.com>
# Licensed under a MIT license. See LICENSE for details
import os
import socket
from ._rscpLib import rscpFrame, rscpEncode, rscpFrameDecode, rscpDecode, rscpFindTag
from ._RSCPEncryptDecrypt import RSCPEncryptDecrypt
import time
import datetime
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG if os.getenv('DEBUG') == 'true' else logging.INFO)

PORT = 5033
BUFFER_SIZE=1024*32

class RSCPAuthenticationError(Exception):
    pass

class RSCPNotAvailableError(Exception):
    pass

class CommunicationError(Exception):
    pass


class E3DC_RSCP_local:
    """A class describing an E3DC system, used to poll the status from the portal
    """
    def __init__(self, username, password, ip, key):
        self.username = username.encode('utf-8')
        self.password = password.encode('utf-8')
        self.ip = ip
        self.key = key.encode('utf-8')
        self.socket = None
        self.encdec = None
        self.processedData = None
        
    def _send(self, plainMsg):
        sendData = rscpFrame( rscpEncode( plainMsg ) )
        encData = self.encdec.encrypt(sendData)
        self.socket.send(encData)

    def _receive(self):
        data = self.socket.recv(BUFFER_SIZE)
        decData = rscpDecode( self.encdec.decrypt(data) )[0]
        return decData  # (strTag, strType, val),
    
    def sendCommand(self, plainMsg):
        self.sendRequest(plainMsg) # same as sendRequest but doesn't return a value
        
    def sendRequest(self, plainMsg) -> tuple:
        """
        Return response as tuple with (Tag, Type, Value)
        """
        try:
            logger.debug(f'Send request: {plainMsg}')
            self._send(plainMsg)
            receive = self._receive()
            logger.debug(f'Decoded request response (in sendRequest, line 58): {receive}')
        except:
            self.disconnect()
            raise CommunicationError

        if receive[1] == 'Error':
            self.disconnect()
            if receive[2] == "RSCP_ERR_ACCESS_DENIED":
                raise RSCPAuthenticationError
            elif receive[2] == "RSCP_ERR_NOT_AVAILABLE":
                raise RSCPNotAvailableError
            else:
                raise CommunicationError(receive[2])
        return receive
        
    def connect(self):
        if self.socket is not None:
            self.disconnect()
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.ip, PORT))
            self.processedData = None
        except:
            self.disconnect()
            raise CommunicationError
        self.encdec = RSCPEncryptDecrypt(self.key)

        decData = self.sendRequest ( ('RSCP_REQ_AUTHENTICATION', 'Container', [
            ('RSCP_AUTHENTICATION_USER', 'CString', self.username),
            ('RSCP_AUTHENTICATION_PASSWORD', 'CString', self.password)]) )
        
    def disconnect(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
    
    def isConnected(self):
        return self.socket is not None
    

