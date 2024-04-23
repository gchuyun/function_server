import os
import sys
import time
import httpx
from types import FrameType
from typing import cast
import logging
from loguru import logger
from .settings import LOG_LEVEL


def init_logger():
    logger.remove()
    
    logger.add(sys.stdout,
               level = LOG_LEVEL,
               format="<green>{time:YYYYMMDD HH:mm:ss}</green> | "  # 颜色>时间
                        "{process.name} | "  # 进程名
                        "{thread.name} | "  # 进程名
                        "<cyan>{module}</cyan>.<cyan>{function}</cyan>"  # 模块名.方法名
                        ":<cyan>{line}</cyan> | "  # 行号
                        "<level>{level}</level>: "  # 等级
                        "<level>{message}</level>",  # 日志内容
            )    
    
    # change handler for default uvicorn logger
    LOGGER_NAMES = ("uvicorn.asgi", "uvicorn.access", "uvicorn")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in LOGGER_NAMES:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
 
        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1
 
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage(),
        )

class Cache:
    def __init__(self, expire_milliseconds: float):
        self.cache = {}
        self.expire_milliseconds = expire_milliseconds
    
    def put(self, key: str, obj, expire_at = None):
        if expire_at is None:
            expire_at = time.time()*1000 + self.expire_milliseconds
        self.cache[key] = (obj, expire_at)
        self.__check_if_expire()
    
    def get(self, key: str):
        self.__check_if_expire()
        t = self.cache.get(key, (None, None))
        return t[0]
    
    def pop(self, key: str):
        self.__check_if_expire()
        t = self.cache.pop(key, (None, None))
        return t[0]
    
    def clear(self):
        self.cache.clear()
    
    def __check_if_expire(self):
        now = time.time()*1000
        keys = []
        for k,v in self.cache.items():
            if now > v[1]:
                keys.append(k)
        for k in keys:
            self.cache.pop(k, None)

class ReReadbleHttpxSuccessfulResponse:    
    def __init__(self, response: httpx.Response):
        self.response = response
        self.is_stream_consumed = True
        self.status_code = self.response.status_code
        self.headers = self.response.headers        
        self.content = self.response.content
    
    async def aclose(self):
        if self.response:
            await self.response.aclose()
            self.response = None            