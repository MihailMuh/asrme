import logging

import aiohttp


class RestService:
    def __init__(self):
        self.__init_logger()
        self.__aio_session: aiohttp.ClientSession = aiohttp.ClientSession()
        self.__logger.debug("RestService initialized")

    async def file_download(self, url: str) -> bytes:
        async with self.__aio_session.get(url) as response:
            return await response.read()

    async def dispose(self):
        await self.__aio_session.close()

    def __init_logger(self):
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)
