import asyncio

from fastapi import WebSocket


class WebsocketService:
    def __init__(self):
        self.__active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.__active_connections.append(websocket)
        while True:
            try:
                await websocket.receive_text()
            except Exception:
                self.__active_connections.remove(websocket)
                return

    async def send_messages(self, messages: list[str]) -> None:
        async def create_task(ws: WebSocket, msg):
            try:
                await ws.send_text(msg)
            except Exception:
                pass

        tasks: list = []
        for message in messages:
            for websocket in self.__active_connections:
                tasks.append(create_task(websocket, message))

        await asyncio.gather(*tasks)

    def has_active_connections(self) -> bool:
        return len(self.__active_connections) > 0

    async def dispose(self) -> None:
        await asyncio.gather(*[websocket.close() for websocket in self.__active_connections])
