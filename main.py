import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.responses import ORJSONResponse
from uvicorn import run

from constants.fastapi_constants import *
from services.rest_service import RestService
from services.whisper_service import WhisperService
from services.ws_service import WebsocketService

logging.basicConfig(format="%(asctime)s %(processName)s %(levelname)s %(funcName)s() --> %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_service, rest_service, websocket_service
    rest_service = RestService()
    websocket_service = WebsocketService()
    whisper_service = WhisperService(websocket_service.send_messages)

    yield

    await websocket_service.dispose()
    await rest_service.dispose()
    whisper_service.dispose()


whisper_service: WhisperService
rest_service: RestService
websocket_service: WebsocketService
app: FastAPI = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)


@app.get("/readyz")
async def readyz() -> str:
    return "OK"


@app.post("/task")
async def create_task(audio: UploadFile = File(...)) -> uuid.UUID:
    return await whisper_service.create_task(await audio.read())


@app.get("/task")
async def create_task(url: str) -> uuid.UUID:
    return await whisper_service.create_task(await rest_service.file_download(url))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_service.connect(websocket)


if __name__ == '__main__':
    run(
        "main:app",
        host=SERVER_HOST, port=SERVER_PORT, workers=1,
        ws="websockets", ws_ping_interval=None, ws_ping_timeout=None
    )
