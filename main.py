import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
from starlette.responses import PlainTextResponse
from uvicorn import run

from constants.fastapi_constants import *
from services.rest_service import RestService
from services.whisper_service import WhisperService

logging.basicConfig(format="%(asctime)s %(processName)s %(levelname)s %(funcName)s() --> %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_service, rest_service
    rest_service = RestService()
    whisper_service = WhisperService()

    yield

    await rest_service.dispose()
    whisper_service.dispose()


whisper_service: WhisperService
rest_service: RestService
app: FastAPI = FastAPI(default_response_class=PlainTextResponse, lifespan=lifespan)


@app.get("/readyz")
async def readyz() -> str:
    return "OK"


@app.post("/task")
async def create_task(audio: UploadFile = File(...)) -> uuid.UUID:
    return await whisper_service.create_task(await audio.read())


@app.post("/task")
async def create_task(url: str) -> uuid.UUID:
    return await whisper_service.create_task(await rest_service.file_download(url))


@app.get("/task")
async def get_task(uid: str) -> str:
    return whisper_service.get_task(uuid.UUID(uid))


if __name__ == '__main__':
    run("main:app", host=SERVER_HOST, port=SERVER_PORT, workers=SERVER_WORKERS)
