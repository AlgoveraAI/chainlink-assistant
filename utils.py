import json
import math
import logging
from fastapi import WebSocket
from pydantic import BaseModel
from typing import Any, List, Dict
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager

from schemas import ChatResponse, Sender, MessageType

USERNAMES = [
    "algovera_admin",
    "1af26bc619c4adf5e3f9a1806879e434ab681281c30528d2a30691226b4f7051",
]

def createLogHandler(job_name, log_file="logs.log"):
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.INFO)

    ## create a file handler ##
    handler = logging.FileHandler(log_file)
    ## create a logging format ##
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: Any):
        if isinstance(data, BaseModel):
            data = data.dict()
        for connection in self.active_connections:
            await connection.send_json(data)


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, connection_manager):
        # self.websocket = websocket
        self.connection_manager = connection_manager

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender=Sender.BOT, message=token, type=MessageType.STREAM)
        await self.connection_manager.broadcast(resp)


class QuestionGenCallbackHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, connection_manager):
        # self.websocket = websocket
        self.connection_manager = connection_manager

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        resp = ChatResponse(
            sender=Sender.BOT, message="Synthesizing question...", type=MessageType.INFO
        )
        await self.connection_manager.broadcast(resp.dict())


# Dependency
async def get_websocket_manager(websocket: WebSocket):
    manager = ConnectionManager()
    await manager.connect(websocket)
    try:
        yield manager
    finally:
        await manager.disconnect(websocket)


def get_stream_manager(manager):
    """Return a new stream manager."""
    stream_handler = StreamingLLMCallbackHandler(manager)
    return AsyncCallbackManager([stream_handler])
