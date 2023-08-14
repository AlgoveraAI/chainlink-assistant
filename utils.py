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
    "algovera_admin_08062023",
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


def get_folders(firebase, context_uuids: List[str]) -> List[str]:
    folders = [
        f"{firebase.get_context(context)['folder']}/{v}"
        for context in context_uuids
        for _, v in json.loads(firebase.get_context(context)["vecdb_uuids"]).items()
    ]
    return folders


def calculate_credits(cb, margin=0.7, costpercredit=0.01):
    total_tokens = cb.total_tokens
    cost = cb.total_cost

    cost = cost * (1 + margin) / costpercredit
    final_cost = math.ceil(cost)

    if final_cost < 1:
        final_cost = 1

    if total_tokens < 1:
        total_tokens = -1
    return total_tokens, final_cost


def get_chat_history(memory_uuid, firebase, new=False):
    if new:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer",
        )
        return memory

    else:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer",
        )
        chats = firebase.get_memory(memory_uuid)["chat_history"]

        assert len(chats) > 0, "No chat history found"

        for i in range(len(chats)):
            if (i == 0) or (i % 2 == 0):
                memory.chat_memory.add_user_message(chats[i])
            else:
                memory.chat_memory.add_ai_message(chats[i])

        return memory


def set_chat_history(memory_uuid, uid, chat_history, firebase):
    chats = [e.content.strip() for e in chat_history]
    firebase.set_memory(memory_uuid, {"chat_history": chats, "uid": uid})


def get_stream_manager(manager):
    """Return a new stream manager."""
    stream_handler = StreamingLLMCallbackHandler(manager)
    return AsyncCallbackManager([stream_handler])
