import json
import uuid
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, WebSocket, Request, Depends, WebSocketDisconnect
from langchain.callbacks import get_openai_callback
from firebase_admin import auth
from schemas import ChatInput, ChatResponse, Sender, MessageType
from utils import (
    ConnectionManager,
    QuestionGenCallbackHandler,
    StreamingLLMCallbackHandler,
    get_websocket_manager,
    calculate_credits,
    USERNAMES,
    get_chat_history,
    set_chat_history,
    createLogHandler,
)
from config import firebase
from workflow.get_chain_no_mem import get_answer
from workflow.get_chain_mem import get_answer_memory


logger = createLogHandler("assistant", "assistant.log")

app = FastAPI()

origins = [
    "*",  # Allow all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

async def authenticate_user(token, manager):
    if not token:
        await manager.broadcast(
            ChatResponse(
                sender=Sender.BOT,
                message=f"Please provide a username",
                type=MessageType.ERROR,
            ).dict()
        )
    elif token not in USERNAMES:
        try:
            decoded_token = auth.verify_id_token(token)
            uid = decoded_token["uid"]
            return uid
        except Exception as e:
            logger.info(e)
            await manager.broadcast(
                ChatResponse(
                    sender=Sender.BOT,
                    message=f"Invalid credential",
                    type=MessageType.ERROR,
                ).dict()
            )
            return None
    elif token in USERNAMES:
        return token


async def check_uuids(context_uuids, assistant_uuid, manager):
    logger.info(f"context_uuids: {context_uuids}, assistant_uuid: {assistant_uuid}")
    if (context_uuids == [""]) and (assistant_uuid == ""):
        await manager.broadcast(
            ChatResponse(
                sender=Sender.BOT,
                message=f"Please provide context or assistant uuid, not both",
                type=MessageType.ERROR,
            ).dict()
        )
        return False
    else:
        return True


@app.websocket("/chat_chainlink")
async def chat_endpoint_chainlink(
    websocket: WebSocket, manager: ConnectionManager = Depends(get_websocket_manager)
):
    try:
        verified = False
        payer_uuid = None
        memory = None
        memory_uuid = None
        while True:
            data = await websocket.receive_text()
            message = ChatInput(**json.loads(data))
            logger.info(message)
            if not verified:
                # Validate user credentials
                logger.info("Validating credentials")
                uid = await authenticate_user(message.username, manager)
                if not uid:
                    continue
                verified = True

            resp = ChatResponse(
                sender=Sender.YOU, message=message.message, type=MessageType.STREAM
            )
            await manager.broadcast(resp)

            # Construct a response
            start_resp = ChatResponse(
                sender=Sender.BOT, message="", type=MessageType.START
            )
            await manager.broadcast(start_resp)

            # get the answer
            if message.memory:
                logger.info("Getting answer with memory")
                if (message.memory_uuid != "") and (message.memory_uuid is not None):
                    if not memory:
                        memory_uuid = message.memory_uuid
                        memory = get_chat_history(
                            memory_uuid, firebase=firebase, new=False
                        )
                        logger.info(memory_uuid)

                else:
                    if not memory:
                        memory_uuid = uuid.uuid4().hex
                        memory = get_chat_history(
                            memory_uuid, firebase=firebase, new=True
                        )
                        logger.info(memory_uuid)

                answer, memory = await get_answer_memory(
                    message.message, memory=memory, manager=manager
                )
                logger.info(answer)

                set_chat_history(
                    memory_uuid=memory_uuid,
                    uid=uid,
                    firebase=firebase,
                    chat_history=memory.chat_memory.messages,
                )
            else:
                logger.info("Getting answer without memory")
                answer = await get_answer(message.message, manager=manager)
                logger.info(answer)

            firebase.remove_credit(payer_uuid, 1)

            end_resp = ChatResponse(
                sender=Sender.BOT,
                message="",
                type=MessageType.END,
                memory_uuid=memory_uuid,
            )
            await manager.broadcast(end_resp)

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected")