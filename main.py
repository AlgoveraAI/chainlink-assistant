import os
import json
import random
from dotenv import load_dotenv
load_dotenv()
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import (
    FastAPI,
    WebSocket,
    Request,
    Depends,
    WebSocketDisconnect,
    status,
    HTTPException,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from schemas import (
    ChatInput,
    ChatResponse,
    Sender,
    MessageType,
    SearchRequestSchema,
    SearchResponseSchema,
)
from utils import get_websocket_manager, ConnectionManager, USERNAMES
from chat.get_chain_no_mem import get_answer
from chat.utils import get_search_retriever, get_retriever_chain
from config import get_logger, WS_HOST, HTTP_HOST

### Secure disabled for FastAPI issues with protected ws ###
# Secure endpoints using a bearer token
#bearer_scheme = HTTPBearer()
#BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
#assert BEARER_TOKEN is not None

#def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
#    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
#        raise HTTPException(status_code=401, detail="Invalid or missing token")
#    return credentials

# Global variables
logger = get_logger(__name__)

API_KEYS = os.environ.get("OPENAI_API_KEYS").split(",")
os.environ["OPENAI_API_KEY"] = random.choice(API_KEYS)


templates = Jinja2Templates(directory="templates")


def initial_setup():
    try:
        chainlink_search_retrevier = get_search_retriever()
    except Exception as err:
        chainlink_search_retrevier = None
        logger.info("Search retriever not loaded: " + str(err))

    try:
        retriever, chain = get_retriever_chain()
    except Exception as err:
        retriever = None
        chain = None
        logger.info("Retriever chain not loaded: " + str(err))

    return chainlink_search_retrevier, retriever, chain

chainlink_search_retrevier, retriever, chain = initial_setup()
# Make sure the retriever is loaded
if chainlink_search_retrevier is None:
    raise Exception("Search retriever not loaded")
if retriever is None:
    raise Exception("Retriever not loaded")
if chain is None:
    raise Exception("Chain not loaded")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-api-key"]
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/chainlink")
async def get_chainlink(request: Request):
    return templates.TemplateResponse("chainlink.html", {"request": request, "ws_host": WS_HOST, "http_host": HTTP_HOST})

@app.websocket("/chat_chainlink")
async def chat_endpoint_chainlink(
    websocket: WebSocket, manager: ConnectionManager = Depends(get_websocket_manager)
):
    try:
        # verified = False
        while True:
            # Select a random OpenAI API key
            if not len(API_KEYS) > 0:
                raise Exception("Not enough API keys. Set OPENAI_API_KEYS")

            api_key = random.choice(API_KEYS)
            os.environ["OPENAI_API_KEY"] = api_key

            data = await websocket.receive_text()
            message = ChatInput(**json.loads(data))
            logger.info(message)
            # if not verified:
            #     # Validate user credentials
            #     logger.info("Validating credentials")
            #     if message.username not in USERNAMES:
            #         logger.info("Invalid username")
            #         await websocket.send_json({"error": "Invalid username"})
            #         await websocket.close()
            #         return
            #     verified = True

            resp = ChatResponse(
                sender=Sender.YOU, message=message.message, type=MessageType.STREAM
            )
            await manager.broadcast(resp)

            # Construct a response
            start_resp = ChatResponse(
                sender=Sender.BOT, message="", type=MessageType.START
            )
            await manager.broadcast(start_resp)

            logger.info("Getting answer without memory")
            try:
                answer = await get_answer(message.message, manager=manager, retriever=retriever, base_chain=chain)
                logger.info(answer)
            except Exception as err:
                logger.info("Error getting answer: " + str(err))
                message = "OpenAI Error. There was an error getting an answer. Please try again."
                await websocket.send_json({"error": message})

            end_resp = ChatResponse(
                sender=Sender.BOT,
                message="",
                type=MessageType.END,
            )
            await manager.broadcast(end_resp)

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected")


@app.post(
    "/search",
    status_code=status.HTTP_200_OK,
    response_model=SearchResponseSchema,
    responses={
        200: {"description": "Successful search."},
        400: {"description": "Bad request."},
        401: {"description": "Unauthorized. Unknown user or Invalid API key."},
        402: {"description": "Insufficient credit."},
        403: {"description": "Forbidden. No permission."},
        500: {"description": "Internal server error."},
    },
)
def search(
    job: SearchRequestSchema,
):
    """Search for documents."""
    global chainlink_search_retrevier
    global new_ingest, new_ingest_time

    # if search retriever is not loaded raise error
    if chainlink_search_retrevier is None:
        raise HTTPException(status_code=500, detail="Search retriever not loaded")

    logger.info("General Search")
    job_dict = job.dict()
    logger.info(job_dict)

    # Get search results
    results = chainlink_search_retrevier.get_relevant_documents(
        query=job_dict["query"], type_=job_dict["type_"]
    )
    logger.info(f"Retrieved {len(results)} documents")
    logger.info(results)

    # Return results
    return SearchResponseSchema(results=results)


@app.post('/refresh')
def refresh():
    global chainlink_search_retrevier, retriever, chain
    try:
        chainlink_search_retrevier, retriever, chain = initial_setup()
        return {"message": "Refreshed."}
    except Exception as err:
        logger.info("Refresh failed: " + str(err))
        return {"message": "Refresh failed: " + str(err)}