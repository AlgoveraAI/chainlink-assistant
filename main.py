import json
from datetime import datetime
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
from fastapi import (
    FastAPI,
    WebSocket,
    Request,
    Depends,
    WebSocketDisconnect,
    BackgroundTasks,
    status,
    HTTPException,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ingest_script import ingest_task
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
from chat.utils import get_search_retriever
from config import get_logger
import os

# Secure endpoints using a bearer token
bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials


# Global variables
logger = get_logger(__name__)

new_ingest = False
new_ingest_time = None
templates = Jinja2Templates(directory="templates")
try:
    chainlink_search_retrevier = get_search_retriever()
except Exception as err:
    chainlink_search_retrevier = None
    logger.info("Search retriever not loaded: " + str(err))

load_dotenv()

app = FastAPI(dependencies=[Depends(validate_token)])


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ingest")
def ingest(background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_task)
    global new_ingest, new_ingest_time
    new_ingest = True
    new_ingest_time = datetime.now()
    return {"message": "Ingestion started."}


@app.get("/chainlink")
async def get_chainlink(request: Request):
    return templates.TemplateResponse("chainlink.html", {"request": request})


@app.websocket("/chat_chainlink")
async def chat_endpoint_chainlink(
    websocket: WebSocket, manager: ConnectionManager = Depends(get_websocket_manager)
):
    try:
        verified = False
        while True:
            data = await websocket.receive_text()
            message = ChatInput(**json.loads(data))
            logger.info(message)
            if not verified:
                # Validate user credentials
                logger.info("Validating credentials")
                if message.username not in USERNAMES:
                    logger.info("Invalid username")
                    await websocket.send_json({"error": "Invalid username"})
                    await websocket.close()
                    return
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

            logger.info("Getting answer without memory")
            answer = await get_answer(message.message, manager=manager)
            logger.info(answer)

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

    # Check if new ingest and if its been 3 hours reload the search retriever
    if new_ingest:
        time_diff = datetime.now() - new_ingest_time
        if time_diff.seconds > 10800:
            logger.info("Reloading search retriever")
            chainlink_search_retrevier = get_search_retriever()
            new_ingest = False
            new_ingest_time = None

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
