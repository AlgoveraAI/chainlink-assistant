from enum import Enum
from typing import Optional, List
from pydantic import BaseModel


# Constants
class Sender(str, Enum):
    BOT = "bot"
    YOU = "you"


class MessageType(str, Enum):
    START = "start"
    STREAM = "stream"
    END = "end"
    ERROR = "error"
    INFO = "info"
    STATUS = "status"


# Models
class ChatInput(BaseModel):
    username: str
    message: str
    memory: Optional[bool] = False
    memory_uuid: Optional[str] = None
    context_uuids: Optional[List[str]] = None
    assistant_uuid: Optional[str] = None


class ChatResponse(BaseModel):
    sender: Sender
    message: str
    type: MessageType
    memory_uuid: Optional[str] = None
