import json
import sys
import os
import logging
from typing import List, Union, Optional, Dict, Any
from collections import OrderedDict
from datetime import datetime
import time
import gc
from pydantic.v1 import BaseModel, validator  # v2 throws warnings
import tiktoken

logger = logging.getLogger("chat-analyzer.history")

DEFAULT_MODEL_SLUG = "gpt-3.5-turbo"


class Author(BaseModel):
    role: str
    name: Optional[str] = None


class ContentPartMetadata(BaseModel):
    dalle: Any = None  # was  Optional[dict]

    class Config:
        extra = "allow"  # ✅ ignore unknown fields like 'fovea' or other nested metadata

class ContentPart(BaseModel):
    content_type: str
    asset_pointer: Optional[str] = None
    size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fovea: Optional[Any] = None
    metadata: Optional[ContentPartMetadata] = None

    class Config:
        extra = "allow"  # ✅ ignore unknown fields like 'fovea' or other nested metadata


class Content(BaseModel):
    content_type: str
    parts: Optional[List[Union[str, ContentPart]]] = None
    text: Optional[str] = None

    @validator("parts", pre=True)
    def validate_parts(cls, value):
        if not isinstance(value, list):
            return value
        processed = []
        for item in value:
            if isinstance(item, dict) and "content_type" in item:
                processed.append(ContentPart(**item))
            else:
                processed.append(item)
        return processed

    class Config:
        extra = "allow"  # ✅ ignore unknown fields like 'fovea' or other nested metadata


class MessageMetadata(BaseModel):
    model_slug: Optional[str] = None
    class Config:
        extra = "allow"  # ✅ ignore unknown fields like 'fovea' or other nested metadata


class Message(BaseModel):
    id: str
    author: Author
    create_time: Optional[float] = None
    update_time: Optional[float] = None
    content: Optional[Content] = None
    metadata: Optional[MessageMetadata] = None

    @property
    def text(self) -> str:
        if self.content:
            if self.content.text:
                return self.content.text
            elif self.content.content_type == 'text' and self.content.parts:
                return " ".join(str(part) for part in self.content.parts)
            elif self.content.content_type == 'multimodal_text':
                return "[TODO: process DALL-E and other multimodal]"
        return ""

    @property
    def role(self) -> str:
        return self.author.role

    @property
    def created(self) -> datetime:
        try:
            return datetime.fromtimestamp(self.create_time) if self.create_time else datetime.now()
        except Exception as e:
            logger.error(f"Error converting timestamp: {e}")
            return datetime.now()

    @property
    def created_str(self) -> str:
        return self.created.strftime('%Y-%m-%d %H:%M:%S')

    @property
    def model_str(self) -> str:
        return self.metadata.model_slug or DEFAULT_MODEL_SLUG

    def count_tokens(self) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.model_str)
        except KeyError:
            encoding = tiktoken.encoding_for_model(DEFAULT_MODEL_SLUG)
        return len(encoding.encode(self.text))

    class Config:
        extra = "allow"  # ✅ ignore unknown fields like 'fovea' or other nested metadata


class MessageMapping(BaseModel):
    id: str
    message: Optional[Message]
    parent: Optional[str] = None


    class Config:
        extra = "allow"  # ✅ ignore unknown fields like 'fovea' or other nested metadata


class Conversation(BaseModel):
    id: str
    title: Optional[str]
    create_time: float
    update_time: float
    mapping: OrderedDict[str, MessageMapping]

    @property
    def messages(self) -> List:
        return [msg.message for k, msg in self.mapping.items() if msg.message and msg.message.text]

    @property
    def created(self) -> datetime:
        return datetime.fromtimestamp(self.create_time)#.strftime('%Y-%m-%d %H:%M:%S')

    @property
    def created_str(self) -> str:
        return self.created.strftime('%Y-%m-%d %H:%M:%S')

    @property
    def updated(self) -> datetime:
        return datetime.fromtimestamp(self.update_time)

    @property
    def updated_str(self) -> str:
        return self.updated.strftime('%Y-%m-%d %H:%M:%S')

    @property
    def title_str(self) -> str:
        return self.title or '[Untitled]'

    @property
    def total_length(self) -> int:
        start_time = self.created
        end_time = max(msg.created for msg in self.messages) if self.messages else start_time
        return (end_time - start_time).total_seconds()

    class Config:
        extra = "allow"  # ✅ ignore unknown fields like 'fovea' or other nested metadata


def load_conversations(path: str) -> List[Conversation]:
    with open(path, 'r') as f:
        conversations_json = json.load(f)

    # Load the JSON data into these models
    try:
        conversations = [Conversation(**conv) for conv in conversations_json]
        success = True
    except Exception as e:
        print(str(e))
        sys.exit(1)

    print(f"-- Loaded {len(conversations)} conversations")
    return conversations
