import json
import os
import tempfile
import uuid
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from shopping_agent import (
    agent,
    get_order_history,
    get_preferences,
    save_preference,
)

app = FastAPI(title="Shopping Assistant API")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    role: str
    content: str


class HealthResponse(BaseModel):
    status: str


class SavePreferenceRequest(BaseModel):
    key: str
    value: str


class OrderResponse(BaseModel):
    order_id: int
    product_id: int
    product_name: str
    price: float
    ordered_at: str


def _extract_message(message: Any) -> dict[str, str]:
    if hasattr(message, "content"):
        return {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", ""),
        }
    if isinstance(message, dict):
        return {
            "role": message.get("role", "assistant"),
            "content": message.get("content", ""),
        }
    raise ValueError("Unexpected message type")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    request_id = str(uuid.uuid4())
    _log(request_id, "POST /chat request received")
    result = _invoke_agent(request_id, [message.dict() for message in request.messages])
    messages = result.get("messages") if isinstance(result, dict) else None

    if not messages:
        _log(request_id, "No response from agent")
        raise HTTPException(status_code=500, detail="No response from agent")

    assistant_message = _extract_message(messages[-1])
    return ChatResponse(**assistant_message)


@app.post("/upload-image", response_model=ChatResponse)
async def upload_image(file: UploadFile = File(...)) -> ChatResponse:
    request_id = str(uuid.uuid4())
    _log(request_id, "POST /upload-image request received")

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    prompt = (
        f"I uploaded a product image. Please analyze it and find similar products in the store. "
        f"Image path: {image_path}"
    )

    try:
        result = _invoke_agent(request_id, [{"role": "user", "content": prompt}])
        messages = result.get("messages") if isinstance(result, dict) else None
        if not messages:
            _log(request_id, "No response from agent")
            raise HTTPException(status_code=500, detail="No response from agent")

        assistant_message = _extract_message(messages[-1])
        return ChatResponse(**assistant_message)
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass

def _log(request_id: str, message: str) -> None:
    print(f"[{request_id}] {message}")


def _invoke_agent(request_id: str, messages: list[dict[str, str]]) -> dict:
    _log(request_id, f"Agent request received: {messages}")
    try:
        return agent.invoke({"messages": messages})
    except Exception as e:
        _log(request_id, f"Agent exception: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/preferences")
async def get_preferences_endpoint() -> dict[str, str]:
    request_id = str(uuid.uuid4())
    _log(request_id, "GET /preferences request received")

    try:
        result = get_preferences()
        return json.loads(result)
    except Exception as e:
        _log(request_id, f"Preferences error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preferences")
async def save_preferences_endpoint(request: SavePreferenceRequest) -> dict[str, str]:
    request_id = str(uuid.uuid4())
    _log(request_id, "POST /preferences request received")

    try:
        result = save_preference(request.key, request.value)
        return {"message": result}
    except Exception as e:
        _log(request_id, f"Save preference error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/orders", response_model=list[OrderResponse])
async def get_orders_endpoint() -> list[OrderResponse]:
    request_id = str(uuid.uuid4())
    _log(request_id, "GET /orders request received")

    try:
        result = get_order_history()
        return [OrderResponse(**order) for order in json.loads(result)]
    except Exception as e:
        _log(request_id, f"Order history error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))