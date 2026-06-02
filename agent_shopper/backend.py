import json
import os
import re
import sqlite3
import tempfile
import uuid
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

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


class ProductResponse(BaseModel):
    id: int
    index: int
    name: str
    price: float
    rating: Optional[float] = None
    is_organic: bool
    image_url: Optional[str] = None


class ReviewResponse(BaseModel):
    rating: float
    reviewer_name: Optional[str] = None
    review_text: Optional[str] = None


class ProductDetailResponse(BaseModel):
    id: int
    name: str
    category: Optional[str] = None
    price: float
    description: Optional[str] = None
    is_organic: bool
    image_url: Optional[str] = None
    rating: Optional[float] = None
    reviews: list[ReviewResponse] = Field(default_factory=list)


class ChatResponse(BaseModel):
    role: str
    content: str
    products: list[ProductResponse] = Field(default_factory=list)


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
    image_url: Optional[str] = None


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


def _extract_product_ids_from_message(content: str) -> list[tuple[int, int]]:
    """
    Extract product ID and index from assistant message.
    Format: #<number>. <name> (ID:<product_id>) — ...
    Returns: list of (index, product_id) tuples in order.
    """
    pattern = r'#(\d+)\..+?\(ID:(\d+)\)'
    matches = re.findall(pattern, content)
    return [(int(idx), int(pid)) for idx, pid in matches]


def _fetch_product_data(product_ids_with_indices: list[tuple[int, int]]) -> list[ProductResponse]:
    """
    Fetch product data with ratings from database.
    Preserves order from the indices in the assistant message.
    """
    if not product_ids_with_indices:
        return []

    db_path = os.path.join(os.path.dirname(__file__), "store.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    product_ids = [pid for _, pid in product_ids_with_indices]
    placeholders = ",".join("?" * len(product_ids))

    cursor.execute(f"""
        SELECT
            p.id,
            p.name,
            p.price,
            p.is_organic,
            p.image_url,
            COALESCE(AVG(r.rating), 0) AS rating
        FROM products p
        LEFT JOIN reviews r ON r.product_id = p.id
        WHERE p.id IN ({placeholders})
        GROUP BY p.id
    """, product_ids)

    rows = cursor.fetchall()
    conn.close()

    # Create a map of product_id -> product data
    product_map = {row[0]: row for row in rows}

    # Build response in original order, indexed by the assistant message numbering
    products = []
    for idx, pid in product_ids_with_indices:
        if pid in product_map:
            row = product_map[pid]
            products.append(
                ProductResponse(
                    id=row[0],
                    index=idx,
                    name=row[1],
                    price=row[2],
                    is_organic=bool(row[3]),
                    image_url=row[4],
                    rating=float(row[5]) if row[5] else None,
                )
            )

    return products


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
    content = assistant_message.get("content", "")

    # Extract product recommendations from content
    product_ids_with_indices = _extract_product_ids_from_message(content)
    products = _fetch_product_data(product_ids_with_indices)
    _log(request_id, f"Extracted {len(products)} products from assistant message")

    return ChatResponse(
        role=assistant_message["role"],
        content=content,
        products=products,
    )


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
        content = assistant_message.get("content", "")

        # Extract product recommendations from content
        product_ids_with_indices = _extract_product_ids_from_message(content)
        products = _fetch_product_data(product_ids_with_indices)
        _log(request_id, f"Extracted {len(products)} products from image analysis")

        return ChatResponse(
            role=assistant_message["role"],
            content=content,
            products=products,
        )
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


def _fetch_all_product_data() -> list[ProductDetailResponse]:
    db_path = os.path.join(os.path.dirname(__file__), "store.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            p.id,
            p.name,
            p.category,
            p.price,
            p.description,
            p.is_organic,
            p.image_url,
            AVG(r.rating) AS rating
        FROM products p
        LEFT JOIN reviews r ON r.product_id = p.id
        GROUP BY p.id
        ORDER BY p.id
        """
    )
    rows = cursor.fetchall()

    product_ids = [row[0] for row in rows]
    reviews_map = {pid: [] for pid in product_ids}
    if product_ids:
        placeholders = ",".join("?" * len(product_ids))
        cursor.execute(
            f"""
            SELECT product_id, rating, reviewer_name, review_text
            FROM reviews
            WHERE product_id IN ({placeholders})
            ORDER BY product_id
            """,
            product_ids,
        )
        for product_id, rating, reviewer_name, review_text in cursor.fetchall():
            reviews_map[product_id].append(
                ReviewResponse(
                    rating=rating,
                    reviewer_name=reviewer_name,
                    review_text=review_text,
                )
            )

    conn.close()

    products = []
    for row in rows:
        product_id = row[0]
        products.append(
            ProductDetailResponse(
                id=product_id,
                name=row[1],
                category=row[2],
                price=row[3],
                description=row[4],
                is_organic=bool(row[5]),
                image_url=row[6],
                rating=float(row[7]) if row[7] is not None else None,
                reviews=reviews_map.get(product_id, []),
            )
        )

    return products


@app.get("/products", response_model=list[ProductDetailResponse])
async def get_products_endpoint() -> list[ProductDetailResponse]:
    request_id = str(uuid.uuid4())
    _log(request_id, "GET /products request received")

    try:
        return _fetch_all_product_data()
    except Exception as e:
        _log(request_id, f"Product list error: {type(e).__name__}: {e}")
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