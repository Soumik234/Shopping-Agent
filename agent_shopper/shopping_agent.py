import base64
import json
import os
import sqlite3
import time
from typing import Optional
print("1. imports starting")

from dotenv import load_dotenv
print("2. dotenv ok")

from langchain.agents import create_agent
print("3. create_agent ok")

from langchain.tools import tool
print("4. tool ok")

from langchain_core.messages import HumanMessage
print("5. HumanMessage ok")

from langchain_groq import ChatGroq
print("6. ChatGroq ok")

from reviews_api import get_product_rating

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

_prefs_cache: dict[str, str] = {}
_prefs_cache_ts: float = 0.0
_PREFS_TTL = 30  # seconds

# NOTE: creating ChatGroq clients and the agent can fail at import time if
# the GROQ API key isn't set. Lazily initialize them when first used.


def _create_agent():
    print("7. creating llm")
    llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.1)  # change this
    vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

    print("8. creating agent")
    agent_obj = create_agent(
        tools=[
            search_products,
            list_inventory,
            get_rating,
            checkout,
            get_order_history,
            save_preference,
            get_preferences,
            describe_product_image
        ],
        model=llm,
      system_prompt=(
            "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
            "IMAGE SEARCH — when the user provides an image path:\n"
            "1. Call describe_product_image with the path to identify the product.\n"
            "2. Use the returned search_query and is_organic to call search_products.\n"
            "3. Continue with the BROWSING flow from step 2 onwards.\n\n"
            "BROWSING — when the user describes what they want to buy:\n"
            "1. Call search_products — ratings are already included in the results (average_rating field).\n"
            "2. DO NOT call get_rating separately — it is redundant and slow.\n"
            "3. Filter by the user's minimum rating if specified using the average_rating field.\n"
            "4. Present qualifying products as a numbered list. For each item use this exact format "
            "   (plain text, no backticks, no code blocks, no bold, no italic):\n\n"
            "   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>\n\n"
            "   Add a blank line between each product entry for readability. "
            "   Always include (ID:X) so you can reference it later.\n"
            "5. If only one product qualifies, still show it in the list and ask: "
            "   'Would you like to order it? Just say yes or give me the number.'\n"
            "6. Do NOT call checkout at this stage.\n\n"
            "INVENTORY CHECK — MANDATORY TOOL USE:\n"
            "When user asks about ingredients for ANY recipe, make ONE call to search_products with a broad query (for overnight oatmeal, an empty query is fine) and use those results to identify available items.\n"
            "Then make additional calls only for ingredients that were not found. NEVER make more than 3 search_products calls per request.\n"
            "NEVER say items are unavailable without calling search_products first.\n"
            "Show only products returned by the tool using BROWSING format with (ID:X).\n\n"
            "ORDERING — when the user confirms they want to buy (e.g. 'yes', 'sure', 'go ahead', "
            "'order number 2', 'the first one', 'get me #3', or explicitly names a product like 'order almond milk'):\n"
            "1. If the user mentions a specific product name that was NOT in your last message (e.g., 'order almond milk'), "
            "   immediately pivot to the BROWSING flow first to find the product ID.\n"
            "2. Otherwise, look at your previous message to find the (ID:X) for the chosen product "
            "   (if only one was listed and the user says 'yes', use that product's ID).\n"
            "3. Call checkout with that product_id (the number from (ID:X)).\n"
            "4. Confirm the order to the user in plain text.\n\n"
            "ORDER HISTORY — when the user asks about previous purchases:\n"
            "1. If the user asks 'What have I ordered before?', 'Show my orders', "
            "'Show my order history', 'Previous purchases', or similar, call "
            "get_order_history.\n"
            "2. Display the returned orders in a numbered list.\n"
            "3. Include product name, order ID, price, and ordered date.\n"
            "4. Do not call checkout when showing order history.\n\n"
            "OFF-TOPIC POLICY — stay shopping related:\n"
            "1. Only answer questions that are directly about shopping, products, prices, orders, categories, ratings, or preferences.\n"
            "2. If the user asks anything unrelated to shopping, politely refuse and say: \"Sorry, I can only help with shopping-related questions. Please ask about products, orders, preferences, or checkout.\"\n"
            "3. Do not try to answer general knowledge, programming, personal, or unrelated questions.\n"
            "4. If the request is off-topic, do not call any tools or provide extra detail beyond the refusal.\n\n"
            "Never place an order unless the user explicitly confirms. "
            "Never guess a product_id — always take it from the (ID:X) in your own previous message."
            "\n\n"
            "USER PREFERENCES — when handling user requests:\n"
            "1. Before searching for products, call get_preferences.\n"
            "2. If the user has saved preferences and does not specify constraints, use those preferences.\n"
            "3. If the user's message contains the words 'remember', 'always', 'from now on', or 'I prefer', call save_preference to save the preference.\n"
            "4. Explicit instructions in the current message override stored preferences.\n"
        ),
    )
    print("9. agent created")
    # attach llm/vision_llm so tools can reference them if needed
    agent_obj._llm = llm
    agent_obj._vision_llm = vision_llm
    return agent_obj


class _LazyAgent:
    def __init__(self):
        self._agent = None

    def _ensure(self):
        if self._agent is None:
            self._agent = _create_agent()

    def invoke(self, *args, **kwargs):
        self._ensure()
        return self._agent.invoke(*args, **kwargs)

    def __getattr__(self, name):
        self._ensure()
        return getattr(self._agent, name)


# Export a lazy agent instance so `from shopping_agent import agent` works without
# requiring GROQ credentials at import time.
agent = _LazyAgent()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def list_inventory(query: Optional[str] = None, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    List inventory items matching the requested query. Treat this as the product availability tool
    for user requests like 'what ingredients are in inventory' or 'what do we have available'.
    Returns the same JSON array format as search_products.
    """
    return search_products.func(query or "", max_price=max_price, is_organic=is_organic)


@tool
def search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    Search the product database by keyword (matched against name, description, and category).
    Optionally filter by maximum price and/or organic status.
    Returns a JSON array of matching products, each with: id, name, category, price,
    description, is_organic, image_url, average_rating, review_count.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = """
        SELECT p.id,
               p.name,
               p.category,
               p.price,
               p.description,
               p.is_organic,
               p.image_url,
               ROUND(COALESCE(AVG(r.rating), 0), 1) AS average_rating,
               COUNT(r.id) AS review_count
        FROM products p
        LEFT JOIN reviews r ON r.product_id = p.id
        WHERE 1=1
    """
    params: list = []

    if query:
        sql += " AND (p.name LIKE ? OR p.description LIKE ? OR p.category LIKE ?)"
        like = f"%{query}%"
        params.extend([like, like, like])

    if max_price is not None:
        sql += " AND p.price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND p.is_organic = ?"
        params.append(1 if is_organic else 0)

    sql += " GROUP BY p.id"

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    products = [
        {
            "id":             row[0],
            "name":           row[1],
            "category":       row[2],
            "price":          row[3],
            "description":    row[4],
            "is_organic":     bool(row[5]),
            "image_url":      row[6],
            "average_rating": row[7],
            "review_count":   row[8],
        }
        for row in rows
    ]
    return json.dumps(products)


@tool
def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(product_id)
    return json.dumps(result)


@tool
def checkout(product_id: int) -> str:
    """
    Place an order for the given product ID. Saves the order to the database and returns
    a confirmation message with the order ID, product name, and price.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if an identical order was placed in the last 5 seconds (idempotency guard)
    cursor.execute("""
        SELECT id FROM orders 
        WHERE product_id = ? 
        AND datetime(ordered_at) > datetime('now', '-5 seconds')
        LIMIT 1
    """, (product_id,))
    
    recent_order = cursor.fetchone()
    if recent_order:
        conn.close()
        return f"Order already placed for this product. Order #{recent_order[0]} is pending."
    
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row
    cursor.execute(
        "INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?)",
        (product_id, name, price),
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been successfully ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. Thank you for shopping with us!"
    )

@tool
def get_order_history(limit: int = 10) -> str:
    """
    Retrieve recent orders.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            o.id,
            o.product_id,
            o.product_name,
            o.price,
            o.ordered_at,
            p.image_url
        FROM orders o
        LEFT JOIN products p ON o.product_id = p.id
        ORDER BY o.ordered_at DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return json.dumps([
        {
            "order_id": row[0],
            "product_id": row[1],
            "product_name": row[2],
            "price": row[3],
            "ordered_at": row[4],
            "image_url": row[5],
        }
        for row in rows
    ])
    
@tool
def save_preference(key: str, value: str) -> str:
    """
    Save a user preference for future shopping sessions.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO user_preferences
        (preference_key, preference_value)
        VALUES (?, ?)
    """, (key, value))

    conn.commit()
    conn.close()

    return f"Saved preference: {key}={value}"

@tool
def get_preferences() -> str:
    """
    Retrieve all saved user preferences.
    """

    global _prefs_cache, _prefs_cache_ts

    if time.time() - _prefs_cache_ts < _PREFS_TTL:
        return json.dumps(_prefs_cache)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT preference_key, preference_value
        FROM user_preferences
    """)

    rows = cursor.fetchall()
    conn.close()

    _prefs_cache = {
        key: value
        for key, value in rows
    }
    _prefs_cache_ts = time.time()

    return json.dumps(_prefs_cache)

@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and return its key attributes as a JSON object.
    Use this when the user uploads a photo of a product they are interested in.
    The returned attributes can be used directly with search_products.
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_data}"},
        },
        {
            "type": "text",
            "text": (
                "Look at this product image and extract its key attributes. "
                "Return ONLY a JSON object with these fields:\n"
                "- product_type: what kind of product it is (e.g. honey, olive oil, almonds)\n"
                "- search_query: a short keyword to search for it (e.g. 'honey', 'olive oil')\n"
                "- is_organic: true if the label says organic, false if not, null if unclear\n"
                "- description: one sentence describing the product"
            ),
        },
    ])

    # Ensure the vision model is initialized (creates agent/clients if needed)
    try:
        agent._ensure()
        vision_model = agent._agent._vision_llm
    except Exception:
        # Fallback: try to use a global if present
        vision_model = globals().get("vision_llm")

    response = vision_model.invoke([message])
    return response.content



if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I want to buy organic honey with 4.5+ rating and less than $20 price."
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)
