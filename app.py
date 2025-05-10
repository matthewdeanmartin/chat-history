from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import sqlite3
import openai
import toml
from datetime import datetime
from markdown import markdown
from collections import defaultdict
import statistics
import logging
import time
import os
import sys

from history import load_conversations
from utils import time_group, human_readable_time
from llms import load_create_embeddings, search_similar, openai_api_cost, TYPE_CONVERSATION, TYPE_MESSAGE
from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin
from mdit_py_plugins.attrs import attrs_plugin
from mdit_py_plugins.deflist import deflist_plugin
from pygments.formatters import HtmlFormatter



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("chat-analyzer")

# Create MarkdownIt parser with all desired plugins
md = (
    MarkdownIt("commonmark", {"highlight": lambda code, lang, _: f'<pre><code class="language-{lang}">{code}</code></pre>'})
    .use(footnote_plugin)
    .use(tasklists_plugin)
    .use(attrs_plugin)
    .use(deflist_plugin)
)

DB_EMBEDDINGS = "data/embeddings.db"
DB_SETTINGS = "data/settings.db"
SECRETS_PATH = "data/secrets.toml"
CONVERSATIONS_PATH = "data/conversations.json"

pygments_css = HtmlFormatter().get_style_defs('.highlight')
with open("static/pygments.css", "w") as f:
    f.write(pygments_css)

# Initialize FastAPI app
app = FastAPI()
api_app = FastAPI(title="API")


# Function to verify database connections
def verify_database_access():
    """Check if database files can be accessed or created"""
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(DB_EMBEDDINGS), exist_ok=True)
        os.makedirs(os.path.dirname(DB_SETTINGS), exist_ok=True)

        # Test database connections
        for db_path in [DB_EMBEDDINGS, DB_SETTINGS]:
            conn = sqlite3.connect(db_path, timeout=10.0)
            conn.close()
        return True
    except Exception as e:
        logger.error(f"Database access verification failed: {e}")
        return False


# Check file size before loading
def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except FileNotFoundError:
        return 0


# Load conversations with timeout protection
def safe_load_conversations():
    """Load conversations with timeouts and progress tracking"""
    file_size = get_file_size_mb(CONVERSATIONS_PATH)
    logger.info(f"Conversations file size: {file_size:.2f} MB")

    if file_size > 100:  # Large file warning
        logger.warning(f"Conversations file is very large ({file_size:.2f} MB). Loading may take time.")

    if not os.path.exists(CONVERSATIONS_PATH):
        logger.error(f"Conversations file not found: {CONVERSATIONS_PATH}")
        return []

    start_time = time.time()
    try:
        conversations = load_conversations(CONVERSATIONS_PATH)
        elapsed = time.time() - start_time
        logger.info(f"Loaded {len(conversations)} conversations in {elapsed:.2f} seconds")
        return conversations
    except Exception as e:
        logger.error(f"Failed to load conversations: {e}")
        return []


# Startup sequence
logger.info("Application starting...")
verify_database_access()

try:
    logger.info("Loading conversations...")
    conversations = safe_load_conversations()
    logger.info(f"Successfully loaded {len(conversations)} conversations")
except Exception as e:
    logger.error(f"Failed during conversation loading: {e}")
    conversations = []

try:
    logger.info("Loading secrets...")
    SECRETS = toml.load(SECRETS_PATH)
    OPENAI_ENABLED = True
    logger.info("OpenAI API access enabled")
except Exception as e:
    logger.warning(f"No secrets found or error loading secrets: {e}")
    logger.info("OpenAI API access disabled")
    OPENAI_ENABLED = False

if OPENAI_ENABLED:
    try:
        openai.organization = SECRETS["openai"]["organization"]
        openai.api_key = SECRETS["openai"]["api_key"]

        logger.info("Loading embeddings (this might take time)...")
        start_time = time.time()
        embeddings, embeddings_ids, embeddings_index = load_create_embeddings(DB_EMBEDDINGS, conversations)
        elapsed = time.time() - start_time
        logger.info(f"Embeddings loaded in {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI or embeddings: {e}")
        OPENAI_ENABLED = False


# All conversation items
@api_app.get("/conversations")
def get_conversations():
    logger.debug("Retrieving all conversations")
    start_time = time.time()

    # try:
        # Get favorites
    conn = connect_settings_db()
    cursor = conn.cursor()
    cursor.execute("SELECT conversation_id FROM favorites WHERE is_favorite = 1")
    rows = cursor.fetchall()
    favorite_ids = [row[0] for row in rows]
    conn.close()

    conversations_data = [{
        "group": time_group(conv.created),
        "id": conv.id,
        "title": conv.title_str,
        "created": conv.created_str,
        "total_length": human_readable_time(conv.total_length, short=True),
        "is_favorite": conv.id in favorite_ids
    } for conv in conversations]

    elapsed = time.time() - start_time
    logger.debug(f"Retrieved {len(conversations_data)} conversations in {elapsed:.2f} seconds")
    return JSONResponse(content=conversations_data)
    # except Exception as e:
    #     logger.error(f"Error retrieving conversations: {e}")
    #     return JSONResponse(content={"error": str(e)}, status_code=500)


# All messages from a specific conversation by its ID
@api_app.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: str):
    logger.debug(f"Retrieving messages for conversation {conv_id}")

    # try:
    conversation = next((conv for conv in conversations if conv.id == conv_id), None)
    if not conversation:
        logger.warning(f"Invalid conversation ID requested: {conv_id}")
        return JSONResponse(content={"error": "Invalid conversation ID"}, status_code=404)

    messages = []
    prev_created = None  # Keep track of the previous message's creation time
    for msg in conversation.messages:
        if not msg:
            continue

        # If there's a previous message and the time difference is 1 hour or more
        if prev_created and (msg.created - prev_created).total_seconds() >= 3600:
            delta = msg.created - prev_created
            time_str = human_readable_time(delta.total_seconds())
            messages.append({
                "text": f"{time_str} passed",
                "role": "internal"
            })

        messages.append({
            "text": f"<article class=\"prose\">{md.render(msg.text)}</article>",
            "role": msg.role,
            "created": msg.created_str
        })

        # Update the previous creation time for the next iteration
        prev_created = msg.created

    response = {
        "conversation_id": conversation.id,
        "messages": messages
    }
    return JSONResponse(content=response)
    # except Exception as e:
    #     logger.error(f"Error retrieving messages for conversation {conv_id}: {e}")
    #     return JSONResponse(content={"error": str(e)}, status_code=500)


@api_app.get("/activity")
def get_activity():
    logger.debug("Retrieving activity data")

    try:
        activity_by_day = defaultdict(int)

        for conversation in conversations:
            for message in conversation.messages:
                day = message.created.date()
                activity_by_day[day] += 1

        activity_by_day = {str(k): v for k, v in sorted(dict(activity_by_day).items())}
        return JSONResponse(content=activity_by_day)
    except Exception as e:
        logger.error(f"Error retrieving activity data: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@api_app.get("/statistics")
def get_statistics():
    logger.debug("Retrieving statistics")

    try:
        # Calculate the min, max, and average lengths
        lengths = []
        for conv in conversations:
            lengths.append((conv.total_length, conv.id))
        # Sort conversations by length
        lengths.sort(reverse=True)

        if lengths:
            min_threshold_seconds = 1
            filtered_min_lengths = [l for l in lengths if l[0] >= min_threshold_seconds]
            min_length = human_readable_time(min(filtered_min_lengths)[0]) if filtered_min_lengths else "N/A"
            max_length = human_readable_time(max(lengths)[0])
            avg_length = human_readable_time(statistics.mean([l[0] for l in lengths]))
        else:
            min_length = max_length = avg_length = "N/A"

        # Generate links for the top 3 longest conversations
        top_3_links = "".join([f"<a href='https://chat.openai.com/c/{l[1]}' target='_blank'>Chat {chr(65 + i)}</a><br/>"
                               for i, l in enumerate(lengths[:3])])

        # Get the last chat message timestamp and backup age
        last_chat_timestamp = max(conv.created for conv in conversations) if conversations else datetime.now()

        return JSONResponse(content={
            "Chat backup age": human_readable_time((datetime.now() - last_chat_timestamp).total_seconds()),
            "Last chat message": last_chat_timestamp.strftime('%Y-%m-%d'),
            "First chat message": min(conv.created for conv in conversations).strftime(
                '%Y-%m-%d') if conversations else "N/A",
            "Shortest conversation": min_length,
            "Longest conversation": max_length,
            "Average chat length": avg_length,
            "Top longest chats": top_3_links
        })
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@api_app.get("/ai-cost")
def get_ai_cost():
    logger.debug("Retrieving AI cost data")

    try:
        tokens_by_month = defaultdict(lambda: {'input': 0, 'output': 0})

        for conv in conversations:
            for msg in conv.messages:
                year_month = msg.created.strftime('%Y-%m')
                token_count = msg.count_tokens()

                if msg.role == "user":
                    tokens_by_month[year_month]['input'] += openai_api_cost(msg.model_str,
                                                                            input=token_count)
                else:
                    tokens_by_month[year_month]['output'] += openai_api_cost(msg.model_str,
                                                                             output=token_count)

        # Make a list of dictionaries
        tokens_list = [
            {'month': month, 'input': int(data['input']), 'output': int(data['output'])}
            for month, data in sorted(tokens_by_month.items())
        ]

        return JSONResponse(content=tokens_list)
    except Exception as e:
        logger.error(f"Error retrieving AI cost data: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Search conversations and messages
@api_app.get("/search")
def search_conversations(query: str = Query(..., min_length=3, description="Search query")):
    logger.debug(f"Searching for: {query}")
    start_time = time.time()

    try:
        def add_search_result(search_results, result_type, conv, msg):
            search_results.append({
                "type": result_type,
                "id": conv.id,
                "title": conv.title_str,
                "text": markdown(msg.text),
                "role": msg.role,
                "created": conv.created_str if result_type == "conversation" else msg.created_str,
            })

        def find_conversation_by_id(conversations, id):
            return next((conv for conv in conversations if conv.id == id), None)

        def find_message_by_id(messages, id):
            return next((msg for msg in messages if msg.id == id), None)

        search_results = []

        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
            query_exact = True
        else:
            query_exact = False

        if OPENAI_ENABLED and not query_exact:
            for _id in search_similar(query, embeddings_ids, embeddings_index):
                conv = find_conversation_by_id(conversations, embeddings[_id]["conv_id"])
                if conv:
                    result_type = embeddings[_id]["type"]
                    if result_type == TYPE_CONVERSATION:
                        msg = conv.messages[0] if conv.messages else None
                    else:
                        msg = find_message_by_id(conv.messages, _id)

                    if msg:
                        add_search_result(search_results, result_type, conv, msg)
        else:
            for conv in conversations:
                query_lower = query.lower()
                if (conv.title or "").lower().find(query_lower) != -1:
                    if conv.messages:
                        add_search_result(search_results, "conversation", conv, conv.messages[0])

                for msg in conv.messages:
                    if msg and msg.text.lower().find(query_lower) != -1:
                        add_search_result(search_results, "message", conv, msg)

                if len(search_results) >= 10:
                    break

        elapsed = time.time() - start_time
        logger.debug(f"Search completed in {elapsed:.2f} seconds, found {len(search_results)} results")
        return JSONResponse(content=search_results)
    except Exception as e:
        logger.error(f"Error during search: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Toggle favorite status
@api_app.post("/toggle_favorite")
def toggle_favorite(conv_id: str):
    logger.debug(f"Toggling favorite status for conversation {conv_id}")

    try:
        conn = connect_settings_db()
        cursor = conn.cursor()

        # Check if the conversation_id already exists in favorites
        cursor.execute("SELECT is_favorite FROM favorites WHERE conversation_id = ?", (conv_id,))
        row = cursor.fetchone()

        if row is None:
            # Insert new entry with is_favorite set to True
            cursor.execute("INSERT INTO favorites (conversation_id, is_favorite) VALUES (?, ?)", (conv_id, True))
            is_favorite = True
        else:
            # Toggle the is_favorite status
            is_favorite = not row[0]
            cursor.execute("UPDATE favorites SET is_favorite = ? WHERE conversation_id = ?", (is_favorite, conv_id))

        conn.commit()
        conn.close()

        return {"conversation_id": conv_id, "is_favorite": is_favorite}
    except Exception as e:
        logger.error(f"Error toggling favorite status: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


def connect_settings_db():
    """Connect to settings database with timeout and error handling"""
    try:
        conn = sqlite3.connect(DB_SETTINGS, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS favorites
                       (
                           conversation_id
                           TEXT
                           PRIMARY
                           KEY,
                           is_favorite
                           BOOLEAN
                       );
                       """)
        conn.commit()
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise


app.mount("/api", api_app)
app.mount("/", StaticFiles(directory="static", html=True), name="Static")

logger.info("Application initialized and ready")