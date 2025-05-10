import openai
import faiss
import numpy as np
import json
import sqlite3
import logging
import time
from tqdm import tqdm

logger = logging.getLogger("chat-analyzer.llms")

TYPE_CONVERSATION = "conversation"
TYPE_MESSAGE = "message"


def get_embedding(text, retries=3, delay=2):
    """Get embedding with retry logic"""
    attempt = 0

    while attempt < retries:
        try:
            logger.debug(f"Getting embedding for text of length {len(text)}")
            result = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return result["data"][0]["embedding"]
        except Exception as e:
            attempt += 1
            if attempt >= retries:
                logger.error(f"Failed to get embedding after {retries} attempts: {e}")
                raise
            logger.warning(f"Embedding attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff


def load_create_embeddings(path: str, conversations):
    """Load and create embeddings with improved error handling and logging"""

    def connect_db(db_name):
        """Connect to SQLite database with timeout"""
        try:
            conn = sqlite3.connect(db_name, timeout=30.0)
            c = conn.cursor()
            c.execute('''
                      CREATE TABLE IF NOT EXISTS embeddings
                      (
                          id
                          TEXT
                          PRIMARY
                          KEY,
                          type
                          TEXT
                          NOT
                          NULL,
                          conv_id
                          TEXT
                          NOT
                          NULL,
                          embedding
                          BLOB
                          NOT
                          NULL
                      );
                      ''')
            conn.commit()
            return conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def load_embeddings(conn):
        """Load embeddings from database with error handling"""
        c = conn.cursor()
        embeddings = {}
        start_time = time.time()
        logger.info("Loading embeddings from database...")

        try:
            query = 'SELECT id, type, conv_id, embedding FROM embeddings'
            c.execute(query)

            # Use fetchmany for memory efficiency
            batch_size = 1000
            fetched = 0

            while True:
                rows = c.fetchmany(batch_size)
                if not rows:
                    break

                for row in rows:
                    _id, _type, conv_id, embedding_bytes = row
                    # Deserialize bytes to NumPy array
                    embedding_array = np.frombuffer(embedding_bytes)
                    embeddings[_id] = {
                        "type": _type,
                        "conv_id": conv_id,
                        "embedding": embedding_array.tolist()
                    }

                fetched += len(rows)
                if fetched % 10000 == 0:
                    logger.info(f"Loaded {fetched} embeddings so far...")

            elapsed = time.time() - start_time
            logger.info(f"Loaded {len(embeddings)} embeddings in {elapsed:.2f} seconds")
        except sqlite3.Error as e:
            logger.error(f"SQLite error while loading embeddings: {e}")

        return embeddings

    def save_embeddings(conn, embeddings):
        """Save embeddings to database with transaction control"""
        if not embeddings:
            return

        c = conn.cursor()
        try:
            # Use a transaction for better performance
            c.execute("BEGIN TRANSACTION")

            for _id, embedding_data in embeddings.items():
                # Serialize NumPy array to bytes
                embedding_bytes = np.array(embedding_data["embedding"]).tobytes()
                try:
                    c.execute("REPLACE INTO embeddings (id, type, conv_id, embedding) VALUES (?, ?, ?, ?)",
                              (_id, embedding_data["type"], embedding_data["conv_id"], embedding_bytes))
                except sqlite3.InterfaceError as e:
                    logger.error(f"Error inserting data into database for ID {_id}: {e}")
                    continue

            conn.commit()
            logger.info(f"Saved {len(embeddings)} embeddings to database")
        except sqlite3.Error as e:
            logger.error(f"Database error while saving embeddings: {e}")
            conn.rollback()
            raise

    def generate_missing_embeddings(db_conn, conversations, embeddings):
        """Generate missing embeddings with progress tracking and error handling"""
        new_embeddings = 0
        embeddings_save = {}
        batch_size = 50  # Save in batches to avoid memory issues

        logger.info("Generating missing embeddings...")

        try:
            for conv in tqdm(conversations):
                if conv.title and conv.id not in embeddings:
                    try:
                        record = {
                            "type": TYPE_CONVERSATION,
                            "conv_id": conv.id,
                            "embedding": get_embedding(conv.title)
                        }
                        embeddings[conv.id] = record
                        embeddings_save[conv.id] = record
                        new_embeddings += 1
                    except Exception as e:
                        logger.error(f"Failed to generate embedding for conversation {conv.id}: {e}")

                for msg in conv.messages:
                    if msg and msg.text and msg.id not in embeddings:
                        try:
                            record = {
                                "type": TYPE_MESSAGE,
                                "conv_id": conv.id,
                                "embedding": get_embedding(msg.text)
                            }
                            embeddings[msg.id] = record
                            embeddings_save[msg.id] = record
                            new_embeddings += 1
                        except Exception as e:
                            logger.error(f"Failed to generate embedding for message {msg.id}: {e}")

                # Save in batches to avoid memory buildup
                if len(embeddings_save) >= batch_size:
                    save_embeddings(db_conn, embeddings_save)
                    embeddings_save = {}

            # Save any remaining embeddings
            if embeddings_save:
                save_embeddings(db_conn, embeddings_save)

            return new_embeddings
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            # Try to save what we have so far
            if embeddings_save:
                try:
                    save_embeddings(db_conn, embeddings_save)
                except:
                    pass
            return new_embeddings

    def build_faiss_index(embeddings):
        """Build FAISS index with error handling"""
        try:
            logger.info("Building FAISS index...")
            start_time = time.time()

            embeddings_ids = list(embeddings.keys())
            embeddings_np = np.array([np.array(embeddings[_id]["embedding"]) for _id in embeddings_ids]).astype(
                'float32')

            d = embeddings_np.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(embeddings_np)

            elapsed = time.time() - start_time
            logger.info(f"Built FAISS index with {index.ntotal} vectors in {elapsed:.2f} seconds")
            return index, embeddings_ids
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            raise

    try:
        db_conn = connect_db(path)
        embeddings = load_embeddings(db_conn)
        logger.info(f"Loaded {len(embeddings)} embeddings")

        new_embeddings = 0
        missing_count = sum(1 for conv in conversations if conv.title and conv.id not in embeddings)
        if missing_count > 0:
            logger.info(f"Found {missing_count} conversations without embeddings. Generating...")
            new_embeddings = generate_missing_embeddings(db_conn, conversations, embeddings)

        if new_embeddings > 0:
            logger.info(f"Created {new_embeddings} new embeddings")

        embeddings_index, embeddings_ids = build_faiss_index(embeddings)
        logger.info(f"Built FAISS index with {embeddings_index.ntotal} embeddings")

        db_conn.close()
        return embeddings, embeddings_ids, embeddings_index
    except Exception as e:
        logger.error(f"Error in embedding process: {e}")
        # Return minimal working data to allow app to start
        return {}, [], faiss.IndexFlatL2(1536)  # 1536 is the embedding dimension for text-embedding-ada-002


def search_similar(query, embeddings_ids, embeddings_index, top_n=10):
    """Search for similar embeddings with error handling"""
    try:
        query_embedding = get_embedding(query)
        query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = embeddings_index.search(query_vector, top_n)

        # Check if indices contains valid values
        valid_indices = [i for i in indices[0] if 0 <= i < len(embeddings_ids)]
        similar_ids = [embeddings_ids[i] for i in valid_indices]

        return similar_ids[:top_n]
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        return []  # Return empty list on error


def openai_api_cost(model, input=0, output=0):
    """Calculate OpenAI API cost"""
    pricing = {
        "gpt-3.5-turbo-4k": {
            "prompt": 0.0015,
            "completion": 0.002,
        },
        "gpt-3.5-turbo-16k": {
            "prompt": 0.003,
            "completion": 0.004,
        },
        "gpt-4-8k": {
            "prompt": 0.03,
            "completion": 0.06,
        },
        "gpt-4-32k": {
            "prompt": 0.06,
            "completion": 0.12,
        },
        "text-embedding-ada-002-v2": {
            "prompt": 0.0001,
            "completion": 0.0001,
        }
    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        if model and 'gpt-4' in model:
            if input + output > 8192:
                model_pricing = pricing["gpt-4-32k"]
            else:
                model_pricing = pricing["gpt-4-8k"]
        elif model and 'gpt-3.5' in model:
            if input + output > 4096:
                model_pricing = pricing["gpt-3.5-turbo-16k"]
            else:
                model_pricing = pricing["gpt-3.5-turbo-4k"]
        else:
            model_pricing = pricing["gpt-3.5-turbo-4k"]

    if input > 0:
        return model_pricing["prompt"] * input / 10  # in cents
    elif output > 0:
        return model_pricing["completion"] * output / 10  # in cents
    else:
        return 0  # Return 0 instead of raising an error