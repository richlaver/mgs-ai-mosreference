"""Manage database operations and web scraping for the MissionHelp Demo application.

This module handles PostgreSQL connections, image storage, and scraping of MissionOS
manual webpages to extract text, images, and videos.
"""

import setup
import base64
import json
import logging
import os
import psycopg2
from typing import List, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from google.cloud.sql.connector import Connector, IPTypes
from google.oauth2 import service_account
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage


# Configure logging for database and scraping events
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scrape_debug.log")
    ]
)
logger = logging.getLogger(__name__)


def connect_to_conversations_db():
    credentials_json = st.secrets["AWS_CREDENTIALS_JSON"]
    credentials = json.loads(credentials_json)
    return psycopg2.connect(
        host=credentials["DB_HOST"],
        database=credentials["DB_NAME"],
        user=credentials["DB_USER"],
        password=credentials["DB_PASSWORD"],
        port=credentials["DB_PORT"]
    )


def create_persistence_tables():
    """Recreate the users, threads and messages tables in the database."""
    conn = connect_to_conversations_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS message_timings CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS messages CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS threads CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS users CASCADE;")
            cursor.execute("""
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255)
                );
            """)
            cursor.execute("""
                CREATE TABLE threads (
                    id UUID PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    title VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_archived BOOLEAN DEFAULT FALSE
                );
            """)
            cursor.execute("""
                CREATE TABLE messages (
                    id SERIAL PRIMARY KEY,
                    thread_id UUID REFERENCES threads(id),
                    user_id INTEGER REFERENCES users(id),
                    content TEXT,
                    is_ai BOOLEAN,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    feedback INTEGER  -- -1 for thumbs-down, 1 for thumbs-up, 0 or NULL for no feedback
                );
            """)
            cursor.execute("""
                CREATE TABLE message_timings (
                    id SERIAL PRIMARY KEY,
                    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
                    timing_key VARCHAR(255) NOT NULL,
                    timing_value FLOAT NOT NULL
                );
            """)
            cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_thread_id ON messages (thread_id);
                """)
            st.toast("Thread persistence tables recreated successfully.", icon=":material/database:")
    finally:
        conn.commit()
        conn.close()


def populate_users_table():
    """Populates the users table with usernames from the database."""
    conn = connect_to_conversations_db()
    try:
        with conn.cursor() as cursor:
            for user_id, username in setup.users.items():
                cursor.execute(
                    """
                    INSERT INTO users (id, username)
                    VALUES (%s, %s)
                    ON CONFLICT (id) DO UPDATE SET username = EXCLUDED.username;
                    """,
                    (user_id, username)
                )
        st.toast("Users table populated successfully.", icon=":material/database:")
    finally:
        conn.commit()
        conn.close()


def generate_user_id_mapping():
    """Generates a mapping of usernames to user IDs from the database.

    Returns:
        Dictionary mapping usernames to user IDs.
    """
    conn = connect_to_conversations_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, username FROM users;")
            users = cursor.fetchall()
            return {user[1]: user[0] for user in users}
    finally:
        conn.close()


def get_user_threads(user_id: int) -> list:
    conn = connect_to_conversations_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, created_at
        FROM threads
        WHERE user_id = %s AND is_archived = FALSE
        ORDER BY updated_at DESC
    """, (user_id,))
    threads = [{"id": str(row[0]), "title": row[1], "created_at": row[2]} for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return threads


def get_thread_messages(thread_id: str, user_id: int) -> list:
    conn = connect_to_conversations_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, content, is_ai, metadata, feedback
            FROM messages
            WHERE thread_id = %s AND user_id = %s
            ORDER BY created_at ASC
        """, (thread_id, user_id))
        rows = cursor.fetchall()
        messages = []
        for row in rows:
            message_id, content, is_ai, metadata_obj, feedback = row
            metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
            if not isinstance(metadata, dict):
                try:
                    metadata = json.loads(metadata_obj) if metadata_obj else {}
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Failed to parse metadata for message {message_id}: {e}")
                    metadata = {}
            metadata["feedback"] = feedback or 0
            additional_kwargs = {
                "image_map": metadata.get("image_map", {}),
                "videos": metadata.get("videos", []),
                "message_id": message_id,
                "feedback": feedback or 0
            }
            if is_ai:
                messages.append(AIMessage(content=content, additional_kwargs=additional_kwargs))
            else:
                messages.append(HumanMessage(content=content, additional_kwargs=additional_kwargs))
        return messages
    except Exception as e:
        logger.error(f"Error retrieving thread messages for thread {thread_id}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def delete_thread(thread_id: str, user_id: int):
    conn = connect_to_conversations_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE threads
            SET is_archived = TRUE
            WHERE id = %s AND user_id = %s
        """, (thread_id, user_id))
        conn.commit()
    finally:
        cursor.close()
        conn.close()
        

def add_thread_to_db(thread_id: str, user_id: int, title: str):
    """Adds a thread to the threads table using a UUID thread_id as the primary key.

    Args:
        thread_id: UUID string to use as the primary key (id).
        user_id: ID of the user creating the thread.
        title: Title of the thread.
    """

    conn = None
    try:
        conn = connect_to_conversations_db()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO threads (id, user_id, title)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                RETURNING id;
                """,
                (thread_id, user_id, title)
            )
            result = cursor.fetchone()
            if result:
                conn.commit()
            else:
                conn.rollback()
    finally:
        if conn:
            conn.close()


def add_message_to_db(
    thread_id: int,
    user_id: int,
    content: str,
    is_ai: bool,
    feedback: Optional[int] = None,
    additional_kwargs: Optional[dict] = None
) -> Optional[int]:
    """Adds a message to the database.

    Args:
        thread_id: ID of the thread to which the message belongs.
        user_id: ID of the user sending the message.
        content: Content of the message.
        is_ai: Boolean indicating if the message is from a bot.
        feedback: Feedback value for the message (-1, 0, 1, or None).

    Returns:
        Optional[int]: The generated message_id if successful, None otherwise.
    """

    message_id = None
    conn = None
    try:
        conn = connect_to_conversations_db()
        with conn.cursor() as cursor:
            metadata = {
                "image_map": additional_kwargs.get("image_map", {}) if additional_kwargs else {},
                "videos": additional_kwargs.get("videos", []) if additional_kwargs else [],
                "feedback": additional_kwargs.get("feedback", 0) if additional_kwargs else 0
            }
            cursor.execute(
                """
                INSERT INTO messages (thread_id, user_id, content, is_ai, metadata, feedback)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (thread_id, user_id, content, is_ai, json.dumps(metadata), feedback)
            )
            result = cursor.fetchone()
            if result:
                message_id = result[0]
            else:
                return None
        conn.commit()
    finally:
        if conn:
            conn.close()

    return message_id


def update_message_feedback(message_id: int, feedback: int) -> None:
    """Updates the feedback value for a specific message in the database.

    Args:
        message_id: ID of the message to update.
        feedback: Feedback value (-1 for thumbs-down, 1 for thumbs-up, 0 for no feedback).
    """
    conn = None
    try:
        conn = connect_to_conversations_db()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE messages
                SET feedback = %s
                WHERE id = %s;
                """,
                (feedback, message_id)
            )
        conn.commit()
    finally:
        if conn:
            conn.close()


def add_message_timings_to_db(
    message_id: int,
    timings: List[dict],
) -> None:
    """Adds message timings to the database.

    Args:        
        message_id: The ID of the message to associate timings with.
        timings: List of dictionaries, each with 'component' (str) and 'time' (float).
            Example: [{"node": "generate", "time": 0.5, "component": "llm_generation"}, ...]
    """
    conn = None
    try:
        conn = connect_to_conversations_db()
        with conn.cursor() as cursor:
            for timing in timings:
                timing_key = timing.get('component')
                timing_value = timing.get('time')
                if not isinstance(timing_value, (int, float)) or timing_value < 0:
                    continue
                cursor.execute(
                    """
                    INSERT INTO message_timings (message_id, timing_key, timing_value)
                    VALUES (%s, %s, %s);
                    """,
                    (message_id, timing_key, float(timing_value))
                )
    finally:
        conn.commit()
        conn.close()


def connect_to_images_db():
    """Establishes a connection to the PostgreSQL database using Google Cloud SQL.

    Returns:
        Database connection object.
    """
    credentials = service_account.Credentials.from_service_account_file("google_credentials.json")
    connector = Connector(credentials=credentials)
    return connector.connect(
        instance_connection_string="intense-age-455102-i9:asia-east2:mgs-web-user-manual",
        driver="pg8000",
        user="langchain-tutorial-rag-service@intense-age-455102-i9.iam",
        enable_iam_auth=True,
        db="postgres",
        ip_type=IPTypes.PUBLIC,
    )


def query_db() -> str:
    """Query the database to retrieve its version.

    Returns:
        The PostgreSQL version string.

    Raises:
        Exception: If the query fails.
    """
    conn = None
    try:
        conn = connect_to_images_db()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        result = cursor.fetchone()
        version_string = result[0]
        return version_string
    finally:
        if conn is not None:
            cursor.close()
            conn.close()


def create_images_table() -> None:
    """Creates or recreates the images table in the database."""
    conn = connect_to_images_db()
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE IF EXISTS images;")
        cursor.execute(
            """
            CREATE TABLE images (
                id SERIAL PRIMARY KEY,
                url VARCHAR(255) NOT NULL,
                image_binary BYTEA NOT NULL,
                caption TEXT
            );
            """
        )
        cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_images_id ON images (id);
            """)
        cursor.execute("GRANT ALL PRIVILEGES ON images TO postgres;")
        conn.commit()
        st.toast("Images table recreated successfully.", icon=":material/database:")
    finally:
        cursor.close()
        conn.close()


def get_images_by_ids(image_ids):
    """Retrieves images from the database by their IDs.
    Args:
        image_ids: List of image IDs to retrieve.
    Returns:
        List of dictionaries containing image ID, base64 encoded image data, and caption.
    """
    conn = connect_to_images_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, image_binary, caption FROM images WHERE id = ANY(%s)",
        (image_ids,)
    )
    rows = cursor.fetchall()
    images = []
    for row in rows:
        id, image_binary, caption = row
        images.append({
            "id": id,
            "base64": base64.b64encode(image_binary).decode("utf-8"),
            "caption": caption
        })
    return images


def generate_webpaths() -> List[str]:
    """Generates URLs for MissionOS manual webpages from a CSV file.

    Returns:
        List of webpage URLs.
    """
    base_url = (
        "https://www.maxwellgeosystems.com/manuals/demo-manual/"
        "manual-web-content-highlight.php?manual_id="
    )
    ids = (
        pd.read_csv("WUM articles.csv", usecols=[0], skip_blank_lines=True)
        .dropna()
        .iloc[:, 0]
        .astype(int)
        .to_list()
    )
    return [base_url + str(id) for id in ids]


def load_cached_docs(cache_dir: str = "scrape_cache") -> List[Document]:
    """Loads cached documents from disk.

    Args:
        cache_dir: Directory containing cached JSON files.

    Returns:
        List of Document objects from cache, or empty list if cache is invalid.
    """
    if not os.path.exists(cache_dir):
        return []
    docs = []
    for filename in os.listdir(cache_dir):
        if filename.endswith(".json"):
            with open(os.path.join(cache_dir, filename)) as f:
                data = json.load(f)
                docs.append(Document(page_content=data["page_content"], metadata=data["metadata"]))
    return docs


def save_cached_docs(docs: List[Document], cache_dir: str = "scrape_cache") -> None:
    """Saves documents to disk as JSON files.

    Args:
        docs: List of Document objects to cache.
        cache_dir: Directory to store JSON files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    for i, doc in enumerate(docs):
        cache_data = {"page_content": doc.page_content, "metadata": doc.metadata}
        with open(os.path.join(cache_dir, f"doc_{i}.json"), "w") as f:
            json.dump(cache_data, f)


def web_scrape(use_cache: bool = True, cache_dir: str = "scrape_cache") -> List[Document]:
    """Scrapes MissionOS manual webpages for text, images, and videos.

    Args:
        use_cache: If True, loads from cache before scraping.
        cache_dir: Directory for cached JSON files.

    Returns:
        List of Document objects with processed content and metadata.
    """
    if use_cache and (docs := load_cached_docs(cache_dir)):
        pass
    else:
        webpaths = generate_webpaths()
        st.toast("Loading webpages...", icon=":material/build:")
        loader = AsyncChromiumLoader(urls=webpaths)
        docs = loader.load()
        save_cached_docs(docs, cache_dir)

    st.toast("Processing webpages...", icon=":material/build:")
    conn = connect_to_images_db()
    cursor = conn.cursor()
    try:
        last_percent = -1
        status_message = st.toast(f"Processing webpages: 0%", icon=":material/build:")
        for i, doc in enumerate(docs):
            base_url = doc.metadata["source"]
            soup = BeautifulSoup(doc.page_content, "html.parser")
            div_print = soup.find("div", id="div_print")
            doc.metadata["videos"] = []

            if div_print:
                # Convert relative URLs to absolute
                for a_tag in div_print.find_all("a"):
                    if (href := a_tag.get("href")) and not href.startswith(("#", "mailto:", "javascript:", "tel:")):
                        a_tag["href"] = urljoin(base_url, href)

                # Extract YouTube videos from iframes
                for iframe in div_print.find_all("iframe"):
                    if "youtube.com" in (iframe_src := iframe.get("src", "")) or "youtu.be" in iframe_src:
                        parsed_url = urlparse(iframe_src)
                        video_id = None
                        if "youtube.com" in parsed_url.netloc:
                            if "/embed/" in parsed_url.path:
                                video_id = parsed_url.path.split("/embed/")[-1].split("?")[0]
                            else:
                                video_id = parse_qs(parsed_url.query).get("v", [None])[0]
                        elif "youtu.be" in parsed_url.netloc:
                            video_id = parsed_url.path.strip("/")

                        if video_id:
                            watch_url = f"https://www.youtube.com/watch?v={video_id}"
                            response = requests.get(
                                watch_url,
                                headers={
                                    "User-Agent": (
                                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                                        "Chrome/91.0.4472.124 Safari/537.36"
                                    )
                                },
                                timeout=10,
                            )
                            response.raise_for_status()
                            iframe_soup = BeautifulSoup(response.text, "html.parser")
                            title = None
                            if (meta_title := iframe_soup.find("meta", attrs={"name": "title"})) and meta_title.get(
                                "content"
                            ):
                                title = meta_title.get("content").strip()
                            elif (
                                og_title := iframe_soup.find("meta", attrs={"property": "og:title"})
                            ) and og_title.get("content"):
                                title = og_title.get("content").strip()
                            elif (
                                (title_tag := iframe_soup.find("title"))
                                and title_tag.get_text(strip=True)
                                and title_tag.get_text(strip=True) != "YouTube"
                            ):
                                title = title_tag.get_text(strip=True)

                            if title:
                                title = title.replace(" - YouTube", "").strip()
                                if not title or title == "-":
                                    title = None
                            if not title:
                                title = f"Untitled Video {video_id}"

                            url_tag = iframe_soup.find("link", rel="canonical")
                            video_url = url_tag.get("href", watch_url) if url_tag else watch_url
                            doc.metadata["videos"].append({"url": video_url, "title": title})

                # Convert specific <p> tags to <h1>
                for p_tag in div_print.find_all("p", class_="headingp page-header"):
                    new_tag = soup.new_tag("h1")
                    new_tag.string = p_tag.get_text()
                    p_tag.replace_with(new_tag)

                # Process and store images
                for img in div_print.find_all("img"):
                    if (src := img.get("src", "")).startswith("data:image/png;base64,"):
                        base64_string = src.split(",")[1]
                        image_binary = base64.b64decode(base64_string)
                        figure = img.find_parent("figure")
                        caption = (
                            figure.find("figcaption").get_text(strip=True) if figure and figure.find("figcaption") else ""
                        )
                        cursor.execute(
                            """
                            INSERT INTO images (url, image_binary, caption)
                            VALUES (%s, %s, %s)
                            RETURNING id
                            """,
                            (base_url, image_binary, caption),
                        )
                        img["src"] = f"db://images/{cursor.fetchone()[0]}"

                doc.page_content = str(div_print.decode_contents())
            else:
                doc.page_content = ""

            # Update toast every 1% progress
            current_percent = int(((i + 1) / len(docs)) * 100)
            if current_percent > last_percent:
                status_message.toast(f"Processing webpages: {current_percent}%")
                last_percent = current_percent

        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return docs


def chunk_text(docs: List[Document]) -> List[Document]:
    """Splits documents into semantic chunks for vector storage.

    Args:
        docs: List of Document objects to chunk.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    st.toast("Chunking text semantically...", icon=":material/build:")
    headers_to_split_on = [
        ("h1", "Heading 1"),
        ("h2", "Heading 2"),
        ("h3", "Heading 3"),
        ("h4", "Heading 4"),
    ]
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=headers_to_split_on,
        max_chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? "],
        preserve_links=True,
        preserve_images=True,
        preserve_videos=True,
        preserve_audio=True,
        stopword_removal=False,
        normalize_text=False,
        elements_to_preserve=["table", "ul", "ol"],
        denylist_tags=["script", "style", "head"],
        preserve_parent_metadata=True,
    )

    all_splits = splitter.transform_documents(documents=docs)

    if not all_splits:
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? "],
        )
        all_splits = fallback_splitter.split_documents(docs)

    if not all_splits:
        all_splits = docs[:]

    st.toast("Chunking complete.", icon=":material/done:")
    return all_splits


def index_chunks(all_splits: List[Document], vector_store) -> None:
    """Index document chunks in the vector store.

    Args:
        all_splits: List of chunked Document objects.
        vector_store: Qdrant vector store instance.
    """
    st.toast("Indexing chunks...", icon=":material/build:")
    vector_store.add_documents(documents=all_splits)