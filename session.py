"""Manages session state for the MissionHelp Demo application.

This module initializes and maintains Streamlit session state variables used
across the application for configuration, conversation history, and multimedia.
"""

import streamlit as st
import logging
import database
import uuid
import pytz
from streamlit_js import st_js_blocking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("session_debug.log")]
)
logger = logging.getLogger(__name__)

def setup_session() -> None:
    """Initializes Streamlit session state variables with default values."""
    # Initialize core components
    if "llm" not in st.session_state:
        st.session_state.llm = False
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = False
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = False
    if "graph" not in st.session_state:
        st.session_state.graph = False

    # Initialize conversation and multimedia
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "webpage_urls" not in st.session_state:
        st.session_state.webpage_urls = []
    if "images" not in st.session_state:
        st.session_state.images = []
    if "videos" not in st.session_state:
        st.session_state.videos = []

    # Initialize retrieval parameters
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 200
    if "retrieval_k" not in st.session_state:
        st.session_state.retrieval_k = 4

    # Ensure session state variables are initialized
    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False
    if "new_message_added" not in st.session_state:
        st.session_state.new_message_added = False
    if "user_id_mapping" not in st.session_state:
        st.session_state.user_id_mapping = database.generate_user_id_mapping()
    if "selected_user_id" not in st.session_state:
        st.session_state.selected_user_id = None
    if "persistence_setup_complete" not in st.session_state:
        st.session_state.persistence_setup_complete = False
    if "image_cache" not in st.session_state:
        st.session_state.image_cache = {}
    if "timings" not in st.session_state:
        st.session_state.timings = []

    if "user_timezone" not in st.session_state:
        try:
            timezone = st_js_blocking(
                code="""return Intl.DateTimeFormat().resolvedOptions().timeZone"""
            )
            if timezone and isinstance(timezone, str):
                pytz.timezone(timezone)
                st.session_state.user_timezone = timezone
            else:
                st.session_state.user_timezone = "UTC"
                st.warning("Failed to detect timezone; using UTC. Please ensure JavaScript is enabled.")
        except pytz.exceptions.UnknownTimeZoneError:
            logger.warning(f"Invalid timezone detected: {timezone}, falling back to UTC")
            st.session_state.user_timezone = "UTC"
            st.warning("Failed to detect timezone; using UTC. Please ensure JavaScript is enabled.")
        except Exception as e:
            logger.error(f"Error detecting timezone: {e}")
            st.session_state.user_timezone = "UTC"
            st.warning("Failed to detect timezone; using UTC. Please ensure JavaScript is enabled.")