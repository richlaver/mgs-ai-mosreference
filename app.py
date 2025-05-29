"""Main entry point for the MissionHelp Demo application.

This Streamlit app initializes the environment, sets up dependencies (LLM, vector
store, LangGraph), and renders the chatbot interface for MissionOS queries.
"""

import streamlit as st
import rag
import session
import setup
import database
import ui
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app_debug.log")]
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Initializes and runs the MissionHelp Demo application."""
    st.set_page_config(
        page_title="MissionOS Reference",
        page_icon="mgs-small-logo.svg",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    session.setup_session()

    # Prompt for user to log in
    if st.session_state.selected_user_id is None:
        ui.select_user()

    # Render initial UI (sidebar, app title, popover, disabled chat input)
    ui.render_initial_ui()

    # Run setup only once per session
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False

    if not st.session_state.setup_complete:
        setup.install_playwright_browsers()
        setup.set_google_credentials()
        st.session_state.llm = setup.get_llm()
        st.session_state.embeddings = setup.get_embeddings()

        if not setup.collection_exists() or not setup.points_exist():
            setup.rebuild_database()

        if not st.session_state.vector_store:
            st.session_state.vector_store = setup.get_vector_store(
                embeddings=st.session_state.embeddings
            )

        st.session_state.graph = rag.build_graph(
            llm=st.session_state.llm,
            vector_store=st.session_state.vector_store,
            k=st.session_state.retrieval_k
        )
        st.session_state.setup_complete = True
        st.toast("Set-up complete!", icon=":material/check_circle:")

    # Render chat messages, history, and enabled chat input
    if st.session_state.graph:
        ui.render_chat_content()


if __name__ == "__main__":
    main()