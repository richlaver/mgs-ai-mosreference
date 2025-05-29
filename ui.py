"""Render the chatbot user interface for the MissionHelp Demo application.

This module defines the Streamlit-based UI, handling chat history display,
user input, multimedia rendering, and new UI components (sidebar, modals).
"""

import database
import base64
import logging
import uuid
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from streamlit_extras.stylable_container import stylable_container


# Configure logging for UI events and errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("ui_debug.log")]
)
logger = logging.getLogger(__name__)


def clear_chat() -> None:
    """Clears the chat history and resets the thread ID."""
    st.session_state.messages = []
    st.session_state.images = []
    st.session_state.videos = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.new_message_added = True
    feedback_keys = [key for key in st.session_state if key.startswith("feedback_")]
    for key in feedback_keys:
        try:
            del st.session_state[key]
            logger.info(f"Deleted feedback key: {key}")
        except:
            logger.error(f"Error deleting feedback key: {key}", exc_info=True)
    st.markdown('<script>window.newMessageAdded = true; scrollToBottom();</script>', unsafe_allow_html=True)
    st.session_state.new_message_added = False


@st.dialog("Select User")
def select_user() -> None:
    """Renders the user selection modal using Streamlit dialog."""
    with st.form("user_form", enter_to_submit=False):
        selected_username = st.selectbox(
            "Select User",
            [None] + list(st.session_state.user_id_mapping.keys()),
            key="user_select",
            label_visibility="visible",
            index=0
        )
        if st.form_submit_button("Log In"):
            if selected_username is not None:
                selected_user_id = st.session_state.user_id_mapping[selected_username]
                st.session_state.selected_user_id = selected_user_id
                st.toast(f"Selected user: {selected_username} (ID: {st.session_state.selected_user_id})", icon=":material/check_circle:")
                st.success(f"Selected user: {selected_username}")
                clear_chat()
                st.rerun()
            else:
                st.error(icon=":material/error:", body="Please select a user")


def render_initial_ui() -> None:
    """Renders the initial UI components (sidebar, app title, popover, disabled chat input) before setup."""
    # Apply custom CSS for layout and styling
    st.markdown(
        """
        <style>
        .stAppHeader {
            background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
            visibility: visible;  /* Ensure the header is visible */
        }
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
            padding-right: 0rem;
        }
        .chat-messages {
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 60px;
            margin-top: 60px; /* Space for app bar */
        }
        .stChatInput {
            position: fixed;
            bottom: 10px;
            width: 100%;
            max-width: 720px;
            margin: 0 auto;
            background-color: white;
            z-index: 1000;
        }
        .inline-image {
            margin: 10px 0;
            max-width: 100%;
        }
        .sidebar-title {
            font-size: 14px;
            font-weight: 600;
            color: #666;
            margin-bottom: 20px;
            text-transform: uppercase;
        }
        .app-title {
            font-size: 14px;
            font-weight: 600;
            color: #666;
            margin-top: 20px;
            text-transform: uppercase;
        }
        [data-testid="stPopover"] {
            width: auto !important;
            max-width: 48px !important;
            padding: 4px !important;
        }
        [data-testid="stPopover"] > div {
            width: 48px !important;
        }
        .popover-buttons {
            display: flex;
            flex-direction: column;
            gap: 4px;
            width: 40px;
        }
        .popover-container {
            display: flex;
            justify-content: flex-end;
        }
        </style>
        <script>
        function scrollToBottom() {
            if (window.newMessageAdded) {
                setTimeout(() => {
                    const chatMessages = document.querySelector('.chat-messages');
                    if (chatMessages) {
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                        window.newMessageAdded = false;
                    }
                }, 100); // Delay to ensure DOM is updated
            }
        }
        document.addEventListener('streamlit:render', scrollToBottom);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Streamlit logo (renders in sidebar and top-left corner)
    st.logo(
        image="mgs-full-logo.svg",
        size="small",
        link="https://www.maxwellgeosystems.com/",
        icon_image="mgs-small-logo.svg"
    )

    # Render sidebar
    with st.sidebar:
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown('<div class="sidebar-title">CONVERSATIONS</div>', unsafe_allow_html=True)
        if cols[1].button(label="", icon=":material/add_comment:", key="new_conversation", help="New conversation", type="primary", use_container_width=True):
            clear_chat()
        
        # Conversation cards
        threads = database.get_user_threads(st.session_state.selected_user_id)
        for i, thread in enumerate(threads):
            with stylable_container(
                key=f'conversation_card_{i}',
                css_styles="""
                {
                    background-color: #ffffff;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 0px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                """
            ):
                cols = st.columns([3, 1, 1])
                cols[0].markdown(thread["title"], unsafe_allow_html=False)
                if cols[1].button(label="", icon=":material/open_in_new:", key=f"open_convo_{i}", help="Open conversation"):
                    with st.spinner("Loading conversation..."):
                        st.session_state.thread_id = thread["id"]
                        messages = database.get_thread_messages(thread["id"], st.session_state.selected_user_id)
                        st.session_state.messages = messages
                        st.session_state.images = []
                        st.session_state.videos = []
                        st.session_state.new_message_added = True
                        st.markdown('<script>window.newMessageAdded = true; scrollToBottom();</script>', unsafe_allow_html=True)
                        st.session_state.new_message_added = False
                        st.rerun()
                if cols[2].button(label="", icon=":material/delete:", key=f"delete_convo_{i}", help="Delete conversation"):
                    with st.spinner("Deleting conversation..."):
                        database.delete_thread(thread["id"], st.session_state.selected_user_id)
                        if st.session_state.thread_id == thread["id"]:
                            clear_chat()
                        st.rerun()
        
        # App title and popover button in two columns
        st.divider()
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown('<div class="app-title">MISSION REFERENCE</div>', unsafe_allow_html=True)
        with cols[1]:
            with st.popover(
                label="",
                icon=":material/menu:",
                use_container_width=False
            ):
                st.button(label="Switch User", icon=":material/account_circle:", key="user_button", help="Change user", on_click=user_modal, use_container_width=True)
                st.button(label="Admin Login", icon=":material/lock_open:", key="admin_button", help="Unlock admin features", on_click=admin_modal, use_container_width=True)
                st.button(label="Run Batch Test", icon=":material/science:", key="test_button", help="Evaluate performance on query set", disabled=not st.session_state.get("admin_logged_in", False), on_click=test_modal, use_container_width=True)
                st.button(label="Re-Build Graph", icon=":material/build:", key="graph_button", help="Re-build LangGraph LLM implementation", disabled=not st.session_state.get("admin_logged_in", False), on_click=graph_modal, use_container_width=True)
                st.button(label="Re-Create Embeddings", icon=":material/content_copy:", key="docs_button", help="Scrape, parse, chunk and index embeddings database", disabled=not st.session_state.get("admin_logged_in", False), on_click=docs_modal, use_container_width=True)
                st.button(label="Erase History", icon=":material/delete_forever:", key="erase_button", help="Clear conversation history", disabled=not st.session_state.get("admin_logged_in", False), on_click=erase_modal, use_container_width=True)

    # Reserve space for chat input
    with st.empty():
        st.chat_input(
            placeholder="Setting up, please wait...",
            disabled=True,
            key="initial_chat_input"
        )


def render_chat_content() -> None:
    """Renders chat messages, history, and enabled chat input after setup is complete."""
    if not st.session_state.get("setup_complete", False):
        return

    # Render chat history in the persistent container
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for msg in st.session_state.get('messages', []):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            has_tool_calls = (hasattr(msg, "tool_calls") and bool(msg.tool_calls)) or \
                             (hasattr(msg, "invalid_tool_calls") and bool(msg.invalid_tool_calls))
            if has_tool_calls:
                continue
            with st.chat_message("assistant"):
                content = msg.content
                images = msg.additional_kwargs.get("images", [])
                videos = msg.additional_kwargs.get("videos", [])

                segments = re.findall(r"((?:[0-9]+\.\s+[^\n]*?(?=(?:[0-9]+\.\s+|\Z)))|[^\n]+)", content, re.DOTALL)
                for segment in segments:
                    segment = segment.strip()
                    if not segment:
                        continue

                    image_refs = re.findall(r"\[Image (\d+)\]", segment)
                    cleaned_segment = re.sub(r"\[Image (\d+)\]\s*", "", segment)
                    cleaned_segment = re.sub(r"\s+([.!?])", r"\1", cleaned_segment)
                    if cleaned_segment.strip():
                        st.markdown(cleaned_segment)

                    for ref in image_refs:
                        idx = int(ref) - 1
                        if 0 <= idx < len(images):
                            caption = images[idx].get("caption", "")
                            cleaned_caption = re.sub(r"^Figure \d+:\s*", "", caption)
                            st.image(
                                base64.b64decode(images[idx]["base64"]),
                                caption=cleaned_caption if cleaned_caption.strip() else None,
                                use_container_width=True,
                                output_format="auto",
                                clamp=True,
                                channels="RGB",
                            )

                for video in videos:
                    st.markdown(f"**Video**: [{video['title']}]({video['url']})")
                    st.video(video["url"])

                
                message_id = msg.additional_kwargs.get("message_id")
                if message_id:
                    feedback_key = f"feedback_{message_id}_{st.session_state.thread_id}"
                    if feedback_key not in st.session_state:
                        stored_feedback = msg.additional_kwargs.get("feedback", 0)
                        st.session_state[feedback_key] = 1 if stored_feedback == 1 else 0 if stored_feedback == -1 else None
                    
                    feedback = st.feedback("thumbs", key=feedback_key)
                    if feedback is not None:
                        new_feedback_value = 1 if feedback == 1 else -1 if feedback == 0 else 0
                        stored_feedback = msg.additional_kwargs.get("feedback", 0)
                        if new_feedback_value != stored_feedback:
                            database.update_message_feedback(message_id, new_feedback_value)
                            for m in st.session_state.messages:
                                if m.additional_kwargs.get("message_id") == message_id and isinstance(m, AIMessage):
                                    m.additional_kwargs["feedback"] = new_feedback_value
                                    break
                    else:
                        stored_feedback = msg.additional_kwargs.get("feedback", 0)
                        if stored_feedback != 0:
                            database.update_message_feedback(message_id, 0)
                            for m in st.session_state.messages:
                                if m.additional_kwargs.get("message_id") == message_id and isinstance(m, AIMessage):
                                    m.additional_kwargs["feedback"] = 0
                                    break

    st.markdown("</div>", unsafe_allow_html=True)

    if question := st.chat_input(
        placeholder="Ask a question about MissionOS:",
        # disabled=not st.session_state.setup_complete,
        key="active_chat_input"
    ):
        st.session_state.images = []
        st.session_state.videos = []
        user_message = HumanMessage(content=question)
        st.session_state.messages.append(user_message)
        st.session_state.new_message_added = True
        with st.chat_message("user"):
            st.markdown(question)

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.selected_user_id
            }
        }
        initial_state = {
            "messages": [user_message],
            "images": [],
            "videos": [],
            "timings": [],
        }

        with st.spinner("Generating..."):
            with st.chat_message("assistant"):
                try:
                    def stream_tokens():
                        accumulated_content = ""
                        for chunk, metadata in st.session_state.graph.stream(initial_state, stream_mode="messages", config=config):
                            if isinstance(chunk, dict) and chunk.get("type") == "ai" and not chunk.get("additional_kwargs", {}).get("chunk"):
                                content = chunk.get("content", "")
                                if content:
                                    accumulated_content += content
                                    yield content
                            elif hasattr(chunk, "content") and chunk.content:
                                yield chunk.content
                            state_messages = [
                                {
                                    "type": msg.type,
                                    "content": msg.content,
                                    "additional_kwargs": msg.additional_kwargs
                                }
                                for msg in st.session_state.graph.get_state(config).values.get("messages", [])
                                if msg.type in ("human", "ai")
                            ]
                            st.session_state.messages = initial_state["messages"] + state_messages
                        return accumulated_content

                    stream_container = st.empty()
                    with stream_container:
                        full_content = stream_container.write_stream(stream_tokens())

                    final_state = st.session_state.graph.get_state(config).values
                    st.session_state.images = final_state.get("images", [])
                    st.session_state.videos = final_state.get("videos", [])
                    st.session_state.timings = final_state.get("timings", [])
                    if final_state:
                        ai_messages = [msg for msg in final_state.get("messages", []) if isinstance(msg, AIMessage) and not getattr(msg, "type", "") == "tool_call"]
                        if ai_messages:
                            final_message = ai_messages[-1]
                            images = final_message.additional_kwargs.get("images", [])
                            videos = final_message.additional_kwargs.get("videos", [])

                            stream_container.empty()
                            pattern = r"(?:\n|^)\s*\(Image (\d+)\)\s*(?:\n|$)"
                            parts = re.split(pattern, final_message.content, flags=re.MULTILINE)
                            for i in range(len(parts)):
                                if i % 2 == 0:  # Text part
                                    part = parts[i].strip()
                                    if part:
                                        st.markdown(part)
                                else:  # Image number
                                    img_idx = int(parts[i]) - 1
                                    if 0 <= img_idx < len(images):
                                        caption = images[img_idx].get("caption", "")
                                        cleaned_caption = re.sub(r"^Figure\s*\d+:\s*", "", caption)
                                        prefixed_caption = f"Image {parts[i]}: {cleaned_caption}" if cleaned_caption.strip() else f"Image {parts[i]}:"
                                        st.image(
                                            base64.b64decode(images[img_idx]["base64"]),
                                            caption=prefixed_caption,
                                            use_container_width=True,
                                            output_format="auto",
                                            clamp=True,
                                            channels="RGB",
                                        )

                            # Render video links
                            for video in videos:
                                st.markdown(f"**Video**: [{video['title']}]({video['url']})")
                                st.video(video["url"])

                            # Feedback
                            response_message_id = final_message.additional_kwargs.get("message_id")
                            if response_message_id:
                                feedback_key = f"feedback_{response_message_id}_{st.session_state.thread_id}"
                                if feedback_key not in st.session_state:
                                    st.session_state[feedback_key] = None
                                feedback = st.feedback("thumbs", key=feedback_key)
                                if feedback is not None:
                                    new_feedback_value = 1 if feedback == 1 else -1 if feedback == 0 else 0
                                    database.update_message_feedback(response_message_id, new_feedback_value)
                                    final_message.additional_kwargs["feedback"] = new_feedback_value
                                else:
                                    stored_feedback = final_message.additional_kwargs.get("feedback", 0)
                                    if stored_feedback != 0:
                                        database.update_message_feedback(response_message_id, 0)
                                        final_message.additional_kwargs["feedback"] = 0

                    # Display latency details
                    with st.expander("Latency Details", expanded=True):
                        timings = final_state.get("timings", [])
                        total_latency = sum(timing["time"] for timing in timings)
                        
                        latency_data = [
                            {
                                "Component": timing["component"],
                                "Latency (s)": f"{timing['time']:.2f}",
                                "Percentage (%)": f"{(timing['time'] / total_latency * 100):.0f}" if total_latency > 0 else "0"
                            }
                            for timing in timings
                        ]
                        
                        df = pd.DataFrame(latency_data)
                        
                        st.markdown(f"**Total Latency: {total_latency:.2f} seconds**")
                        
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True
                        )

                        if timings and total_latency > 0:
                            fig, ax = plt.subplots()
                            components = [timing["component"] for timing in timings]
                            percentages = [(timing["time"] / total_latency * 100) for timing in timings]
                            pastel_colors = cm.Pastel1(range(len(components)))
                            ax.pie(percentages, labels=components, autopct='%1.0f%%', startangle=90, colors=pastel_colors)
                            ax.axis('equal')
                            st.pyplot(fig)
                            plt.close(fig)

                except Exception as e:
                    st.error(f"Error streaming response: {e}")
                    logger.error(f"Streaming error: {e}", exc_info=True)


@st.dialog("Admin Login")
def admin_modal():
    """Renders the admin login modal using Streamlit dialog."""
    with st.form("admin_login_form", enter_to_submit=False):
        st.markdown("### Admin Login")
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter admin password"
        )
        if st.form_submit_button("Login"):
            if password == st.secrets.admin_password:
                st.session_state.admin_logged_in = True
                st.toast("Logged in as admin", icon=":material/check_circle:")
                st.success("Logged in as admin")
                st.rerun()
            else:
                st.error("Incorrect password")


@st.dialog("Change User")
def user_modal():
    """Renders the user selection modal using Streamlit dialog."""
    with st.form("user_form", enter_to_submit=False):
        all_usernames = list(st.session_state.user_id_mapping.keys())
        selected_username = next((key for key, value in st.session_state.user_id_mapping.items() if value == st.session_state.selected_user_id), None)
        default_index = all_usernames.index(selected_username)
        selected_username = st.selectbox(
            "Change User",
            list(st.session_state.user_id_mapping.keys()),
            key="user_select",
            label_visibility="visible",
            index=default_index
        )
        if st.form_submit_button("Select"):
            if selected_username:
                selected_user_id = st.session_state.user_id_mapping[selected_username]
                st.session_state.selected_user_id = selected_user_id
                st.toast(f"Selected user: {selected_username} (ID: {selected_user_id})", icon=":material/check_circle:")
                st.success(f"Selected user: {selected_username}")
                clear_chat()
                st.rerun()
            else:
                st.error("Please select a user")


@st.dialog("Batch Testing")
def test_modal():
    """Renders the batch testing modal using Streamlit dialog."""
    results = []
    with st.form("batch_test_form", enter_to_submit=False):
        st.markdown("### Batch Testing")
        test_csv = st.file_uploader(
            "Upload Test CSV",
            type="csv",
            disabled=not st.session_state.vector_store or not st.session_state.graph
        )
        if st.form_submit_button("Run Batch Test"):
            with st.spinner("Running batch test..."):
                from setup import run_batch_test  # Import here to avoid circular import
                progress_text = st.empty()
                
                # results = []
                for current_query, total_queries, batch_results in run_batch_test(
                    test_csv,
                    st.session_state.graph,
                    st.session_state.vector_store,
                ):
                    results = batch_results
                    progress_text.text(f"Processing query {current_query} of {total_queries}...")
                
                progress_text.empty()
                st.toast("Batch test completed.", icon=":material/check_circle:")
                st.success("Batch test completed.")
    if results:
        st.download_button(
            label="Download Results CSV",
            data=pd.DataFrame(results).to_csv(index=False).encode('utf-8'),
            file_name="test_results.csv",
            mime="text/csv"
        )


@st.dialog("Database Parameters")
def docs_modal():
    """Renders the database parameters modal using Streamlit dialog."""
    with st.form("docs_form", enter_to_submit=False):
        st.markdown("### Database Parameters")
        col1, col2 = st.columns(2)
        with col1:
            new_chunk_size = st.number_input(
                "Chunk size",
                min_value=100,
                max_value=5000,
                value=st.session_state.get("chunk_size", 1000),
                step=100
            )
        with col2:
            new_chunk_overlap = st.number_input(
                "Chunk overlap",
                min_value=0,
                max_value=1000,
                value=st.session_state.get("chunk_overlap", 200),
                step=50
            )
        if st.form_submit_button("Update Database"):
            from setup import set_google_credentials, get_embeddings, rebuild_database, get_vector_store  # Import here to avoid circular import
            st.session_state.chunk_size = new_chunk_size
            st.session_state.chunk_overlap = new_chunk_overlap
            set_google_credentials()
            embeddings = get_embeddings()
            rebuild_database()
            st.session_state.vector_store = get_vector_store(embeddings)
            st.toast(f"Database updated with chunk_size={new_chunk_size}, overlap={new_chunk_overlap}", icon=":material/check_circle:")
            st.success(f"Database updated with chunk_size={new_chunk_size}, overlap={new_chunk_overlap}")
            st.rerun()


@st.dialog("RAG Parameters")
def graph_modal():
    """Renders the RAG parameters modal for reconfiguring the LangGraph graph using Streamlit dialog."""
    with st.form("graph_form", enter_to_submit=False):
        st.markdown("### RAG Parameters")
        new_k = st.number_input(
            "Number of chunks to retrieve (k)",
            min_value=1,
            max_value=20,
            value=st.session_state.get("retrieval_k", 4),
            step=1
        )
        if st.form_submit_button("Reconfigure RAG"):
            from rag import build_graph  # Import here to avoid circular import
            st.session_state.retrieval_k = new_k
            st.session_state.graph = build_graph(
                llm=st.session_state.llm,
                vector_store=st.session_state.vector_store,
                k=st.session_state.retrieval_k,
            )
            st.toast(f"RAG updated with number of chunks={new_k}", icon=":material/check_circle:")
            st.success(f"RAG updated with number of chunks={new_k}")
            st.rerun()


@st.dialog("Erase History")
def erase_modal():
    """Renders the erase history modal using Streamlit dialog."""
    with st.form("erase_form", enter_to_submit=False):
        st.markdown("### Erase History")
        st.warning(icon=":material/warning:", body="Are you sure you want to erase the conversation history? This action cannot be undone.")
        if st.form_submit_button("Delete Forever"):
            with st.spinner("Erasing history... This may take a moment."):
                database.create_persistence_tables()
                database.populate_users_table()
                clear_chat()
            st.rerun()