"""Render the chatbot user interface for the MissionHelp Demo application.

This module defines the Streamlit-based UI, handling chat history display,
user input, multimedia rendering, and new UI components (sidebar, modals).
"""

import base64
import logging
import re
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
        st.markdown('<div class="sidebar-title">CONVERSATIONS</div>', unsafe_allow_html=True)
        
        # Conversation cards
        for i, convo in enumerate([
            {"title": "Project Setup Guide"},
            {"title": "Data Analysis Tips"},
            {"title": "Troubleshooting Sensors"}
        ]):
            with stylable_container(
                key='conversation_card_' + str(i),
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
                cols[0].markdown(convo["title"], unsafe_allow_html=False)
                if cols[1].button(label="", icon=":material/open_in_new:", key=f"open_convo_{i}", help="Open conversation"):
                    st.session_state[f"convo_{i}_opened"] = True
                if cols[2].button(label="", icon=":material/delete:", key=f"delete_convo_{i}", help="Delete conversation"):
                    st.session_state[f"convo_{i}_deleted"] = True
        
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
                st.button(label="", icon=":material/account_circle:", key="user_button", help="Change user", on_click=user_modal)
                st.button(label="", icon=":material/lock_open:", key="admin_button", help="Unlock admin features", on_click=admin_modal)
                st.button(label="", icon=":material/science:", key="test_button", help="Run batch test", disabled=not st.session_state.get("admin_logged_in", False), on_click=test_modal)
                st.button(label="", icon=":material/build:", key="graph_button", help="Re-build graph", disabled=not st.session_state.get("admin_logged_in", False), on_click=graph_modal)
                st.button(label="", icon=":material/content_copy:", key="docs_button", help="Re-write database", disabled=not st.session_state.get("admin_logged_in", False), on_click=docs_modal)

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
            with st.chat_message("assistant"):
                if "tool_calls" in msg.additional_kwargs and msg.additional_kwargs["tool_calls"]:
                    st.markdown(f"{msg.content} (Calling tool...)")
                else:
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

        config = {"configurable": {"thread_id": f"{st.session_state.thread_id}"}}
        initial_state = {
            "messages": [user_message],
            "images": [],
            "videos": [],
            "timings": [],
        }

        with st.spinner("Generating..."):
            for step in st.session_state.graph.stream(initial_state, stream_mode="values", config=config):
                new_messages = [msg for msg in step["messages"] if msg not in st.session_state.messages]
                for msg in new_messages:
                    st.session_state.messages.append(msg)
                    if isinstance(msg, AIMessage):
                        with st.chat_message("assistant"):
                            if "tool_calls" in msg.additional_kwargs and msg.additional_kwargs["tool_calls"]:
                                st.markdown(f"{msg.content} (Calling tool...)")
                            else:
                                content = msg.content
                                images = msg.additional_kwargs.get("images", [])
                                videos = msg.additional_kwargs.get("videos", [])

                                segments = re.findall(
                                    r"((?:[0-9]+\.\s+[^\n]*?(?=(?:[0-9]+\.\s+|\Z)))|[^\n]+)", content, re.DOTALL
                                )
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

                final_state = step

            if final_state:
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

        st.session_state.new_message_added = True
        st.markdown('<script>window.newMessageAdded = true; scrollToBottom();</script>', unsafe_allow_html=True)
        st.session_state.new_message_added = False


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


@st.dialog("Select User")
def user_modal():
    """Renders the user selection modal using Streamlit dialog."""
    with st.form("user_form", enter_to_submit=False):
        user = st.selectbox(
            "Select User",
            ["Ananthu", "Rich", "Ronac", "Sandeep"],
            key="user_select",
            label_visibility="visible"
        )
        if st.form_submit_button("Select"):
            if user:
                st.session_state.selected_user = user
                st.toast(f"Selected user: {user}", icon=":material/check_circle:")
                st.success(f"Selected user: {user}")
                st.rerun()
            else:
                st.error("Please select a user")


@st.dialog("Batch Testing")
def test_modal():
    """Renders the batch testing modal using Streamlit dialog."""
    with st.form("batch_test_form", enter_to_submit=False):
        st.markdown("### Batch Testing")
        test_csv = st.file_uploader(
            "Upload Test CSV",
            type="csv",
            disabled=not st.session_state.vector_store or not st.session_state.graph
        )
        if test_csv:
            st.session_state.test_csv = test_csv
            st.rerun()  # Force rerender to update button state
        if st.form_submit_button("Run Batch Test", disabled=not st.session_state.get("test_csv", False)):
            with st.spinner("Running batch test..."):
                from setup import run_batch_test  # Import here to avoid circular import
                progress_text = st.empty()
                
                results = []
                for current_query, total_queries, batch_results in run_batch_test(
                    st.session_state.test_csv,
                    st.session_state.graph,
                    st.session_state.vector_store,
                ):
                    results = batch_results
                    progress_text.text(f"Processing query {current_query} of {total_queries}...")
                
                progress_text.empty()
                st.toast("Batch test completed.", icon=":material/check_circle:")
                st.success("Batch test completed.")
        if st.session_state.get("results_csv", False):
            st.download_button(
                label="Download Results CSV",
                data=st.session_state.results_csv,
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