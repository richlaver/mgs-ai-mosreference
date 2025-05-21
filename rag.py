"""Configures the LangGraph workflow for query processing in the MissionHelp Demo.

This module defines the retrieval-augmented generation pipeline, integrating the
language model and vector store.
"""

import base64
import csv
import logging
import re
from typing import List, Tuple

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
import time

import database
from classes import State

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def build_graph(llm, vector_store, k) -> StateGraph:
    """Builds the LangGraph workflow for query processing.

    Args:
        llm: Language model instance.
        vector_store: Qdrant vector store instance.
        k: Number of documents to retrieve.

    Returns:
        Compiled LangGraph instance.
    """
    st.toast("Building LangGraph workflow...", icon=":material/build:")

    @tool(response_format="content_and_artifact")
    def retrieve(query: str) -> Tuple[str, dict]:
        """Retrieves MissionOS information including text, images, and videos.

        Args:
            query: User query string.

        Returns:
            Tuple of serialized documents and artifact dictionary.
        """
        start_search = time.time()
        retrieved_docs = vector_store.similarity_search(query, k)
        search_time = time.time() - start_search

        serialized_docs = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )

        image_ids = []
        videos_set = set()
        for doc in retrieved_docs:
            for video in doc.metadata.get("videos", []):
                videos_set.add((video["url"], video["title"]))
            image_ids.extend(int(id) for id in re.findall(r"db://images/(\d+)", doc.page_content))

        videos = [{"url": url, "title": title} for url, title in videos_set]
        start_image_fetch = time.time()
        images = []
        with database.getconn() as conn:
            cursor = conn.cursor()
            try:
                if image_ids:
                    cursor.execute(
                        "SELECT id, image_binary, caption FROM images WHERE id = ANY(%s)",
                        (image_ids,),
                    )
                    image_map = {
                        f"db://images/{img[0]}": {
                            "base64": base64.b64encode(img[1]).decode("utf-8"),
                            "caption": img[2],
                        }
                        for img in cursor.fetchall()
                    }
                    for doc in retrieved_docs:
                        for img_ref, img_data in image_map.items():
                            if img_ref in doc.page_content or (
                                img_data["caption"] and img_data["caption"] in doc.page_content
                            ):
                                images.append(img_data)
                image_fetch_time = time.time() - start_image_fetch
            finally:
                cursor.close()

        artifact = {
            "docs": retrieved_docs,
            "images": images,
            "videos": videos,
            "timings": {"search_time": search_time, "image_fetch_time": image_fetch_time},
        }
        return serialized_docs, artifact
    

    def query_or_respond(state: State) -> dict:
        """Decides whether to query tools or respond directly.

        Args:
            state: Current state with messages and metadata.

        Returns:
            Updated state with new messages and timings.
        """
        start_time = time.time()
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are an assistant for MissionOS. Use the 'retrieve' tool for "
                    "info-seeking queries about MissionOS. For other requests, respond directly."
                ),
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}"),
        ])
        llm_with_tools = llm.bind_tools([retrieve])
        chain = prompt | llm_with_tools
        response = chain.invoke(
            {"history": state["messages"], "query": state["messages"][-1].content},
            config={"configurable": {"thread_id": f"{st.session_state.thread_id}"}},
        )
        response_time = time.time() - start_time
        new_state = state.copy()
        new_state["messages"].append(response)
        new_state["timings"].append({"node": "query_or_respond", "time": response_time, "component": "LLM decision"})
        return new_state
    

    def tools_node(state: State) -> dict:
        """Executes tools and updates state with results.

        Args:
            state: Current state with messages and metadata.

        Returns:
            Updated state with tool results and timings.
        """
        start_time = time.time()
        tool_result = ToolNode([retrieve]).invoke(state)
        updated_state = state.copy()
        updated_state["messages"].extend(tool_result["messages"])
        tool_time = time.time() - start_time
        updated_state["timings"].append({"node": "tools", "time": tool_time, "component": "Tool execution"})
        return updated_state
    

    def tools_condition(state: State) -> str:
        """Routes based on presence of tool calls in the last message.

        Args:
            state: Current state with messages.

        Returns:
            'tools' if tool calls exist, else END.
        """
        last_message = state["messages"][-1]
        return "tools" if hasattr(last_message, "tool_calls") and last_message.tool_calls else END
    

    def generate(state: State) -> dict:
        """Generates a response using retrieved context and multimedia.

        Args:
            state: Current state with messages and metadata.

        Returns:
            Updated state with response and multimedia.
        """
        start_time = time.time()
        query = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "unknown")
        tool_messages = [msg for msg in reversed(state["messages"]) if msg.type == "tool"][::-1]
        csv_file = "retrieved_chunks.csv"
        csv_headers = ["Query", "Qdrant Point ID", "Page Content", "URL"]
        csv_rows = []

        for msg in tool_messages:
            if not hasattr(msg, "artifact") or not msg.artifact:
                continue
            artifact = msg.artifact
            retrieved_docs = artifact.get("docs", [])
            chunks = re.findall(
                r"Source: (https?://[^\n]+)\nContent: (.*?)(?=(Source:|$))",
                msg.content,
                re.DOTALL,
            ) or [(artifact.get("source", "unknown"), msg.content, "")]

            for i, (source, content, _) in enumerate(chunks, 1):
                point_id = "unknown"
                for doc in retrieved_docs:
                    doc_content = doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")
                    doc_metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
                    if doc_content.strip() == content.strip():
                        point_id = doc_metadata.get("_id", "unknown")
                        break
                csv_rows.append({
                    "Query": query,
                    "Qdrant Point ID": point_id,
                    "Page Content": content.strip(),
                    "URL": source,
                })

        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            if f.tell() == 0:
                writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)

        retrieved_content = "\n\n".join(msg.content for msg in tool_messages if msg.content)
        images = []
        videos = []
        for msg in tool_messages:
            if hasattr(msg, "artifact") and msg.artifact:
                images.extend(msg.artifact.get("images", []))
                videos.extend(msg.artifact.get("videos", []))

        system_message_content = (
            "You are a polite and helpful assistant providing information on MissionOS "
            "to users of MissionOS. The user's query is provided "
            "in the messages that follow this instruction. Use the following pieces "
            "of retrieved context, images, and videos to provide information that is "
            "directly relevant to the user's request. Respond using simple language "
            "that is easy to understand. The image captions provide clues how you can "
            "reference the images in your response. The video titles provide clues how "
            "you can reference the videos in your response. Treat the user as if all "
            "he or she knows about MissionOS is that it is a construction and "
            "instrumentation data platform. Provide options for further requests for "
            "information. Start each response with an overview of the topic raised in "
            "the question. The overview should introduce the topic and its context. "
            "Order your response in a logical way and use bullet points or numbered "
            "lists where appropriate. If the user asks a question that is definitely "
            "not related to MissionOS, provide a polite response indicating that you "
            "cannot assist. If an image is relevant, reference it using [Image N] "
            "(e.g., [Image 1], [Image 2]) at the end of a sentence or logical break, "
            "ensuring the reference enhances the explanation without disrupting "
            "sentence flow. Do not place [Image N] mid-sentence unless absolutely "
            "necessary, and avoid trailing punctuation (e.g., '.', ',') after "
            "[Image N]. Number images sequentially based on their order (1 for first, "
            "2 for second, etc.). If you don't know the answer, say so clearly.\n\n"
            f"Context:\n{retrieved_content}\n\n"
            f"Available images: {len(images)} image(s)\n"
            f"Available videos: {len(videos)} video(s)"
        )

        conversation_messages = [
            message for message in state["messages"] if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(content=system_message_content)] + conversation_messages
        response = llm.invoke(prompt, config={"configurable": {"thread_id": f"{st.session_state.thread_id}"}})
        response.additional_kwargs["images"] = images
        response.additional_kwargs["videos"] = videos
        generate_time = time.time() - start_time

        new_state = state.copy()
        new_state["messages"].append(response)
        new_state["images"] = images
        new_state["videos"] = videos
        new_state["timings"].extend(
            [
                {"node": "generate", "time": generate_time, "component": "LLM generation"},
                *[
                    {"node": "retrieve", "time": time_val, "component": component}
                    for component, time_val in (
                        tool_messages[-1].artifact.get("timings", {})
                        if tool_messages and hasattr(tool_messages[-1], "artifact") and tool_messages[-1].artifact
                        else {}
                    ).items()
                ],
            ]
        )
        return new_state
    

    # Initialize the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", tools_node)
    graph_builder.add_node("generate", generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        source="query_or_respond",
        path=tools_condition,
        path_map={END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile(checkpointer=MemorySaver())