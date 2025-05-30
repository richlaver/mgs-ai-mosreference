"""Configures the LangGraph workflow for query processing in the MissionHelp Demo.

This module defines the retrieval-augmented generation pipeline, integrating the
language model and vector store.
"""

import base64
import csv
import logging
import re
from typing import Tuple, Generator

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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


def generate_thread_title(query: str, response: str, llm) -> str:
    """
    Generate a unique thread title using an LLM.
    Title is max 40 chars and descriptive.

    Args:
        query: User query string.
        response: LLM response string.
        llm: Language model instance.

    Returns:
        Thread title string.
    """
    prompt = ChatPromptTemplate.from_template(
        "Generate a single, descriptive title (max 35 chars) for a chat thread "
        "based on this query: '{query}' and response: '{response}'. "
        "Focus on the most relevant key concepts, use concise language, "
        "and avoid introductory phrases like 'Here are' or 'Options'. "
        "Return only the title."
        "Omit mentioning MissionOS or Maxwell GeoSystems in the title."
    )
    chain = prompt | llm
    result = chain.invoke({"query": query[:100], "response": response[:100]})
    return result.content[:40].strip()


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
        with database.connect_to_images_db() as conn:
            cursor = conn.cursor()
            try:
                if image_ids:
                    cursor.execute(
                        "SELECT id, image_binary, caption FROM images WHERE id = ANY(%s)",
                        (image_ids,),
                    )
                    image_map = {
                        f"db://images/{img[0]}": {
                            "id": img[0],
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
            "timings": {"search": search_time, "image_fetch": image_fetch_time},
        }
        return serialized_docs, artifact
    

    def query_or_respond(state: State, config: dict) -> dict:
        """Decides whether to query tools or respond directly.

        Args:
            state: Current state with messages and metadata.

        Returns:
            Updated state with new messages and timings.
        """
        start_time = time.time()
        thread_id = config["configurable"]["thread_id"]
        user_id = config["configurable"]["user_id"]
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are a polite and helpful assistant for MissionOS."
                    "Use the retrieve tool to fetch relevant information to answer the query."
                    "If the query is ambiguous, assume it relates to MissionOS and use the tool."
                ),
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}"),
        ])
        llm_with_tools = llm.bind_tools([retrieve])
        chain = prompt | llm_with_tools
        response = chain.invoke(
            {"history": state["messages"], "query": state["messages"][-1].content},
            config={"configurable": {"thread_id": thread_id}},
        )
        response_time = time.time() - start_time

        database.add_thread_to_db(
            thread_id=thread_id,
            user_id=user_id,
            title=generate_thread_title(
                query=state["messages"][-1].content,
                response=response.content,
                llm=llm
            ),
        )
        query_message_id = database.add_message_to_db(
            thread_id=thread_id,
            user_id=user_id,
            content=state["messages"][-1].content,
            is_ai=False
        )
        state["messages"][-1].additional_kwargs["message_id"] = query_message_id

        has_tool_calls = (hasattr(response, "tool_calls") and bool(response.tool_calls)) or \
                     (hasattr(response, "invalid_tool_calls") and bool(response.invalid_tool_calls))
        if not has_tool_calls:
            response_message_id = database.add_message_to_db(
                thread_id=thread_id,
                user_id=user_id,
                content=response.content,
                is_ai=True
            )
            response.additional_kwargs["message_id"] = response_message_id

        new_state = state.copy()
        new_state["messages"].append(response)
        new_state["timings"].append({"node": "query_or_respond", "time": response_time, "component": "llm_decision"})
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
        updated_state["timings"].append({"node": "tools", "time": tool_time, "component": "tool_execution"})
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
    

    def generate(state: State, config: dict) -> Generator[State, None, None]:
        """Generates a response using retrieved context and multimedia.

        Args:
            state: Current state with messages and metadata.

        Returns:
            Generator yielding updated states with response and multimedia.
        """
        start_time = time.time()
        thread_id = config["configurable"]["thread_id"]
        user_id = config["configurable"]["user_id"]
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

        image_info = "\n".join(
            f"Image ID:{img['id']}: {img['caption'] or 'No caption'}"
            for img in images
        ) if images else "No images available."
        system_message_content = (
            "You are a polite and helpful assistant providing information to MissionOS users. "
            "The user's query is provided in the messages that follow this instruction. "
            "Use the following pieces of retrieved context, images, and videos to provide information directly relevant to the user's request. "
            "Respond using simple language that is easy to understand. "
            "All that the user knows about MissionOS is that it is a construction and instrumentation data platform. "
            "Provide options for further information requests. "
            "Start responses with an overview and context of the queried topic. "
            "Order your response in a logical way and use bullet points or numbered lists where appropriate. "
            "For questions definitely not related to MissionOS, politely respond that you cannot assist. "
            "The image and video captions provide clues how you can reference images and videos in your response. "
            "Assign a unique image number to each image you reference in your response. "
            "Image numbers start from 1 and are sequentially numbered based on their order of insertion. "
            "Reference images naturally in the response, e.g. 'See Image 1 below for details.', '...as shown in Image 2', 'Image 3 illustrates that...' "
            "A placeholder marks where an image will be rendered. "
            "An image will only be rendered once in the response, even if referenced multiple times, and therefore there will only be one placeholder per image. "
            "Placeholders are enclosed in curved brackets like (Image 1), (Image 2), etc. "
            "Insert each placeholder on a new line, one placeholder for one line. "
            "Ensure placeholder insertion enhances explanation without disrupting flow of reading. "
            "Avoid trailing punctuation after placeholders. "
            "If you don't know the answer, say so clearly.\n\n"
            f"Context:\n{retrieved_content}\n\n"
            f"Available images:\n{image_info}\n\n"
            f"Available videos: {len(videos)} video(s)"
        )

        conversation_messages = [
            message for message in state["messages"] if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(content=system_message_content)] + conversation_messages

        accumulated_content = ""
        for chunk in llm.stream(prompt, config={"configurable": {"thread_id": thread_id}}):
            new_state = state.copy()
            new_state["messages"] = new_state["messages"] + [AIMessage(
                content=chunk.content,
                additional_kwargs={}
            )]
            accumulated_content += chunk.content
            yield new_state

        placeholder_nums = []
        seen_nums = set()
        for match in re.finditer(r"\(Image (\d+)\)", accumulated_content):
            num = match.group(1)
            if num not in seen_nums:
                placeholder_nums.append(num)
                seen_nums.add(num)

        image_map = {}
        ordered_images = [None] * len(placeholder_nums)
        remaining_images = images.copy()
        used_image_ids = set()

        for num in placeholder_nums:
            if not remaining_images:
                break
            img = remaining_images.pop(0)
            image_map[num] = img["id"]
            ordered_images[int(num) - 1] = img
            used_image_ids.add(img["id"])

        for img in remaining_images:
            if img["id"] not in used_image_ids:
                ordered_images.append(img)
                image_map[str(len(image_map) + 1)] = img["id"]

        generate_time = time.time() - start_time
        final_response = AIMessage(
            content=accumulated_content,
            additional_kwargs={
                "images": ordered_images,
                "videos": videos,
                "image_map": image_map
            }
        )
        response_message_id = database.add_message_to_db(
            thread_id=thread_id,
            user_id=user_id,
            content=accumulated_content,
            is_ai=True
        )
        final_response.additional_kwargs["message_id"] = response_message_id
        database.add_message_timings_to_db(
            message_id=response_message_id,
            timings=[
                {"node": "generate", "time": generate_time, "component": "llm_generation"},
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

        new_state = state.copy()
        new_state["messages"] = new_state["messages"] + [final_response]
        new_state["images"] = ordered_images
        new_state["videos"] = videos
        new_state["timings"].extend(
            [
                {"node": "generate", "time": generate_time, "component": "llm_generation"},
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
        yield new_state
    

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