"""Set up dependencies and configuration for the MissionHelp Demo application.

This module initializes the LLM, embeddings, Qdrant vector store, and database,
ensuring all components are ready for the RAG pipeline.
"""

import os
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.messages import HumanMessage, AIMessage

import database


# Qdrant configuration
QDRANT_INFO = {
    "client": QdrantClient(**st.secrets.qdrant_client_credentials),
    "collection_name": "manual-text",
}


def install_playwright_browsers() -> None:
    """Install Playwright browsers for web scraping.

    Checks if browsers are already installed and installs them if needed.
    """
    playwright_dir = os.path.expanduser("~/.cache/ms-playwright")
    if not os.path.exists(playwright_dir) or not os.listdir(playwright_dir):
        st.toast("Installing Playwright browsers...", icon=":material/build:")
        try:
            subprocess.run(["playwright", "install"], check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to install Playwright browsers: {e}")


def get_llm() -> ChatOpenAI:
    """Initialize the Grok 3 Beta language model.

    Returns:
        A ChatOpenAI instance configured with xAI API.
    """
    st.toast("Setting up the Gemini 2.0 Flash LLM...", icon=":material/build:")
    return ChatVertexAI(
        model="gemini-2.0-flash-001"
    )


def set_google_credentials() -> None:
    """Set Google Cloud credentials for database access.

    Writes credentials from secrets to a temporary file and sets the environment variable.
    """
    st.toast("Setting Google credentials...", icon=":material/build:")
    credentials_json = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    temp_file_path = "google_credentials.json"
    with open(temp_file_path, "w") as f:
        f.write(credentials_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path


def get_embeddings() -> VertexAIEmbeddings:
    """Initialize the Gemini text embeddings model.

    Returns:
        A VertexAIEmbeddings instance for text-embedding-004.
    """
    st.toast("Setting up the Gemini text-embedding-004 model...", icon=":material/build:")
    return VertexAIEmbeddings(
        model="text-embedding-004",
        project=st.secrets.google_project_id,
    )


def delete_collection() -> None:
    """Delete the existing Qdrant collection."""
    st.toast("Deleting existing Qdrant collection...", icon=":material/build:")
    client = QDRANT_INFO["client"]
    client.delete_collection(collection_name=QDRANT_INFO["collection_name"])


def create_collection() -> None:
    """Create a new Qdrant collection for vector storage."""
    st.toast("Creating new Qdrant collection...", icon=":material/build:")
    client = QDRANT_INFO["client"]
    client.create_collection(
        collection_name=QDRANT_INFO["collection_name"],
        vectors_config=models.VectorParams(
            size=768,  # Matches text-embedding-004
            distance=models.Distance.COSINE,
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            ),
        ),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=12),
    )


def collection_exists() -> bool:
    """Check if the Qdrant collection exists.

    Returns:
        True if the collection exists, False otherwise.
    """
    client = QDRANT_INFO["client"]
    existing_collections = client.get_collections()
    return any(
        col.name == QDRANT_INFO["collection_name"]
        for col in existing_collections.collections
    )


def points_exist() -> bool:
    """Check if the Qdrant collection contains points.

    Returns:
        True if points exist, False otherwise.
    """
    client = QDRANT_INFO["client"]
    collection_info = client.get_collection(QDRANT_INFO["collection_name"])
    return collection_info.points_count is not None and collection_info.points_count > 0


def get_vector_store(embeddings: VertexAIEmbeddings) -> QdrantVectorStore:
    """Initializes the Qdrant vector store.

    Args:
        embeddings: Embeddings model for vectorization.

    Returns:
        Configured QdrantVectorStore instance.
    """
    st.toast("Setting up Qdrant vector store...", icon=":material/build:")
    return QdrantVectorStore(
        client=QDRANT_INFO["client"],
        collection_name=QDRANT_INFO["collection_name"],
        embedding=embeddings,
    )


def rebuild_database() -> None:
    """Rebuild the database and vector store from scratch."""
    delete_collection()
    create_collection()
    database.create_images_table()

    docs = database.web_scrape()
    all_splits = database.chunk_text(docs=docs)
    embeddings = get_embeddings()
    st.session_state.vector_store = get_vector_store(embeddings)
    database.index_chunks(
        all_splits=all_splits,
        vector_store=st.session_state.vector_store,
    )


def run_batch_test(test_csv, graph, vector_store):
    """Processes batch test queries and generates results CSV with NDCG, MAP, MRR, and relevance scores, yielding progress updates.

    Args:
        test_csv: Uploaded CSV file with test queries.
        graph: LangGraph instance for query processing.
        vector_store: Qdrant vector store for retrieval.

    Yields:
        Tuple of (current_query, total_queries, results) where results is a list of result dictionaries.
    """
    def dcg_at_k(ranks):
        """Calculate Discounted Cumulative Gain (DCG)."""
        return np.sum([rel / np.log2(rank + 2) for rank, rel in enumerate(ranks)])

    def ndcg_at_k(true_relevance, predicted_scores):
        """Calculate Normalized DCG (NDCG)."""
        if not true_relevance or not predicted_scores:
            return 0.0
        actual_dcg = dcg_at_k([true_relevance[i] for i in np.argsort(predicted_scores)[::-1]])
        ideal_dcg = dcg_at_k(sorted(true_relevance, reverse=True))
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def average_precision(true_relevance, predicted_scores):
        """Calculate Average Precision (AP) for MAP."""
        if not true_relevance or not predicted_scores:
            return 0.0
        sorted_indices = np.argsort(predicted_scores)[::-1]
        relevant = 0
        precision_sum = 0.0
        for i, idx in enumerate(sorted_indices):
            if true_relevance[idx] == 1:
                relevant += 1
                precision_sum += relevant / (i + 1)
        return precision_sum / sum(true_relevance) if sum(true_relevance) > 0 else 0.0

    def reciprocal_rank(true_relevance, predicted_scores):
        """Calculate Reciprocal Rank (RR) for MRR."""
        if not true_relevance or not predicted_scores:
            return 0.0
        sorted_indices = np.argsort(predicted_scores)[::-1]
        for i, idx in enumerate(sorted_indices):
            if true_relevance[idx] == 1:
                return 1.0 / (i + 1)
        return 0.0

    expected_timings = {
        "Vector store retrieval": 0.0,
        "Image fetch": 0.0,
        "Tool execution": 0.0,
        "LLM decision": 0.0,
        "LLM generation": 0.0
    }

    df = pd.read_csv(test_csv)
    grouped = df.groupby(['query_id', 'query_text'])
    total_queries = len(grouped)
    results = []
    last_percent = -1

    status_message = st.toast(f"Processing batch test: 0%", icon=":material/build:")
    for i, ((query_id, query_text), group) in enumerate(grouped):
        ground_truth_ids = set(group['point_id'].astype(str))
        config = {"configurable": {"thread_id": f"test_query_{query_id}"}}
        initial_state = {
            "messages": [
                HumanMessage(content=query_text),
            ],
            "images": [],
            "videos": [],
            "timings": [],
        }
        final_state = list(graph.stream(initial_state, stream_mode="values", config=config))[-1]
        response_message = [
            msg for msg in final_state["messages"]
            if isinstance(msg, AIMessage) and not msg.tool_calls
        ][-1]
        response_text = response_message.content
        tool_message = [msg for msg in final_state["messages"] if msg.type == "tool"]
        retrieved_docs = tool_message[-1].artifact["docs"] if tool_message else []
        timings = final_state["timings"]

        retrieved_results = vector_store.similarity_search_with_relevance_scores(
            query_text, k=len(retrieved_docs) if retrieved_docs else 4
        )
        retrieved_results = sorted(retrieved_results, key=lambda x: x[1], reverse=True)
        retrieved_docs = [doc for doc, _ in retrieved_results]
        predicted_scores = [score for _, score in retrieved_results]
        true_relevance = [
            1 if doc.metadata.get('_id', 'unknown') in ground_truth_ids else 0
            for doc in retrieved_docs
        ]

        ndcg = ndcg_at_k(true_relevance, predicted_scores)
        ap = average_precision(true_relevance, predicted_scores)
        rr = reciprocal_rank(true_relevance, predicted_scores)

        timing_dict = expected_timings.copy()
        for timing in timings:
            if timing["component"] in timing_dict:
                timing_dict[timing["component"]] = timing["time"]

        for doc, score in retrieved_results:
            chunk_id = doc.metadata.get('_id', 'unknown')
            chunk_text = doc.page_content
            chunk_url = doc.metadata.get('source', 'unknown')
            is_in_gt = chunk_id in ground_truth_ids
            results.append({
                'Test query ID': query_id,
                'Query text': query_text,
                'Response text': response_text,
                'Retrieved chunk ID': chunk_id,
                'Retrieved chunk text': chunk_text,
                'Retrieved chunk page URL': chunk_url,
                'Is in ground truth': is_in_gt,
                'Relevance score': score,
                'NDCG': ndcg,
                'MAP': ap,
                'MRR': rr,
                **{f"{component} (s)": time for component, time in timing_dict.items()}
            })

        # Update toast every 1% progress
        current_percent = int(((i + 1) / total_queries) * 100)
        if current_percent > last_percent:
            status_message.toast(f"Processing batch test: {current_percent}%")
            last_percent = current_percent

        yield i + 1, total_queries, results
    yield total_queries, total_queries, results