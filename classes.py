"""Defines state classes for the MissionHelp Demo application.

This module provides a typed dictionary for structuring the application's state
in the LangGraph workflow.
"""

from typing import List, Optional
from langgraph.graph import MessagesState


class State(MessagesState):
    """State class for managing conversation and multimedia in LangGraph.

    Attributes:
        messages: List of conversation messages (human, AI, or tool).
        images: Optional list of dictionaries with base64 image data and captions.
        videos: Optional list of dictionaries with video URLs and titles.
        timings: List of dictionaries with timing details for nodes and components.
    """
    images: Optional[List[dict]]
    videos: Optional[List[dict]]
    timings: List[dict]