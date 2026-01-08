# Exercise - Create a Router with LangGraph - STARTER
# In this exercise, you will build a router using LangGraph to dynamically control the flow of your application.

# Challenge

# You're building a text processing application that can:

# Reverse a string (e.g., "hello" → "olleh")
# Convert a string to uppercase (e.g., "hello" → "HELLO")
# Your application should:

# Accept user input and an action type.
# Route to the appropriate node (reverse or upper) based on the action.
# Handle invalid actions gracefully.
# This will be achieved by routing the input through LangGraph nodes using a conditional edge.

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
