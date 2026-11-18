import os

import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM
# =========================

my_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)


# =========================
# State  - shared memory
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    route: str              # "chat" or "travel" — set by router_agent
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# =========================
# Router Agent — classifies intent
# =========================

ROUTER_PROMPT = (
    "You are a router. If the user is asking about travel planning, booking flights, "
    "hotels, itinerary, trips, destinations, or anything travel-related, respond with "
    "exactly: travel\n"
    "For everything else (greetings, chitchat, questions about you, general questions, "
    "etc.), respond with exactly: chat\n"
    "Reply with ONLY one word: travel or chat"
)

def router_agent(state: TravelState):
    query = state["user_query"]
    response = my_llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=query),
    ])
    route = response.content.strip().lower()
    # Normalize — anything not exactly "chat" defaults to "travel"
    if route not in ("chat", "travel"):
        route = "travel" if any(w in query.lower() for w in
            ["trip", "flight", "hotel", "travel", "book", "itinerary", "destination",
             "fly", "stay", "vacation", "tour", "airport", "airline"]) else "chat"

    return {
        "route": route,
        "messages": [AIMessage(content=f"[Router → {route}]")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def route_decision(state: TravelState):
    return state.get("route", "travel")


# =========================
# Chat Agent — casual conversation
# =========================

CHAT_SYSTEM = (
    "You are Trip Planner, a friendly AI travel planning assistant. "
    "The user is chatting casually. Respond warmly and helpfully. "
    "If they ask about travel, tell them you can help plan trips — just describe "
    "where they want to go, their budget, and how many days. Keep responses concise."
)

def chat_agent(state: TravelState):
    response = my_llm.invoke([
        SystemMessage(content=CHAT_SYSTEM),
        HumanMessage(content=state["user_query"]),
    ])
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Flight Agent
# =========================

def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Itinerary Agent
# =========================

def _truncate(text: str, max_chars: int = 1500) -> str:
    """Keep text under max_chars to avoid Groq TPM limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n… (truncated)"


def itinerary_agent(state: TravelState):
    flights = _truncate(state['flight_results'], 1200)
    hotels  = _truncate(state['hotel_results'],  1200)

    prompt = f"""\
Create a concise travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{flights}

Hotel Results:
{hotels}

Make the itinerary practical, budget-aware, and easy to follow. Keep it concise.
"""

    response = my_llm.invoke([
        SystemMessage(content="You are an expert travel planner. Keep your response concise."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =========================
# Final Response Agent
# =========================


def final_agent(state: TravelState):
    flights  = _truncate(state['flight_results'],  1200)
    hotels   = _truncate(state['hotel_results'],   1200)
    itinerary = _truncate(state['itinerary'],      2000)

    final_prompt = f"""\
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{flights}

Hotels:
{hotels}

Itinerary:
{itinerary}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Be concise — keep each section focused.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
"""

    response = my_llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant. Keep your response concise and well-structured."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("router_agent", router_agent)
graph.add_node("chat_agent", chat_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

# Router decides the path
graph.add_edge(START, "router_agent")
graph.add_conditional_edges("router_agent", route_decision, {
    "travel": "flight_agent",
    "chat": "chat_agent",
})

# Travel pipeline
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

# Chat exits directly
graph.add_edge("chat_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# Function for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "route": "",
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "route": result.get("route", "travel"),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }