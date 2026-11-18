
import traceback
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import run_travel_agent, travel_graph, _conn

import json
import uuid

from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

AGENT_ORDER = ["flight_agent", "hotel_agent", "itinerary_agent", "final_agent"]


# =========================
# FastAPI Application
# =========================

app = FastAPI(
    title="TripMate AI API",
    description="LangGraph Multi-Agent Travel Planner API",
    version="1.0.0"
)


# =========================
# CORS Configuration
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Request Schema
# =========================

class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


# =========================
# Travel API Endpoint
# =========================

@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):

    try:
        user_message = request_data.message.strip()

        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty."
            )

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id
        )

        return {
            "success": True,
            "thread_id": result["thread_id"],
            "answer": result["answer"],
            "flight_results": result["flight_results"],
            "hotel_results": result["hotel_results"],
            "itinerary": result["itinerary"],
            "llm_calls": result["llm_calls"]
        }

    except HTTPException:
        raise

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your travel request."
        )


# =========================
# Health Check Endpoint
# =========================

@app.get("/health")
async def health_check():

    return {
        "status": "ok",
        "message": "Trip Multi-Agent is running"
    }


# =========================
# History Endpoint  (paginated trips)
# =========================

@app.get("/api/history/{thread_id}")
async def get_history(thread_id: str, page: int = 1, limit: int = 3):
    """
    Fetch paginated trip history for a given thread_id.
    Messages are grouped into individual trips (each HumanMessage starts a trip).
    Returns newest trips first.
    """
    EMPTY = {
        "thread_id": thread_id,
        "found": False,
        "trips": [],
        "total": 0,
        "page": 1,
        "limit": limit,
        "total_pages": 0,
    }

    try:
        config = {"configurable": {"thread_id": thread_id}}

        # get_state may return None or a snapshot with empty values
        try:
            state = travel_graph.get_state(config)
        except Exception:
            return EMPTY

        if state is None:
            return EMPTY

        values = getattr(state, "values", None) or {}
        raw_messages = values.get("messages", [])

        if not raw_messages:
            return EMPTY

        # Group messages into trips — each HumanMessage starts a new trip
        trips = []
        current = None

        for msg in raw_messages:
            cls = msg.__class__.__name__
            if cls == "HumanMessage":
                if current:
                    trips.append(current)
                current = {
                    "query": msg.content,
                    "answer": "",
                    "answer_preview": "",
                }
            elif cls == "AIMessage" and current:
                # Only the long final answer (skip short agent hand-off messages)
                if len(msg.content) > 200:
                    current["answer"] = msg.content
                    current["answer_preview"] = (
                        msg.content[:300].rstrip()
                        + ("…" if len(msg.content) > 300 else "")
                    )

        if current:
            trips.append(current)

        # Newest first
        trips.reverse()

        total       = len(trips)
        limit       = max(1, min(limit, 20))
        page        = max(1, page)
        total_pages = max(1, (total + limit - 1) // limit) if total else 0
        page        = min(page, total_pages) if total_pages else 1

        start     = (page - 1) * limit
        paginated = trips[start : start + limit]

        return {
            "thread_id":   thread_id,
            "found":       total > 0,
            "trips":       paginated,
            "total":       total,
            "page":        page,
            "limit":       limit,
            "total_pages": total_pages,
        }

    except Exception as e:
        print("HISTORY ERROR:", e)
        traceback.print_exc()
        # Return empty instead of crashing with 500
        return EMPTY


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/travel/stream")
def travel_stream(request_data: TravelRequest):  # plain `def`: FastAPI runs it in a worker thread
    message = request_data.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    thread_id = request_data.thread_id or f"user_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    def events():
        try:
            merged = {"flight_results": "", "hotel_results": "", "itinerary": "", "llm_calls": 0}
            answer = ""

            yield _sse("agent_started", {"agent": AGENT_ORDER[0]})

            for chunk in travel_graph.stream(
                {
                    "messages": [HumanMessage(content=message)],
                    "user_query": message,
                    "flight_results": "",
                    "hotel_results": "",
                    "itinerary": "",
                    "llm_calls": 0,
                },
                config=config,
                stream_mode="updates",  # one chunk per finished node
            ):
                for node, update in chunk.items():
                    if node not in AGENT_ORDER:
                        continue
                    for key in merged:
                        if key in update:
                            merged[key] = update[key]
                    if node == "final_agent":
                        answer = update["messages"][-1].content

                    yield _sse("agent_done", {"agent": node})
                    nxt = AGENT_ORDER.index(node) + 1
                    if nxt < len(AGENT_ORDER):
                        yield _sse("agent_started", {"agent": AGENT_ORDER[nxt]})

            yield _sse("result", {"thread_id": thread_id, "answer": answer, **merged})

        except Exception:
            traceback.print_exc()
            yield _sse("error", {"message": "Something went wrong while processing your travel request."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =========================
# Run Server
# =========================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
