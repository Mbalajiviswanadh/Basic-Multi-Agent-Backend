
import traceback
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import run_travel_agent


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
# Run Server
# =========================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
