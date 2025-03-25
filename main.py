from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Gemini 2.0 Flash via v1beta (for MakerSuite keys)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Session memory (in-memory per session_id)
session_histories = {}

class GeminiRequest(BaseModel):
    session_id: str
    message: Optional[str] = None
    parent_context: Optional[str] = None
    sub_question: Optional[str] = None
    mode: Optional[str] = "default"

@app.post("/gemini")
async def gemini_handler(req: GeminiRequest):
    session_id = req.session_id
    mode = req.mode or "default"

    if session_id not in session_histories:
        session_histories[session_id] = []

    history = session_histories[session_id]

    # Build message
    if req.parent_context and req.sub_question:
        message = f"Context:\n{req.parent_context}\n\nNow answer this part:\n{req.sub_question}"
    elif req.message:
        message = req.message
    else:
        return {"error": "Missing message or sub-question."}

    # Append user input to session history
    history.append({
        "role": "user",
        "parts": [{"text": message}]
    })

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            gemini_response = await client.post(
                GEMINI_ENDPOINT,
                json={"contents": history}
            )

        if gemini_response.status_code != 200:
            return {
                "error": "Gemini API failed",
                "status_code": gemini_response.status_code,
                "details": gemini_response.text
            }

        reply = gemini_response.json()["candidates"][0]["content"]["parts"][0]["text"]

        # Append model reply to memory
        history.append({
            "role": "model",
            "parts": [{"text": reply}]
        })

        return {"reply": reply}

    except Exception as e:
        return {
            "error": "Internal server error",
            "details": str(e)
        }

@app.post("/reset")
async def reset_session(req: GeminiRequest):
    session_histories.pop(req.session_id, None)
    return {"success": True}

