from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

# log conversation history per session
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

# init sess mem
    if session_id not in session_histories:
        session_histories[session_id] = []

    history = session_histories[session_id]


    if req.parent_context and req.sub_question:
        user_message = f"""Based on the following question context:\n\n{req.parent_context}\n\nNow answer this part:\n{req.sub_question}"""
    elif req.message:
        user_message = req.message
    else:
        return {"error": "You must provide either `message` or (`parent_context` and `sub_question`)."}

    # append user message to history
    history.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    # send entire conv to gemi
    async with httpx.AsyncClient() as client:
        gemini_response = await client.post(
            GEMINI_ENDPOINT,
            json={"contents": history}
        )

    if gemini_response.status_code != 200:
        return {"error": "Gemini API failed", "details": gemini_response.text}

    reply = gemini_response.json()["candidates"][0]["content"]["parts"][0]["text"]

    # add reply to session history
    history.append({
        "role": "model",
        "parts": [{"text": reply}]
    })

    return {"reply": reply}

@app.post("/reset")
async def reset_session(req: GeminiRequest):
    session_histories.pop(req.session_id, None)
    return {"success": True}