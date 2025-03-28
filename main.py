from fastapi import FastAPI, Request, Depends, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx
import os
from dotenv import load_dotenv
from mark_answer import router as mark_router

load_dotenv()

app = FastAPI()
app.include_router(mark_router)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_SECRET = os.getenv("API_SECRET")
session_histories = {}

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API Key")

class GeminiRequest(BaseModel):
    session_id: str
    message: Optional[str] = None
    parent_context: Optional[str] = None
    sub_question: Optional[str] = None
    mode: Optional[str] = "default"

@app.post("/gemini", dependencies=[Depends(verify_api_key)])
async def gemini_handler(req: GeminiRequest):
    session_id = req.session_id
    if session_id not in session_histories:
        session_histories[session_id] = []

    history = session_histories[session_id]

    if req.parent_context and req.sub_question:
        message = f"Context:\n{req.parent_context}\n\nNow answer this part:\n{req.sub_question}"
    elif req.message:
        message = req.message
    else:
        return {"error": "Missing message or sub-question."}

    history.append({
        "role": "user",
        "parts": [{"text": message}]
    })

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            gemini_response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                json={"contents": history}
            )

        if gemini_response.status_code != 200:
            return {
                "error": "Gemini API failed",
                "status_code": gemini_response.status_code,
                "details": gemini_response.text
            }

        reply = gemini_response.json()["candidates"][0]["content"]["parts"][0]["text"]
        reply = reply.replace("\\n", "\n").strip()

        history.append({
            "role": "model",
            "parts": [{"text": reply}]
        })

        return {"reply": reply}

    except Exception as e:
        return {"error": "Internal server error", "details": str(e)}

@app.post("/reset", dependencies=[Depends(verify_api_key)])
async def reset_session(req: GeminiRequest):
    session_histories.pop(req.session_id, None)
    return {"success": True}
