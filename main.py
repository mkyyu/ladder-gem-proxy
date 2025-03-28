from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from dotenv import load_dotenv
from memory_store import get_memory, append_user, append_ai
from mark_answer import router as mark_router

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_SECRET = os.getenv("API_SECRET")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mark_router)

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.post("/gemini", dependencies=[Depends(verify_api_key)])
async def gemini_handler(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    message = data.get("message")
    parent_context = data.get("parent_context")
    sub_question = data.get("sub_question")

    if not session_id or not message:
        raise HTTPException(status_code=400, detail="Missing session_id or message")

    # Inject optional context into the message
    if parent_context:
        message += f"\n\n(For context: the previous question was \"{parent_context}\")"
    if sub_question:
        message += f"\n\n(This is part of sub-question {sub_question}.)"

    append_user(session_id, message)
    full_history = get_memory(session_id)

    payload = {"contents": full_history}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)

    reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    # Cleanup response
    cleaned = reply.strip("`").replace("```json", "").replace("```", "").strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    append_ai(session_id, cleaned)

    return {"reply": cleaned}
