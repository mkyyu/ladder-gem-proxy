from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_SECRET = os.getenv("API_SECRET")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API Key")

class MarkRequest(BaseModel):
    session_id: str
    question_number: str
    marks: int
    question_content: str
    markscheme: str
    student_answer: str
    model: Optional[str] = "openai"

@router.post("/mark-answer", dependencies=[Depends(verify_api_key)])
async def mark_answer(req: MarkRequest):
    prompt = f"""
You are a strict exam marker. Given the following:

- Question: {req.question_content}
- Markscheme: {req.markscheme}
- Max Marks: {req.marks}
- Student Answer: {req.student_answer}

Please:
1. Award marks out of the maximum.
2. Give concise feedback on what was correct or missing.

Respond in JSON format like:
{{
  "final_marks": <int>,
  "feedback": "<feedback text>"
}}
""".strip()

    try:
        if req.model.lower() == "gemini":
            return await call_gemini(prompt, req.session_id)
        else:
            return await call_openai(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def call_openai(prompt: str):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": OPENAI_MODEL,
        "messages": [
            { "role": "system", "content": "You are a strict exam marker." },
            { "role": "user", "content": prompt }
        ],
        "temperature": 0.2
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=json_data)
        if response.status_code != 200:
            raise Exception(response.text)
        content = response.json()["choices"][0]["message"]["content"]
        return safe_json(content)

async def call_gemini(prompt: str, session_id: str):
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(endpoint, json=payload)
        if response.status_code != 200:
            raise Exception(response.text)
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return safe_json(content)

def safe_json(text):
    import json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "final_marks": 0,
            "feedback": f"AI returned an unstructured response: {text}"
        }
