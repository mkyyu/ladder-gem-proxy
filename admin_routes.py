# admin_routes.py
from fastapi import APIRouter, Request, HTTPException, Header, Depends
from typing import Dict
from memory_store import memory_store

ADMIN_KEY = "admin-yueming"

admin_router = APIRouter()

def verify_admin_key(x_api_key: str = Header(...)):
    if x_api_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")

@admin_router.get("/admin/stats", dependencies=[Depends(verify_admin_key)])
async def get_stats():
    return {
        "total_sessions": len(memory_store),
        "total_messages": sum(len(session) for session in memory_store.values())
    }

@admin_router.get("/admin/sessions", dependencies=[Depends(verify_admin_key)])
async def list_sessions():
    return {"sessions": list(memory_store.keys())}

@admin_router.get("/admin/session/{session_id}", dependencies=[Depends(verify_admin_key)])
async def view_session(session_id: str):
    session = memory_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": session}

@admin_router.get("/admin/env", dependencies=[Depends(verify_admin_key)])
async def get_env():
    return {
        "default_openai_model": os.getenv("OPENAI_MODEL"),
        "default_gemini_model": os.getenv("GEMINI_MODEL")
    }

@admin_router.get("/admin/health", dependencies=[Depends(verify_admin_key)])
async def health_check():
    return {"status": "ok"}
