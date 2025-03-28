
from collections import defaultdict

# session_id -> list of messages
session_memory = defaultdict(list)

def get_memory(session_id: str):
    return session_memory.get(session_id, [])

def append_user(session_id: str, message: str):
    session_memory[session_id].append({"role": "user", "parts": [{"text": message}]})

def append_ai(session_id: str, message: str):
    session_memory[session_id].append({"role": "model", "parts": [{"text": message}]})
