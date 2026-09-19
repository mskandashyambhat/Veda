from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.inference import VedaProvider
from backend.storage import Storage
from backend.tools import PermissionDenied, WorkspaceTools


app = FastAPI(title="Veda API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:1420", "http://127.0.0.1:1420"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
storage = Storage(os.getenv("VEDA_DB", ".veda/veda.db"))
provider = VedaProvider()
tools = WorkspaceTools(os.getenv("VEDA_WORKSPACE", "."), allow_execute=os.getenv("VEDA_ALLOW_EXECUTE", "0") == "1")


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=20000)
    history: list[dict[str, str]] = Field(default_factory=list)


class CommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=4000)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model": "Veda v0.3", "local_checkpoint": provider.ready, "device": str(provider.device)}


@app.get("/api/conversations")
def conversations() -> list[dict]:
    return storage.list_conversations()


@app.get("/api/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: str) -> list[dict]:
    return storage.messages(conversation_id)


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    conversation_id = request.conversation_id or storage.create_conversation(request.message[:60])
    storage.add_message(conversation_id, "user", request.message)
    prompt_parts = [f"<|user|>\n{item['content']}" for item in request.history[-6:] if item.get("content")]
    prompt_parts.append(f"<|user|>\n{request.message}\n<|assistant|>\n")
    response = provider.generate("\n".join(prompt_parts))
    storage.add_message(conversation_id, "assistant", response, {"provider": "veda-local", "device": str(provider.device)})
    return {"conversation_id": conversation_id, "response": response, "model": "Veda v0.3"}


@app.websocket("/ws/chat")
async def chat_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            message = str(payload.get("message", "")).strip()
            if not message:
                await websocket.send_json({"type": "error", "message": "Message is required"})
                continue
            response = await asyncio.to_thread(provider.generate, message)
            for word in response.split(" "):
                await websocket.send_json({"type": "token", "value": f"{word} "})
                await asyncio.sleep(0.01)
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        return


@app.get("/api/files")
def list_files() -> dict:
    return {"files": tools.list_files()}


@app.get("/api/files/{relative_path:path}")
def read_file(relative_path: str) -> dict:
    try:
        return {"path": relative_path, "content": tools.read_file(relative_path)}
    except (FileNotFoundError, PermissionDenied) as error:
        raise HTTPException(status_code=403, detail=str(error)) from error


@app.post("/api/terminal")
def run_command(request: CommandRequest) -> dict:
    try:
        return tools.run_command(request.command)
    except PermissionDenied as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
