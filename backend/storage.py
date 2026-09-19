from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class Storage:
    def __init__(self, path: str | Path = ".veda/veda.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_conversation(self, title: str = "New conversation") -> str:
        conversation_id = str(uuid.uuid4())
        timestamp = self.now()
        self.connection.execute(
            "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, title, timestamp, timestamp),
        )
        self.connection.commit()
        return conversation_id

    def add_message(self, conversation_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        self.connection.execute(
            "INSERT INTO messages(id, conversation_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), conversation_id, role, content, json.dumps(metadata or {}), self.now()),
        )
        self.connection.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (self.now(), conversation_id))
        self.connection.commit()

    def list_conversations(self) -> list[dict]:
        rows = self.connection.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def messages(self, conversation_id: str) -> list[dict]:
        rows = self.connection.execute("SELECT role, content, metadata, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at", (conversation_id,)).fetchall()
        return [dict(row) for row in rows]
