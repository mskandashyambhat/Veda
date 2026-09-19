from __future__ import annotations

import subprocess
from pathlib import Path


class PermissionDenied(Exception):
    pass


class WorkspaceTools:
    def __init__(self, workspace: str | Path, allow_execute: bool = False) -> None:
        self.workspace = Path(workspace).resolve()
        self.allow_execute = allow_execute

    def _safe_path(self, relative_path: str) -> Path:
        candidate = (self.workspace / relative_path).resolve()
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise PermissionDenied("Path is outside the workspace")
        return candidate

    def list_files(self, relative_path: str = ".") -> list[str]:
        directory = self._safe_path(relative_path)
        ignored = {".git", ".venv", "node_modules", "dist", "build", "__pycache__"}
        return sorted(str(path.relative_to(self.workspace)) for path in directory.rglob("*") if path.is_file() and not ignored.intersection(path.parts))

    def read_file(self, relative_path: str) -> str:
        return self._safe_path(relative_path).read_text()

    def write_file(self, relative_path: str, content: str) -> None:
        path = self._safe_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def run_command(self, command: str) -> dict[str, str | int]:
        if not self.allow_execute:
            raise PermissionDenied("Terminal execution requires explicit permission")
        result = subprocess.run(command, cwd=self.workspace, shell=True, capture_output=True, text=True, timeout=120)
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
