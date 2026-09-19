import { useEffect, useState } from "react";
import Editor from "@monaco-editor/react";
import {
  Bot,
  Boxes,
  CircleDot,
  Code2,
  Command,
  FileCode2,
  FolderOpen,
  GitBranch,
  GitCommitHorizontal,
  MessageSquare,
  Play,
  Plus,
  Send,
  Settings2,
  TerminalSquare,
  Wrench,
  X
} from "lucide-react";

type Mode = "chat" | "editor" | "agent" | "files" | "git";
type Message = { role: "user" | "assistant"; content: string };
type Conversation = { id: string; title: string; updated_at: string; messages: Message[] };

const initialCode = `from fastapi import FastAPI\n\napp = FastAPI()\n\n\n@app.get("/")\ndef home():\n    return {"status": "online", "model": "veda"}\n`;

const navItems: { id: Mode; label: string; icon: typeof MessageSquare }[] = [
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "editor", label: "Editor", icon: Code2 },
  { id: "agent", label: "Agent", icon: Bot },
  { id: "files", label: "Files", icon: FolderOpen },
  { id: "git", label: "Git", icon: GitBranch }
];

function App() {
  const [mode, setMode] = useState<Mode>("chat");
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedFile, setSelectedFile] = useState("main.py");
  const [code, setCode] = useState(initialCode);
  const [isSending, setIsSending] = useState(false);

  useEffect(() => {
    fetch("/api/conversations")
      .then((response) => response.json())
      .then((items: Conversation[]) => setConversations(items))
      .catch(() => setConversations([]));
  }, []);

  async function sendMessage() {
    const trimmed = message.trim();
    if (!trimmed || isSending) return;
    const userMessage: Message = { role: "user", content: trimmed };
    setMessages((current) => [...current, userMessage]);
    setMessage("");
    setIsSending(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, history: messages })
      });
      const data = await response.json();
      setMessages((current) => [...current, { role: "assistant", content: data.response ?? data.detail ?? "No response." }]);
    } catch {
      setMessages((current) => [...current, { role: "assistant", content: "Backend offline. Start FastAPI on port 8000 to connect local Veda." }]);
    } finally {
      setIsSending(false);
    }
  }

  function renderMain() {
    if (mode === "editor") {
      return <EditorView code={code} setCode={setCode} selectedFile={selectedFile} />;
    }
    if (mode === "agent") return <AgentView />;
    if (mode === "files") return <FilesView onOpen={(file) => { setSelectedFile(file); setMode("editor"); }} />;
    if (mode === "git") return <GitView />;
    return <ChatView messages={messages} message={message} setMessage={setMessage} sendMessage={sendMessage} isSending={isSending} />;
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">V</span><span>VEDA</span><span className="slash">/</span><span className="workspace-name">WORKSPACE: VEDA</span></div>
        <div className="topbar-status"><span className="status-dot" /> LOCAL <span className="version">v0.3</span><button className="icon-button" title="Command palette"><Command size={16} /></button><button className="icon-button" title="Settings"><Settings2 size={16} /></button></div>
      </header>
      <div className="workspace">
        <aside className="sidebar">
          <div className="sidebar-label">WORKBENCH</div>
          {navItems.map(({ id, label, icon: Icon }) => <button key={id} className={`nav-item ${mode === id ? "active" : ""}`} onClick={() => setMode(id)}><Icon size={17} /><span>{label}</span></button>)}
          <div className="sidebar-spacer" />
          <div className="local-card"><span className="status-dot" /><div><strong>VEDA LOCAL</strong><small>READY / MPS</small></div></div>
        </aside>
        <main className="main-column">
          <div className="view-header"><div><span className="eyebrow">{mode.toUpperCase()}</span><h1>{mode === "chat" ? "Ask Veda" : mode === "editor" ? selectedFile : mode === "agent" ? "Active task" : mode === "files" ? "Project files" : "Source control"}</h1></div><div className="header-actions"><button className="outline-button"><CircleDot size={14} /> LOCAL MODEL</button><button className="icon-button"><Boxes size={16} /></button></div></div>
          <div className="main-content">{renderMain()}</div>
        </main>
        <aside className="right-rail"><AgentRail mode={mode} /></aside>
      </div>
      <footer className="terminal-bar"><div className="terminal-tabs"><span className="terminal-tab active"><TerminalSquare size={14} /> TERMINAL</span><span className="terminal-tab">PROBLEMS <b className="problem-count">0</b></span><span className="terminal-tab">OUTPUT</span><span className="terminal-tab">AGENT LOG</span></div><div className="terminal-line"><span className="prompt">$</span><span>veda status</span><span className="terminal-result">local model ready / workspace clean</span></div></footer>
    </div>
  );
}

function ChatView({ messages, message, setMessage, sendMessage, isSending }: { messages: Message[]; message: string; setMessage: (value: string) => void; sendMessage: () => void; isSending: boolean }) {
  return <section className="chat-view"><div className="conversation-list"><button className="new-chat" onClick={() => window.location.reload()}><Plus size={15} /> NEW CHAT</button><div className="list-heading">TODAY</div><div className="conversation-row selected"><MessageSquare size={14} /><span>{messages[0]?.content.slice(0, 24) || "New conversation"}</span></div><div className="list-heading muted">RECENT</div><div className="conversation-row"><MessageSquare size={14} /><span>Veda model roadmap</span></div><div className="conversation-row"><MessageSquare size={14} /><span>Training diagnostics</span></div></div><div className="chat-panel">{messages.length === 0 ? <div className="empty-chat"><span className="empty-glyph">V</span><h2>What are you building?</h2><p>Ask Veda about your project, code, or the next experiment.</p><div className="suggestions"><button onClick={() => setMessage("Explain this authentication flow")}>Explain authentication</button><button onClick={() => setMessage("Find the current training status")}>Check training status</button><button onClick={() => setMessage("Create a plan for this project")}>Create a project plan</button></div></div> : <div className="message-stack">{messages.map((item, index) => <div className={`message ${item.role}`} key={`${item.role}-${index}`}><div className="message-meta">{item.role === "user" ? "YOU" : "VEDA"}</div><div className="message-body">{item.content}</div></div>)}</div>}<div className="composer"><textarea value={message} onChange={(event) => setMessage(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(); } }} placeholder="Ask Veda..." rows={2} /><div className="composer-footer"><span>⌘↵ SEND</span><button className="send-button" onClick={sendMessage} disabled={isSending}>{isSending ? <span className="spinner" /> : <Send size={16} />}</button></div></div></div></section>;
}

function EditorView({ code, setCode, selectedFile }: { code: string; setCode: (value: string) => void; selectedFile: string }) {
  return <section className="editor-view"><div className="editor-tabs"><span className="editor-tab active"><FileCode2 size={14} /> {selectedFile}<X size={13} /></span><span className="editor-tab">app.py<X size={13} /></span><span className="editor-tab">git.diff<X size={13} /></span></div><div className="editor-frame"><Editor height="100%" language="python" theme="vs-dark" value={code} onChange={(value) => setCode(value ?? "")} options={{ minimap: { enabled: false }, fontSize: 13, fontFamily: "'IBM Plex Mono', monospace", padding: { top: 18 }, lineNumbersMinChars: 3, smoothScrolling: true, tabSize: 4 }} /></div></section>;
}

function AgentView() { return <section className="agent-view"><div className="task-banner"><div><span className="eyebrow">TASK / VEDA-001</span><h2>Build JWT authentication</h2><p>Agent plan ready for review. Changes are isolated to the workspace.</p></div><button className="primary-button"><Play size={14} /> START AGENT</button></div><div className="plan-list"><PlanRow state="done" label="Inspect project structure" detail="12 files indexed" /><PlanRow state="done" label="Identify authentication layer" detail="auth.py · middleware.py" /><PlanRow state="current" label="Implement authentication" detail="Waiting for approval" /><PlanRow state="pending" label="Add tests" detail="" /><PlanRow state="pending" label="Run test suite" detail="" /></div></section>; }
function PlanRow({ state, label, detail }: { state: "done" | "current" | "pending"; label: string; detail: string }) { return <div className={`plan-row ${state}`}><span className="plan-state">{state === "done" ? "✓" : state === "current" ? "◉" : "○"}</span><div><strong>{label}</strong>{detail && <small>{detail}</small>}</div>{state === "done" && <span className="plan-time">DONE</span>}</div>; }
function FilesView({ onOpen }: { onOpen: (file: string) => void }) { return <section className="files-view"><div className="file-toolbar"><button className="outline-button"><FolderOpen size={14} /> OPEN FOLDER</button><button className="icon-button"><Wrench size={16} /></button></div>{["src", "tests", ".veda", "README.md", "main.py", "requirements.txt"].map((file) => <button className="file-row" key={file} onClick={() => onOpen(file)}><FileCode2 size={15} /><span>{file}</span><small>{file === "src" || file === "tests" || file === ".veda" ? "DIR" : "PY"}</small></button>)}</section>; }
function GitView() { return <section className="git-view"><div className="git-summary"><GitBranch size={16} /><strong>main</strong><span>SYNCED</span></div><div className="list-heading">CHANGES / 0</div><div className="empty-state"><GitCommitHorizontal size={22} /><p>Workspace clean</p><small>Your next commit will appear here.</small></div></section>; }
function AgentRail({ mode }: { mode: Mode }) { return <div className="agent-rail"><div className="rail-heading"><span>VEDA AGENT</span><span className="rail-status"><span className="status-dot" /> READY</span></div><div className="rail-section"><span className="eyebrow">CONTEXT</span><div className="context-row"><FileCode2 size={14} /><span>{mode === "editor" ? "main.py" : "workspace"}</span><small>1 file</small></div><div className="context-row"><GitBranch size={14} /><span>Working tree</span><small>clean</small></div></div><div className="rail-section"><span className="eyebrow">QUICK ACTIONS</span><button className="rail-action">Explain selection <span>⌘K</span></button><button className="rail-action">Find related files <span>⌘P</span></button><button className="rail-action">Create task <span>+</span></button></div><div className="rail-section agent-note"><span className="eyebrow">MODEL</span><strong>Veda v0.3</strong><small>26.5M parameters · local</small></div></div>; }

export default App;
