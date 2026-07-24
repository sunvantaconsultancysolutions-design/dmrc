import { useEffect, useRef, useState } from "react";
import Message from "./components/Message.jsx";
import ChatInput from "./components/ChatInput.jsx";
import StatusBadge from "./components/StatusBadge.jsx";
import { askQuestion } from "./api.js";

const SUGGESTIONS = [
  "What are the contractor's obligations?",
  "Who is responsible for maintenance during the defects liability period?",
  "Which stations are covered under this contract?",
  "What training must the contractor provide to employer staff?",
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [pending, setPending] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, pending]);

  async function handleSend(query) {
    const userMsg = { role: "user", content: query };
    setMessages((m) => [...m, userMsg]);
    setPending(true);

    try {
      const { answer, sources, confidence } = await askQuestion(query);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: answer, sources, confidence },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Something went wrong reaching the model: ${err.message}. Check that the backend pod is running and ALLOWED_ORIGINS includes this site.`,
          isError: true,
        },
      ]);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__brand">
          <span className="app-header__mark">DM</span>
          <div>
            <h1>DMRC Contract Intelligence</h1>
            <p>Grounded Q&amp;A over the ECS / BMS Scope of Work</p>
          </div>
        </div>
        <StatusBadge />
      </header>

      <main className="chat-scroll" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="empty-state">
            <p className="empty-state__eyebrow">Ask about the contract</p>
            <h2>Query any clause, obligation, or scope item directly.</h2>
            <div className="suggestion-grid">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => handleSend(s)} className="suggestion-card">
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="msg-list">
            {messages.map((m, i) => (
              <Message key={i} {...m} />
            ))}
            {pending && <Message role="assistant" isLoading />}
          </div>
        )}
      </main>

      <footer className="app-footer">
        <ChatInput onSend={handleSend} disabled={pending} />
        <p className="app-footer__note">
          Answers are grounded in retrieved contract clauses; verify against the source
          document before relying on them.
        </p>
      </footer>
    </div>
  );
}
