import SourceChip from "./SourceChip.jsx";

export default function Message({ role, content, sources, isError, isLoading }) {
  const isUser = role === "user";

  return (
    <div className={`msg-row ${isUser ? "msg-row--user" : "msg-row--assistant"}`}>
      <div className="msg-avatar" aria-hidden="true">
        {isUser ? "You" : "DM"}
      </div>

      <div className="msg-column">
        <div className={`msg-bubble ${isError ? "msg-bubble--error" : ""}`}>
          {isLoading ? (
            <span className="typing-dots" aria-label="Generating answer">
              <span />
              <span />
              <span />
            </span>
          ) : (
            <p>{content}</p>
          )}
        </div>

        {!isUser && sources && sources.length > 0 && (
          <div className="sources-rail">
            <span className="sources-rail__label">Sourced from</span>
            <div className="sources-rail__track">
              {sources.map((s, i) => (
                <SourceChip key={s.chunk_id || i} source={s} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
