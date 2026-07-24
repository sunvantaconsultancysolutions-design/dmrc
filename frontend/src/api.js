// Talks to the FastAPI backend defined in src/app.py. Endpoints and field
// names below are copied 1:1 from that file's Pydantic models -- keep them
// in sync if app.py's contract changes.
//
//   POST /ask     { query } -> { answer, sources: SourceItem[], confidence }
//   GET  /status  -> { status, embedding_model, reranker_model,
//                      dense_model_loaded, reranker_model_loaded,
//                      chromadb_connected }
//
// SourceItem: { clause, page, document, item_number, retrieval_source,
//               reranker_score, chunk_id }  -- every field is optional,
// since not all chunk types (contract clause vs. BOQ row) carry all of them.

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function parseErrorDetail(res) {
  try {
    const body = await res.json();
    return body.detail || res.statusText;
  } catch {
    return res.statusText;
  }
}

export async function askQuestion(query, signal) {
  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });

  if (!res.ok) {
    throw new Error(await parseErrorDetail(res));
  }
  return res.json(); // { answer, sources, confidence }
}

export async function getStatus() {
  const res = await fetch(`${API_URL}/status`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}

export { API_URL };
