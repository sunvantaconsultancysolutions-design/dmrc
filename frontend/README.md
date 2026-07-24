# DMRC Contract Intelligence — Frontend

React (Vite) chat UI for the FastAPI backend in `../src/app.py`. Talks to
exactly two endpoints: `POST /ask` and `GET /status` — see `src/api.js`.

## Local development

```bash
cd frontend
npm install
cp .env.example .env       # point VITE_API_URL at your backend
npm run dev                # http://localhost:5173
```

If you're running the backend locally too (`uvicorn src.app:app` from the
repo root), the default `.env.example` value already points at it.

## Deploy (Vercel)

```bash
npm i -g vercel
cd frontend
vercel deploy --prod
```

When prompted (or in the Vercel dashboard → Project → Settings →
Environment Variables), set:

```
VITE_API_URL = https://<your-runpod-pod-id>-8000.proxy.runpod.net
```

(or whatever public URL your GPU host exposes port 8000 on).

Then, on the **backend**, set `ALLOWED_ORIGINS` to the Vercel URL Vercel
gives you (e.g. `https://dmrc-contract-intelligence.vercel.app`) so CORS
allows the browser to call it — see the `CORSMiddleware` block in
`src/app.py`.

## Notes

- No streaming: `gemma_inference.generate_answer()` is a single blocking
  call, so the UI shows a typing indicator until the full answer returns,
  rather than a token-by-token stream. Wiring real streaming later means
  switching the backend to `model.generate(..., streamer=...)` behind a
  `StreamingResponse`, and swapping `askQuestion()` in `api.js` for an
  `EventSource`/`fetch` reader — out of scope for this pass.
- The "Sourced from" chips render whatever `sources[]` the backend
  returns (`clause`, `page`, `document`, `reranker_score`) — if you add
  fields to `SourceItem` in `app.py`, extend `SourceChip.jsx` to match.
