import { useEffect, useState } from "react";
import { getStatus } from "../api.js";

export default function StatusBadge() {
  const [state, setState] = useState("checking"); // checking | ready | warming | offline

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const s = await getStatus();
        if (cancelled) return;
        const ready =
          s.dense_model_loaded && s.reranker_model_loaded && s.chromadb_connected;
        setState(ready ? "ready" : "warming");
      } catch {
        if (!cancelled) setState("offline");
      }
    }

    poll();
    const id = setInterval(poll, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const copy = {
    checking: "Connecting…",
    ready: "Models ready",
    warming: "Warming up GPU…",
    offline: "Backend unreachable",
  }[state];

  return (
    <div className={`status-badge status-badge--${state}`}>
      <span className="status-badge__dot" />
      {copy}
    </div>
  );
}
