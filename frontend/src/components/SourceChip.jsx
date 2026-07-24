export default function SourceChip({ source }) {
  const { clause, page, document, item_number, reranker_score } = source;

  const label = clause || item_number || "Unlabelled clause";

  return (
    <div className="source-chip">
      <span className="source-chip__marker" aria-hidden="true" />
      <div className="source-chip__body">
        <span className="source-chip__clause">{label}</span>
        <span className="source-chip__meta">
          {document ? document : "Scope of Work"}
          {page != null ? ` · p.${page}` : ""}
        </span>
      </div>
      {reranker_score != null && (
        <span className="source-chip__score" title="Reranker relevance score">
          {(reranker_score * 100).toFixed(0)}%
        </span>
      )}
    </div>
  );
}
