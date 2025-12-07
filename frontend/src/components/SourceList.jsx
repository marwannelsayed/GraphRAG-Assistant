function SourceList({ sources }) {
  // TODO: Implement source list display
  // - Fetch sources from API
  // - Display ingested documents
  // - Show metadata (name, date, status)
  // - Allow deletion of sources

  return (
    <div className="source-list-container">
      <h2>Ingested Sources</h2>
      {sources.length === 0 ? (
        <p>No sources ingested yet. Upload documents to get started.</p>
      ) : (
        <ul>
          {sources.map((source, idx) => (
            <li key={idx}>
              <strong>{source.name}</strong>
              <span>{source.date}</span>
              <button>Delete</button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default SourceList;

