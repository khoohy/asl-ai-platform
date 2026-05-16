export default function HealthStatus({
  health,
  isLoading,
  error,
  onRefresh,
}) {
  const statusTone = error ? "danger" : health?.status === "ok" ? "good" : "idle";
  const statusLabel = error
    ? "Unavailable"
    : health?.status === "ok"
      ? "Healthy"
      : "Checking";

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Backend</p>
          <h2>Health status</h2>
        </div>
        <button
          type="button"
          className="secondary-button"
          onClick={onRefresh}
          disabled={isLoading}
        >
          {isLoading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      <div className={`status-pill status-pill--${statusTone}`}>{statusLabel}</div>

      <dl className="meta-grid">
        <div>
          <dt>Service</dt>
          <dd>{health?.service ?? "Waiting for backend"}</dd>
        </div>
        <div>
          <dt>Environment</dt>
          <dd>{health?.environment ?? "Unknown"}</dd>
        </div>
        <div>
          <dt>Version</dt>
          <dd>{health?.version ?? "Unknown"}</dd>
        </div>
      </dl>

      <p className={`inline-message ${error ? "inline-message--error" : ""}`}>
        {error ?? "This panel confirms that the FastAPI backend is reachable."}
      </p>
    </section>
  );
}
