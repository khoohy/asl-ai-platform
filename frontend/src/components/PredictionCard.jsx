const STABILIZATION_LABELS = {
  mock: "Mock response",
  raw_only: "Raw only",
  raw_predicted: "Raw predicted",
  collecting_votes: "Collecting votes",
  holding_context: "Holding context",
  waiting_for_hands: "Waiting for hands",
  idle: "Idle",
  stable: "Stable",
  stabilized: "Stable",
  peak_accepted: "Peak accepted",
  held_confusion: "Held for confusion",
  low_confidence: "Low confidence",
  motion_required: "Motion required",
  no_landmarks: "No landmarks",
  warming_up: "Warming up",
  reset: "Session reset",
};

export default function PredictionCard({ prediction, runtimeStats }) {
  const isIdleLike =
    prediction?.status === "waiting_for_hands" ||
    prediction?.status === "idle";
  const isHoldingContext = prediction?.status === "holding_context";
  const topKPredictions =
    isIdleLike || isHoldingContext
      ? []
      : Array.isArray(prediction?.top_k)
        ? prediction.top_k
        : [];
  const stablePrediction = prediction?.stable_prediction;
  const fallbackPrediction = prediction?.prediction;
  const stableOrFallback = isIdleLike
    ? "Waiting for hands"
    : isHoldingContext
      ? "Holding context"
      : stablePrediction || fallbackPrediction || "Waiting for frames";
  const displayConfidence =
    isIdleLike || isHoldingContext
      ? null
      : stablePrediction
        ? prediction?.stable_confidence
        : prediction?.confidence;
  const stabilizationLabel =
    STABILIZATION_LABELS[prediction?.stabilization_status] ??
    STABILIZATION_LABELS[prediction?.status] ??
    "Waiting";
  const framesLabel =
    typeof prediction?.frames_collected === "number" &&
    typeof prediction?.sequence_length === "number"
      ? `${prediction.frames_collected}/${prediction.sequence_length}`
      : "0/30";
  const statusClass = getStatusClass(prediction);

  return (
    <section className="panel prediction-card prediction-card--sticky">
      <div className="panel-header prediction-card__header">
        <div>
          <p className="eyebrow">Live prediction</p>
          <h2>Inference output</h2>
        </div>
        <div className={`status-pill status-pill--prediction status-pill--${getStatusTone(prediction)} ${statusClass}`}>
          {stabilizationLabel}
        </div>
      </div>

      <div className="prediction-hero">
        <div className="prediction-hero__primary">
          <span className="prediction-kicker">Stable prediction</span>
          <div className="prediction-value">{stableOrFallback}</div>
        </div>

        <div className="prediction-badges">
          <div className="prediction-badge">
            <span>Frames</span>
            <strong className="metric-code">{framesLabel}</strong>
          </div>
          <div className="prediction-badge">
            <span>Confidence</span>
            <strong className="metric-code">
              {typeof displayConfidence === "number"
                ? `${Math.round(displayConfidence * 100)}%`
                : "N/A"}
            </strong>
          </div>
        </div>
      </div>

      <div className="prediction-subline">
        <span className="prediction-subline__label">Raw</span>
        <strong>
          {isIdleLike || isHoldingContext
            ? "Not accepted"
            : prediction?.raw_prediction ?? "Not available"}
        </strong>
        <span className="prediction-subline__meta metric-code">
          {isIdleLike || isHoldingContext
            ? "N/A"
            : prediction?.raw_prediction && typeof prediction?.raw_confidence === "number"
            ? `${Math.round(prediction.raw_confidence * 100)}%`
            : "N/A"}
        </span>
      </div>

      <dl className="meta-grid meta-grid--prediction meta-grid--compact">
        <div>
          <dt>Status</dt>
          <dd className="metric-code">{prediction?.status ?? "Waiting"}</dd>
        </div>
        <div>
          <dt>Stabilization</dt>
          <dd className="metric-code">{prediction?.stabilization_status ?? "Not available"}</dd>
        </div>
        <div>
          <dt>Votes</dt>
          <dd className="metric-code">
            {typeof prediction?.vote_count === "number" &&
            typeof prediction?.vote_window_size === "number"
              ? `${prediction.vote_count}/${prediction.vote_window_size}`
              : "Not available"}
          </dd>
        </div>
        <div>
          <dt>Model source</dt>
          <dd className="metric-code">{prediction?.model_source ?? "Not available"}</dd>
        </div>
        <div>
          <dt>Stable confidence</dt>
          <dd className="metric-code">
            {!isIdleLike &&
            !isHoldingContext &&
            prediction?.stable_prediction &&
            typeof prediction?.stable_confidence === "number"
              ? `${Math.round(prediction.stable_confidence * 100)}%`
              : "Not available"}
          </dd>
        </div>
      </dl>

      <p className="inline-message prediction-note">
        {prediction?.note ??
          "Run the mock inference request to populate this card."}
      </p>

      <div className="topk-block">
        <p className="topk-title">Top 5 raw model candidates</p>
        {topKPredictions.length > 0 ? (
          <ul className="topk-list">
            {topKPredictions.map((item) => (
              <li key={`${item.label}-${item.confidence}`} className="topk-item">
                <div className="topk-item__meta">
                  <span>{item.label}</span>
                  <strong className="metric-code">{Math.round(item.confidence * 100)}%</strong>
                </div>
                <div className="topk-bar">
                  <span
                    className="topk-bar__fill"
                    style={{ width: `${Math.max(item.confidence * 100, 4)}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="inline-message inline-message--compact">
            Top-K predictions appear after the 30-frame buffer is full.
          </p>
        )}
      </div>

      <details className="runtime-details">
        <summary>Runtime stats</summary>
        <dl className="meta-grid meta-grid--single meta-grid--secondary">
          <div>
            <dt>Request loop</dt>
            <dd className="metric-code">
              {`interval ${runtimeStats?.captureIntervalMs ?? 200}ms, in flight ${runtimeStats?.isRequestInFlight ? "yes" : "no"}`}
            </dd>
          </div>
          <div>
            <dt>Camera</dt>
            <dd className="metric-code">
              {runtimeStats?.isCameraActive ? "active" : "idle"} |{" "}
              {runtimeStats?.isInferenceSessionRunning ? "session running" : "session stopped"}
            </dd>
          </div>
          <div>
            <dt>Frame traffic</dt>
            <dd className="metric-code">
              {`sent ${runtimeStats?.framesSent ?? 0}, ok ${runtimeStats?.successfulResponses ?? 0}, failed ${runtimeStats?.failedResponses ?? 0}`}
            </dd>
          </div>
        </dl>
      </details>
    </section>
  );
}

function getStatusTone(prediction) {
  const status = prediction?.status;
  const stabilizationStatus = prediction?.stabilization_status;

  if (status === "stabilized" || stabilizationStatus === "stable" || stabilizationStatus === "peak_accepted") {
    return "good";
  }

  if (
    status === "low_confidence" ||
    status === "held_confusion" ||
    status === "motion_required" ||
    status === "no_landmarks"
  ) {
    return "danger";
  }

  if (status === "holding_context" || status === "waiting_for_hands" || status === "idle") {
    return "idle";
  }

  return "idle";
}

function getStatusClass(prediction) {
  const status = prediction?.status;
  const stabilizationStatus = prediction?.stabilization_status;

  if (status === "warming_up") {
    return "status-pill--warming";
  }
  if (status === "holding_context") {
    return "status-pill--collecting";
  }
  if (status === "waiting_for_hands" || status === "idle") {
    return "status-pill--neutral";
  }
  if (status === "no_landmarks") {
    return "status-pill--warning";
  }
  if (stabilizationStatus === "collecting_votes") {
    return "status-pill--collecting";
  }
  if (status === "stabilized" || stabilizationStatus === "stable" || stabilizationStatus === "peak_accepted") {
    return "status-pill--stabilized";
  }
  if (
    status === "low_confidence" ||
    status === "held_confusion" ||
    status === "motion_required"
  ) {
    return "status-pill--error";
  }
  return "status-pill--neutral";
}
