export default function PredictionCard({ prediction }) {
  const topKPredictions = Array.isArray(prediction?.top_k) ? prediction.top_k : [];

  return (
    <section className="panel prediction-card">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Result</p>
          <h2>Prediction output</h2>
        </div>
      </div>

      <div className="prediction-value">
        {prediction?.prediction ? prediction.prediction : "No prediction yet"}
      </div>

      <dl className="meta-grid meta-grid--prediction">
        <div>
          <dt>Status</dt>
          <dd>{prediction?.status ?? "Waiting"}</dd>
        </div>
        <div>
          <dt>Confidence</dt>
          <dd>
            {typeof prediction?.confidence === "number"
              ? `${Math.round(prediction.confidence * 100)}%`
              : "Not available"}
          </dd>
        </div>
        <div>
          <dt>Frames</dt>
          <dd>
            {typeof prediction?.frames_collected === "number" &&
            typeof prediction?.sequence_length === "number"
              ? `${prediction.frames_collected}/${prediction.sequence_length}`
              : "Not available"}
          </dd>
        </div>
        <div>
          <dt>Model source</dt>
          <dd>{prediction?.model_source ?? "Not available"}</dd>
        </div>
      </dl>

      <p className="inline-message">
        {prediction?.note ??
          "Run the mock inference request to populate this card."}
      </p>

      <div className="topk-block">
        <p className="topk-title">Top K</p>
        {topKPredictions.length > 0 ? (
          <ul className="topk-list">
            {topKPredictions.map((item) => (
              <li key={`${item.label}-${item.confidence}`} className="topk-item">
                <span>{item.label}</span>
                <strong>{Math.round(item.confidence * 100)}%</strong>
              </li>
            ))}
          </ul>
        ) : (
          <p className="inline-message inline-message--compact">
            Top-K predictions appear after the 30-frame buffer is full.
          </p>
        )}
      </div>
    </section>
  );
}
