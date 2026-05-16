export default function PredictionCard({ prediction }) {
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
          <dt>Confidence</dt>
          <dd>
            {typeof prediction?.confidence === "number"
              ? `${Math.round(prediction.confidence * 100)}%`
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
    </section>
  );
}
