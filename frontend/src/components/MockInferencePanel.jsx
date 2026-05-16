export default function MockInferencePanel({
  isLoading,
  error,
  onRunInference,
}) {
  return (
    <section className="panel panel--accent">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Mock endpoint</p>
          <h2>Inference test</h2>
        </div>
      </div>

      <p className="panel-copy">
        Send a placeholder base64 string to the backend mock inference route and
        display the returned prediction payload. This remains available as a
        separate non-webcam test path.
      </p>

      <div className="request-preview">
        <span>POST</span>
        <code>/api/inference/mock</code>
      </div>

      <button
        type="button"
        className="primary-button"
        onClick={onRunInference}
        disabled={isLoading}
      >
        {isLoading ? "Running mock inference..." : "Run mock inference"}
      </button>

      <p className={`inline-message ${error ? "inline-message--error" : ""}`}>
        {error ??
          'This button uses the placeholder payload "abcdefghijklmnop".'}
      </p>
    </section>
  );
}
