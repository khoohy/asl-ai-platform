import { useEffect, useRef, useState } from "react";

const CAPTURE_INTERVAL_MS = 200;

export default function WebcamPanel({
  isLoading,
  isResetting,
  error,
  inferenceResult,
  onCaptureAndRunRealInference,
  onResetRealInferenceSession,
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const requestInFlightRef = useRef(false);
  const runRealInferenceRef = useRef(onCaptureAndRunRealInference);

  const [cameraStatus, setCameraStatus] = useState(
    "Camera is idle. Start the camera to enable capture.",
  );
  const [captureStatus, setCaptureStatus] = useState(
    "No frame captured yet.",
  );
  const [isStartingCamera, setIsStartingCamera] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isInferenceSessionRunning, setIsInferenceSessionRunning] = useState(false);
  const [isRequestInFlight, setIsRequestInFlight] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [framesSent, setFramesSent] = useState(0);
  const [successfulResponses, setSuccessfulResponses] = useState(0);
  const [failedResponses, setFailedResponses] = useState(0);

  useEffect(() => {
    runRealInferenceRef.current = onCaptureAndRunRealInference;
  }, [onCaptureAndRunRealInference]);

  function stopInferenceSession(options = {}) {
    const { updateStatus = true } = options;

    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    requestInFlightRef.current = false;
    setIsRequestInFlight(false);
    setIsInferenceSessionRunning(false);

    if (updateStatus) {
      setCaptureStatus("Real inference session stopped.");
    }
  }

  function releaseCamera(shouldUpdateState = true) {
    stopInferenceSession({ updateStatus: false });

    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    if (shouldUpdateState) {
      setIsCameraActive(false);
      setCameraError("");
      setCameraStatus("Camera stopped. You can start it again at any time.");
    }
  }

  function stopCamera() {
    releaseCamera(true);
  }

  async function startCamera() {
    setIsStartingCamera(true);
    setCameraError("");
    setCameraStatus("Requesting webcam access...");

    try {
      releaseCamera(false);

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not support webcam access.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsCameraActive(true);
      setCameraStatus("Camera ready.");
      setCaptureStatus("Camera ready for frame capture.");
    } catch (error) {
      const message = getCameraErrorMessage(error);
      setCameraError(message);
      setCameraStatus(message);
      setIsCameraActive(false);
    } finally {
      setIsStartingCamera(false);
    }
  }

  async function captureAndSendFrame({ source }) {
    if (!videoRef.current || !canvasRef.current || !isCameraActive) {
      const message = "Camera is not ready yet.";
      setCaptureStatus(message);
      throw new Error(message);
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video.videoWidth || !video.videoHeight) {
      const message = "Capture failed because the video frame is not ready.";
      setCaptureStatus(message);
      throw new Error(message);
    }

    try {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("Canvas rendering is unavailable in this browser.");
      }

      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageBase64 = canvas.toDataURL("image/jpeg", 0.92);
      setFramesSent((count) => count + 1);

      const response = await runRealInferenceRef.current(imageBase64);
      setSuccessfulResponses((count) => count + 1);

      if (source === "manual") {
        setCaptureStatus("Manual real inference frame submitted.");
      } else {
        setCaptureStatus(
          `Continuous real inference running at ${CAPTURE_INTERVAL_MS}ms intervals.`,
        );
      }

      return response;
    } catch (error) {
      setFailedResponses((count) => count + 1);
      const message =
        error instanceof Error ? error.message : "Capture failed.";
      setCaptureStatus(message);
      throw error;
    }
  }

  async function handleManualRealInference() {
    requestInFlightRef.current = true;
    setIsRequestInFlight(true);

    try {
      await captureAndSendFrame({ source: "manual" });
    } catch (error) {
      return;
    } finally {
      requestInFlightRef.current = false;
      setIsRequestInFlight(false);
    }
  }

  function handleStartInferenceSession() {
    if (!isCameraActive) {
      setCaptureStatus("Start the camera before starting a real inference session.");
      return;
    }

    if (intervalRef.current !== null) {
      setCaptureStatus("Real inference session is already running.");
      return;
    }

    setCaptureStatus(
      `Continuous real inference started at ${CAPTURE_INTERVAL_MS}ms intervals.`,
    );
    setIsInferenceSessionRunning(true);

    intervalRef.current = window.setInterval(async () => {
      if (requestInFlightRef.current) {
        return;
      }

      requestInFlightRef.current = true;
      setIsRequestInFlight(true);

      try {
        await captureAndSendFrame({ source: "loop" });
      } catch (error) {
        return;
      } finally {
        requestInFlightRef.current = false;
        setIsRequestInFlight(false);
      }
    }, CAPTURE_INTERVAL_MS);
  }

  async function handleResetSession() {
    const response = await onResetRealInferenceSession();
    if (response?.status === "reset") {
      setFramesSent(0);
      setSuccessfulResponses(0);
      setFailedResponses(0);
      setCaptureStatus("Real inference session reset.");
    }
  }

  useEffect(() => {
    return () => {
      releaseCamera(false);
    };
  }, []);

  return (
    <section className="panel panel--webcam">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Webcam real flow</p>
          <h2>Camera capture</h2>
        </div>
      </div>

      <p className="panel-copy">
        Open the browser camera, capture a single frame, convert it to base64,
        and send it to the 30-frame model buffer endpoint with runtime
        stabilization layered on top of the raw model output.
      </p>

      <div className={`webcam-stage ${isCameraActive ? "webcam-stage--live" : ""}`}>
        <video
          ref={videoRef}
          className="webcam-video"
          autoPlay
          muted
          playsInline
        />
        {!isCameraActive ? (
          <div className="webcam-placeholder">
            <strong>Camera preview</strong>
            <span>Start the camera to see the live feed here.</span>
          </div>
        ) : null}
      </div>

      <div className="button-row">
        <button
          type="button"
          className="secondary-button"
          onClick={startCamera}
          disabled={isStartingCamera || isCameraActive}
        >
          {isStartingCamera ? "Starting camera..." : "Start Camera"}
        </button>

        <button
          type="button"
          className="secondary-button"
          onClick={stopCamera}
          disabled={!isCameraActive}
        >
          Stop Camera
        </button>
      </div>

      <div className="button-row">
        <button
          type="button"
          className="primary-button"
          onClick={handleManualRealInference}
          disabled={!isCameraActive || isLoading || isRequestInFlight}
        >
          {isLoading || isRequestInFlight
            ? "Capturing manual real frame..."
            : "Capture frame and run real inference"}
        </button>

        <button
          type="button"
          className="primary-button primary-button--accent"
          onClick={handleStartInferenceSession}
          disabled={!isCameraActive || isInferenceSessionRunning}
        >
          {isInferenceSessionRunning
            ? "Real inference session running"
            : "Start Real Inference Session"}
        </button>

        <button
          type="button"
          className="secondary-button"
          onClick={() => stopInferenceSession({ updateStatus: true })}
          disabled={!isInferenceSessionRunning}
        >
          Stop Real Inference Session
        </button>

        <button
          type="button"
          className="secondary-button"
          onClick={handleResetSession}
          disabled={isResetting}
        >
          {isResetting ? "Resetting session..." : "Reset Real Inference Session"}
        </button>
      </div>

      <dl className="meta-grid meta-grid--single">
        <div>
          <dt>Camera status</dt>
          <dd>{cameraStatus}</dd>
        </div>
        <div>
          <dt>Capture status</dt>
          <dd>{captureStatus}</dd>
        </div>
        <div>
          <dt>Live session state</dt>
          <dd>
            {inferenceResult?.status ?? "idle"} |{" "}
            {typeof inferenceResult?.frames_collected === "number" &&
            typeof inferenceResult?.sequence_length === "number"
              ? `${inferenceResult.frames_collected}/${inferenceResult.sequence_length}`
              : "0/30"}
          </dd>
        </div>
        <div>
          <dt>Stabilization state</dt>
          <dd>
            {inferenceResult?.stabilization_status ?? "raw_only"} |{" "}
            {typeof inferenceResult?.vote_count === "number" &&
            typeof inferenceResult?.vote_window_size === "number"
              ? `${inferenceResult.vote_count}/${inferenceResult.vote_window_size}`
              : "0/10"}
          </dd>
        </div>
        <div>
          <dt>Runtime stats</dt>
          <dd>
            {`frames sent ${framesSent}, responses ok ${successfulResponses}, failed ${failedResponses}, interval ${CAPTURE_INTERVAL_MS}ms, in flight ${isRequestInFlight ? "yes" : "no"}`}
          </dd>
        </div>
      </dl>

      <p
        className={`inline-message ${
          cameraError || error ? "inline-message--error" : ""
        }`}
      >
        {cameraError ||
          error ||
          (inferenceResult?.status && inferenceResult.status !== "mock"
            ? inferenceResult.note
            : "") ||
          "Allow browser camera permission and capture multiple frames to fill the 30-frame buffer."}
      </p>

      <canvas ref={canvasRef} className="hidden-canvas" aria-hidden="true" />
    </section>
  );
}

function getCameraErrorMessage(error) {
  if (!error || typeof error !== "object") {
    return "Unable to access the webcam.";
  }

  if ("name" in error) {
    switch (error.name) {
      case "NotAllowedError":
      case "PermissionDeniedError":
        return "Camera permission denied. Allow access in the browser and try again.";
      case "NotFoundError":
      case "DevicesNotFoundError":
        return "No camera found on this device.";
      case "NotReadableError":
      case "TrackStartError":
        return "The camera is already in use by another application.";
      case "OverconstrainedError":
        return "Requested camera settings are not supported on this device.";
      default:
        break;
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return "Unable to access the webcam.";
}
