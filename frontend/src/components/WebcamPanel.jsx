import { useEffect, useRef, useState } from "react";

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

  const [cameraStatus, setCameraStatus] = useState(
    "Camera is idle. Start the camera to enable capture.",
  );
  const [captureStatus, setCaptureStatus] = useState(
    "No frame captured yet.",
  );
  const [isStartingCamera, setIsStartingCamera] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");

  function releaseCamera(shouldUpdateState = true) {
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

  async function handleCapture() {
    if (!videoRef.current || !canvasRef.current || !isCameraActive) {
      const message = "Camera is not ready yet.";
      setCaptureStatus(message);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video.videoWidth || !video.videoHeight) {
      const message = "Capture failed because the video frame is not ready.";
      setCaptureStatus(message);
      return;
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

      setCaptureStatus("Frame captured and submitted for real inference.");
      await onCaptureAndRunRealInference(imageBase64);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Capture failed.";
      setCaptureStatus(message);
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
        and send it to the raw 30-frame model buffer endpoint.
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

      <button
        type="button"
        className="primary-button"
        onClick={handleCapture}
        disabled={!isCameraActive || isLoading}
      >
        {isLoading
          ? "Capturing and running real inference..."
          : "Capture frame and run real inference"}
      </button>

      <button
        type="button"
        className="secondary-button secondary-button--inline"
        onClick={onResetRealInferenceSession}
        disabled={isResetting}
      >
        {isResetting ? "Resetting session..." : "Reset real inference session"}
      </button>

      <dl className="meta-grid meta-grid--single">
        <div>
          <dt>Camera status</dt>
          <dd>{cameraStatus}</dd>
        </div>
        <div>
          <dt>Capture status</dt>
          <dd>{captureStatus}</dd>
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
