export function getWebcamControlState({
  isStartingCamera = false,
  isStoppingCamera = false,
  isCameraActive = false,
  isRecognitionActive = false,
  isResetting = false,
} = {}) {
  return {
    startCameraDisabled: isStartingCamera || isCameraActive || isStoppingCamera,
    stopCameraDisabled: !isCameraActive || isStoppingCamera,
    startRecognitionDisabled:
      isStartingCamera || isStoppingCamera || !isCameraActive || isRecognitionActive,
    stopRecognitionDisabled: !isRecognitionActive,
    resetDisabled: isResetting || isRecognitionActive || isStoppingCamera,
  };
}

export function getRecognitionStartTransition({
  isCameraActive = false,
  isRecognitionActive = false,
  isRequestInFlight = false,
  isStoppingCamera = false,
} = {}) {
  if (!isCameraActive) {
    return {
      shouldStart: false,
      nextRecognitionActive: false,
      shouldKickImmediateCapture: false,
      reason: "camera_inactive",
    };
  }

  if (isStoppingCamera) {
    return {
      shouldStart: false,
      nextRecognitionActive: false,
      shouldKickImmediateCapture: false,
      reason: "camera_stopping",
    };
  }

  if (isRecognitionActive) {
    return {
      shouldStart: false,
      nextRecognitionActive: true,
      shouldKickImmediateCapture: false,
      reason: "already_running",
    };
  }

  return {
    shouldStart: true,
    nextRecognitionActive: true,
    shouldKickImmediateCapture: !isRequestInFlight,
    reason: isRequestInFlight ? "queued_for_next_frame" : "start_now",
  };
}
