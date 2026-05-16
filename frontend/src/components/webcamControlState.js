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
    startRecognitionDisabled: !isCameraActive || isRecognitionActive,
    stopRecognitionDisabled: !isRecognitionActive,
    resetDisabled: isResetting || isRecognitionActive || isStoppingCamera,
  };
}
