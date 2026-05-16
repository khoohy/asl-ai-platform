const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function parseJsonResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail =
      typeof data.detail === "string"
        ? data.detail
        : "The backend request failed.";
    throw new Error(detail);
  }

  return data;
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  return parseJsonResponse(response);
}

export async function runMockInference(imageBase64 = "abcdefghijklmnop") {
  const response = await fetch(`${API_BASE_URL}/api/inference/mock`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      image_base64: imageBase64,
    }),
  });

  return parseJsonResponse(response);
}

export async function runRealInference(imageBase64, sessionId = "default") {
  const response = await fetch(`${API_BASE_URL}/api/inference/frame`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      image_base64: imageBase64,
      session_id: sessionId,
    }),
  });

  return parseJsonResponse(response);
}

export async function resetRealInferenceSession(sessionId = "default") {
  const response = await fetch(`${API_BASE_URL}/api/inference/reset-session`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
    }),
  });

  return parseJsonResponse(response);
}

export { API_BASE_URL };
