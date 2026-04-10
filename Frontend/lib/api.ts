export type ApiChatRole = "user" | "assistant";

export type ApiChatMessage = {
  role: ApiChatRole;
  content: string;
};

type HealthResponse = {
  status: string;
  cors_origins: string[];
};

type ChatResponse = {
  message: string;
};

type ErrorResponse = {
  detail?: string;
};

type StreamChunk = {
  delta?: string;
};

type StreamDone = Record<string, never>;

function buildUrl(apiBaseUrl: string, path: string) {
  return `${apiBaseUrl}${path}`;
}

async function readErrorMessage(response: Response) {
  try {
    const payload = (await response.json()) as ErrorResponse;
    if (typeof payload.detail === "string" && payload.detail.trim()) {
      return payload.detail;
    }
  } catch {}

  return `Backend request failed with status ${response.status}.`;
}

export async function getBackendHealth(apiBaseUrl: string) {
  const response = await fetch(buildUrl(apiBaseUrl, "/api/health"), {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return (await response.json()) as HealthResponse;
}

export async function sendChatMessage(
  apiBaseUrl: string,
  message: string,
  history: ApiChatMessage[],
) {
  const response = await fetch(buildUrl(apiBaseUrl, "/api/chat"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      history,
    }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  const payload = (await response.json()) as ChatResponse;
  if (!payload.message.trim()) {
    throw new Error("Backend returned an empty response.");
  }

  return payload.message;
}

function parseSseFrame(frame: string) {
  const lines = frame
    .split("\n")
    .map((line) => line.trimEnd())
    .filter(Boolean);

  let event = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
      continue;
    }

    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  return {
    event,
    data: dataLines.join("\n"),
  };
}

export async function streamChatMessage(
  apiBaseUrl: string,
  message: string,
  history: ApiChatMessage[],
  onDelta: (delta: string) => void,
) {
  const response = await fetch(buildUrl(apiBaseUrl, "/api/chat/stream"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      history,
    }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  if (!response.body) {
    throw new Error("Backend stream did not return a readable body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalMessage = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });

    let separatorIndex = buffer.indexOf("\n\n");
    while (separatorIndex !== -1) {
      const frame = buffer.slice(0, separatorIndex).replace(/\r/g, "");
      buffer = buffer.slice(separatorIndex + 2);

      if (frame.trim()) {
        const parsed = parseSseFrame(frame);

        if (parsed.event === "message" && parsed.data) {
          const payload = JSON.parse(parsed.data) as StreamChunk;
          if (typeof payload.delta === "string" && payload.delta) {
            finalMessage += payload.delta;
            onDelta(payload.delta);
          }
        }

        if (parsed.event === "done") {
          JSON.parse(parsed.data || "{}") as StreamDone;
          return finalMessage;
        }
      }

      separatorIndex = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }

  if (!finalMessage.trim()) {
    throw new Error("Backend stream ended without a response.");
  }

  return finalMessage;
}
