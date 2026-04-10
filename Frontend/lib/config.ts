const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  process.env.NEXT_PUBLIC_BACKEND_URL ??
  "http://localhost:8000";

export const API_BASE_URL = apiBaseUrl.replace(/\/+$/, "");
