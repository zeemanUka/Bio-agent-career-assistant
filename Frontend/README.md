# Bio Agent Frontend

Next.js frontend for the Bio Agent project.

## Stack

- Next.js with the App Router
- TypeScript
- ESLint

## Environment

Copy `.env.example` to `.env.local` and set the backend URL:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Local development

From `Frontend/`:

```bash
npm install
npm run dev
```

The frontend runs on `http://localhost:3000` by default.

## Current scope

The frontend now includes a streaming chat interface and sends requests directly to the backend `POST /api/chat/stream` endpoint. For the UI to function end to end, the backend API must be running at the URL defined in `NEXT_PUBLIC_API_BASE_URL`.
