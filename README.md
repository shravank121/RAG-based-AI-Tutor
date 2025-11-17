# RAG-Based AI Tutor (Sound) – Node.js

## Overview
An AI Tutor that ingests a chapter PDF and answers questions grounded in the content. Each answer includes one relevant local diagram from the `Sound/` folder. Text answers prefer Gemini; when the LLM is unavailable, it falls back to an extractive answer from retrieved chunks. The UI is with a sticky composer, loading indicators, and line‑by‑line rendering.

- Backend: Node + Express
- LLM: Gemini REST v1 preferred (via `GEMINI_API_KEY`); extractive fallback
- Retrieval: Local TF‑IDF‑like vectors + cosine similarity (no external embedding API)
- Images: Local `Sound/` diagrams with keyword vectors; one image per answer
- Frontend: Minimal HTML/JS with modern layout (no framework)

## Quick Start
1) Prerequisites
- Node 18+ (tested on Node 22)
- npm

2) Install
```bash
npm install
```

3) Environment
Create `.env` in the project root. Minimal example:
```
GEMINI_API_KEY=AI...your_key...
# Optional overrides
GEMINI_MODEL=gemini-2.5-flash
USE_OPENAI_FALLBACK=false
```
Notes:
- If you only have a Gemini key and not OpenAI, leave `USE_OPENAI_FALLBACK=false`.
- The server auto-discovers a working Gemini model and falls back to extractive answers on errors (503/404).

4) Run
```bash
npm run dev
# open http://localhost:3000
```

## How to Use
1) Click Upload and select the chapter PDF (e.g., the Sound chapter).
2) After processing, the UI shows “Extracted N chunks”.
3) Ask a question in the composer at the bottom.
4) The bot shows a thinking indicator and renders the answer line‑by‑line. A relevant local diagram is shown under the AI message.

## Endpoints
- POST `/upload` (multipart/form-data)
  - Body: `pdf` (file)
  - Action: extract text via `pdf-parse`, chunk, build vectors; persist under `data/<topicId>.json`
  - Response: `{ topicId, chunkCount }`
- POST `/chat` (application/json)
  - Body: `{ topicId, question }`
  - Action: retrieve top‑K chunks; prefer Gemini for generation; on failure, use extractive answer; always attach one relevant local image
  - Response: `{ answer, image, providerUsed }`
- GET `/images/:topicId`
  - Returns metadata for local images. The current logic is topic‑agnostic and picks images by matching the user question.
- GET `/health`
  - Returns status and provider availability flags (Gemini reachable, OpenAI fallback enabled, etc.).

## Architecture
- PDF ingestion: `pdf-parse` extracts text; sentence‑level chunking; chunks stored with simple TF‑IDF‑like vectors in JSON.
- Retrieval: cosine similarity between query vector and chunk vectors; K=5 by default.
- Generation:
  - Gemini (REST v1) is preferred; models are discovered at runtime and cached.
  - On 503/404 or missing key, the server produces a grounded extractive answer from top chunks.
  - Provider used is returned in the API response but hidden in the UI per requirements.
- Images: local folder `Sound/` is scanned once; each file gets metadata and a small keyword vector. On each `/chat`, one best‑match image is returned based on the user’s question.
- Logging: each chat turn is appended to `logs/chat.log` with question, answer, provider, and image metadata.

## RAG Pipeline Details
1) Chunking
- Splits extracted text into sentence‑level chunks; filters very short/noisy lines; normalizes whitespace.

2) Vectorization
- Tokenizes and stems terms; builds TF and IDF; stores per‑chunk vectors.

3) Retrieval
- Converts query to the same vector space and ranks by cosine similarity; selects top‑K (default 5).

4) Answering
- Prompt (Gemini): concise, student‑friendly, use only provided contexts; say “I don’t know” if not present.
- Extractive fallback: concatenates/re‑phrases the most relevant sentences.

## Image Retrieval Logic
- On startup, scan `Sound/` and create metadata: `{ id, filename, title, keywords }`.
- Build tiny vectors from `title + keywords`.
- At answer time, pick a single image using similarity with the user’s question.
- The UI displays the image inline beneath the AI bubble.

## Frontend Behavior
- Chat‑style layout: centered thread with avatars (Y/AI), sticky composer.
- Loading states: spinner on upload; rotating “AI is thinking…” messages during chat.
- Streaming effect: client renders the server’s answer line‑by‑line.
- Welcome card: “What do you want to learn today?” is shown until the first message.
- Provider label is hidden per requirement; Topic ID is tracked internally after upload.

## Troubleshooting
- Gemini 503: “The model is overloaded. Please try again later.”
  - The app automatically falls back to extractive answers and keeps responding.
- Gemini 404 (v1beta vs v1):
  - The app uses REST v1 and lists available models; ensure `GEMINI_API_KEY` is valid and not restricted.
- No image shown:
  - Ensure your `Sound/` folder contains the expected diagrams (PNG). Filenames used include: ReflectionOfSound.png, VocalCordsDiagram.png, CompressionAndRefraction.png, SchoolBellVibration.png, VibrationOfRubberBand.png, MusicalInstrumentsVibrationChart.png.
- Windows CRLF warnings in Git:
  - Harmless. You can set `git config core.autocrlf true` if desired.

## Project Structure
```

├─ public/
│  └─ index.html          # UI (upload + chat)
├─ Sound/                 # Local diagrams
├─ data/                  # Created on upload (per‑topic chunk store)
├─ logs/
│  └─ chat.log            # Chat turn logs
├─ server.js              # Express server (endpoints + RAG + Gemini)
├─ package.json
└─ .env                   # GEMINI_API_KEY, optional overrides
```

## Deploy / Run
- Local: `npm run dev` (nodemon)
- Production (example):
  - Set `PORT` and env keys; run `node server.js` behind a reverse proxy.

## Contributing / Pushing to GitHub
1) Initialize and commit
```bash
git init
git add .
git commit -m "Initial commit"
```
2) Create repo and push (without GitHub CLI):
```bash
git remote add origin https://github.com/<your-username>/RAG-based-AI-Tutor.git
git branch -M main
git push -u origin main
```
3) Or with GitHub CLI (after `gh auth login`):
```bash
gh repo create <your-username>/RAG-based-AI-Tutor --public --source . --remote origin --push
```

## Roadmap (Optional)
- Swap TF‑IDF for neural embeddings (`@xenova/transformers` MiniLM) and store vectors in JSON.
- Add true streaming via Server‑Sent Events (SSE) on `/chat`.
- Persist `data/images.json` metadata explicitly.
