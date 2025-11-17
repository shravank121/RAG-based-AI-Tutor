# RAG-Based AI Tutor (Sound) - Node.js

## Overview
A minimal RAG chatbot that ingests a chapter PDF and answers questions grounded in the content, returning a relevant image from the `Sound` folder.

- Backend: Node + Express
- Retrieval: Local TF-like vectorization and cosine similarity
- LLM: Optional OpenAI (via `OPENAI_API_KEY`) with fallback to extractive answer
- Images: Scanned from `Sound/` with basic keyword vectors
- Frontend: Minimal HTML/JS

## Endpoints
- POST `/upload` (multipart/form-data: `pdf`)
  - Extracts text via `pdf-parse`, chunks it, builds vectors, stores under `data/<topicId>.json`
  - Returns `{ topicId, chunkCount }`
- POST `/chat` (JSON: `{ topicId, question }`)
  - Retrieves top chunks, generates answer (OpenAI if available), picks one relevant image
  - Returns `{ answer, image }`
- GET `/images/:topicId`
  - Returns image metadata for `Sound/` images (topic-agnostic)

## Run Locally
1. Ensure Node 18+
2. Install deps:
   ```bash
   npm install
   ```
3. Optional: set OpenAI key:
   ```bash
   echo OPENAI_API_KEY=sk-... > .env
   ```
4. Start server:
   ```bash
   npm run dev
   # open http://localhost:3000
   ```

## RAG Pipeline
- PDF → Text extraction (`pdf-parse`)
- Text → Sentence-based chunking (≈180 tokens)
- Embedding: simple term-frequency vector (stemmed tokens)
- Store: `data/<topicId>.json` with `chunks` and `vectors`
- Retrieval: cosine similarity against query vector
- Answer generation:
  - If `OPENAI_API_KEY` exists: prompt with top contexts
  - Else: extractive summary from top sentences

## Image Retrieval Logic
- On startup, scan `Sound/` for images and create metadata: id, filename, title, keywords
- Vectorize keywords+description with same tokenizer
- On each answer, compute similarity between answer text and image vectors; return best match

## Prompts Used
System: `You are a helpful AI tutor. Stick to provided sources.`
User: `Using only the sources below, answer the question clearly and concisely for a student. If the answer is not in the sources, say you don't know.` plus retrieved sources and question.

## Notes
- This example uses very light-weight vectors for simplicity. You can replace vectors with real embeddings (e.g., OpenAI text-embedding-3-small or local models) and store in FAISS.
- No additional styling beyond basic layout.
- Demo video: record a short screen capture showing upload, ask 2–3 questions, and image display.
