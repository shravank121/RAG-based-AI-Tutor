import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import pdfParse from 'pdf-parse';
import dotenv from 'dotenv';
import natural from 'natural';
const { TfIdf, PorterStemmer } = natural;
import { fileURLToPath } from 'url';
import OpenAI from 'openai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

const upload = multer({ dest: path.join(__dirname, 'uploads') });
const DATA_DIR = path.join(__dirname, 'data');
const SOUND_DIR = path.join(__dirname, 'Sound');
const PUBLIC_DIR = path.join(__dirname, 'public');
const LOG_DIR = path.join(__dirname, 'logs');
const CHAT_LOG = path.join(LOG_DIR, 'chat.log');

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR);
if (!fs.existsSync(PUBLIC_DIR)) fs.mkdirSync(PUBLIC_DIR);
if (!fs.existsSync(path.join(__dirname, 'uploads'))) fs.mkdirSync(path.join(__dirname, 'uploads'));
if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR);

// Serve static assets
app.use('/static', express.static(PUBLIC_DIR));
app.use('/Sound', express.static(SOUND_DIR));

// Simple tokenizer
function tokenize(text) {
  if (!text) return [];
  const cleaned = text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ');
  const tokens = cleaned.split(/\s+/).filter(Boolean);
  return tokens.map(t => PorterStemmer.stem(t));
}

function chunkText(text, maxTokens = 180) {
  const sentences = text.split(/(?<=[.!?])\s+/);
  const chunks = [];
  let cur = [];
  let count = 0;
  for (const s of sentences) {
    const toks = tokenize(s);
    if (count + toks.length > maxTokens && cur.length) {
      chunks.push(cur.join(' '));
      cur = [];
      count = 0;
    }
    cur.push(s.trim());
    count += toks.length;
  }
  if (cur.length) chunks.push(cur.join(' '));
  return chunks;
}

function cosineSim(a, b) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  let dot = 0, na = 0, nb = 0;
  for (const k of keys) {
    const va = a[k] || 0;
    const vb = b[k] || 0;
    dot += va * vb;
    na += va * va;
    nb += vb * vb;
  }
  return dot === 0 ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function termCounts(text) {
  const terms = tokenize(text);
  const counts = {};
  for (const t of terms) counts[t] = (counts[t] || 0) + 1;
  return { counts, total: terms.length || 1 };
}

function buildTfIdfIndex(chunks) {
  const N = chunks.length || 1;
  const perDoc = chunks.map(ch => termCounts(ch));
  const df = {};
  for (const { counts } of perDoc) {
    for (const t of Object.keys(counts)) {
      df[t] = (df[t] || 0) + 1;
    }
  }
  const idf = (t) => Math.log(1 + N / (1 + (df[t] || 0)));
  const vectors = perDoc.map(({ counts, total }) => {
    const vec = {};
    for (const [t, c] of Object.entries(counts)) {
      const tf = c / total;
      vec[t] = tf * idf(t);
    }
    return vec;
  });
  return { vectors, df, N };
}

function buildTfIdf(docs) {
  const tfidf = new TfIdf();
  docs.forEach(d => tfidf.addDocument(tokenize(d).join(' ')));
  return tfidf;
}

function saveJSON(file, obj) {
  fs.writeFileSync(file, JSON.stringify(obj, null, 2), 'utf-8');
}

function readJSON(file, fallback) {
  if (!fs.existsSync(file)) return fallback;
  return JSON.parse(fs.readFileSync(file, 'utf-8'));
}

// --- Image metadata and vectors ---
const IMAGES_META_PATH = path.join(DATA_DIR, 'images.json');
function splitNameToWords(name) {
  // Insert spaces before capitals (CamelCase -> words) and split on non-letters
  const spaced = name.replace(/([a-z])([A-Z])/g, '$1 $2');
  return spaced.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}
function augmentKeywordsFromFilename(name) {
  const base = splitNameToWords(name);
  const set = new Set(base);
  const s = base.join(' ');
  // Add some domain-specific hints
  if (/bell/.test(s)) ['sound','vibration','bell','ring','oscillation'].forEach(k=>set.add(k));
  if (/vocal|cord/.test(s)) ['vocal','cords','voice','larynx','sound','vibration'].forEach(k=>set.add(k));
  if (/rubber|band/.test(s)) ['rubber','band','stretch','vibration','frequency','amplitude','sound'].forEach(k=>set.add(k));
  if (/reflection/.test(s)) ['reflection','echo','sound','wave','law','angle'].forEach(k=>set.add(k));
  if (/compression|refraction/.test(s)) ['compression','rarefaction','refraction','longitudinal','wave','sound','pressure'].forEach(k=>set.add(k));
  if (/instrument|musical/.test(s)) ['musical','instrument','frequency','pitch','loudness','timbre','sound'].forEach(k=>set.add(k));
  return Array.from(set);
}
function scanImages() {
  const files = fs.existsSync(SOUND_DIR) ? fs.readdirSync(SOUND_DIR) : [];
  const images = files.filter(f => /\.(png|jpg|jpeg|gif|webp)$/i.test(f));
  const items = images.map((filename, i) => {
    const rawName = path.parse(filename).name;
    const title = rawName.replace(/[_-]/g, ' ');
    const keywords = augmentKeywordsFromFilename(rawName);
    const description = title;
    return {
      id: `img_${String(i + 1).padStart(3, '0')}`,
      filename,
      title,
      keywords,
      description
    };
  });
  return items;
}

function buildImageVectors(meta) {
  return meta.map(m => ({
    id: m.id,
    filename: m.filename,
    title: m.title,
    vec: termCounts([...(m.keywords||[]), m.description||''].join(' ')).counts
  }));
}

let imageMeta = readJSON(IMAGES_META_PATH, null);
if (!imageMeta) {
  imageMeta = scanImages();
  saveJSON(IMAGES_META_PATH, imageMeta);
}
let imageVectors = buildImageVectors(imageMeta);

// --- Gemini REST helpers ---
let cachedGeminiModel = null;
async function resolveGeminiModel(apiKey) {
  if (process.env.GEMINI_MODEL && process.env.GEMINI_MODEL.trim()) {
    return process.env.GEMINI_MODEL.trim();
  }
  if (cachedGeminiModel) return cachedGeminiModel;
  try {
    const url = `https://generativelanguage.googleapis.com/v1/models?key=${encodeURIComponent(apiKey)}`;
    const resp = await fetch(url);
    const data = await resp.json();
    const models = Array.isArray(data.models) ? data.models : [];
    // Prefer flash then pro; must support generateContent
    const candidates = models
      .filter(m => Array.isArray(m.supportedGenerationMethods) && m.supportedGenerationMethods.includes('generateContent'))
      .map(m => (typeof m.name === 'string' ? m.name.split('/').pop() : null))
      .filter(Boolean);
    const pick = candidates.find(n => /gemini-1\.5-flash/.test(n))
      || candidates.find(n => /gemini-1\.5-pro/.test(n))
      || candidates.find(n => /gemini-1\.0-pro/.test(n))
      || candidates[0];
    if (pick) {
      cachedGeminiModel = pick;
      console.log('[gemini] resolved model', pick);
      return pick;
    }
  } catch (e) {
    console.warn('[gemini] model list failed', e?.message || e);
  }
  // fallback guess
  return 'gemini-1.5-flash';
}

async function callGeminiGenerateContent(apiKey, model, prompt) {
  const url = `https://generativelanguage.googleapis.com/v1/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const body = {
    contents: [
      {
        role: 'user',
        parts: [{ text: prompt }]
      }
    ]
  };
  const resp = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Gemini HTTP ${resp.status}: ${txt.slice(0,200)}`);
  }
  const data = await resp.json();
  const parts = data?.candidates?.[0]?.content?.parts;
  const text = Array.isArray(parts) ? parts.map(p => p.text).filter(Boolean).join('\n').trim() : '';
  return text;
}

function pickImageForAnswer(answer) {
  if (!answer || imageVectors.length === 0) return null;
  const text = String(answer).toLowerCase();
  // Rule-based mapping for higher precision
  const prefer =
    /\b(bell|ring)\b/.test(text) ? (n => /bell/i.test(n)) :
    /\b(vocal|larynx|voice|cord)\b/.test(text) ? (n => /vocal|cord/i.test(n)) :
    /\b(reflection|echo)\b/.test(text) ? (n => /reflection/i.test(n)) :
    /\b(compression|rarefaction|refraction|longitudinal)\b/.test(text) ? (n => /compression|refraction/i.test(n)) :
    /\b(rubber\s*band)\b/.test(text) ? (n => /rubber|band/i.test(n)) :
    /\b(musical|instrument|pitch|timbre)\b/.test(text) ? (n => /musical|instrument/i.test(n)) : null;
  if (prefer) {
    const hit = imageMeta.find(m => prefer(m.filename) || prefer(m.title));
    if (hit) return { ...hit, url: `/Sound/${hit.filename}` };
  }
  // Similarity fallback
  const qvec = termCounts(answer).counts;
  let best = { score: -1, item: null };
  for (const iv of imageVectors) {
    const s = cosineSim(iv.vec, qvec);
    if (s >= best.score) best = { score: s, item: iv };
  }
  if (!best.item) {
    const m0 = imageMeta[0];
    return m0 ? { ...m0, url: `/Sound/${m0.filename}` } : null;
  }
  const meta = imageMeta.find(m => m.id === best.item.id);
  return meta ? { ...meta, url: `/Sound/${meta.filename}` } : null;
}

// --- Topic store (per uploaded PDF) ---
function topicPath(topicId) { return path.join(DATA_DIR, `${topicId}.json`); }

// POST /upload: PDF -> text -> chunks -> store
app.post('/upload', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    const raw = fs.readFileSync(req.file.path);
    const pdf = await pdfParse(raw);
    const text = pdf.text || '';
    const chunks = chunkText(text, 180);
    const { vectors, df, N } = buildTfIdfIndex(chunks);

    const topicId = path.parse(req.file.originalname).name.replace(/\W+/g, '_').toLowerCase();
    const store = { topicId, chunks, vectors, df, N };
    saveJSON(topicPath(topicId), store);

    // cleanup tmp
    fs.unlink(req.file.path, () => {});

    res.json({ topicId, chunkCount: chunks.length });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Failed to process PDF' });
  }
});

function retrieve(topicStore, query, k = 5) {
  let qvec;
  if (topicStore.df && topicStore.N) {
    const { counts, total } = termCounts(query);
    const idf = (t) => Math.log(1 + topicStore.N / (1 + (topicStore.df[t] || 0)));
    qvec = {};
    for (const [t, c] of Object.entries(counts)) {
      const tf = c / (total || 1);
      qvec[t] = tf * idf(t);
    }
  } else {
    // fallback to simple tf vector for backward compatibility
    qvec = termCounts(query).counts;
  }
  const scored = topicStore.vectors.map((v, idx) => ({ idx, score: cosineSim(v, qvec) }));
  scored.sort((a,b) => b.score - a.score);
  const top = scored.slice(0, k).map(s => ({ text: topicStore.chunks[s.idx], score: s.score }));
  return top;
}

async function generateAnswer(query, contexts) {
  let geminiKey = process.env.GEMINI_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;
  // If GEMINI_API_KEY is not set but OPENAI_API_KEY actually contains a Gemini key, use it.
  if (!geminiKey && openaiKey && /^(AI|AIza)/.test(openaiKey)) {
    geminiKey = openaiKey;
  }
  const contextText = contexts.map((c,i) => `Source ${i+1}:\n${c.text}`).join('\n\n');
  const prompt = `You are an AI tutor. Using only the sources below, answer the question clearly and concisely for a student. If the answer is not in the sources, say you don't know.\n\n${contextText}\n\nQuestion: ${query}\nAnswer:`;
  if (geminiKey) {
    try {
      const modelId = await resolveGeminiModel(geminiKey);
      const text = await callGeminiGenerateContent(geminiKey, modelId, prompt);
      if (text) {
        console.log('[gen] provider=gemini model=' + modelId);
        return { text, usedLLM: true, providerUsed: 'gemini:' + modelId };
      }
    } catch (e) {
      console.warn('Gemini failed, trying OpenAI', e?.message || e);
    }
  }
  const useOpenAI = String(process.env.USE_OPENAI_FALLBACK || '').toLowerCase() === 'true';
  if (useOpenAI && openaiKey) {
    try {
      const openai = new OpenAI({ apiKey: openaiKey });
      const chat = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: 'You are a helpful AI tutor. Stick to provided sources.' },
          { role: 'user', content: prompt }
        ],
        temperature: 0.2,
      });
      const text = chat.choices?.[0]?.message?.content?.trim() || '';
      if (text) {
        console.log('[gen] provider=openai');
        return { text, usedLLM: true, providerUsed: 'openai' };
      }
    } catch (e) {
      console.warn('OpenAI failed, falling back to extractive answer');
    }
  }
  // Fallback: extractive answer (clean sentences, pick concise top ones)
  const joined = contexts.map(c => c.text).join(' ');
  let sentences = joined.split(/(?<=[.!?])\s+/);
  const key = tokenize(query).filter(t => t.length > 2).join(' ');
  const clean = (s) => s
    .replace(/[•◦\-]+\s*/g, '')
    .replace(/\s+/g, ' ')
    .replace(/\s([.,!?;:])/g, '$1')
    .trim();
  sentences = sentences
    .map(clean)
    .filter(s => s.length > 20 && !/^\d+\s*\.|^(q\.|question)/i.test(s));
  const rankedList = sentences
    .map(s => ({ s, score: cosineSim(termCounts(s).counts, termCounts(key).counts) }))
    .sort((a,b) => b.score - a.score)
    .slice(0, 3)
    .map(r => r.s);
  let finalText = rankedList.join(' ');
  if (!finalText || finalText.length < 60) {
    finalText = clean(contexts.map(c => c.text).slice(0,1).join(' ')).slice(0, 500);
  }
  console.log('[gen] provider=extractive');
  return { text: finalText || 'I do not know based on the provided material.', usedLLM: false, providerUsed: 'extractive' };
}

// POST /chat: retrieve -> answer -> image
app.post('/chat', async (req, res) => {
  try {
    const { topicId, question } = req.body || {};
    if (!topicId || !question) return res.status(400).json({ error: 'topicId and question are required' });
    const store = readJSON(topicPath(topicId));
    if (!store) return res.status(404).json({ error: 'Unknown topicId. Upload a PDF first.' });
    const contexts = retrieve(store, question, 5);
    const { text: answer, usedLLM, providerUsed } = await generateAnswer(question, contexts);
    // Always choose image from local system based on the user's question (topic)
    const image = pickImageForAnswer(question);
    try { fs.appendFileSync(CHAT_LOG, JSON.stringify({ ts: new Date().toISOString(), topicId, question, answer, usedLLM, providerUsed, image: image ? { id: image.id, filename: image.filename, title: image.title } : null }) + '\n'); } catch {}
    res.json({ answer, image, providerUsed });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Failed to generate answer' });
  }
});

// GET /images/:topicId -> return image metadata (not topic-bound, but per spec)
app.get('/images/:topicId', (req, res) => {
  res.json({ topicId: req.params.topicId, images: imageMeta.map(m => ({ ...m, url: `/Sound/${m.filename}` })) });
});

app.get('/', (req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

// Health check endpoint
app.get('/health', (req, res) => {
  const up = true;
  const geminiConfigured = !!(process.env.GEMINI_API_KEY || (process.env.OPENAI_API_KEY && /^(AI|AIza)/.test(process.env.OPENAI_API_KEY)));
  const openaiConfigured = !!process.env.OPENAI_API_KEY;
  const useOpenAI = String(process.env.USE_OPENAI_FALLBACK || '').toLowerCase() === 'true';
  res.json({ status: up ? 'ok' : 'down', geminiConfigured, openaiConfigured, useOpenAIFallback: useOpenAI });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
