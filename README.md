# Shubham Chatbot API

This repository contains the backend for a **personal chatbot** that responds as *me* — a human software engineer — rather than a generic AI assistant.

The goal of this project is to build a **reliable, low-hallucination chatbot** that answers questions about my background, skills, and projects in a conversational way, while staying truthful and constrained.

---

## What this chatbot does

- Responds in a **human, first-person style**
- Avoids inventing skills, projects, or experience
- Answers only what is asked, concisely
- Supports **streaming responses**
- Enforces **API authentication and rate limiting**

This is not a demo bot — it’s meant to behave predictably and honestly.

---

## Tech stack

- **Python**
- **FastAPI** – HTTP API layer
- **Ollama** – local LLM runtime
- **Qwen 2.5 (7B)** – language model
- **Modal** – containerized deployment
- **SlowAPI** – rate limiting
- **HTTPX** – async HTTP client

---

## Model choice

The chatbot uses **`qwen2.5:7b`** via Ollama.

This model was chosen because:
- Smaller models (1–2B) hallucinate too easily
- Larger models exceed Modal’s disk limits
- Qwen 2.5 7B provides a good balance of:
  - persona stability
  - low hallucination
  - reasonable CPU performance

---

## API Endpoints

### POST /chat
Streaming chat endpoint using Server-Sent Events (SSE).

**Response:**  
Streamed text chunks.

---

### POST /chat-simple
Non-streaming chat endpoint.

**Response:**  
Returns a single response string.

---

### GET /health
Health check endpoint.

Returns:
- Ollama running status
- Available models

---

## Authentication

All chat endpoints require an API key.

- **Header name:** `X-API-Key`
- **Environment variable:**
  ```text
  CHATBOT_API_KEY

## Authentication behavior

Requests without a valid API key will return:


---

## Rate limiting

Rate limiting is enforced using **SlowAPI**.

Current limits:
- `/chat`: 10 requests per minute
- `/chat-simple`: 5 requests per minute
- `/health`: 20 requests per minute

This protects the service from abuse and accidental overload.

---

## Running locally

### Prerequisites
- Python 3.10+
- Ollama installed and running
- `qwen2.5:7b` pulled in Ollama

```bash
ollama serve
ollama pull qwen2.5:7b
pip install -r requirements.txt
uvicorn app:web_app --reload
```
## Deployment

The service is deployed using **Modal**.

Key points:
- CPU-only containers
- Ollama runs inside the container
- Model is pulled at runtime
- Disk usage stays within Modal’s function limits
- Designed for low cost and predictable behavior

---

## Notes

This chatbot is intentionally constrained.

If it does not know something or has not done something, it will say so instead of guessing.  
That behavior is by design.
