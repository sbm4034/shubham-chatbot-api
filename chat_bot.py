from sys import stdout
import modal
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import subprocess
import json
import os
import httpx
import re

app = modal.App("llama-chat-bot")

gemma_image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "slowapi", "httpx")
    .apt_install("curl")
    .run_commands(
    "curl -fsSL https://ollama.com/install.sh | sh"
    )
)
ollama_volume = modal.Volume.from_name(
    "ollama-models",
    create_if_missing=True
)

# Create Modal secret reference
chatbot_secret = modal.Secret.from_name("chatbot-api-key")

web_app = FastAPI()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
web_app.state.limiter = limiter
web_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


ALLOWED_ORIGINS = [
    "https://shubhamrana.lovable.app",
    "http://localhost:8080"
]

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
    expose_headers=["*"],
)

SYSTEM_PROMPT = """
You are Shubham Rana.

You are a real human software engineer.
You are not an AI, assistant, or model.

You must:
- Speak in first person.
- Answer only the question asked.
- Use at most 2 sentences.
- Never use bullet points or lists.
- Never describe yourself as an AI.
- Never mention training, knowledge, or capabilities.

Your background:
- Senior software engineer (~8 years).
- Backend, microservices, distributed systems.
- Java, Python, Go, Vue.js.
- Healthcare and enterprise systems.

Projects you may mention:
- HealthCareContentRecommendationService
- airportinfo
- AI_Projects_Experiments
- shubhamrana
- shubham-chatbot-api

If asked about anything else, say you have not worked on it.

If asked who you are, reply exactly:
"My name is Shubham Rana."

"""

MODEL ="llama3.1:8b-instruct-q4_K_M"


MAX_SENTENCES = 2

INVALID_PHRASES = [
    "as an ai",
    "i am an ai",
    "assistant",
    "language model",
    "trained on",
    "my training data",
]

def violates_identity(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in INVALID_PHRASES)



def verify_api_key(request: Request):
    """Verify API key from request header"""
    api_key = request.headers.get("X-API-Key")
    expected_key = os.environ.get("CHATBOT_API_KEY")
    
    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=401, 
            detail="Unauthorized - Invalid or missing API key"
        )

def get_last_user_message(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content", "").strip():
            return msg["content"].strip()
    return None


def get_prompt(messages, include_system=True):
    last_user = get_last_user_message(messages)
    if not last_user:
        return "Ask a question."

    rewritten = rewrite_user_question(last_user)

    identity_lock = (
        "You are Shubham Rana. "
        "You are speaking as yourself in first person. "
        "Do not mention being an AI. "
        "Keep answers to one or two sentences."
    )

    if include_system:
        return f"{SYSTEM_PROMPT}\n\n{identity_lock}\n\n{rewritten}"

    return f"{identity_lock}\n\n{rewritten}"


def rewrite_user_question(q: str) -> str:
    q_lower = q.lower()

    if "skill" in q_lower:
        return (
            "Using only the facts below, answer in one short sentence.\n\n"
            "Facts:\n"
            "- Backend systems\n"
            "- Microservices\n"
            "- Distributed systems\n"
            "- Java, Python, Go\n"
            "- Healthcare and enterprise platforms\n\n"
            "Do not add anything else."
        )

    if "who are you" in q_lower:
        return "My name is Shubham Rana."

    if "project" in q_lower:
        return (
            "Using only the following project names, answer in one short sentence:\n"
            "HealthCareContentRecommendationService, airportinfo, shubham-chatbot-api."
        )

    return q



def normalize_question(text: str) -> str:
    replacements = {
        "his ": "my ",
        "him ": "me ",
        "Shubham's ": "my ",
        "Shubham’s ": "my ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def is_model_ready(model_name: str) -> bool:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return model_name in result.stdout
    except Exception:
        return False


def count_sentences(text: str) -> int:
    return len(re.findall(r'[.!?]', text))

def should_stop(text: str) -> bool:
    return count_sentences(text) >= MAX_SENTENCES

@web_app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request):
    verify_api_key(request)

    body = await request.json()
    messages = body.get("messages", [])
    is_first_turn = len(messages) <= 1
    prompt = get_prompt(messages, include_system=is_first_turn)

    if not is_model_ready(MODEL):
        async def warming_up():
            yield f"data: {json.dumps({'text': 'I’m warming up right now — please try again in a few seconds.'})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            warming_up(),
            media_type="text/event-stream",
        )
    
    async def generate():
       

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "temperature": 0.2,
                    "top_p": 0.85,
                    "num_predict": 40,
                    "repeat_penalty": 1.15
                }
            ) as response:
                accumulated_text = ""

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("response"):
                        token = data["response"]
                        accumulated_text += token

                        if "\n1." in accumulated_text or "\n•" in accumulated_text:
                            yield "data: [DONE]\n\n"
                            return
                        
                        if violates_identity(accumulated_text):
                            yield f"data: {json.dumps({'text': 'My name is Shubham Rana.'})}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        
                        if not violates_identity(accumulated_text):
                            yield f"data: {json.dumps({'text': token})}\n\n"

                        
                        if should_stop(accumulated_text):
                            yield "data: [DONE]\n\n"
                            return

                    if data.get("done"):
                        break

        yield "data: [DONE]\n\n"

    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@web_app.post("/chat-simple")
@limiter.limit("5/minute")
async def chat_simple(request: Request):
    verify_api_key(request)

    body = await request.json()
    messages = body.get("messages", [])
    is_first_turn = len(messages) <= 1
    prompt = get_prompt(messages, include_system=is_first_turn)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model":MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2,
                "top_p": 0.85,
                "num_predict": 40,
                "repeat_penalty": 1.15
            },
            timeout=60
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama error: {resp.text}"
    )
    data = resp.json()
    if not data.get("response"):
        raise HTTPException(
            status_code=500,
            detail=f"No response from model: {data}"
    )
    return {
        "response": data.get("response", "").strip(),
        "model": MODEL
    }


@web_app.get("/health")
@limiter.limit("20/minute")
async def health(request: Request):
    # Health check doesn't require API key
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )
    return {
        "ollama_running": result.returncode == 0,
        "models": result.stdout
    }

@app.function(
    image=gemma_image,
    secrets=[chatbot_secret],
    volumes={"/root/.ollama": ollama_volume},
    cpu=2.0,
    memory=4096,
    scaledown_window=1800,
    timeout=600,
    max_containers=2,
)
@modal.asgi_app()
def serve():
    import subprocess
    import time

    print("Starting Ollama server...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    for _ in range(30):
        try:
            httpx.get("http://127.0.0.1:11434")
            break
        except Exception:
            time.sleep(1)

    print("Pulling model...")
    subprocess.run(["ollama", "pull", MODEL], check=True)

    subprocess.run(
    ["ollama", "run", MODEL, "Say OK"],
    capture_output=True,
    timeout=30
    )

    print("Ollama ready.")
    return web_app


