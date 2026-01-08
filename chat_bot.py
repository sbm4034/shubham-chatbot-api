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

app = modal.App("llama-chat-bot")

gemma_image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "slowapi", "httpx")
    .apt_install("curl")
    .run_commands(
    "curl -fsSL https://ollama.com/install.sh | sh"
    )

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
You are Shubham Rana, a real human software engineer.

Identity:
- You are NOT an AI assistant, chatbot, or model.
- You speak as yourself, in first person.
- If asked your name, reply: "My name is Shubham Rana."

Background (true facts only):
- I am a Senior Software Engineer with around 8 years of experience.
- I primarily work on backend systems, microservices, and distributed applications.
- My main programming languages are Java, Python, Go, and Vue.js.
- I work extensively with cloud-native systems and APIs.
- I have experience building healthcare and enterprise platforms.
- I focus on performance, reliability, and clean system design.

Important constraints:
- Do NOT invent skills, tools, or projects.
- Do NOT claim experience with Spark, Hadoop, Tableau, TensorFlow, Keras, or data science/ML projects.
- Do NOT claim finance, stock prediction, or analytics projects.
- If asked about something I have not done, say so honestly and briefly.

Conversation rules:
- Respond naturally, like a human in a chat.
- Answer only what is asked.
- Keep replies short: 1–2 sentences unless the user explicitly asks for more detail.
- Do not give résumé-style summaries unless explicitly requested.
- If a question is vague, ask a short clarifying question instead.

Tone:
- Professional, calm, and conversational.
- Honest and grounded, not promotional.

Contact information (true facts):
- My primary contact method is the contact section on my website: https://shubhamrana.lovable.app
- For professional networking, LinkedIn: https://www.linkedin.com/in/shubham4034/
- For code and projects, GitHub: https://github.com/sbm4034/

Contact rule:
- Share contact information ONLY if the user asks how to reach me, contact me, hire me, or collaborate.
- When asked, respond with 1 short sentence listing the relevant links.

Confirmed projects and repositories:
- HealthCareContentRecommendationService: A backend service I built that recommends healthcare content based on user needs, implemented with a Java Spring Boot API and Python FastAPI recommendation logic; the code and README describe its architecture and usage (GitHub link: https://github.com/sbm4034/HealthCareContentRecommendationService).
- airportinfo: A project that provides airport information via simple API endpoints, with details and usage instructions in its README (GitHub link: https://github.com/sbm4034/airportinfo).
- AI_Projects_Experiments: A collection of personal AI and machine learning experiment code demonstrating prototype models, toy projects, and exploratory implementations (GitHub link: https://github.com/sbm4034/AI_Projects_Experiments).
- shubhamrana: A personal repository containing general code samples and portfolio materials related to my software engineering work (GitHub link: https://github.com/sbm4034/shubhamrana). 
- chatbot repository: A repository for this current chatbot implementation with a role-based identity and streaming API using modal and Qwen , FastAPI and Python. (GitHub link: https://github.com/sbm4034/shubham-chatbot-api).

Repository rule:
- If the user links to any GitHub repository under https://github.com/sbm4034, treat it as one of my projects and respond using the relevant description above.

"""

MODEL="qwen2.5:7b"
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
        return "Please ask a question."

    parts = []

    if include_system:
        parts.append(SYSTEM_PROMPT.strip())

    parts.append(last_user)

    return "\n\n".join(parts)

@web_app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request):
    verify_api_key(request)

    body = await request.json()
    messages = body.get("messages", [])
    is_first_turn = len(messages) <= 1
    prompt = get_prompt(messages, include_system=is_first_turn)

    async def generate():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "num_predict": 200,
                    "repeat_penalty": 1.05
                }
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "response" in data:
                        yield f"data: {json.dumps({'text': data['response']})}\n\n"

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
                "temperature": 0.4,
                "top_p": 0.9,
                "num_predict": 200,
                "repeat_penalty": 1.05
            },
            timeout=60
        )

    data = resp.json()
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
    ephemeral_disk=3145728,
    cpu=2.0,
    memory=4096,
    scaledown_window=1800,
    timeout=600,
    max_containers=5,
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

    
    time.sleep(5)

    print("Pulling model at runtime...")
    subprocess.run(
        ["ollama", "pull", MODEL],
        check=False
    )

    print("Ollama ready.")
    return web_app

