import modal
import json
import os
import httpx
import subprocess
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


app = modal.App("llama-chat-bot")

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "slowapi", "httpx")
    .apt_install("curl")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
)

ollama_volume = modal.Volume.from_name("ollama-models", create_if_missing=True)
secret = modal.Secret.from_name("chatbot-api-key")


web_app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
web_app.state.limiter = limiter
web_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

MODEL = "llama3.2:3b"


FACTS = {
    "who": "My name is Shubham Rana.",
    "skills": (
        "I work mainly on backend systems, microservices, and distributed "
        "applications using Java, Python, and Go, AWS, GCP, Azure, Event-Driven Architecutre. For more info : download my resumeß"
    ),
    "projects": (
        "I’ve worked on HealthCareContentRecommendationService, airportinfo, "
        "and this chatbot API."
    ),
    "reach": (
        "Goto Contact section of the website or simply go to my linkedin: https://www.linkedin.com/in/shubham4034/"
    ),
    "Github": (
        "Goto my github repository to know about chat bot code or the portfolio code or anything else: https://github.com/sbm4034/"
    ),
    "linkedin": ("https://www.linkedin.com/in/shubham4034/"),
    "connect": ("Goto Contact section of the website or simply go to my linkedin: https://www.linkedin.com/in/shubham4034/")
    
}


def verify_api_key(request: Request):
    if request.headers.get("X-API-Key") != os.environ.get("CHATBOT_API_KEY"):
        raise HTTPException(status_code=401, detail="Unauthorized")

def deterministic_answer(q: str) -> str | None:
    q = q.lower()
    if "who are you" in q:
        return FACTS["who"]
    if "skill" in q:
        return FACTS["skills"]
    if "project" in q:
        return FACTS["projects"]
    return None



@web_app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request):
    verify_api_key(request)
    body = await request.json()
    messages = body.get("messages", [])
    user_q = messages[-1]["content"]

    fixed = deterministic_answer(user_q)
    if fixed:
        async def stream_fixed():
            yield f"data: {json.dumps({'text': fixed})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_fixed(), media_type="text/event-stream")


    async def stream_llm():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": MODEL,
                    "prompt": (
                        "You are Shubham Rana. Speak in first person. "
                        "Answer briefly in one sentence.\n\n"
                        f"{user_q}"
                    ),
                    "stream": True,
                    "temperature": 0.2,
                    "num_predict": 40,
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if data.get("response"):
                            yield f"data: {json.dumps({'text': data['response']})}\n\n"
                        if data.get("done"):
                            break
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_llm(), media_type="text/event-stream")


@web_app.get("/health")
async def health():
    r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    return {"ollama": r.returncode == 0, "models": r.stdout}


@app.function(
    image=image,
    secrets=[secret],
    volumes={"/root/.ollama": ollama_volume},
    cpu=2,
    memory=4096,
    timeout=600,
)
@modal.asgi_app()
def serve():
    subprocess.Popen(["ollama", "serve"])
    subprocess.run(["ollama", "pull", MODEL], check=True)
    return web_app
