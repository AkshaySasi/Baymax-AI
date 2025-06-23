from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.baymax_chain import BaymaxChain
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Baymax.AI v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    user_id: str
    user_input: str

baymax_chain = BaymaxChain()

@app.on_event("startup")
async def startup_event():
    logger.debug("Server starting up...")
    try:
        # Test BaymaxChain initialization with a minimal API call
        test_response = baymax_chain.generate_emotional_reply("test_user", "test initialization")
        logger.debug("BaymaxChain initialized successfully with response: %s", test_response)
    except Exception as e:
        logger.error("Startup error: %s. Ensure GEMINI_API_KEY is set and internet is available.", str(e))
        raise HTTPException(status_code=503, detail="Service unavailable: Check API key and network")

@app.post("/chat")
async def chat(request: Request, chat_req: ChatRequest):
    if request.headers.get("X-Consent", "false").lower() != "true":
        logger.warning("Consent header missing or invalid for user %s", chat_req.user_id)
        raise HTTPException(status_code=403, detail="Consent required.")
    try:
        response = baymax_chain.generate_emotional_reply(chat_req.user_id, chat_req.user_input)
        logger.info("User %s: %s -> Response: %s", chat_req.user_id, chat_req.user_input, response)
        return {"response": response}
    except Exception as e:
        logger.error("Chat error for user %s: %s", chat_req.user_id, str(e))
        if "API key" in str(e) or "401" in str(e):
            raise HTTPException(status_code=401, detail="Invalid API key")
        elif "429" in str(e):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        else:
            raise HTTPException(status_code=500, detail="Internal server error")
