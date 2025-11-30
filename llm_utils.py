# llm_utils.py
import os
from dotenv import load_dotenv
load_dotenv()

# Try to import Groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# Try to import OpenAI fallback
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "")

def get_llm_client():
    # 1. Prioritize OpenAI if an OpenAI model (starting with 'gpt') is configured
    if OPENAI_AVAILABLE and OPENAI_API_KEY and LLM_MODEL.startswith("gpt"):
        openai.api_key = OPENAI_API_KEY
        return openai

    # 2. Use Groq if a Groq key is available and it wasn't an OpenAI model request
    if GROQ_AVAILABLE and GROQ_API_KEY:
        return Groq(api_key=GROQ_API_KEY)
        
    # 3. Fallback to OpenAI if a key exists
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
        return openai
        
    return None

def generate_answer(client, messages):
    # If Groq client
    if GROQ_AVAILABLE and isinstance(client, type(getattr(__import__("groq"), "Groq")())):
        # Fallback to a safe Groq model
        model_name = LLM_MODEL if LLM_MODEL and not LLM_MODEL.startswith("gpt") else "llama3-8b-8192" 
        resp = client.chat.completions.create(model=model_name, messages=messages)
        return resp.choices[0].message.content
        
    # If OpenAI client
    if OPENAI_AVAILABLE and client == openai:
        # Fallback to a safe OpenAI model
        model_name = LLM_MODEL or "gpt-3.5-turbo" 
        resp = openai.ChatCompletion.create(model=model_name, messages=messages)
        return resp.choices[0].message.content
        
    raise RuntimeError("No LLM client configured")

# Simple transcribe wrapper
def transcribe_audio(audio_bytes):
    # If you have a Groq whisper model endpoint:
    if GROQ_AVAILABLE and GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        try:
            resp = client.audio.transcriptions.create(file=("audio.wav", audio_bytes, "audio/wav"),
                                                     model="whisper-large-v3") 
            return resp.text
        except Exception:
            return ""
    # OpenAI whisper (if key and openai lib available)
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        try:
            from io import BytesIO
            audio_file = BytesIO(audio_bytes)
            # Note: This may need adjustment based on your specific OpenAI SDK version
            resp = openai.Audio.transcribe("whisper-1", audio_file) 
            return resp["text"]
        except Exception:
            return ""
    # fallback
    return ""