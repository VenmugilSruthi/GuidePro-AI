import os
from dotenv import load_dotenv
load_dotenv()

# Import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except:
    GROQ_AVAILABLE = False

# Import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "")


# --------------------------------------------------------
# GET CLIENT (Groq or OpenAI)
# --------------------------------------------------------
def get_llm_client():

    # 1. If model starts with "gpt" -> force OpenAI
    if LLM_MODEL.startswith("gpt") and OPENAI_AVAILABLE and OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY)

    # 2. Otherwise prefer Groq
    if GROQ_AVAILABLE and GROQ_API_KEY:
        return Groq(api_key=GROQ_API_KEY)

    # 3. Fallback OpenAI
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY)

    return None


# --------------------------------------------------------
# GENERATE RESPONSE
# --------------------------------------------------------
def generate_answer(client, messages):

    # GROQ CLIENT
    if GROQ_AVAILABLE and isinstance(client, Groq):

        model_name = (
            LLM_MODEL 
            if LLM_MODEL and not LLM_MODEL.startswith("gpt")
            else "llama-3.1-8b-instant"
        )

        resp = client.chat.completions.create(
            model=model_name,
            messages=messages
        )

        return resp.choices[0].message.content


    # OPENAI CLIENT
    if OPENAI_AVAILABLE and isinstance(client, OpenAI):

        model_name = LLM_MODEL if LLM_MODEL else "gpt-3.5-turbo"

        resp = client.chat.completions.create(
            model=model_name,
            messages=messages
        )

        return resp.choices[0].message.content


    raise RuntimeError("No LLM client configured")


# --------------------------------------------------------
# TRANSCRIBE AUDIO
# --------------------------------------------------------
def transcribe_audio(audio_bytes):

    # Groq Whisper
    if GROQ_AVAILABLE and GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        try:
            return client.audio.transcriptions.create(
                file=("audio.wav", audio_bytes, "audio/wav"),
                model="whisper-large-v3"
            ).text
        except:
            pass

    # OpenAI Whisper fallback
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        try:
            from io import BytesIO
            audio_file = BytesIO(audio_bytes)
            output = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1"
            )
            return output.text
        except:
            pass

    return ""
