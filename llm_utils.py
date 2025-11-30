import os
from dotenv import load_dotenv
load_dotenv()

# Import Groq
from groq import Groq

# Import OpenAI
from openai import OpenAI

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "")


# --------------------------------------------------------
# GET CLIENT (Groq or OpenAI)
# --------------------------------------------------------
def get_llm_client():

    # If user selected GPT model
    if LLM_MODEL.startswith("gpt") and OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY)

    # Groq first
    if GROQ_API_KEY:
        return Groq(api_key=GROQ_API_KEY)

    # Fallback OpenAI
    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY)

    return None


# --------------------------------------------------------
# GENERATE RESPONSE
# --------------------------------------------------------
def generate_answer(client, messages):

    # GROQ
    if isinstance(client, Groq):

        model_name = (
            LLM_MODEL
            if LLM_MODEL and not LLM_MODEL.startswith("gpt")
            else "llama-3.1-8b-instant"
        )

        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return resp.choices[0].message.content

    # OPENAI
    if isinstance(client, OpenAI):

        model_name = LLM_MODEL if LLM_MODEL else "gpt-3.5-turbo"

        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return resp.choices[0].message.content

    raise RuntimeError("No valid LLM client configured")


# --------------------------------------------------------
# TRANSCRIBE AUDIO
# --------------------------------------------------------
def transcribe_audio(audio_bytes):

    # Groq whisper
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        try:
            out = client.audio.transcriptions.create(
                file=("audio.wav", audio_bytes, "audio/wav"),
                model="whisper-large-v3",
            )
            return out.text
        except:
            pass

    # OpenAI fallback
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        try:
            from io import BytesIO
            audio_file = BytesIO(audio_bytes)
            out = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
            )
            return out.text
        except:
            pass

    return ""
