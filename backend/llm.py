import os
import logging
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "gemma-7b-it"

async def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio using Groq's Whisper model.
    """
    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        return transcription.text
    except Exception as e:
        logger.error(f"Groq Whisper transcription failed: {e}")
        raise

async def generate_response(text: str) -> str:
    """
    Generate text reply using Groq's Gemma model.
    """
    try:
        chat = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ]
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq LLM generation failed: {e}")
        raise

async def synthesize_speech(text: str, output_path: str) -> str:
    """
    Stub: Plug in TTS engine like Coqui or Piper.
    """
    with open(output_path, "w") as f:
        f.write(f"(TTS not yet implemented)\n{text}")
    return output_path
