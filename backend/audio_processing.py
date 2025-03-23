import os
import logging
from dotenv import load_dotenv
from groq import Groq
import base64
import voicerss_tts
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


VOICERSS_API_KEY = os.getenv("VOICERSS_API_KEY") 

MODEL_NAME = "llama-3.3-70b-versatile"

# LLM Testing 
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    # Test connection to LLM by making a simple request
    test_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "Test connection"}]
    )
    print("LLM connection successful: Test response received.")
except Exception as e:
    print(f"Failed to connect to LLM: {e}")
    raise

#STT
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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
        return(f"Groq Whisper transcription failed: {e}")

#Response
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_response(text: str) -> str:
    """
    Generate text reply using Groq's Gemma model.
    """
    try:
        chat = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Please analyze the data and proviude an appropriate response"},
                {"role": "user", "content": text}
            ]
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq LLM generation failed: {e}")
        raise

#TTS
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def synthesize_speech(text: str, output_path: str) -> str:
    try:
        payload = {
            'key': VOICERSS_API_KEY,
            'hl': 'en-us',
            'v': 'Linda',
            'src': f'{text}',
            'r': '0',
            'c': 'mp3',
            'f': '44khz_16bit_stereo',
            'ssml': 'false',
            'b64': 'true'
        }
        voice = voicerss_tts.speech(payload)
        audio_bytes = base64.b64decode(voice['response'])
        
        with open(output_path, "wb") as f:
            f.write(audio_bytes)    
        return output_path

    except Exception as e:
        logger.error(f"VoiceRSS TTS failed: {e}")
        raise
