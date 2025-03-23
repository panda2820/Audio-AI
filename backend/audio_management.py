import os
import shutil
import uuid
from fastapi import UploadFile, File
from audio_processing import transcribe_audio, generate_response, synthesize_speech

UPLOAD_DIR = "uploads"
RESPONSE_DIR = "responses"

async def save_uploaded_file(upload_file: UploadFile, destination: str):   
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

async def handle_audio_request(file: UploadFile):
    file_ext = os.path.splitext(file.filename)[-1]
    if file_ext.lower() not in [".mp3", ".wav", ".m4a"]:
        raise ValueError("Unsupported file type")

    input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{file_ext}")
    await save_uploaded_file(file, input_path)

    transcript = await transcribe_audio(input_path)

    reply_text = await generate_response(transcript)

    output_filename = f"response_{uuid.uuid4()}.mp3"
    output_path = os.path.join(RESPONSE_DIR, output_filename)
    await synthesize_speech(reply_text, output_path)

    return output_path, reply_text
