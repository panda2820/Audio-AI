from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from audio_management import handle_audio_request

app = FastAPI(
    title="Audio AI",
    description="Flow:\nUpload audio → transcribe → LLM response → audio reply",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
RESPONSE_DIR = "responses"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESPONSE_DIR, exist_ok=True)


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        print("Processing started")

        output_path, reply_text = await handle_audio_request(file)
        sanitized_reply_text = reply_text.replace('\n', ' ').replace('\r', ' ')

        return FileResponse(
            output_path,
            media_type="audio/mpeg",
            filename=os.path.basename(output_path),
            headers={"X-Reply-Text": sanitized_reply_text}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return JSONResponse({"message": "Audio Response processing using LLM."})
