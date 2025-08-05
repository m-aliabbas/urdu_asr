# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
import tempfile
import shutil
import os
import torch 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

app = FastAPI()

# Load your Whisper model once
pipe = pipeline("automatic-speech-recognition", model="m-aliabbas1/whisper-large-ur1",device=device)

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("m-aliabbas1/whisper-large-ur1"),
    response_format: str = Form("json")  # Support extensibility
):
    if not file.filename.endswith((".wav", ".mp3", ".m4a", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_audio:
        shutil.copyfileobj(file.file, temp_audio)
        temp_audio_path = temp_audio.name

    # Run inference
    try:
        result = pipe(temp_audio_path,generate_kwargs={"language": "ur", "task": "transcribe"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR model failed: {e}")
    finally:
        os.remove(temp_audio_path)  # Clean up

    return JSONResponse(content={"text": result["text"]})
