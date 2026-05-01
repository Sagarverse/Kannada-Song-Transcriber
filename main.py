import os
import shutil
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transcriber import Transcriber
import uuid
import mlflow
import time

app = FastAPI()

# Setup MLflow
mlflow.set_experiment("Kannada_Song_Transcription")

# Setup directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize transcriber
transcriber = Transcriber()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    start_time = time.time()
    
    try:
        with mlflow.start_run():
            mlflow.log_param("filename", file.filename)
            mlflow.log_param("file_size_mb", os.path.getsize(input_path) / (1024 * 1024))
            
            lyrics = transcriber.transcribe(input_path)
            
            duration = time.time() - start_time
            mlflow.log_metric("transcription_time_sec", duration)
            mlflow.log_metric("segment_count", len(lyrics))
            
            lrc_path = os.path.join(OUTPUT_DIR, f"{file_id}.lrc")
            txt_path = os.path.join(OUTPUT_DIR, f"{file_id}.txt")
            
            transcriber.save_as_lrc(lyrics, lrc_path)
            transcriber.save_as_txt(lyrics, txt_path)
            
            return {
                "lyrics": lyrics,
                "lrc_url": f"/download/{file_id}/lrc",
                "txt_url": f"/download/{file_id}/txt"
            }
    except Exception as e:
        print(f"Transcription error: {e}")
        mlflow.log_param("error", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)

@app.get("/download/{file_id}/{ext}")
async def download_file(file_id: str, ext: str):
    file_path = os.path.join(OUTPUT_DIR, f"{file_id}.{ext}")
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=f"transcription.{ext}")
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
