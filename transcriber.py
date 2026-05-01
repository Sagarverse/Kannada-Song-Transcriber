import torch
from transformers import pipeline
import datetime
import os

class Transcriber:
    def __init__(self, model_name="ARTPARK-IISc/whisper-small-vaani-kannada"):
        print(f"Loading Specialized Kannada Model: {model_name}...")
        # Detect device
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1 # CPU
            
        # Using the transformers pipeline which was the only one that gave correct Kannada script
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            stride_length_s=5,
            device=device,
            return_timestamps=True
        )
        print("Model loaded successfully.")

    def format_timestamp(self, seconds: float) -> str:
        if seconds is None: return "[00:00.00]"
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        minutes = total_seconds // 60
        secs = total_seconds % 60
        millis = int((td.total_seconds() - total_seconds) * 100)
        return f"[{minutes:02d}:{secs:02d}.{millis:02d}]"

    def transcribe(self, audio_path: str):
        print(f"Starting specialized transcription for: {audio_path}")
        
        # We use generate_kwargs to force the language and task
        # The ARTPARK model is fine-tuned for Kannada, so this should stay in script.
        result = self.pipe(
            audio_path, 
            generate_kwargs={"language": "kannada", "task": "transcribe"}
        )
        
        chunks = result.get("chunks", [])
        lyrics_with_timestamps = []
        
        for chunk in chunks:
            start_time = chunk["timestamp"][0]
            timestamp = self.format_timestamp(start_time)
            text = chunk["text"].strip()
            
            # Print to console for debugging
            print(f"{timestamp} {text}")
            
            if text:
                lyrics_with_timestamps.append(f"{timestamp} {text}")
            
        print(f"Transcription finished. Total segments: {len(lyrics_with_timestamps)}")
        return lyrics_with_timestamps

    def save_as_lrc(self, lyrics: list, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            for line in lyrics:
                f.write(line + "\n")

    def save_as_txt(self, lyrics: list, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            for line in lyrics:
                f.write(line + "\n")
