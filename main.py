from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel
import os
import tempfile

app = FastAPI()

# Contabo VPS (CPU) အတွက် Model ကို base (သို့) small သုံးတာ အကောင်းဆုံးပါ
model_size = "base"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # ရောက်လာတဲ့ အသံဖိုင်ကို ယာယီသိမ်းရန်
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        # Whisper ဖြင့် အသံကို စာသားနှင့် အချိန်ပြောင်းရန်
        segments, info = model.transcribe(temp_path, beam_size=5)
        
        result = []
        for segment in segments:
            result.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        
        return {"status": "success", "language": info.language, "segments": result}
    
    finally:
        # ယာယီဖိုင်ကို ပြန်ဖျက်ရန်
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
def read_root():
    return {"message": "Whisper API is running smoothly on Port 8001"}
