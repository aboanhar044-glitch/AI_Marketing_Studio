import os
import uuid
import socket
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from presets_catalog import (
    get_all_video_styles,
    STORE_NEEDS_OPTIONS,
    VOICE_PROFILES,
    STORE_DISPLAY_MODES,
)
from video_engine import produce_complete_marketing_video

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="AI Viral Marketing Video Studio")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

VIDEO_JOBS: dict[str, dict] = {}


def _set_job_state(job_id: str, **updates):
    if job_id in VIDEO_JOBS:
        VIDEO_JOBS[job_id].update(updates)


def _cleanup_uploads(paths: list[str]):
    for p in paths or []:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


async def _process_video_job(job_id: str, payload: dict, saved_paths: list[str]):
    try:
        _set_job_state(
            job_id,
            status="processing",
            progress=15,
            stage="تحليل الصور وإعداد السيناريو والصوت البشري...",
        )
        result = await produce_complete_marketing_video(job_id, payload, saved_paths, OUTPUT_DIR)
        fname = os.path.basename(result["final_mp4"])
        _set_job_state(
            job_id,
            status="completed",
            progress=100,
            stage="اكتمل إنتاج الفيديو بنجاح",
            video_url=f"/outputs/{fname}",
            filename=fname,
            voiceover_script=result["voiceover_script"],
            layout_mode=result["layout_mode"],
            duration_seconds=result["duration_seconds"],
            ai_clips_generated=result.get("ai_clips_generated", 0),
        )
    except Exception as exc:
        _set_job_state(
            job_id,
            status="failed",
            stage="تعذر إتمام إنتاج الفيديو",
            error=str(exc),
        )
    finally:
        _cleanup_uploads(saved_paths)


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/catalog")
async def catalog():
    local_ip = get_local_ip()
    return {
        "styles": get_all_video_styles(),
        "needs": STORE_NEEDS_OPTIONS,
        "voices": VOICE_PROFILES,
        "display_modes": STORE_DISPLAY_MODES,
        "device_urls": {
            "desktop": "http://localhost:8000",
            "mobile_lan": f"http://{local_ip}:8000",
        },
    }


@app.post("/api/create-video")
async def create_video(
    background_tasks: BackgroundTasks,
    store_name: str = Form(...),
    phone: str = Form(...),
    address: str = Form(...),
    product_name: str = Form(...),
    product_details: str = Form(...),
    video_style: str = Form(...),
    store_need: str = Form(...),
    display_mode: str = Form("auto"),
    extra_notes: str = Form(""),
    voice_id: str = Form("ar-SA-HamedNeural"),
    duration: str = Form("30"),
    images: List[UploadFile] = File(...),
):
    if not images:
        return JSONResponse(
            {"status": "error", "message": "يرجى اختيار صورة واحدة على الأقل للمنتج."},
            status_code=400,
        )

    job_id = str(uuid.uuid4())[:8]
    saved_paths = []

    for idx, upload in enumerate(images):
        ext = os.path.splitext(upload.filename or "image.jpg")[1] or ".jpg"
        dest = os.path.join(UPLOAD_DIR, f"{job_id}_{idx}{ext}")
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_paths.append(dest)

    payload = {
        "store_name": store_name,
        "phone": phone,
        "address": address,
        "product_name": product_name,
        "product_details": product_details,
        "video_style": video_style,
        "store_need": store_need,
        "display_mode": display_mode,
        "extra_notes": extra_notes,
        "voice_id": voice_id,
        "duration": duration,
    }

    VIDEO_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 8,
        "stage": "تم استلام الطلب وبدء تجهيز المشاهد...",
        "error": None,
    }

    background_tasks.add_task(_process_video_job, job_id, payload, saved_paths)

    return JSONResponse(
        {
            "status": "queued",
            "job_id": job_id,
            "message": "تم بدء إنتاج الفيديو في الخلفية",
        }
    )


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    job = VIDEO_JOBS.get(job_id)
    if not job:
        return JSONResponse({"status": "error", "message": "المهمة غير موجودة"}, status_code=404)
    return JSONResponse(job)


if __name__ == "__main__":
    import uvicorn

    ip = get_local_ip()
    print("Desktop URL: http://localhost:8000")
    print(f"Mobile (same Wi-Fi): http://{ip}:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
