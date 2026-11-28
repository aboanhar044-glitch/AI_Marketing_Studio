import os
import re
import uuid
import time
import random
import socket
import shutil
import subprocess
from typing import List

import imageio
import imageio_ffmpeg
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from presets_catalog import (
    get_all_video_styles,
    STORE_NEEDS_OPTIONS,
    VOICE_PROFILES,
    STORE_DISPLAY_MODES,
)
from video_engine import (
    produce_complete_marketing_video,
    analyze_product_images_with_ai,
    generate_script_and_dialogue,
    generate_voice_mp3,
    generate_sound_effects_wav,
    _build_static_overlay,
    prepare_canvas_background,
    render_frame_with_store_layout,
    mux_audio_with_video,
    try_generate_ai_scene_video,
)

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
        W, H = 720, 1280
        FPS = 20
        max_ai_budget_sec = 40.0
        ai_start_ts = time.monotonic()
        ai_disabled_for_budget = False

        _set_job_state(
            job_id,
            status="processing",
            progress=20,
            stage="تحليل صور المنتج بالذكاء الاصطناعي...",
        )
        visual_context = analyze_product_images_with_ai(
            image_paths=saved_paths,
            product_name=payload.get("product_name", ""),
            extra_notes=payload.get("extra_notes", ""),
        )

        _set_job_state(
            job_id,
            status="processing",
            progress=40,
            stage="تأليف السيناريو والحوار التسويقي...",
        )
        script_obj = generate_script_and_dialogue(payload, visual_context=visual_context)
        voice_text = script_obj["full_voiceover"]
        scenes = script_obj.get("scenes", [])
        if not scenes:
            scenes = [
                {
                    "caption": payload["product_name"],
                    "sub_badge": payload["store_name"],
                    "ai_motion_prompt": payload["product_name"],
                }
            ]

        _set_job_state(
            job_id,
            status="processing",
            progress=60,
            stage="توليد التعليق الصوتي والمؤثرات الصوتية...",
        )
        voice_mp3 = os.path.join(OUTPUT_DIR, f"{job_id}_voice.mp3")
        sfx_wav = os.path.join(OUTPUT_DIR, f"{job_id}_sfx.wav")
        silent_mp4 = os.path.join(OUTPUT_DIR, f"{job_id}_silent.mp4")
        final_mp4 = os.path.join(OUTPUT_DIR, f"{job_id}_final.mp4")

        await generate_voice_mp3(voice_text, payload["voice_id"], voice_mp3)

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        probe = subprocess.run([ffmpeg_exe, "-i", voice_mp3], capture_output=True, text=True)
        dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", probe.stderr)
        if dur_match:
            h, m, s = int(dur_match.group(1)), int(dur_match.group(2)), float(dur_match.group(3))
            voice_duration = h * 3600 + m * 60 + s
        else:
            voice_duration = float(payload.get("duration", 25))

        requested_max = float(payload.get("duration", 60))
        total_duration = max(
            8.0,
            min(60.0, max(voice_duration + 0.6, min(requested_max, voice_duration + 1.5))),
        )

        generate_sound_effects_wav(total_duration, sfx_wav)

        _set_job_state(
            job_id,
            status="processing",
            progress=75,
            stage="توليد المشاهد المتحركة وتصميم بطاقات المتجر...",
        )

        requested_mode = str(payload.get("display_mode", "auto")).strip()
        if requested_mode.isdigit():
            base_layout = int(requested_mode) % 6
        else:
            base_layout = random.randint(0, 5)

        scene_duration = total_duration / len(scenes)
        ai_clips_generated = 0
        valid_image_paths = [p for p in (saved_paths or []) if p and os.path.exists(p)]

        writer = imageio.get_writer(
            silent_mp4,
            fps=FPS,
            codec="libx264",
            quality=7,
            pixelformat="yuv420p",
        )
        try:
            total_scenes = len(scenes)
            for s_idx, scene in enumerate(scenes):
                scene_pct = 75 + int(((s_idx + 1) / max(1, total_scenes)) * 15)
                _set_job_state(
                    job_id,
                    status="processing",
                    progress=min(90, scene_pct),
                    stage=f"معالجة المشهد {s_idx + 1} من {total_scenes}...",
                )

                img_path = valid_image_paths[s_idx % len(valid_image_paths)] if valid_image_paths else None
                ai_vid_path = os.path.join(OUTPUT_DIR, f"{job_id}_ai_{s_idx}.mp4")

                remaining_budget = max_ai_budget_sec - (time.monotonic() - ai_start_ts)
                if not ai_disabled_for_budget and remaining_budget >= 10.0:
                    per_scene_timeout = min(30, max(10, int(remaining_budget)))
                    used_ai_clip = try_generate_ai_scene_video(
                        img_path,
                        scene.get("ai_motion_prompt", ""),
                        ai_vid_path,
                        timeout_sec=per_scene_timeout,
                    )
                    if not used_ai_clip and (time.monotonic() - ai_start_ts) >= max_ai_budget_sec:
                        ai_disabled_for_budget = True
                else:
                    used_ai_clip = False

                ai_frames = None
                if used_ai_clip and os.path.exists(ai_vid_path):
                    reader = imageio.get_reader(ai_vid_path)
                    ai_frames = [
                        Image.fromarray(f).resize((W, H), Image.Resampling.BILINEAR)
                        for f in reader
                    ]
                    reader.close()
                    if ai_frames:
                        ai_clips_generated += 1
                    else:
                        ai_frames = None
                    try:
                        os.remove(ai_vid_path)
                    except Exception:
                        pass

                if not ai_frames and img_path:
                    bg_img, fg_img = prepare_canvas_background(img_path, W, H)
                elif not ai_frames:
                    bg_img = Image.new("RGB", (W, H), (14, 20, 34))
                    fg_img = Image.new("RGB", (1, 1))

                caption_val = scene.get("caption", payload["product_name"])
                cached_overlay = _build_static_overlay(
                    W,
                    H,
                    caption_val,
                    payload["store_name"],
                    payload["phone"],
                    payload["address"],
                )

                n_frames = max(1, int(round(scene_duration * FPS)))
                for f_i in range(n_frames):
                    t_local = f_i / FPS
                    if ai_frames:
                        base_frame = ai_frames[f_i % len(ai_frames)]
                        frame_arr = render_frame_with_store_layout(
                            base_frame,
                            Image.new("RGB", (1, 1)),
                            W,
                            H,
                            t_local,
                            scene_duration,
                            caption=caption_val,
                            badge_text=scene.get("sub_badge", payload["store_name"]),
                            store_name=payload["store_name"],
                            phone=payload["phone"],
                            address=payload["address"],
                            layout_mode=base_layout + s_idx,
                            cached_overlay=cached_overlay,
                        )
                    else:
                        frame_arr = render_frame_with_store_layout(
                            bg_img,
                            fg_img,
                            W,
                            H,
                            t_local,
                            scene_duration,
                            caption=caption_val,
                            badge_text=scene.get("sub_badge", payload["store_name"]),
                            store_name=payload["store_name"],
                            phone=payload["phone"],
                            address=payload["address"],
                            layout_mode=base_layout + s_idx,
                            cached_overlay=cached_overlay,
                        )
                    writer.append_data(frame_arr)
        finally:
            writer.close()

        _set_job_state(
            job_id,
            status="processing",
            progress=94,
            stage="دمج الصوت والمؤثرات وإخراج ملف MP4 النهائي...",
        )
        mux_audio_with_video(silent_mp4, voice_mp3, sfx_wav, final_mp4)

        for temp_f in (silent_mp4, voice_mp3, sfx_wav):
            try:
                if os.path.exists(temp_f):
                    os.remove(temp_f)
            except Exception:
                pass

        fname = os.path.basename(final_mp4)
        _set_job_state(
            job_id,
            status="completed",
            progress=100,
            stage="اكتمل إنتاج الفيديو بنجاح",
            video_url=f"/outputs/{fname}",
            filename=fname,
            voiceover_script=voice_text,
            layout_mode=base_layout,
            duration_seconds=round(total_duration, 1),
            ai_clips_generated=ai_clips_generated,
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
