import os
import re
import json
import wave
import random
import subprocess
import base64
import textwrap
import math
import requests
import edge_tts
import numpy as np
import imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import arabic_reshaper
from bidi.algorithm import get_display
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
FAL_KEY = os.getenv("FAL_KEY", "").strip()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def fix_arabic(text: str) -> str:
    if not text:
        return ""
    try:
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)
    except Exception:
        return str(text)


def load_font(size: int):
    candidates = [
        "Cairo-Bold.ttf",
        "static/Cairo-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def generate_script_and_dialogue(data: dict) -> dict:
    store_name = data["store_name"].strip()
    phone = data["phone"].strip()
    address = data["address"].strip()
    product_name = data["product_name"].strip()
    product_details = data["product_details"].strip()
    video_style = data["video_style"].strip()
    store_need = data["store_need"].strip()
    extra_notes = data.get("extra_notes", "").strip()
    duration = int(data.get("duration", 30))

    scene_count = max(4, min(10, duration // 5))

    prompt = f"""
أنت مخرج إعلانات ومؤلف حوارات تسويقية احترافية لمنصات TikTok و Reels و Shorts.
قم ببناء سيناريو إعلاني متكامل وحوار بشري حيوي باللغة العربية بناءً على البيانات التالية:
- اسم المتجر / الشركة: {store_name}
- أرقام التواصل: {phone}
- العنوان: {address}
- المنتج / الخدمة: {product_name}
- مواصفات وتفاصيل المنتج: {product_details}
- نوع وأسلوب الفيديو: {video_style}
- احتياج صاحب المتجر الأساسي: {store_need}
- ملاحظات إضافية: {extra_notes if extra_notes else "لا يوجد"}
- المدة المستهدفة: {duration} ثانية (مقسمة إلى {scene_count} مشاهد).

أعد النتيجة بصيغة JSON فقط بهذا الشكل:
{{
  "full_voiceover": "النص الكامل للتعليق الصوتي والحوار البشري متضمنا اسم المتجر وأرقام التواصل والعنوان",
  "scenes": [
    {{
      "caption": "عنوان أو جملة قصيرة تظهر على الشاشة (3 إلى 7 كلمات)",
      "sub_badge": "وسم قصير للمشهد",
      "ai_motion_prompt": "Detailed cinematic prompt in English describing realistic human presenter or dynamic product movement in 9:16 vertical frame"
    }}
  ]
}}
"""

    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(prompt)
            raw = resp.text.strip()
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            parsed = json.loads(raw)
            if parsed.get("full_voiceover") and parsed.get("scenes"):
                return parsed
        except Exception as e:
            print("Gemini fallback triggered:", e)

    default_scenes = [
        {
            "caption": f"ليش الكل يتكلم عن {product_name}؟",
            "sub_badge": "بداية قوية",
            "ai_motion_prompt": f"Dynamic commercial shot of {product_name}, dramatic rim lighting, smooth camera push in, 9:16"
        },
        {
            "caption": f"الحل العملي من {store_name}",
            "sub_badge": store_name,
            "ai_motion_prompt": f"Modern lifestyle showcase of {product_name} with warm studio lighting and subtle motion blur"
        },
        {
            "caption": f"{product_details[:45]}" if product_details else "جودة واعتمادية عالية",
            "sub_badge": "مميزات حقيقية",
            "ai_motion_prompt": f"Close up detail view of {product_name} with sleek highlight sweep and depth of field"
        },
        {
            "caption": f"تواصل الآن: {phone}",
            "sub_badge": "اطلب الآن",
            "ai_motion_prompt": f"Energetic promotional product highlight for {product_name} with bright modern accents"
        },
        {
            "caption": f"زورونا في {address}",
            "sub_badge": address[:28],
            "ai_motion_prompt": f"Clean brand finale shot of {product_name} ready for delivery and customer satisfaction"
        }
    ]

    voiceover = (
        f"أهلاً بك! إذا كنت تبحث عن {product_name} بمستوى يعتمد عليه، فهذا الفيديو لك. "
        f"في {store_name} نوفر لك {product_name}، حيث {product_details if product_details else 'نهتم بأدق التفاصيل لتقديم أفضل تجربة'}. "
        f"هدفنا الأساسي هو {store_need}. "
        f"{('ملاحظة مهمة: ' + extra_notes + '. ') if extra_notes else ''}"
        f"للتواصل والحجز أو الاستفسار اتصل بنا مباشرة في {store_name} على الأرقام {phone}، "
        f"أو تفضل بزيارتنا في عنواننا: {address}. ننتظرك بكل ترحيب!"
    )

    return {
        "full_voiceover": voiceover,
        "scenes": default_scenes[:scene_count]
    }


async def generate_voice_mp3(text: str, voice_id: str, output_mp3: str):
    communicate = edge_tts.Communicate(text, voice_id)
    await communicate.save(output_mp3)


def generate_sound_effects_wav(duration_sec: float, output_wav: str, sample_rate: int = 44100):
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    bpm = 118.0
    beat_period = 60.0 / bpm
    beat_phase = (t % beat_period) / beat_period
    kick_env = np.exp(-beat_phase * 18.0)
    kick_freq = 55.0 + 65.0 * np.exp(-beat_phase * 25.0)
    sub_pulse = 0.045 * kick_env * np.sin(2 * np.pi * kick_freq * t)

    whoosh = np.zeros(total_samples, dtype=np.float64)
    rng = np.random.default_rng(42)
    white_noise = rng.normal(0, 1, total_samples)
    for transition_t in np.arange(0.0, duration_sec, 5.0):
        mask = np.abs(t - transition_t) < 0.25
        if np.any(mask):
            local_env = np.exp(-((t[mask] - transition_t) / 0.09) ** 2)
            freq_mod = 0.5 + 0.5 * np.sin(2 * np.pi * 12 * (t[mask] - transition_t))
            whoosh[mask] += 0.08 * local_env * white_noise[mask] * freq_mod

    mix = np.clip(sub_pulse + whoosh, -0.8, 0.8)
    pcm = (mix * 32767.0).astype(np.int16)

    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def try_generate_ai_scene_video(image_path: str, prompt: str, out_mp4: str) -> bool:
    if not FAL_KEY:
        return False
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        headers = {
            "Authorization": f"Key {FAL_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "image_url": f"data:image/jpeg;base64,{b64}",
            "aspect_ratio": "9:16",
async def generate_voice_mp3(text: str, voice_id: str, output_mp3: str):
    try:
        communicate = edge_tts.Communicate(text, voice_id)
        await communicate.save(output_mp3)
        if os.path.exists(output_mp3) and os.path.getsize(output_mp3) > 500:
            return
    except Exception as e:
        print("edge-tts error, using fallback audio:", e)

    sample_rate = 44100
    est_seconds = max(6.0, min(55.0, len(text) / 14.0))
    total_samples = int(est_seconds * sample_rate)
    t = np.linspace(0, est_seconds, total_samples, endpoint=False)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    bed = 0.03 * envelope * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 330 * t))
    pcm = (np.clip(bed, -0.7, 0.7) * 32767.0).astype(np.int16)

    tmp_wav = output_mp3 + ".wav"
    with wave.open(tmp_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [ffmpeg_exe, "-y", "-i", tmp_wav, "-codec:a", "libmp3lame", "-q:a", "4", output_mp3],
        capture_output=True,
        text=True
    )
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)
    if proc.returncode != 0 or not os.path.exists(output_mp3):
        raise RuntimeError(f"Fallback audio generation failed: {proc.stderr}")


def generate_sound_effects_wav(duration_sec: float, output_wav: str, sample_rate: int = 44100):
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    bpm = 118.0
    beat_period = 60.0 / bpm
    beat_phase = (t % beat_period) / beat_period
    kick_env = np.exp(-beat_phase * 18.0)
    kick_freq = 55.0 + 65.0 * np.exp(-beat_phase * 25.0)
    sub_pulse = 0.045 * kick_env * np.sin(2 * np.pi * kick_freq * t)

    whoosh = np.zeros(total_samples, dtype=np.float64)
    rng = np.random.default_rng(42)
    white_noise = rng.normal(0, 1, total_samples)
    for transition_t in np.arange(0.0, duration_sec, 5.0):
        mask = np.abs(t - transition_t) < 0.25
        if np.any(mask):
            local_env = np.exp(-((t[mask] - transition_t) / 0.09) ** 2)
            freq_mod = 0.5 + 0.5 * np.sin(2 * np.pi * 12 * (t[mask] - transition_t))
            whoosh[mask] += 0.08 * local_env * white_noise[mask] * freq_mod

    mix = np.clip(sub_pulse + whoosh, -0.8, 0.8)
    pcm = (mix * 32767.0).astype(np.int16)

    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def try_generate_ai_scene_video(image_path: str, prompt: str, out_mp4: str) -> bool:
    if not FAL_KEY:
        return False
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        headers = {
            "Authorization": f"Key {FAL_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "image_url": f"data:image/jpeg;base64,{b64}",
            "aspect_ratio": "9:16",
            "duration": "5"
        }
        r = requests.post(
            "https://fal.run/fal-ai/kling-video/v1.6/standard/image-to-video",
            json=payload,
            headers=headers,
            timeout=180
        )
        if r.status_code == 200:
            url = r.json().get("video", {}).get("url")
            if url:
                vid_bytes = requests.get(url, timeout=120).content
                with open(out_mp4, "wb") as out:
                    out.write(vid_bytes)
                return True
    except Exception as e:
        print("Fal AI video error:", e)
    return False



def prepare_canvas_background(img_path: str, width: int, height: int) -> tuple[Image.Image, Image.Image]:
    src = Image.open(img_path).convert("RGB")

    bg_ratio = max(width / src.width, height / src.height)
    bg = src.resize((int(src.width * bg_ratio), int(src.height * bg_ratio)), Image.Resampling.LANCZOS)
    left = (bg.width - width) // 2
    top = (bg.height - height) // 2
    bg = bg.crop((left, top, left + width, top + height)).filter(ImageFilter.GaussianBlur(radius=28))

    dark_layer = Image.new("RGBA", (width, height), (8, 12, 24, 125))
    bg_rgba = Image.alpha_composite(bg.convert("RGBA"), dark_layer).convert("RGB")

    fg_max_w = int(width * 0.84)
    fg_max_h = int(height * 0.46)
    fg_ratio = min(fg_max_w / src.width, fg_max_h / src.height)
    fg = src.resize((max(1, int(src.width * fg_ratio)), max(1, int(src.height * fg_ratio))), Image.Resampling.LANCZOS)

    return bg_rgba, fg


def render_frame_with_store_layout(
    bg_img: Image.Image,
    fg_img: Image.Image,
    width: int,
    height: int,
    t: float,
    scene_duration: float,
    caption: str,
    badge_text: str,
    store_name: str,
    phone: str,
    address: str,
    layout_mode: int
) -> np.ndarray:
    progress = min(1.0, max(0.0, t / max(0.1, scene_duration)))
    pulse = 1.0 + 0.025 * math.sin(2 * math.pi * t * 2.2)

    zoom_bg = 1.04 + 0.06 * progress
    bw, bh = int(width * zoom_bg), int(height * zoom_bg)
    bg_moved = bg_img.resize((bw, bh), Image.Resampling.BILINEAR)
    bx = (bw - width) // 2 + int(18 * math.sin(t * 1.4))
    by = (bh - height) // 2 + int(14 * math.cos(t * 1.1))
    frame = bg_moved.crop((bx, by, bx + width, by + height)).convert("RGBA")

    if fg_img.width > 2 and fg_img.height > 2:
        fg_scale = (0.95 + 0.08 * progress) * pulse
        fw = max(10, int(fg_img.width * fg_scale))
        fh = max(10, int(fg_img.height * fg_scale))
        fg_resized = fg_img.resize((fw, fh), Image.Resampling.BILINEAR)
        fx = (width - fw) // 2 + int(14 * math.sin(t * 2.0))
        fy = max(230, min(height - 760 - fh // 2, (height // 2 - 40) - fh // 2 + int(12 * math.sin(t * 1.7))))
        frame.paste(fg_resized, (fx, fy))

    draw = ImageDraw.Draw(frame)

    palettes = [
        {"accent": (244, 63, 94, 240), "card": (15, 23, 42, 225), "highlight": (251, 191, 36, 255), "text": (255, 255, 255, 255)},
        {"accent": (16, 185, 129, 240), "card": (6, 38, 32, 225), "highlight": (110, 231, 183, 255), "text": (255, 255, 255, 255)},
        {"accent": (245, 158, 11, 240), "card": (24, 18, 12, 228), "highlight": (253, 224, 71, 255), "text": (255, 255, 255, 255)},
        {"accent": (56, 189, 248, 240), "card": (12, 25, 48, 225), "highlight": (125, 211, 252, 255), "text": (255, 255, 255, 255)},
        {"accent": (192, 132, 252, 240), "card": (28, 16, 48, 225), "highlight": (233, 213, 255, 255), "text": (255, 255, 255, 255)},
        {"accent": (251, 113, 133, 240), "card": (20, 20, 28, 228), "highlight": (254, 205, 211, 255), "text": (255, 255, 255, 255)},
    ]
    pal = palettes[layout_mode % len(palettes)]

    f_store = load_font(40)
    f_caption = load_font(52)
    f_contact = load_font(36)
    f_addr = load_font(31)
    f_badge = load_font(28)

    ar_store = fix_arabic(store_name)
    ar_phone = fix_arabic(f"تواصل: {phone}")
    ar_address = fix_arabic(f"العنوان: {address}")
    wrapped_cap = "\n".join(textwrap.wrap(caption, width=24))
    ar_caption = fix_arabic(wrapped_cap)
    ar_badge = fix_arabic(badge_text)

    mode = layout_mode % 6

    top_y = 135
    if mode in (0, 2, 4):
        draw.rounded_rectangle([55, top_y, width - 55, top_y + 86], radius=24, fill=pal["card"], outline=pal["accent"], width=3)
        draw.text((width // 2, top_y + 43), ar_store, font=f_store, fill=pal["highlight"], anchor="mm")
    else:
        draw.rounded_rectangle([55, top_y, width - 55, top_y + 84], radius=42, fill=pal["accent"])
        draw.text((width // 2, top_y + 42), ar_store, font=f_store, fill=pal["text"], anchor="mm")

    if badge_text:
        draw.rounded_rectangle([75, top_y + 96, 360, top_y + 142], radius=16, fill=pal["card"], outline=pal["highlight"], width=2)
        draw.text((217, top_y + 119), ar_badge, font=f_badge, fill=pal["text"], anchor="mm")

    cap_box_top = height - 650
    cap_box_bottom = height - 475
    draw.rounded_rectangle([55, cap_box_top, width - 55, cap_box_bottom], radius=26, fill=pal["card"], outline=pal["highlight"], width=2)
    draw.multiline_text((width // 2, (cap_box_top + cap_box_bottom) // 2), ar_caption, font=f_caption, fill=pal["text"], anchor="mm", align="center")

    right_safe = width - 125
    center_safe_x = (55 + right_safe) // 2

    if mode in (0, 1, 3):
        draw.rounded_rectangle([55, height - 455, right_safe, height - 265], radius=24, fill=pal["accent"])
        draw.text((center_safe_x, height - 405), ar_phone, font=f_contact, fill=pal["text"], anchor="mm")
        draw.line([(95, height - 355), (right_safe - 40, height - 355)], fill=pal["text"], width=2)
        draw.text((center_safe_x, height - 308), ar_address, font=f_addr, fill=pal["text"], anchor="mm")
    else:
        draw.rounded_rectangle([55, height - 455, right_safe, height - 365], radius=18, fill=pal["card"], outline=pal["accent"], width=3)
        draw.text((center_safe_x, height - 410), ar_phone, font=f_contact, fill=pal["highlight"], anchor="mm")

        draw.rounded_rectangle([55, height - 352, right_safe, height - 265], radius=18, fill=pal["accent"])
        draw.text((center_safe_x, height - 308), ar_address, font=f_addr, fill=pal["text"], anchor="mm")

    return np.array(frame.convert("RGB"))


def mux_audio_with_video(silent_mp4: str, voice_mp3: str, sfx_wav: str, final_mp4: str):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", silent_mp4,
        "-i", voice_mp3,
        "-i", sfx_wav,
        "-filter_complex", "[1:a]volume=1.15[a1];[2:a]volume=0.85[a2];[a1][a2]amix=inputs=2:duration=first[aout]",
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        final_mp4
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg mux error: {proc.stderr}")


async def produce_complete_marketing_video(job_id: str, data: dict, image_paths: list[str], output_dir: str) -> dict:
    W, H = 1080, 1920
    FPS = 24

    script_obj = generate_script_and_dialogue(data)
    voice_text = script_obj["full_voiceover"]
    scenes = script_obj.get("scenes", [])
    if not scenes:
        scenes = [{"caption": data["product_name"], "sub_badge": data["store_name"], "ai_motion_prompt": data["product_name"]}]

    voice_mp3 = os.path.join(output_dir, f"{job_id}_voice.mp3")
    sfx_wav = os.path.join(output_dir, f"{job_id}_sfx.wav")
    silent_mp4 = os.path.join(output_dir, f"{job_id}_silent.mp4")
    final_mp4 = os.path.join(output_dir, f"{job_id}_final.mp4")

    await generate_voice_mp3(voice_text, data["voice_id"], voice_mp3)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([ffmpeg_exe, "-i", voice_mp3], capture_output=True, text=True)
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", probe.stderr)
    if dur_match:
        h, m, s = int(dur_match.group(1)), int(dur_match.group(2)), float(dur_match.group(3))
        voice_duration = h * 3600 + m * 60 + s
    else:
        voice_duration = float(data.get("duration", 25))

    requested_max = float(data.get("duration", 60))
    total_duration = max(8.0, min(60.0, max(voice_duration + 0.6, min(requested_max, voice_duration + 1.5))))

    generate_sound_effects_wav(total_duration, sfx_wav)

    requested_mode = str(data.get("display_mode", "auto")).strip()
    if requested_mode.isdigit():
        base_layout = int(requested_mode) % 6
    else:
        base_layout = random.randint(0, 5)

    scene_duration = total_duration / len(scenes)

    writer = imageio.get_writer(silent_mp4, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p")
    try:
        for s_idx, scene in enumerate(scenes):
            img_path = image_paths[s_idx % len(image_paths)]
            ai_vid_path = os.path.join(output_dir, f"{job_id}_ai_{s_idx}.mp4")
            used_ai_clip = try_generate_ai_scene_video(img_path, scene.get("ai_motion_prompt", ""), ai_vid_path)

            if used_ai_clip and os.path.exists(ai_vid_path):
                reader = imageio.get_reader(ai_vid_path)
                ai_frames = [Image.fromarray(f).resize((W, H), Image.Resampling.LANCZOS) for f in reader]
                reader.close()
                if not ai_frames:
                    bg_img, fg_img = prepare_canvas_background(img_path, W, H)
                    ai_frames = None
            else:
                bg_img, fg_img = prepare_canvas_background(img_path, W, H)
                ai_frames = None

            n_frames = max(1, int(round(scene_duration * FPS)))
            for f_i in range(n_frames):
                t_local = f_i / FPS
                if ai_frames:
                    base_frame = ai_frames[f_i % len(ai_frames)]
                    overlay_arr = render_frame_with_store_layout(
                        base_frame, Image.new("RGB", (1, 1)),
                        W, H, t_local, scene_duration,
                        caption=scene.get("caption", data["product_name"]),
                        badge_text=scene.get("sub_badge", data["store_name"]),
                        store_name=data["store_name"],
                        phone=data["phone"],
                        address=data["address"],
                        layout_mode=base_layout + s_idx
                    )
                    writer.append_data(overlay_arr)
                else:
                    frame_arr = render_frame_with_store_layout(
                        bg_img, fg_img, W, H, t_local, scene_duration,
                        caption=scene.get("caption", data["product_name"]),
                        badge_text=scene.get("sub_badge", data["store_name"]),
                        store_name=data["store_name"],
                        phone=data["phone"],
                        address=data["address"],
                        layout_mode=base_layout + s_idx
                    )
                    writer.append_data(frame_arr)
    finally:
        writer.close()

    mux_audio_with_video(silent_mp4, voice_mp3, sfx_wav, final_mp4)

    return {
        "final_mp4": final_mp4,
        "voiceover_script": voice_text,
        "layout_mode": base_layout,
        "duration_seconds": round(total_duration, 1)
    }
