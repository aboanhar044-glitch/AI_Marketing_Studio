import io
import os
import re
import json
import time
import wave
import random
import subprocess
import base64
import textwrap
import requests
import edge_tts
import numpy as np
import imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import arabic_reshaper
from bidi.algorithm import get_display

try:
    from google import genai as genai_new
except Exception:
    genai_new = None

try:
    import google.generativeai as genai_legacy
except Exception:
    genai_legacy = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
FAL_KEY = re.sub(r"\s+", "", os.getenv("FAL_KEY", "") or "")

if GEMINI_API_KEY and genai_legacy is not None:
    try:
        genai_legacy.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print("Legacy Gemini configure warning:", e)


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
        "Cairo-Regular.ttf",
        "static/Cairo-Regular.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _call_gemini_multimodal(prompt: str, image_paths: list[str] | None = None) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
    valid_imgs = [p for p in (image_paths or []) if p and os.path.exists(p)][:3]

    if genai_new is not None:
        client = genai_new.Client(api_key=GEMINI_API_KEY)
        contents: list = [prompt]
        for img_p in valid_imgs:
            with Image.open(img_p) as im:
                pil_img = im.convert("RGB")
                pil_img.thumbnail((896, 896))
                contents.append(pil_img)

        last_err = None
        for model_name in models_to_try:
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
                if getattr(resp, "text", None):
                    return resp.text.strip()
            except Exception as e:
                last_err = e
        if last_err:
            raise last_err

    if genai_legacy is not None:
        last_err = None
        for model_name in models_to_try:
            try:
                model = genai_legacy.GenerativeModel(model_name)
                parts: list = [prompt]
                for img_p in valid_imgs:
                    with Image.open(img_p) as im:
                        pil_img = im.convert("RGB")
                        pil_img.thumbnail((896, 896))
                        parts.append(pil_img)
                resp = model.generate_content(parts)
                if getattr(resp, "text", None):
                    return resp.text.strip()
            except Exception as e:
                last_err = e
        if last_err:
            raise last_err

    raise RuntimeError("No Gemini SDK available")


def analyze_product_images_with_ai(image_paths: list[str], product_name: str, extra_notes: str) -> str:
    valid_imgs = [p for p in (image_paths or []) if p and os.path.exists(p)]
    if not valid_imgs or not GEMINI_API_KEY:
        return f"Product: {product_name}. Scene idea: {extra_notes}".strip()

    vision_prompt = f"""
Analyze the uploaded product image(s) carefully for a 9:16 vertical commercial video.
Product name: {product_name}
User scene / notes: {extra_notes if extra_notes else "None"}

Describe concisely in English:
1. Exact visual appearance of the product (type, shape, dominant colors, any visible labels/logos/style).
2. Best realistic commercial scenes and human actions that fit the product and user notes.
Keep it factual to the image and suitable for video generation prompts.
"""
    try:
        return _call_gemini_multimodal(vision_prompt, valid_imgs)
    except Exception as e:
        print("Gemini vision analysis fallback:", e)
        return f"Product: {product_name}. Scene idea: {extra_notes}".strip()


def generate_script_and_dialogue(data: dict, visual_context: str = "") -> dict:
    store_name = data["store_name"].strip()
    phone = data["phone"].strip()
    address = data["address"].strip()
    product_name = data["product_name"].strip()
    product_details = data["product_details"].strip()
    video_style = data["video_style"].strip()
    store_need = data["store_need"].strip()
    extra_notes = data.get("extra_notes", "").strip()
    duration = int(data.get("duration", 30))

    scene_count = max(3, min(6, duration // 6))

    prompt = f"""
أنت مخرج إعلانات ومؤلف حوارات تسويقية احترافية لمنصات TikTok و Reels و Shorts بأسلوب إنتاج سينمائي.
صمم سيناريو إعلانياً قصصياً واقعياً ومشوّقاً باللغة العربية بناءً على:
- اسم المتجر / الشركة: {store_name}
- أرقام التواصل: {phone}
- العنوان: {address}
- المنتج / الخدمة: {product_name}
- مواصفات وتفاصيل المنتج: {product_details}
- نوع وأسلوب الفيديو: {video_style}
- احتياج صاحب المتجر الأساسي: {store_need}
- فكرة المشاهد المطلوبة من المستخدم: {extra_notes if extra_notes else "لا يوجد"}
- التحليل البصري للمنتج من الصور: {visual_context if visual_context else product_name}
- المدة المستهدفة: {duration} ثانية (مقسمة إلى {scene_count} مشاهد متسلسلة).

أعد النتيجة بصيغة JSON فقط بهذا الشكل:
{{
  "full_voiceover": "نص تعليق صوتي وحوار تسويقي مترابط وطبيعي يضم اسم المتجر والمنتج والتواصل والعنوان",
  "scenes": [
    {{
      "caption": "جملة قصيرة واضحة (3 إلى 7 كلمات)",
      "sub_badge": "وسم مختصر",
      "ai_motion_prompt": "Detailed cinematic vertical 9:16 video motion prompt in English"
    }}
  ]
}}
"""

    if GEMINI_API_KEY:
        try:
            raw = _call_gemini_multimodal(prompt, None)
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            parsed = json.loads(raw)
            if parsed.get("full_voiceover") and isinstance(parsed.get("scenes"), list) and parsed["scenes"]:
                return parsed
        except Exception as e:
            print("Gemini script fallback triggered:", e)

    story_context = extra_notes if extra_notes else f"استخدام {product_name} عملياً وتقديم تجربة موثوقة"

    default_scenes = [
        {
            "caption": f"ليش {product_name} خيارك الأذكى؟",
            "sub_badge": "البداية",
            "ai_motion_prompt": f"Cinematic 9:16 vertical commercial shot of {product_name} ({visual_context[:100]}), smooth dolly-in camera movement, studio lighting"
        },
        {
            "caption": f"تفاصيل تصنع الفرق في {product_name}",
            "sub_badge": "المواصفات",
            "ai_motion_prompt": f"Smooth close-up camera shot around {product_name} highlighting build quality, natural commercial lighting, 9:16 vertical"
        },
        {
            "caption": story_context[:42],
            "sub_badge": "مشهد واقعي",
            "ai_motion_prompt": f"Realistic 9:16 vertical scene showing practical use of {product_name}: {story_context[:100]}, smooth cinematic motion"
        },
        {
            "caption": f"اطلب من {store_name}: {phone}",
            "sub_badge": "تواصل معنا",
            "ai_motion_prompt": f"Clean commercial showcase of {product_name} representing {store_name}, bright modern lighting, 9:16 vertical"
        },
        {
            "caption": f"ننتظركم في {address}",
            "sub_badge": "العنوان",
            "ai_motion_prompt": f"Polished brand finale shot of {product_name} ready for delivery, warm lighting, 9:16 vertical"
        }
    ]

    voiceover = (
        f"لو كنت تدور على {product_name} يعتمد عليه وقت الجد، فهذا العرض لك. "
        f"في {store_name} نوفر لك {product_name}، {product_details if product_details else 'بمواصفات عملية وأداء ثابت'}. "
        f"هدفنا نلبي احتياجك في {store_need}. "
        f"{(extra_notes + '. ') if extra_notes else ''}"
        f"تواصل معنا مباشرة على {phone} أو زرنا في {address}."
    )

    return {
        "full_voiceover": voiceover,
        "scenes": default_scenes[:scene_count]
    }


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
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    bed = 0.02 * envelope * (np.sin(2 * np.pi * 196 * t) + 0.4 * np.sin(2 * np.pi * 293.66 * t))
    pcm = (np.clip(bed, -0.5, 0.5) * 32767.0).astype(np.int16)

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

    bpm = 112.0
    beat_period = 60.0 / bpm
    beat_phase = (t % beat_period) / beat_period
    kick_env = np.exp(-beat_phase * 18.0)
    kick_freq = 52.0 + 55.0 * np.exp(-beat_phase * 24.0)
    sub_pulse = 0.025 * kick_env * np.sin(2 * np.pi * kick_freq * t)

    whoosh = np.zeros(total_samples, dtype=np.float64)
    rng = np.random.default_rng(42)
    white_noise = rng.normal(0, 1, total_samples)
    for transition_t in np.arange(0.0, duration_sec, 5.0):
        mask = np.abs(t - transition_t) < 0.22
        if np.any(mask):
            local_env = np.exp(-((t[mask] - transition_t) / 0.08) ** 2)
            freq_mod = 0.5 + 0.5 * np.sin(2 * np.pi * 10 * (t[mask] - transition_t))
            whoosh[mask] += 0.045 * local_env * white_noise[mask] * freq_mod

    mix = np.clip(sub_pulse + whoosh, -0.6, 0.6)
    pcm = (mix * 32767.0).astype(np.int16)

    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _download_video_url(video_url: str, out_mp4: str, timeout_sec: int = 35) -> bool:
    if not video_url:
        return False
    vid_resp = requests.get(video_url, timeout=timeout_sec)
    if vid_resp.status_code == 200 and len(vid_resp.content) > 1000:
        with open(out_mp4, "wb") as out:
            out.write(vid_resp.content)
        return True
    return False


def try_generate_ai_scene_video(
    image_path: str | None,
    prompt: str,
    out_mp4: str,
    timeout_sec: int = 35,
) -> bool:
    clean_fal_key = re.sub(r"\s+", "", FAL_KEY or "")
    if not clean_fal_key:
        return False

    headers = {
        "Authorization": f"Key {clean_fal_key}",
        "Content-Type": "application/json",
    }

    endpoints: list[tuple[str, dict]] = []

    if image_path and os.path.exists(image_path):
        with Image.open(image_path) as im:
            img_rgb = im.convert("RGB")
            img_rgb.thumbnail((720, 1280))
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=82)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        data_uri = f"data:image/jpeg;base64,{b64}"
        endpoints.append((
            "https://fal.run/fal-ai/kling-video/v1.6/standard/image-to-video",
            {
                "prompt": prompt,
                "image_url": data_uri,
                "aspect_ratio": "9:16",
                "duration": "5",
            },
        ))
    else:
        endpoints.append((
            "https://fal.run/fal-ai/kling-video/v1.6/standard/text-to-video",
            {
                "prompt": prompt,
                "aspect_ratio": "9:16",
                "duration": "5",
            },
        ))

    for url_endpoint, payload in endpoints:
        try:
            r = requests.post(
                url_endpoint,
                json=payload,
                headers=headers,
                timeout=timeout_sec,
            )
            if r.status_code == 200:
                data_json = r.json()
                vid_url = (
                    (data_json.get("video") or {}).get("url")
                    or data_json.get("video_url")
                    or ""
                )
                if _download_video_url(vid_url, out_mp4, timeout_sec=timeout_sec):
                    return True
            else:
                print(f"Fal AI endpoint {url_endpoint} returned {r.status_code}: {r.text[:160]}")
        except Exception as e:
            print(f"Fal AI error on {url_endpoint}:", e)

    return False


def prepare_canvas_background(img_path: str, width: int, height: int) -> tuple[Image.Image, Image.Image]:
    with Image.open(img_path) as im:
        src = im.convert("RGB")

    bg_ratio = max(width / src.width, height / src.height)
    bg = src.resize((int(src.width * bg_ratio), int(src.height * bg_ratio)), Image.Resampling.BILINEAR)
    left = (bg.width - width) // 2
    top = (bg.height - height) // 2
    bg = bg.crop((left, top, left + width, top + height)).filter(ImageFilter.GaussianBlur(radius=14))

    dark_layer = Image.new("RGBA", (width, height), (10, 14, 24, 95))
    bg_rgba = Image.alpha_composite(bg.convert("RGBA"), dark_layer).convert("RGB")

    fg_max_w = int(width * 0.90)
    fg_max_h = int(height * 0.56)
    fg_ratio = min(fg_max_w / src.width, fg_max_h / src.height)
    fg = src.resize((max(1, int(src.width * fg_ratio)), max(1, int(src.height * fg_ratio))), Image.Resampling.BILINEAR)

    return bg_rgba, fg


def _build_static_overlay(width: int, height: int, caption: str, store_name: str, phone: str, address: str) -> Image.Image:
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    top_h = int(height * 0.14)
    for y in range(top_h):
        alpha = int(145 * (1.0 - y / top_h))
        odraw.line([(0, y), (width, y)], fill=(8, 12, 20, alpha))

    bot_start = int(height * 0.72)
    bot_h = height - bot_start
    for y in range(bot_start, height):
        ratio = (y - bot_start) / max(1, bot_h)
        alpha = int(175 * ratio)
        odraw.line([(0, y), (width, y)], fill=(8, 12, 20, alpha))

    scale = width / 1080.0
    f_store = load_font(max(20, int(38 * scale)))
    f_caption = load_font(max(24, int(48 * scale)))
    f_contact = load_font(max(18, int(32 * scale)))

    ar_store = fix_arabic(store_name)
    wrapped_cap = "\n".join(textwrap.wrap(caption, width=26))
    ar_caption = fix_arabic(wrapped_cap)
    ar_contact = fix_arabic(f"{phone}  |  {address}")

    odraw.text((width // 2, int(height * 0.086)), ar_store, font=f_store, fill=(255, 215, 90, 255), anchor="mm")
    odraw.multiline_text(
        (width // 2, int(height * 0.784)),
        ar_caption,
        font=f_caption,
        fill=(255, 255, 255, 255),
        anchor="mm",
        align="center"
    )
    odraw.text(
        (width // 2, int(height * 0.846)),
        ar_contact,
        font=f_contact,
        fill=(220, 230, 245, 245),
        anchor="mm"
    )
    return overlay


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
    layout_mode: int,
    cached_overlay: Image.Image | None = None,
) -> np.ndarray:
    progress = min(1.0, max(0.0, t / max(0.1, scene_duration)))
    smooth_ease = progress * progress * (3.0 - 2.0 * progress)
    motion_type = layout_mode % 4

    if motion_type == 0:
        zoom_bg = 1.03 + 0.05 * smooth_ease
        pan_x = int(18 * (smooth_ease - 0.5))
        pan_y = int(10 * (smooth_ease - 0.5))
    elif motion_type == 1:
        zoom_bg = 1.08 - 0.04 * smooth_ease
        pan_x = int(-18 * (smooth_ease - 0.5))
        pan_y = int(-10 * (smooth_ease - 0.5))
    elif motion_type == 2:
        zoom_bg = 1.04 + 0.03 * smooth_ease
        pan_x = int(24 * (smooth_ease - 0.5))
        pan_y = 0
    else:
        zoom_bg = 1.05 + 0.03 * smooth_ease
        pan_x = int(-24 * (smooth_ease - 0.5))
        pan_y = int(8 * (smooth_ease - 0.5))

    bw, bh = int(width * zoom_bg), int(height * zoom_bg)
    bg_moved = bg_img.resize((bw, bh), Image.Resampling.BILINEAR)
    bx = max(0, min(bw - width, (bw - width) // 2 + pan_x))
    by = max(0, min(bh - height, (bh - height) // 2 + pan_y))
    frame = bg_moved.crop((bx, by, bx + width, by + height)).convert("RGBA")

    if fg_img.width > 2 and fg_img.height > 2:
        fg_scale = 0.97 + 0.04 * smooth_ease
        fw = max(10, int(fg_img.width * fg_scale))
        fh = max(10, int(fg_img.height * fg_scale))
        fg_resized = fg_img.resize((fw, fh), Image.Resampling.BILINEAR)
        fx = max(24, min(width - fw - 24, (width - fw) // 2 + int(pan_x * 0.35)))
        fy = max(int(height * 0.14), min(int(height * 0.68) - fh, (height // 2 - int(height * 0.03)) - fh // 2 + int(pan_y * 0.35)))
        frame.paste(fg_resized, (fx, fy))

    overlay = cached_overlay or _build_static_overlay(width, height, caption, store_name, phone, address)
    frame = Image.alpha_composite(frame, overlay)
    return np.array(frame.convert("RGB"))


def mux_audio_with_video(silent_mp4: str, voice_mp3: str, sfx_wav: str, final_mp4: str):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", silent_mp4,
        "-i", voice_mp3,
        "-i", sfx_wav,
        "-filter_complex", "[1:a]volume=1.2[a1];[2:a]volume=0.55[a2];[a1][a2]amix=inputs=2:duration=first[aout]",
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "24",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        final_mp4
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg mux error: {proc.stderr}")


async def produce_complete_marketing_video(job_id: str, data: dict, image_paths: list[str], output_dir: str) -> dict:
    W, H = 720, 1280
    FPS = 20
    max_ai_budget_sec = 45.0
    ai_start_ts = time.monotonic()
    ai_disabled_for_budget = False

    visual_context = analyze_product_images_with_ai(
        image_paths=image_paths,
        product_name=data.get("product_name", ""),
        extra_notes=data.get("extra_notes", "")
    )
    script_obj = generate_script_and_dialogue(data, visual_context=visual_context)
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
    ai_clips_generated = 0
    valid_image_paths = [p for p in (image_paths or []) if p and os.path.exists(p)]

    writer = imageio.get_writer(silent_mp4, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p")
    try:
        for s_idx, scene in enumerate(scenes):
            img_path = valid_image_paths[s_idx % len(valid_image_paths)] if valid_image_paths else None
            ai_vid_path = os.path.join(output_dir, f"{job_id}_ai_{s_idx}.mp4")

            remaining_budget = max_ai_budget_sec - (time.monotonic() - ai_start_ts)
            if not ai_disabled_for_budget and remaining_budget >= 12.0:
                per_scene_timeout = min(35, max(12, int(remaining_budget)))
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
                ai_frames = [Image.fromarray(f).resize((W, H), Image.Resampling.BILINEAR) for f in reader]
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

            caption_val = scene.get("caption", data["product_name"])
            cached_overlay = _build_static_overlay(
                W, H, caption_val, data["store_name"], data["phone"], data["address"]
            )

            n_frames = max(1, int(round(scene_duration * FPS)))
            for f_i in range(n_frames):
                t_local = f_i / FPS
                if ai_frames:
                    base_frame = ai_frames[f_i % len(ai_frames)]
                    frame_arr = render_frame_with_store_layout(
                        base_frame, Image.new("RGB", (1, 1)),
                        W, H, t_local, scene_duration,
                        caption=caption_val,
                        badge_text=scene.get("sub_badge", data["store_name"]),
                        store_name=data["store_name"],
                        phone=data["phone"],
                        address=data["address"],
                        layout_mode=base_layout + s_idx,
                        cached_overlay=cached_overlay,
                    )
                else:
                    frame_arr = render_frame_with_store_layout(
                        bg_img, fg_img, W, H, t_local, scene_duration,
                        caption=caption_val,
                        badge_text=scene.get("sub_badge", data["store_name"]),
                        store_name=data["store_name"],
                        phone=data["phone"],
                        address=data["address"],
                        layout_mode=base_layout + s_idx,
                        cached_overlay=cached_overlay,
                    )
                writer.append_data(frame_arr)
    finally:
        writer.close()

    mux_audio_with_video(silent_mp4, voice_mp3, sfx_wav, final_mp4)

    for temp_f in (silent_mp4, voice_mp3, sfx_wav):
        try:
            if os.path.exists(temp_f):
                os.remove(temp_f)
        except Exception:
            pass

    return {
        "final_mp4": final_mp4,
        "voiceover_script": voice_text,
        "layout_mode": base_layout,
        "duration_seconds": round(total_duration, 1),
        "ai_clips_generated": ai_clips_generated,
        "visual_analysis": visual_context
    }
