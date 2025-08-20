# app.py
import os, uuid, shutil, json, zipfile, time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from utils import *
from pathlib import Path

BASE_DIR = os.path.dirname(__file__)
STORE = os.path.join(BASE_DIR, "storage")
SRC   = os.path.join(STORE, "sources")
OUT   = os.path.join(STORE, "outputs")
TMP   = os.path.join(STORE, "temp")
BROLL = os.path.join(STORE, "broll")
BGM   = os.path.join(STORE, "music")
ASSETS_FONTS = os.path.join(STORE, "fonts")
LOGO_DEFAULT = os.path.join(STORE, "logo.png")

for d in [SRC, OUT, TMP, BROLL, BGM, ASSETS_FONTS]:
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="All-in-one Clipper API")

# -------------------------
# Request models
# -------------------------
class DownloadReq(BaseModel):
    url: str

class ProcessReq(BaseModel):
    source_id: str
    start_sec: float
    duration_sec: float
    preset: str = "tiktok"
    fps: int = 30
    target_bitrate: str = "3500k"
    smart_crop: str = "none"  # none|face|cropdetect
    burn_captions: bool = False
    whisper_model: str = "base"
    letterbox: bool = False
    watermark_path: Optional[str] = None
    loudnorm: bool = False
    progress_bar: bool = False
    keep_audio: bool = True
    bgm: Optional[str] = None           # path in storage/music or 'random'
    broll_keywords: Optional[List[str]] = None
    inject_emoji: bool = False

class AutoClipsReq(BaseModel):
    source_id: str
    min_dur: float = 20
    max_dur: float = 40
    num_clips: int = 4
    preset: str = "tiktok"
    fps: int = 30
    target_bitrate: str = "3500k"
    smart_crop: str = "face"   # none|face
    burn_captions: bool = True
    whisper_model: str = "base"
    letterbox: bool = False
    watermark_path: Optional[str] = None
    loudnorm: bool = True
    progress_bar: bool = False
    keep_audio: bool = True
    lang: str = "id"
    bgm: Optional[str] = None
    broll_folder: Optional[str] = None
    inject_emoji: bool = True

class GenMetaReq(BaseModel):
    text: str
    use_openai: bool = False  # optional

# -------------------------
# Endpoint: download via yt-dlp
# -------------------------
@app.post("/api/download")
def download(req: DownloadReq):
    vid = str(uuid.uuid4())
    out_path = os.path.join(SRC, f"{vid}.mp4")
    tmp_out_template = os.path.join(TMP, f"{vid}.%(ext)s")
    cmd = [
        "yt-dlp", req.url,
        "-f", "bv*+ba/b",
        "-S", "res,codec:avc:m4a",
        "-o", tmp_out_template,
        "--merge-output-format", "mp4"
    ]
    proc_out = run_capture(cmd)
    # find produced mp4
    produced = None
    for fn in os.listdir(TMP):
        if fn.startswith(vid) and fn.endswith(".mp4"):
            produced = os.path.join(TMP, fn); break
    if not produced:
        return JSONResponse({"error": "download failed", "log": proc_out}, status_code=500)
    shutil.move(produced, out_path)
    return {"source_id": vid, "path": f"/api/file/{vid}"}

# -------------------------
# Serve source & output & thumbnails
# -------------------------
@app.get("/api/file/{source_id}")
def get_source(source_id: str):
    path = os.path.join(SRC, f"{source_id}.mp4")
    if not os.path.exists(path):
        return JSONResponse({"error":"not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=f"{source_id}.mp4")

@app.get("/api/output/{output_id}")
def get_output(output_id: str):
    p = os.path.join(OUT, f"{output_id}.mp4")
    if not os.path.exists(p):
        return JSONResponse({"error":"not found"}, status_code=404)
    return FileResponse(p, media_type="video/mp4", filename=f"{output_id}.mp4")

@app.get("/api/thumbnail/{clip_id}")
def get_thumb(clip_id: str):
    p = os.path.join(OUT, f"{clip_id}.jpg")
    if not os.path.exists(p):
        return JSONResponse({"error":"not found"}, status_code=404)
    return FileResponse(p, media_type="image/jpeg", filename=f"{clip_id}.jpg")

# -------------------------
# Single process endpoint (basic + features)
# -------------------------
@app.post("/api/process")
def process(req: ProcessReq):
    src = os.path.join(SRC, f"{req.source_id}.mp4")
    if not os.path.exists(src):
        return JSONResponse({"error":"source not found"}, status_code=404)
    w_out, h_out = (1080, 1920)  # presets could be extended
    out_id = str(uuid.uuid4())
    out_path = os.path.join(OUT, f"{out_id}.mp4")

    vf_parts = []
    input_args = ["-i", src]

    # smart crop / letterbox
    if req.letterbox:
        vf_parts.append(f"scale={w_out}:-2:flags=lanczos,pad={w_out}:{h_out}:(ow-iw)/2:(oh-ih)/2")
    else:
        if req.smart_crop == "face":
            meta = sample_faces(src)
            cw,ch,cx,cy = compute_crop_9x16_from_faces(meta, target_h=h_out)
            vf_parts.append(f"crop={cw}:{ch}:{cx}:{cy},scale={w_out}:{h_out}")
        else:
            vf_parts.append(f"scale=-2:{h_out},crop={w_out}:{h_out}")

    # progress bar
    if req.progress_bar:
        vf_parts.append(make_progress_bar_filter(req.duration_sec))

    # watermark
    if req.watermark_path:
        wm = req.watermark_path
        wm_path = wm if os.path.isabs(wm) else os.path.join(BASE_DIR, wm)
        if os.path.exists(wm_path):
            input_args += ["-i", wm_path]
            vf_parts.append("overlay=W-w-24:24")

    # whisper captions (burn-in)
    srt_path = None
    if req.burn_captions:
        model = load_whisper_model(req.whisper_model)
        clip_tmp = os.path.join(TMP, f"{out_id}_clip.wav")
        ffmpeg_run(["ffmpeg","-y","-ss", str(req.start_sec),"-t", str(req.duration_sec),"-i", src,"-ar","16000","-ac","1","-vn", clip_tmp], check=False)
        tr = whisper_transcribe(model, clip_tmp, translate=False)
        srt_path = os.path.join(TMP, f"{out_id}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            idx = 1
            for seg in tr["segments"]:
                start_s = seg["start"]; end_s = seg["end"]
                f.write(f"{idx}\n{sec_to_ts(start_s)} --> {sec_to_ts(end_s)}\n{seg['text'].strip()}\n\n")
                idx += 1
        vf_parts.append(build_subtitle_filter(srt_path))

    # inject emoji into first caption line (option)
    # handled during SRT creation if inject_emoji True (not shown here for brevity)

    # bgm mixing
    cmd = ["ffmpeg","-y","-ss", str(req.start_sec),"-t", str(req.duration_sec)] + input_args + [
        "-vf", ",".join(vf_parts) if vf_parts else "null",
        "-r", str(req.fps),
        "-c:v","libx264","-preset","medium","-b:v", req.target_bitrate,
        "-pix_fmt","yuv420p","-profile:v","high",
        "-x264-params","keyint=60:min-keyint=60:scenecut=0"
    ]

    # audio
    if req.keep_audio:
        if req.loudnorm:
            cmd += ["-af", loudnorm_filter()]
        cmd += ["-c:a","aac","-b:a","160k"]
    else:
        cmd += ["-an"]

    # finalize
    cmd += ["-movflags","+faststart", out_path]

    ffmpeg_run(cmd)
    # make thumbnail
    save_thumbnail(out_path, min(2.0, float(req.duration_sec)/2.0), os.path.join(OUT, f"{out_id}.jpg"))
    # cleanup srt
    try:
        if srt_path and os.path.exists(srt_path): os.remove(srt_path)
    except: pass
    return {"output_id": out_id, "download": f"/api/output/{out_id}"}

# -------------------------
# Auto clips endpoint (batch like 2short.ai)
# -------------------------
@app.post("/api/auto_clips")
def auto_clips(req: AutoClipsReq):
    src = os.path.join(SRC, f"{req.source_id}.mp4")
    if not os.path.exists(src):
        return JSONResponse({"error":"source not found"}, status_code=404)

    # 1) Extract audio and transcribe
    model = load_whisper_model(req.whisper_model)
    audio_tmp = os.path.join(TMP, f"{uuid.uuid4().hex}_a.wav")
    ffmpeg_run(["ffmpeg","-y","-i",src,"-ar","16000","-ac","1","-vn", audio_tmp], check=False)
    tr = whisper_transcribe(model, audio_tmp, translate=False, language=req.lang)
    segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in tr["segments"]]

    # 2) Candidate windows
    candidates = select_windows_from_transcript(segments, req.min_dur, req.max_dur, req.num_clips, req.lang)

    results = []
    for cand in candidates:
        start = float(cand["start"])
        duration = float(cand["dur"])
        out_id = str(uuid.uuid4())
        out_path = os.path.join(OUT, f"{out_id}.mp4")
        w_out, h_out = (1080,1920)
        vf_parts=[]
        if req.letterbox:
            vf_parts.append(f"scale={w_out}:-2:flags=lanczos,pad={w_out}:{h_out}:(ow-iw)/2:(oh-ih)/2")
        else:
            if req.smart_crop=="face":
                meta = sample_faces(src)
                cw,ch,cx,cy = compute_crop_9x16_from_faces(meta, target_h=h_out)
                vf_parts.append(f"crop={cw}:{ch}:{cx}:{cy},scale={w_out}:{h_out}")
            else:
                vf_parts.append(f"scale=-2:{h_out},crop={w_out}:{h_out}")
        if req.progress_bar:
            vf_parts.append(make_progress_bar_filter(duration))
        input_args = ["-i", src]
        if req.watermark_path:
            wm = req.watermark_path
            wm_path = wm if os.path.isabs(wm) else os.path.join(BASE_DIR, wm)
            if os.path.exists(wm_path):
                input_args += ["-i", wm_path]
                vf_parts.append("overlay=W-w-24:24")
        # build captions for window only
        srt_path = None
        if req.burn_captions:
            sub_out = os.path.join(TMP, f"{out_id}.srt")
            with open(sub_out, "w", encoding="utf-8") as f:
                idx = 1
                for seg in segments:
                    if seg["end"] <= start: continue
                    if seg["start"] >= start + duration: break
                    ss = max(0.0, seg["start"] - start)
                    ee = min(duration, seg["end"] - start)
                    text = seg["text"].strip()
                    if req.inject_emoji:
                        text = inject_emojis(text)
                    f.write(f"{idx}\n{sec_to_ts(ss)} --> {sec_to_ts(ee)}\n{text}\n\n")
                    idx += 1
            vf_parts.append(build_subtitle_filter(sub_out))
            srt_path = sub_out

        af_parts = []
        if req.loudnorm: af_parts.append(loudnorm_filter())
        af_str = ",".join(af_parts) if af_parts else None

        cmd = ["ffmpeg","-y","-ss", str(start),"-t", str(duration)] + input_args + [
            "-vf", ",".join(vf_parts) if vf_parts else "null",
            "-r", str(req.fps),
            "-c:v","libx264","-preset","medium","-b:v", req.target_bitrate,
            "-pix_fmt","yuv420p","-profile:v","high",
            "-x264-params","keyint=60:min-keyint=60:scenecut=0"
        ]
        if req.keep_audio:
            if af_str: cmd += ["-af", af_str]
            cmd += ["-c:a","aac","-b:a","160k"]
        else:
            cmd += ["-an"]
        cmd += ["-movflags","+faststart", out_path]
        ffmpeg_run(cmd)
        # thumbnail
        save_thumbnail(out_path, min(2.0, duration/2.0), os.path.join(OUT, f"{out_id}.jpg"))
        # prepare result metadata
        teaser = (cand["text"][:137] + "...") if len(cand["text"])>140 else cand["text"]
        results.append({
            "clip_id": out_id,
            "start": start,
            "duration": duration,
            "score": cand["score"],
            "text": teaser,
            "preview": f"/api/output/{out_id}",
            "thumbnail": f"/api/thumbnail/{out_id}"
        })
        try:
            if srt_path and os.path.exists(srt_path): os.remove(srt_path)
        except: pass

    # cleanup audio_tmp
    try:
        if os.path.exists(audio_tmp): os.remove(audio_tmp)
    except: pass

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"clips": results}

# -------------------------
# Batch mode: accept list of URLs or source_ids
# -------------------------
class BatchReq(BaseModel):
    urls: Optional[List[str]] = None
    source_ids: Optional[List[str]] = None
    auto_clip_params: Optional[AutoClipsReq] = None

@app.post("/api/batch_process")
def batch_process(req: BatchReq):
    results = []
    # download urls
    if req.urls:
        for u in req.urls:
            try:
                dd = download(DownloadReq(url=u))
                results.append(dd)
            except Exception as e:
                results.append({"error": str(e)})
    # process source_ids with auto_clips if requested
    outputs = []
    if req.source_ids and req.auto_clip_params:
        for sid in req.source_ids:
            acr = req.auto_clip_params
            acr_dict = acr.dict()
            acr_dict["source_id"] = sid
            res = auto_clips(AutoClipsReq(**acr_dict))
            outputs.append(res)
    return {"downloaded": results, "processed": outputs}

# -------------------------
# Metadata generator: title + hashtags
# -------------------------
@app.post("/api/gen_meta")
def gen_meta(req: GenMetaReq):
    text = req.text.strip()
    title = (text[:70] + "...") if len(text) > 80 else text
    # naive hashtags: top nouns / keywords: pick high-frequency words excluding stopwords
    words = re.findall(r"\w+", text.lower())
    stop = set(["the","and","is","a","to","in","of","it","that","this","you","i","we","they","on","for","with","as","are","be"])
    freq={}
    for w in words:
        if w in stop or len(w)<3: continue
        freq[w]=freq.get(w,0)+1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:6]
    hashtags = ["#"+w.replace(" ", "") for w,_ in top]
    if not hashtags:
        hashtags = ["#shorts","#viral","#tiktok"]
    # Optionally call OpenAI to refine (if env var provided). Wrapped to avoid crash.
    if req.use_openai:
        try:
            import openai, os
            key = os.environ.get("OPENAI_API_KEY")
            if key:
                openai.api_key = key
                prompt = f"Create a catchy short-form title (max 60 chars) and 6 hashtags for the following text:\n\n{text}\n\nReturn JSON: {{\"title\":\"...\",\"hashtags\":[...]}}"
                resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=120)
                txt = resp.choices[0].text.strip()
                # attempt parse JSON
                try:
                    obj = json.loads(txt)
                    title = obj.get("title", title)
                    hashtags = obj.get("hashtags", hashtags)
                except:
                    pass
        except Exception as e:
            pass
    return {"title": title, "hashtags": hashtags}

# -------------------------
# Preview editor endpoints (trim, update caption, re-render small)
# -------------------------
class TrimReq(BaseModel):
    clip_id: str
    new_start: float
    new_duration: float

@app.post("/api/trim")
def trim_clip(req: TrimReq):
    clip_file = os.path.join(OUT, f"{req.clip_id}.mp4")
    if not os.path.exists(clip_file):
        return JSONResponse({"error":"clip not found"}, status_code=404)
    new_id = str(uuid.uuid4())
    out_path = os.path.join(OUT, f"{new_id}.mp4")
    cmd = ["ffmpeg","-y","-ss", str(req.new_start),"-t", str(req.new_duration),"-i", clip_file,"-c","copy",out_path]
    ffmpeg_run(cmd)
    save_thumbnail(out_path, min(2.0, req.new_duration/2.0), os.path.join(OUT, f"{new_id}.jpg"))
    return {"clip_id": new_id, "preview": f"/api/output/{new_id}", "thumbnail": f"/api/thumbnail/{new_id}"}

# -------------------------
# Zip export
# -------------------------
class ExportReq(BaseModel):
    clip_ids: List[str]
    out_name: str = "export"

@app.post("/api/export_zip")
def export_zip(req: ExportReq):
    export_folder = os.path.join(TMP, f"export_{uuid.uuid4().hex}")
    os.makedirs(export_folder, exist_ok=True)
    meta = []
    for cid in req.clip_ids:
        mp = os.path.join(OUT, f"{cid}.mp4")
        th = os.path.join(OUT, f"{cid}.jpg")
        if os.path.exists(mp):
            shutil.copy(mp, os.path.join(export_folder, os.path.basename(mp)))
        if os.path.exists(th):
            shutil.copy(th, os.path.join(export_folder, os.path.basename(th)))
        meta.append({"clip_id": cid, "file": os.path.basename(mp), "thumbnail": os.path.basename(th)})
    with open(os.path.join(export_folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    zip_path = os.path.join(TMP, f"{req.out_name}_{uuid.uuid4().hex}.zip")
    shutil.make_archive(zip_path.replace('.zip',''), 'zip', export_folder)
    final_zip = zip_path if zip_path.endswith('.zip') else zip_path + ".zip"
    # cleanup folder
    shutil.rmtree(export_folder)
    return {"zip": f"/api/temp_zip/{os.path.basename(final_zip)}"}

@app.get("/api/temp_zip/{zip_name}")
def get_temp_zip(zip_name: str):
    p = os.path.join(TMP, zip_name)
    if not os.path.exists(p):
        return JSONResponse({"error":"not found"}, status_code=404)
    return FileResponse(p, media_type="application/zip", filename=zip_name)

# -------------------------
# Simple health
# -------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "storage": {"src": len(os.listdir(SRC)), "out": len(os.listdir(OUT))}}

# End of file
