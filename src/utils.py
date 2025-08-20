# utils.py
import os, re, subprocess, uuid, json, math, random, shutil
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
from tqdm import tqdm

# -------------------------
# Basic shell helpers
# -------------------------
def run_capture(cmd: List[str], check: bool=False) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return proc.stdout

def ffmpeg_run(cmd: List[str], check: bool=True) -> str:
    return run_capture(cmd, check=check)

def ffprobe_duration(path: str) -> float:
    out = run_capture([
        "ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=nw=1:nk=1", path
    ])
    try:
        return float(out.strip())
    except:
        return 0.0

# -------------------------
# Face sampling & smart crop
# -------------------------
def sample_faces(video_path: str, max_frames=120, step=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for face sampling")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    centers, sizes = [], []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = 0
    sampled = 0
    while sampled < max_frames and idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48,48))
        if len(faces) > 0:
            x,y,fw,fh = sorted(faces, key=lambda b:b[2]*b[3], reverse=True)[0]
            centers.append((x + fw/2, y + fh/2))
            sizes.append(max(fw, fh))
        idx += step
        sampled += 1
    cap.release()
    return {"w": w, "h": h, "centers": centers, "sizes": sizes}

def clamp(a, lo, hi): return max(lo, min(hi, a))

def compute_crop_9x16_from_faces(meta, target_h):
    iw, ih = meta["w"], meta["h"]
    if not meta["centers"]:
        out_h = min(ih, iw * 16/9)
        out_w = out_h * 9/16
        x = (iw - out_w) / 2
        y = (ih - out_h) / 2
        return int(out_w), int(out_h), int(x), int(y)
    centers = np.array(meta["centers"])
    sizes = np.array(meta["sizes"]) if meta["sizes"] else np.array([min(iw,ih)//4]*len(centers))
    cx = float(np.median(centers[:,0]))
    cy = float(np.median(centers[:,1]))
    face_size = float(np.percentile(sizes, 75))
    desired_h = clamp(face_size * 4.0, ih*0.45, ih*0.95)
    out_h = desired_h
    out_w = out_h * 9/16
    x = clamp(cx - out_w/2, 0, iw - out_w)
    y = clamp(cy - out_h*0.45, 0, ih - out_h)
    return int(out_w), int(out_h), int(x), int(y)

# -------------------------
# Whisper helpers
# -------------------------
def load_whisper_model(name="base"):
    import whisper
    return whisper.load_model(name)

def whisper_transcribe(model, audio_path, translate=False, language=None):
    # model.transcribe returns dict with segments
    kwargs = {}
    if translate: kwargs["task"] = "translate"
    if language: kwargs["language"] = language
    return model.transcribe(audio_path, **kwargs)

# -------------------------
# Hook keywords + scoring
# -------------------------
KEYWORDS_HOOK = {
    "id": [
        "rahasia","tips","trik","penting","kunci","caranya","gini","contoh","story","kisah",
        "pelajaran","kesalahan","jangan","hindari","ternyata","fakta","mindset","strategi","mudah","cepat","gratis"
    ],
    "en": [
        "secret","tip","trick","important","key","here's","story","example","lesson","mistake","avoid",
        "turns out","fact","mindset","strategy","easy","quick","free","surprising","announce"
    ]
}

def select_windows_from_transcript(segments, min_dur=20, max_dur=40, top_k=4, lang="id"):
    windows=[]
    n = len(segments)
    for i in range(n):
        start = segments[i]['start']
        j = i
        text = ""
        end = start
        while j < n and (segments[j]['end'] - start) <= max_dur:
            end = segments[j]['end']
            text += " " + segments[j]['text'].strip()
            dur = end - start
            if dur >= min_dur:
                words = len(text.split())
                wps = words / max(0.0001, dur)
                hooks = KEYWORDS_HOOK.get(lang,[]) + KEYWORDS_HOOK["en"]
                bonus = sum(1 for k in hooks if k.lower() in text.lower())
                score = wps * 2.0 + bonus * 1.5 + min(dur, max_dur)/max_dur
                windows.append({"start": start, "end": end, "dur": dur, "text": text.strip(), "score": score})
            j += 1
    # NMS-like pick
    windows.sort(key=lambda x: x["score"], reverse=True)
    picks=[]
    def overlap(a,b):
        return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"])) / max(a["dur"], b["dur"])
    for w in windows:
        if all(overlap(w,p) < 0.35 for p in picks):
            picks.append(w)
        if len(picks) >= top_k: break
    return picks

# -------------------------
# Scene detection (scene change)
# -------------------------
def scene_score(video_path, threshold=0.3, max_samples=200):
    # count scene changes using ffmpeg select='gt(scene,threshold)'
    out = run_capture([
        "ffprobe","-hide_banner","-loglevel","error","-show_frames",
        "-of","compact=p=0","-f","lavfi", f"movie={video_path},select=gt(scene\\,{threshold})"
    ])
    # approximate: count "frame=" occurrences
    return out.count("pkt_pts_time=")  # heuristic

# -------------------------
# Silence detect & loudness
# -------------------------
def detect_silences(video_path, noise_floor=-35.0):
    log = run_capture([
        "ffmpeg","-i",video_path,"-af",f"silencedetect=noise={noise_floor}dB:d=0.4","-f","null","-"
    ])
    starts, ends = [], []
    for line in log.splitlines():
        if "silence_start" in line:
            m = re.search(r"silence_start:\s*([0-9\.]+)", line)
            if m: starts.append(float(m.group(1)))
        if "silence_end" in line:
            m = re.search(r"silence_end:\s*([0-9\.]+)", line)
            if m: ends.append(float(m.group(1)))
    # build speech intervals
    dur = ffprobe_duration(video_path)
    if not starts and not ends:
        return [(0.0, dur)]
    intervals=[]
    cur = 0.0
    idx_s, idx_e = 0, 0
    while cur < dur:
        next_s = starts[idx_s] if idx_s < len(starts) else None
        if next_s and next_s > cur:
            intervals.append((cur, next_s))
            cur = ends[idx_e] if idx_e < len(ends) else dur
            idx_s += 1
            idx_e += 1
        else:
            # jump forward
            cur = (ends[idx_e] if idx_e < len(ends) else dur)
            idx_e += 1
    return intervals

# -------------------------
# B-roll picker (local folder)
# -------------------------
def pick_broll(keywords: List[str], assets_folder: str, max_items=2):
    # simple: scan filenames and pick those containing keywords
    files = []
    for root, _, fns in os.walk(assets_folder):
        for fn in fns:
            if fn.lower().endswith(('.mp4','.mov','.mkv','.webm')):
                files.append(os.path.join(root, fn))
    scored=[]
    for f in files:
        score = 0
        name = os.path.basename(f).lower()
        for k in keywords:
            if k.lower() in name: score += 2
        if score == 0: score = 0.1
        scored.append((score, f))
    scored.sort(reverse=True)
    picks = [p for _,p in scored[:max_items]]
    # fallback random
    while len(picks) < max_items and files:
        cand = random.choice(files)
        if cand not in picks: picks.append(cand)
    return picks

# -------------------------
# Emoji captioning (simple mapping)
# -------------------------
EMOJI_MAP = {
    "🔥": ["hebat","luar biasa","amazing","hot","viral"],
    "💡": ["tips","ide","cara","trik","cara"],
    "😂": ["haha","lucu","funny","ngakak"],
    "❗": ["penting","jangan","ingat","perhatian"],
    "💰": ["uang","bisnis","profit","duit","ekonomi"],
    "🎮": ["game","gaming","play"],
    "🍔": ["makan","makanan","food","resepi"],
}

def inject_emojis(text: str, max_emojis=2) -> str:
    tokens = text.lower()
    added=[]
    for emoji, keys in EMOJI_MAP.items():
        for k in keys:
            if k in tokens and emoji not in added:
                added.append(emoji)
                break
        if len(added) >= max_emojis: break
    if added:
        return text + " " + " ".join(added)
    return text

# -------------------------
# BGM auto-mix / ducking filter builder
# -------------------------
def build_bgm_duck_filter(bgm_stream_index=1, speech_stream_index=0, duck_level_db= -12.0):
    # Use sidechaincompress or volume with adelay; here we approximate with sidechaincompress
    # Format: [bgm][speech]sidechaincompress=threshold=0.05:ratio=9:attack=5:release=100
    # BUT sidechaincompress works with two input streams in filter_complex.
    # We'll build filter_complex string elsewhere when mixing is used.
    return f"sidechaincompress=threshold=0.05:ratio=9:attack=5:release=100:gain={duck_level_db}"

# -------------------------
# Thumbnail save
# -------------------------
def save_thumbnail(video_path: str, t: float, out_path: str):
    ffmpeg_run(["ffmpeg","-ss",str(t),"-i",video_path,"-vframes","1","-q:v","3","-y",out_path], check=False)

# -------------------------
# Zip export
# -------------------------
def zip_results(folder: str, out_zip: str):
    shutil.make_archive(out_zip.replace('.zip',''), 'zip', folder)
    return out_zip

# -------------------------
# Small text utilities
# -------------------------
def sec_to_ts(sec: float):
    ms = int((sec - int(sec)) * 1000)
    s = int(sec) % 60
    m = (int(sec) // 60) % 60
    h = int(sec) // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"
