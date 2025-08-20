# 🎬 AI Shorts Clipper Backend

Backend ini adalah **Video Shorts Generator** mirip **2short.ai** atau **Opus Clip**, dibangun dengan **FastAPI + yt-dlp + ffmpeg + Whisper**.  
Bisa otomatis download video YouTube, deteksi momen menarik, potong jadi clip 9:16, tambahkan caption, watermark, musik, dan langsung siap diupload ke **TikTok / YouTube Shorts / IG Reels**.

---

## ✨ Fitur Utama
- ✅ Download video YouTube (via `yt-dlp`)  
- ✅ Auto deteksi hook + scene change → generate clip otomatis  
- ✅ Smart crop (9:16) dengan deteksi wajah / objek  
- ✅ Auto captions (Whisper) + emoji injection 😎  
- ✅ Multi-language subtitles (EN/ID/ES, dll)  
- ✅ AI Title + Hashtags Generator  
- ✅ Watermark & brand template preset  
- ✅ Background music auto-mix (sidechain ducking)  
- ✅ Progress bar & dynamic captions  
- ✅ Batch mode (playlist / multi URL)  
- ✅ Output: MP4, thumbnails, metadata JSON  

---

## 📂 Struktur Folder
```
backend/
│── app.py              # FastAPI utama
│── utils.py            # fungsi bantu (ffmpeg, crop, dsb)
│── requirements.txt    # dependensi python
│── README.md           # dokumentasi
│── storage/
│    ├── sources/       # hasil download yt
│    ├── outputs/       # hasil final clip
│    ├── temp/          # file sementara
```

---

## ⚙️ Instalasi

### 1. Clone repo & masuk folder
```bash
git clone https://github.com/jhodypedia/ytshorts.git
cd ytshorts
```

### 2. Install dependensi
```bash
pip install -r requirements.txt
```

### 3. Jalankan server
```bash
uvicorn app:app --reload --port 8000
```
Server akan jalan di: `http://localhost:8000`

---

## 📡 Endpoint API

### 1. Download Video
```http
POST /api/download
Content-Type: application/json

{ "url": "https://www.youtube.com/watch?v=XXXX" }
```
Respon:
```json
{ "source_id": "uuid", "path": "/api/file/uuid" }
```

---

### 2. Proses Jadi Shorts
```http
POST /api/process
Content-Type: application/json

{
  "source_id": "uuid",
  "start_sec": 30,
  "duration_sec": 20,
  "smart_crop": "face",
  "burn_captions": true,
  "loudnorm": true,
  "progress_bar": true,
  "watermark_path": "storage/logo.png",
  "bgm": "random"
}
```
Respon:
```json
{ "output_id": "uuid", "download": "/api/output/uuid" }
```

---

### 3. Ambil File
```http
GET /api/output/{output_id}
```
Respon: file `.mp4` siap download.

---

## 🎯 Rencana Fitur Lanjutan
- Editor frontend (React) untuk preview & manual trim.  
- Upload langsung ke TikTok API / YouTube API.  
- Template preset untuk niche tertentu (podcast, gaming, edukasi).  

---

## 📝 Lisensi
MIT License.
