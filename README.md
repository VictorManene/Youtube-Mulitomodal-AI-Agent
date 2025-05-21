**VideoInsight AI**
*A comprehensive YouTube multimodal analysis & AI agent platform*

---

## 🚀 Project Overview

VideoInsight AI is an AI‑powered virtual assistant that transforms YouTube videos into actionable insights. Whether you need to understand an interview, summarize a tutorial, or extract on‑screen text, our platform automates the entire workflow:

1. **Download** YouTube videos (full audio & video).
2. **Extract & convert** audio to clean `.wav` (16 kHz, mono).
3. **Transcribe** speech using Whisper for near‑human accuracy.
4. **OCR** key video frames for on‑screen text.
5. **Embed & index** transcript + OCR snippets in a Chroma vector store.
6. **Query** content via Groq’s LLM API or follow-up chat.
7. **AI Agent** (ReAct) automates analysis, summarization, and Q\&A.
8. **Text‑to‑Speech** playback of AI responses.
9. **Web UI** built with Gradio for seamless multimodal & agent modes, you can sign in/sign up to toggle between multimodal and agent model

### Problems Solved

* **Video understanding without watching**: Get the gist instantly.

* **Accessibility & inclusivity**: Captions, summaries, and OCR benefit hearing-impaired users.

* **Time-saving**: Skip to the highlights with summaries and smart Q\&A.

* **Cross-language Q\&A**: Ask questions in any language—our platform auto-detects and translates inputs/outputs, breaking communication barriers.

* **Video understanding without watching**: Get the gist instantly.

* **Accessibility & inclusivity**: Captions, summaries, and OCR benefit hearing‑impaired users.

* **Time‑saving**: Skip to the highlights with summaries and smart Q\&A.

---

## ✨ Key Features

* **Audio Download & Conversion**: Fetch YouTube audio and convert to 16 kHz WAV using `yt_dlp` and FFmpeg.
* **Speech-to-Text**: Leverage `faster_whisper` for accurate, low-latency transcription.
* **Optical Character Recognition**: Extract on-screen text at regular intervals with Tesseract OCR.
* **Vector Embeddings**: Chunk transcripts & OCR snippets, embed via `sentence-transformers`, and persist in Chroma for similarity search.
* **LLM Q\&A & Summaries**: Contextual retrieval + Groq LLM gives human-like answers and concise summaries.
* **ReAct Agent** Tools\*\*:

  * `WatchVideo`: Download media.
  * `SummarizeContent`: Full multimodal summarization.
  * `ReadSlides`: Answer questions about specific moments.
  * `Speak`: TTS playback using `pyttsx3` + `pygame`.
* **Web App (Gradio)**:

  * **Multimodal Mode**: URL + question → transcript, OCR, summary, audio, follow-up chat.
  * **Agent Mode**: Turn-based AI agent analysis via React agent.
  * Dark mode, custom CSS, interactive chat UI.

---

## 📦 Tech Stack & Dependencies

**Backend (Python)**

* Python 3.8+
* `yt_dlp`, `ffmpeg`
* `faster_whisper` (CPU int8), `sentence-transformers`, `torch`
* `opencv-python`, `pytesseract` (requires Tesseract on PATH)
* `langchain`, `langchain_community (Chroma, embeddings)`, `deep_translator`, `langdetect`
* `pyttsx3`, `pygame`
* `requests`, `logging`, `uuid`, `dotenv`

**Frontend (Gradio)**

* `gradio`
* Custom CSS for theming & animations

**APIs & Services**

* YouTube Data API (via API key & optional cookie file)
* Groq LLM API (via `GROQ_MODEL` & `GROQ_API_URL`)

---

## ⚙️ Environment & Configuration

1. **Clone & install**

   ```bash
   git clone https://github.com/<your-username>/youtube-multimodal-ai-agent.git
   cd youtube-multimodal-ai-agent
   pip install -r requirements.txt
   ```

2. **Install system tools**

   * [FFmpeg](https://ffmpeg.org/download.html) in your PATH.
   * [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and accessible.

3. **Environment variables**  (create a `.env` in root)

   ```dotenv
   YOUTUBE_API_KEY=<your_youtube_api_key>
   YT_COOKIES_PATH=<optional_path_to_youtube_cookies.txt>
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

4. **Optional**: for the Gradio UI, set `OPENAI_API_KEY` (alias for `YOUTUBE_API_KEY`).

---

## 📁 Directory Structure

```
├── backend/
│   ├── main.py            # Core pipeline & agent definitions
│   ├── requirements.txt   # Python dependencies
│   └── files/             # Auto‑generated: audio, video, text, chroma_db
├── frontend/
│   ├── app.py             # Gradio UI code
│   └── styles.css         # Custom CSS for UI
├── .env                   # YouTube & Groq API keys
└── README.md
```

---

## ▶️ Usage Examples

### 1. Multimodal (pipeline) mode

```bash
# Run analysis without TTS playback
python backend/main.py --url <YouTube_URL> --question "What happens in this video?" --no-audio
```

### 2. Agent (ReAct) mode

```bash
python backend/main.py --agent True
```

### 3. Gradio Web App

```bash
python frontend/app.py
# Opens http://localhost:7860 with both modes selectable
```

---

## 🛠 Troubleshooting

* **`ValueError` in `summarize_tool`**: Ensure you updated the function to handle `input_data=None` (grab latest WAV).
* **`TesseractNotFoundError`**: Install Tesseract and verify `tesseract --version` works.
* **Audio playback issues**: Check `pygame` setup and sound drivers on your OS.

---

## 🤝 Contributing & License

Contributions welcome! Please open issues or pull requests on GitHub.
Licensed under MIT. See [LICENSE](LICENSE) for details.
