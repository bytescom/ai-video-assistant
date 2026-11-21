# 🎬 AI Video Assistant: Multimodal Meeting Intelligence with RAG

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Speech-to-Text](https://img.shields.io/badge/STT-OpenAI%20Whisper-purple.svg)](https://github.com/openai/whisper)
[![Indic-STT](https://img.shields.io/badge/Indic%20STT-Sarvam%20AI-orange.svg)](https://www.sarvam.ai/)
[![LLM](https://img.shields.io/badge/LLM-Mistral%20AI-red.svg)](https://mistral.ai/)
[![Vector DB](https://img.shields.io/badge/Vector%20Store-ChromaDB-blueviolet.svg)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-HuggingFace%20all--MiniLM--L6--v2-yellow.svg)](https://huggingface.co/)
[![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

A privacy-focused, 100% free multimodal meeting intelligence assistant that ingests YouTube videos and local media recordings, transcribes and translates audio using local Whisper or Sarvam AI, performs Map-Reduce summarization, extracts structured deliverables (action items, decisions, open questions), and enables conversational vector search through a local ChromaDB RAG pipeline.

---

## 🌟 Key Features

- **Multimodal Ingestion:** Automatically downloads and extracts audio from YouTube URLs via `yt-dlp` or normalizes local video/audio files (MP4, MP3, WAV, WEBM) to 16 kHz mono WAV with `pydub`.
- **Hybrid Speech-to-Text & Translation:**
  - **Local Whisper (`small`):** Free, local STT running on CPU/GPU without external cloud costs or privacy risks.
  - **Sarvam AI (`saaras:v2.5`):** Specialized Indic/Hinglish speech-to-text translation endpoint with an automated micro-chunking engine to bypass API length limits.
- **Map-Reduce Summarization:** Processes long meeting transcripts (10,000+ words) using LangChain's Map-Reduce pattern to eliminate context-window overflow.
- **Structured Deliverables Extraction:** Automatically generates:
  - 📌 **Meeting Title:** Short, actionable title ($\le 8$ words).
  - 📋 **Executive Summary:** Bulleted synthesis of discussion points.
  - ✅ **Action Items:** Numbered tasks with responsible owners and deadlines.
  - 🔑 **Key Decisions:** Finalized group agreements and architectural choices.
  - ❓ **Open Questions:** Unresolved topics requiring follow-up.
- **Conversational RAG Chat:** Embeds transcripts locally with HuggingFace `all-MiniLM-L6-v2` into a persistent **ChromaDB** vector database. Enables contextual chat with negative grounding to eliminate hallucinations.
- **Streamlit Web UI:** Dark futuristic dashboard with live stage trackers (`Audio` ➔ `Transcript` ➔ `Title` ➔ `Summary` ➔ `Extract` ➔ `RAG`), split-view cards, and real-time chat bubbles.

---

## 🏗️ System Architecture

```
================ PHASE 1: INGESTION, TRANSCRIPTION & SYNTHESIS ================

[YouTube URL or Local Media]
             │
             ▼
  ┌────────────────────────┐
  │ utils/audio_processor  │ ──▶ yt-dlp + FFmpeg / pydub (16 kHz Mono WAV)
  └────────────────────────┘      ──▶ Slices into 10-Minute Macro Chunks
             │
             ▼
  ┌────────────────────────┐
  │   core/transcriber     │ ──▶ [English]  ──▶ Local OpenAI Whisper (small)
  └────────────────────────┘ ──▶ [Hinglish] ──▶ Micro-chunks (25s) ──▶ Sarvam AI STT
             │
             ▼ (Full Aggregated Transcript)
  ┌────────────────────────┴────────────────────────┐
  │                                                 │
  ▼                                                 ▼
┌────────────────────────┐        ┌──────────────────────────────────┐
│    core/summarizer     │        │          core/extractor          │
│ Map-Reduce Summarizer  │        │ Reusable LCEL Extraction Chains  │
│ - Title (<= 8 words)   │        │ - Action Items (Owner, Deadline) │
│ - Executive Summary    │        │ - Key Decisions                  │
└────────────────────────┘        │ - Open Questions / Follow-ups    │
                                  └──────────────────────────────────┘

======================== PHASE 2: LOCAL RAG & CHAT ========================

Full Transcript
       │
       ▼
┌────────────────────────┐
│   core/vector_store    │ ──▶ Recursive Splitter (500 chars / 50 overlap)
└────────────────────────┘ ──▶ HuggingFace all-MiniLM-L6-v2 Embeddings (Local CPU)
                           ──▶ ChromaDB Persistent Store (./vector_db)
       │
       ▼
┌────────────────────────┐
│    core/rag_engine     │ ──▶ Similarity Retriever (k=4)
└────────────────────────┘ ──▶ LCEL Grounded Prompt + Mistral AI
       │
       ▼
Streamlit Interactive Dashboard / CLI Terminal Chat Loop
```

---

## 🛠️ Tech Stack

| Domain | Technology / Tool | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core application runtime |
| **Audio Processing** | `yt-dlp`, FFmpeg, `pydub` | Media ingestion, format conversion, and audio chunking |
| **Speech Recognition** | OpenAI Whisper (Local) | On-device transcription without external API fees |
| **Indic Translation** | Sarvam AI (`saaras:v2.5`) | High-accuracy Hinglish / Hindi speech-to-English translation |
| **LLM & Reasoning** | Mistral AI (`mistral-small-latest`) | Map-Reduce summarization, entity extraction, and Q&A |
| **Orchestration** | LangChain & LCEL Runnables | Dynamic prompt piping and sequence chains |
| **Vector Store** | ChromaDB (`langchain-chroma`) | Persistent local vector index in `./vector_db` |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Lightweight, CPU-friendly local embeddings |
| **Frontend** | Streamlit, Custom CSS | Interactive web dashboard and real-time chat interface |

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.10 to 3.12** installed on your system.
2. **FFmpeg installed on your system PATH** (Required for audio extraction):
   - **Windows:**
     ```powershell
     winget install Gyan.FFmpeg
     ```
   - **macOS:**
     ```bash
     brew install ffmpeg
     ```
   - **Linux (Ubuntu/Debian):**
     ```bash
     sudo apt update && sudo apt install -y ffmpeg
     ```
   *(Restart your terminal after installation so FFmpeg is detected in your PATH).*

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mepan/video_agent.git
   cd video_agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Using uv:
   uv venv
   .venv\Scripts\activate          # On Windows PowerShell
   # source .venv/bin/activate     # On Linux/macOS

   # Or using standard venv:
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r Requirements.txt
   # Or with uv: uv pip install -r Requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   MISTRAL_API_KEY="your-mistral-api-key"
   SARVAM_API_KEY="your-sarvam-api-key"      # Optional: only needed for Hinglish mode
   WHISPER_MODEL="small"                     # tiny / small / medium / large
   SARVAM_STT_MODEL="saaras:v2.5"
   ```

---

## 💻 Usage

### Option 1: Streamlit Web Dashboard (Recommended)

Start the web application:
```bash
streamlit run app.py
```
1. Open **`http://localhost:8501`** in your browser.
2. Paste any YouTube URL or path to a local media file in the sidebar.
3. Select the language (`english` or `hinglish`) and click **⚡ Analyse**.
4. Review the generated executive summary, action items, and chat with your meeting.

### Option 2: Terminal / CLI Mode

Run the terminal pipeline with interactive Q&A:
```bash
python main.py
```
Follow the console prompts to input a media source and converse with the transcript directly in your terminal.

---

## 📂 Project Structure

```
video_agent/
├── .env                        # API keys (MISTRAL_API_KEY, SARVAM_API_KEY - never commit)
├── .gitignore                  # Excludes .env, venvs, audio downloads, and vector_db
├── Requirements.txt            # Project dependencies
├── utils/
│   └── audio_processor.py      # Media downloader, 16kHz WAV converter, and chunker
├── core/
│   ├── transcriber.py          # Local Whisper + Sarvam AI translation with micro-slicing
│   ├── summarizer.py           # Mistral Map-Reduce summarization & title generator
│   ├── extractor.py            # Extraction chains for action items, decisions, questions
│   ├── vector_store.py         # ChromaDB indexing with HuggingFace embeddings
│   └── rag_engine.py           # Grounded LCEL RAG chain & Q&A handler
├── main.py                     # CLI entry point with interactive chat loop
├── app.py                      # Streamlit web application
├── test.py                     # Developer test runner
├── STEPS.md                    # In-depth engineering build log and notes
└── README.md                   # Project documentation
```

---

## 💡 Engineering Highlights

- **Overcoming the 30-Second API Limitation:** Sarvam's synchronous STT API rejects payloads larger than 30 seconds. The transcriber subdivides 10-minute macro chunks into 25-second micro-slices, dispatches them sequentially, and cleans up temporary files in a `finally` block to protect local disk space.
- **Local & Privacy-Preserving Architecture:** Audio transcription and vector embeddings execute locally on CPU using Whisper and HuggingFace, avoiding recurring API costs and ensuring private meeting recordings remain confidential.
- **Negative Prompt Guardrails:** The RAG system prompt enforces strict context adherence: *"If the answer is not found in the context, say: 'I could not find this information in the meeting transcript.'"* This prevents speculative hallucinations when users ask about topics omitted from the meeting.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
