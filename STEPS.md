# Building AI Video Assistant: Multimodal Meeting Intelligence & RAG
*Developer Build Log & Engineering Architecture Notes*

---

## 1. Project Motivation & Core Vision

Reviewing hour-long video recordings, Zoom meetings, and YouTube lectures is one of the biggest productivity drains for engineers, students, and teams. While commercial tools like Fireflies.ai or Otter.ai exist, they suffer from notable drawbacks:
- Expensive recurring subscriptions.
- Cloud lock-in and privacy concerns (uploading confidential audio to third-party servers).
- Inability to handle mixed languages (e.g., Hinglish / conversational Indic code-switching).

I set out to build **AI Video Assistant**—a fully autonomous, privacy-conscious multimodal meeting intelligence system. 

The application takes any **YouTube URL or local video/audio file (MP4, MP3, WAV, WEBM)**, normalizes the audio, transcribes and translates it using local or Indic-specialized models, extracts structured meeting deliverables (action items, decisions, follow-ups), and indexes the entire transcript into a local vector database for **conversational RAG Q&A**.

---

## 2. High-Level System Architecture

The pipeline is split into two distinct operational phases:

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

## 3. Step-by-Step Implementation Details

### Step 1: Environment & Tooling Configuration

I configured an isolated environment using `uv` for high-speed package management:

```bash
uv venv
source .venv/Scripts/activate  # On Windows PowerShell: .venv\Scripts\activate
```

#### Secrets Management (`.env`)
The system requires minimal external dependencies by relying on local compute where possible:
- `MISTRAL_API_KEY`: Powers the reasoning, summarization, and RAG chat (`mistral-small-latest`).
- `SARVAM_API_KEY`: Powers the speech-to-text translation endpoint for Indic/Hinglish audio.
- `WHISPER_MODEL`: Configurable local model size (`small` by default, balancing speed and accuracy).
- `SARVAM_STT_MODEL`: `saaras:v2.5` speech-to-text translation model.

*Security rule: All credentials are isolated in `.env` and excluded from version control via `.gitignore`.*

---

### Step 2: Audio Extraction & Ingestion Pipeline (`utils/audio_processor.py`)

Raw video and audio files vary wildly in format, sample rate, and channel layout. I built an ingestion router that standardizes any input into speech-ready WAV files:

1. **YouTube Extraction (`download_youtube_audio`):**
   - Configured `yt-dlp` with `FFmpegExtractAudio` post-processing.
   - Extracts audio at 192 kbps WAV directly into a local `downloades/` directory.
2. **Local Media Normalization (`convert_to_wav`):**
   - Uses `pydub.AudioSegment` to automatically parse MP4, MP3, M4A, or WEBM.
   - Converts stereo channels down to **single-channel mono** and downsamples/upsamples to **16 kHz**—the optimal sample rate for Whisper acoustic models.
3. **Macro Chunking (`chunk_audio`):**
   - Long recordings (30 to 90 minutes) cause memory spikes if processed in a single pass.
   - I built a slicing algorithm that breaks the audio into 10-minute WAV segments (`wav_path_chunk_{i}.wav`).
4. **Unified Input Router (`process_input`):**
   - Inspects the source string. If it begins with `http://` or `https://`, it routes through `yt-dlp`; otherwise, it routes to `pydub`.

---

### Step 3: Hybrid Speech-to-Text & Translation Engine (`core/transcriber.py`)

To deliver high-accuracy transcription across both standard English and multilingual Indian contexts, I implemented a dual-engine architecture:

#### Engine A: Local OpenAI Whisper (English / Global)
- **Zero API Costs:** Runs entirely locally via `openai-whisper` and PyTorch.
- **Singleton Model Caching:** Implemented `load_model()` with a global `_model` instance so the 460 MB Whisper weights are loaded into RAM only once per process lifetime.

#### Engine B: Sarvam AI STT-Translate (Hinglish / Indic)
While Whisper is capable, it frequently stumbles on colloquial Hinglish and code-switched audio. Sarvam AI's `saaras:v2.5` model handles Hinglish nuances with high fidelity and translates directly to English.

**Engineering Workaround for Sarvam's 30-Second API Ceiling:**
Sarvam's synchronous STT API strictly rejects audio files longer than 30 seconds. To overcome this limitation:
1. I built a micro-chunker (`transcribe_chunk_sarvam`) that subdivides each 10-minute audio segment into **25-second micro-WAV slices** (adding a 5-second buffer).
2. Slices are posted to `api.sarvam.ai/speech-to-text-translate` via HTTP POST multipart requests.
3. Transcripts are aggregated sequentially.
4. **Clean Disk Hygiene:** All temporary micro-WAV files are deleted inside a `try/finally` block to prevent disk bloat.

---

### Step 4: Map-Reduce Summarization (`core/summarizer.py`)

Passing a 1-hour meeting transcript (10,000+ words) directly into an LLM prompt degrades reasoning quality and risks context-window exhaustion. 

I implemented an **LCEL Map-Reduce Summarization Architecture**:
1. **Splitting:** The transcript is sliced into 3,000-character segments with 200-character overlap using `RecursiveCharacterTextSplitter`.
2. **Map Phase:** Each chunk is summarized independently using an LCEL chain (`map_prompt | llm | StrOutputParser()`).
3. **Reduce Phase:** All partial summaries are joined and passed into an executive prompt (`combined_prompt`) that synthesizes them into a unified bullet-point meeting brief.
4. **Title Generation:** A focused LCEL chain analyzes the first 2,000 characters of the transcript to output an actionable meeting title ($\le 8$ words).

---

### Step 5: Structured Meeting Deliverables Extraction (`core/extractor.py`)

Summaries alone often bury key operational details. I created a reusable chain factory (`build_chain`) to extract three critical deliverables:

1. **Action Items:** Extracts a numbered list containing:
   - Task description
   - Owner (who is accountable)
   - Deadline (if mentioned, otherwise explicitly noted as *Not specified*)
2. **Key Decisions:** Extracts finalized agreements and technical decisions made during the call.
3. **Open Questions:** Identifies unresolved blockers, topics needing follow-up, or unanswered audience questions.

---

### Step 6: Local Vector Store & Grounded RAG Pipeline (`core/vector_store.py` & `core/rag_engine.py`)

To allow users to converse directly with their meeting recording, I built an end-to-end Retrieval-Augmented Generation (RAG) system:

#### 1. Vector Indexing (`core/vector_store.py`)
- **Granular Chunking:** Slices the transcript into 500-character chunks with 50-character overlap, attaching a `chunk_index` metadata tag to preserve chronological sequence.
- **Local Embeddings:** Uses HuggingFace's `all-MiniLM-L6-v2` running on CPU via `sentence-transformers`. This keeps embedding generation fast, lightweight, and completely free.
- **Persistent Storage:** Saves embeddings directly into a local ChromaDB collection (`meeting_transcript`) in `./vector_db`.

#### 2. Grounded RAG Chain (`core/rag_engine.py`)
- **Retrieval:** Configures a similarity retriever fetching top-$k=4$ most relevant chunks.
- **Negative Prompt Constraint:** To prevent the LLM from speculating or answering from general knowledge, the system prompt explicitly enforces:
  > *"Answer based ONLY on the meeting transcript context provided below. If the answer is not found in the context, say: 'I could not find this information in the meeting transcript.'"*
- **LCEL Pipe Structure:**
  ```python
  rag_chain = (
      {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )
  ```

---

### Step 7: Dual Interface (CLI & Streamlit Dashboard)

#### CLI Entry Point (`main.py`)
Provides a complete terminal interface that prints meeting minutes with clean ASCII dividers and launches an interactive while-loop chat (`You:` / `Assistant:`).

#### Production Streamlit Application (`app.py`)
Built a custom, responsive web application:
- **Design System:** Custom dark aesthetic (`#0a0a0f`), animated SVG grid, glassmorphism cards, and typography pairing `Syne` (headers) with `JetBrains Mono` (code/body).
- **Sidebar & Live Status Monitor:** Displays input controls and a 6-stage pulsing visual status indicator:
  - 🔊 Audio Processing ➔ 📝 Transcription ➔ 🏷️ Title Generation ➔ 📋 Summarisation ➔ 🔍 Extraction ➔ 🧠 RAG Engine
- **Multi-Card Layout:** Dedicated cards for Session Title, Executive Summary, Full Collapsible Transcript, Action Items, Key Decisions, and Open Questions.
- **Interactive RAG Chat:** Threaded message history with stylized chat bubbles and a clear-chat button.

---

## 4. Key Engineering Challenges & Solutions

### 1. The 30-Second API Limitation on Cloud STT
- **Challenge:** Sarvam's free-tier translation endpoint is optimal for Hinglish but rejects any audio file over 30 seconds.
- **Solution:** Sliced the 10-minute audio chunks into 25-second micro-segments, dispatched them sequentially, concatenated the transcript strings, and cleaned up temporary WAV slices in a `finally` block to protect local disk space.

### 2. Audio Processing System Dependencies
- **Challenge:** Python libraries like `pydub` and `yt-dlp` depend on native FFmpeg binaries on the operating system, which aren't bundled with pip.
- **Solution:** Standardized FFmpeg installation via Windows `winget` (`winget install Gyan.FFmpeg`) and added runtime checks to verify audio codec availability before processing.

### 3. Mitigating Hallucinations in Conversational Q&A
- **Challenge:** When asked about topics not discussed in the meeting, standard LLMs often hallucinate plausible-sounding answers based on broad internet training data.
- **Solution:** Implemented strict negative prompting in the LCEL RAG prompt and verified that the model responds with the fallback notice whenever the vector search returns irrelevant chunks.

---

## 5. How to Run the Project

### Prerequisites
Make sure FFmpeg is installed and accessible in your system PATH:
```powershell
winget install Gyan.FFmpeg
```

### Running the Web Interface (Streamlit)
```bash
cd video_agent
.venv\Scripts\activate       # On Linux/macOS: source .venv/bin/activate
streamlit run app.py
```
Visit **`http://localhost:8501`** (or `8502`), paste a YouTube link or local video file, choose the language, and click **⚡ Analyse**.

### Running via Terminal (CLI)
```bash
cd video_agent
.venv\Scripts\activate
python main.py
```

---

## 6. Future Enhancements

- [ ] **Speaker Diarization:** Identify and tag individual speakers (e.g., Speaker 1 vs Speaker 2) using `pyannote.audio`.
- [ ] **Automated Export Integrations:** One-click export of Action Items directly to Notion databases, Trello boards, or Slack channels.
- [ ] **Async Batch Processing:** Parallelize the 25-second Sarvam API calls using `asyncio` to reduce transcription latency on long videos.
