# Emotion-Aware Conversational Voice Agent — Build Roadmap

**Course**: DTSC 5082 · UNT · Fall 2026
**Team**: 06

---

## Completed

- **Phase 1 — Baseline classifiers** (RF 0.625, XGBoost 0.719 in current env)
  - Feature extraction: 39 features × 576 clips, speaker-independent split (16/4/4 actors)
  - Random Forest + XGBoost, confusion matrices, feature importance
  - Per-clip probability export (`clip_probabilities.csv`)
- **MoodManager v1** — EMA-smoothed mood state (γ=0.7, persistence 2 turns)
  - Real-data experiment: 12-turn scripted conversation, 1000 runs
  - Reactive switches 4.45±1.74 vs MoodManager 1.36±0.62
- **Research poster** — presented Oct 6

---

## Build Sequence (what's next, in order)

### A. Thin end-to-end agent — DONE (2026-10-31)
Wire the minimum viable pipeline printed to the terminal:
1. Whisper ASR (tiny.en) → text
2. XGBoost probabilities → emotion label + distress score
3. MoodManager → tone mode (DEFAULT / EMPATHETIC / CALMING / POSITIVE_ENCOURAGING)
4. `build_system_prompt(tone, transcript)` → prompt
5. Gemini (or template fallback) → reply text
6. Print to screen (no TTS yet)

**Goal**: Prove the loop works. No memory, no streaming, no audio output.

**Result** (`scripts/agent.py --script`, 12-turn label script, test speaker 16):
- MoodManager: 1 mode switch (DEFAULT → EMPATHETIC on turn 7, ignoring the one-off angry spike on turn 3) vs reactive baseline (`--no-mood`): 4 switches
- Average turn latency ~2.5 s (LLM ~1.3–3 s, ASR ~0.45 s, emotion ~0.15 s); 5.1 s on a re-run where gemini-3.6-flash was rate-limited (429) on 10/12 turns
- Shipped as built: 2 modes only (DEFAULT / EMPATHETIC, MoodManager v1); `build_system_prompt(mode, perceived_emotion, memory_facts)`; tone driven by mood state only, never the per-turn label
- **Confound**: the Gemini fallback chain (3.6-flash → 3.5-flash-lite → 3.1-flash-lite) can change model mid-conversation, so tone differences may partly come from the model. Every attempt is logged per turn; pin one model for the Task F evaluation
- RAVDESS transcripts are lexically neutral ("Kids are talking by the door"), so reply content is uninformative; tone is the only signal in scripted runs

### B. MemoryManager — JSON scaffold
- Conversation buffer (last N turns)
- User-fact store (name, preferences) with emotion-gated write
- JSON file backend — no vector DB yet

### C. Bi-LSTM + CNN classifiers
- CNN on mel-spectrograms (3-class)
- Bi-LSTM on feature sequences (3-class)
- Compare against XGBoost 0.719 baseline in same environment
- Update model_comparison.csv with all three

### D. RAG upgrade (MemoryManager v2)
- Replace JSON fact store with Chroma/FAISS
- MiniLM embeddings + emotion-weighted retrieval
- Integrate into agent loop

### E. TTS — CSM-1B on Colab
- Google Colab notebook running Sesame CSM-1B
- REST endpoint the agent calls for audio output
- Tone-conditioned voice (map mood mode → speaker style)

### F. Evaluation & demo
- End-to-end evaluation: emotion accuracy, response quality, mood stability
- Prepare demo script and recording

### G. Repo restructure
- Clean folder layout per `Modular Project Structure.pdf`
- Final README, docstrings, reproducibility check

---

## Environment

- Python 3.12.10, CPU-only torch 2.15.0
- Venv: `(emotion-agent-312)` in `Code/.venv312/`
- Activate (PowerShell, from `Code/`): `.\.venv312\Scripts\Activate.ps1`
- Pinned deps: `requirements.txt` (unpinned originals in `requirements.in`)
- The original Python 3.14.5 venv (`Code/.venv/`) is kept as a backup and is no longer used
