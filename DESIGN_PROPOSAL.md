# STT Arena - Návrh rozšírení

## 1. PROBLÉM: WER hodnoty príliš nízke

### Root Cause
WER (Word Error Rate) hodnoty sú umelecky nízke, pretože:
- **Golden Transcript je generovaný Z provider transkriptov** (Gemini syntetizuje z nich)
- Keď porovnávame provider transkript s golden transkriptom → WER je nízky
- **Nie je to skutočný "Ground Truth"** (human-verified reference)

### Riešenie
```
Pridať podporu pre Ground Truth Reference:
├── Užívateľ nahrá skutočný prepis (human-verified)
├── WER sa počíta proti Ground Truth (nie proti Golden)
├── Golden Transcript = pomocný output (AI syntéza)
└── Ground Truth = skutočná referencia pre metriky
```

**Implementácia:**
1. V sidebar pridať: "📋 Ground Truth Reference (Optional)"
2. Rozlíšiť medzi:
   - **Manual Transcript** → ďalší provider na porovnanie
   - **Ground Truth** → referencia pre WER/CER výpočet
3. Ak Ground Truth existuje → použiť ho namiesto Golden
4. Ak neexistuje → použiť Golden (ako teraz)

---

## 2. DATABASE: Ukladanie hodnotení a prepisov

### Architektúra: SQLite
```
stt_arena.db
├── audio_files          # Metadata o audio súboroch
├── transcriptions       # STT výsledky od všetkých providerov
├── evaluations          # WER/CER/Cost metriky
├── qa_sessions          # QA Arena sessions
└── qa_evaluations       # LLM odpovede a hodnotenia
```

### Schema

#### Table: audio_files
```sql
CREATE TABLE audio_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    duration_seconds REAL,
    language TEXT,
    file_size_mb REAL,
    audio_hash TEXT UNIQUE  -- SHA256 pre deduplikáciu
);
```

#### Table: transcriptions
```sql
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file_id INTEGER,
    provider_name TEXT NOT NULL,
    transcript_text TEXT,
    processing_time_seconds REAL,
    success BOOLEAN,
    error_message TEXT,
    metadata JSON,  -- diarization, segments, confidence, etc.
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (audio_file_id) REFERENCES audio_files(id)
);
```

#### Table: evaluations
```sql
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file_id INTEGER,
    evaluation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ground_truth_text TEXT,  -- Human-verified reference
    golden_transcript TEXT,   -- AI-generated (Gemini)
    golden_generator TEXT,    -- 'Gemini 2.5' alebo 'Groq (fallback)'
    FOREIGN KEY (audio_file_id) REFERENCES audio_files(id)
);
```

#### Table: evaluation_scores
```sql
CREATE TABLE evaluation_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER,
    transcription_id INTEGER,
    provider_name TEXT,
    wer_score REAL,
    cer_score REAL,
    cost_usd REAL,
    cost_effectiveness_score REAL,  -- Performance/Cost ratio
    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id),
    FOREIGN KEY (transcription_id) REFERENCES transcriptions(id)
);
```

#### Table: qa_sessions
```sql
CREATE TABLE qa_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file_id INTEGER,
    transcript_source TEXT,  -- Ktorý provider transcript sa použil
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (audio_file_id) REFERENCES audio_files(id)
);
```

#### Table: qa_questions
```sql
CREATE TABLE qa_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    qa_session_id INTEGER,
    question_text TEXT NOT NULL,
    question_type TEXT,  -- 'factual', 'summary', 'sentiment', 'entity_extraction'
    ground_truth_answer TEXT,  -- Optional: human-verified answer
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (qa_session_id) REFERENCES qa_sessions(id)
);
```

#### Table: qa_answers
```sql
CREATE TABLE qa_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER,
    llm_provider TEXT,  -- 'Gemini', 'Groq', 'Mistral'
    llm_model TEXT,     -- 'gemini-2.5-flash-lite', 'llama-3.3-70b', etc.
    answer_text TEXT,
    processing_time_seconds REAL,
    cost_usd REAL,
    tokens_used INTEGER,
    quality_score REAL,  -- Porovnanie s ground truth alebo golden answer
    metadata JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (question_id) REFERENCES qa_questions(id)
);
```

### Výhody SQLite
- ✅ Žiadna externá DB infraštruktúra
- ✅ Single file database (`stt_arena.db`)
- ✅ Relačné dotazy (JOIN, agregácie)
- ✅ Export do CSV/JSON
- ✅ Funguje na Streamlit Cloud
- ✅ ACID compliance

---

## 3. QA ARENA: Testovanie LLM na Q&A

### Architektúra
```
QA Arena Pipeline:
├── Input: STT Transcript (from best provider)
├── Question Generation:
│   ├── Auto-generate: Gemini vytvorí 5-10 otázok z prepisu
│   └── Manual: Užívateľ zadá vlastné otázky
├── LLM Testing:
│   ├── Gemini (gemini-2.5-flash-lite, gemini-2.5-flash)
│   ├── Groq (llama-3.3-70b-versatile, mixtral-8x7b)
│   └── Mistral API (mistral-small, mistral-large)
├── Answer Generation:
│   └── Každý LLM odpovedá na všetky otázky
├── Evaluation:
│   ├── Golden Answer: Gemini Opus vytvorí referenciu
│   ├── Similarity Score: Semantic similarity (embeddings)
│   ├── Factual Accuracy: Fact-checking proti transcriptu
│   └── Cost Tracking: $/1K tokens
└── Output: LLM Leaderboard (Quality × Cost Effectiveness)
```

### Typy otázok
1. **Factual Questions** - "Koľko zákazníkov volalo minulý týždeň?"
2. **Summarization** - "Zhrň hlavné body rozhovoru"
3. **Sentiment Analysis** - "Aká bola nálada zákazníka?"
4. **Entity Extraction** - "Kto bol volajúci a o čo išlo?"
5. **Problem Resolution** - "Ako bol vyriešený problém zákazníka?"

### LLM Providers a modely

#### Gemini API (Google)
```python
Models:
- gemini-2.5-flash-lite  # FREE: 10 RPM, 4M TPM
- gemini-2.5-flash       # FREE: 5 RPM, 4M TPM
- gemini-2.5-pro         # FREE: 2 RPM, 50K TPD

Pricing (Paid tier):
- Flash Lite: $0.00001/1K tokens input, $0.00004/1K output
- Flash: $0.00001/1K tokens input, $0.00004/1K output
- Pro: $0.00125/1K tokens input, $0.005/1K output
```

#### Groq API
```python
Models:
- llama-3.3-70b-versatile   # FREE: 30 RPM, 20K TPM
- llama-3.1-8b-instant      # FREE: 30 RPM, 20K TPM
- mixtral-8x7b-32768        # FREE: 30 RPM, 20K TPM

Pricing: FREE tier (rate limited)
```

#### Mistral API
```python
Models (Free tier):
- mistral-small-latest   # FREE: ~25 RPM limit
- mistral-large-latest   # FREE: ~10 RPM limit
- open-mistral-7b        # FREE: ~30 RPM

Pricing (Paid):
- Small: €0.001/1K tokens
- Medium: €0.0027/1K tokens
- Large: €0.008/1K tokens
```

### Metriky hodnotenia

#### Quality Score (0-100)
```python
quality_score = (
    semantic_similarity * 0.4 +      # Embedding similarity s golden answer
    factual_accuracy * 0.4 +         # Fact-checking proti transcriptu
    completeness * 0.2               # Či odpoveď pokrýva všetky aspekty
)
```

#### Cost Effectiveness Score
```python
cost_effectiveness = (quality_score / 100) / cost_per_answer
# Higher is better: high quality, low cost
```

#### Leaderboard Metrics
- **Quality Score** (0-100)
- **Avg Response Time** (seconds)
- **Cost per Answer** ($)
- **Cost Effectiveness** (Quality/$)
- **Success Rate** (% správnych odpovedí)

---

## 4. UI CHANGES

### Nová štruktúra aplikácie
```
app.py (Main)
├── Tab 1: STT Arena (existing)
│   ├── Ground Truth upload (NEW)
│   └── WER proti Ground Truth (FIXED)
├── Tab 2: QA Arena (NEW)
│   ├── Select transcript source
│   ├── Question generation/upload
│   ├── LLM provider selection
│   └── Results comparison
├── Tab 3: Analytics & Database (NEW)
│   ├── Historical data
│   ├── Provider leaderboards
│   ├── Cost analysis over time
│   └── Export to CSV/JSON
└── Tab 4: Settings
    └── API keys, DB settings
```

### Sidebar Updates
```python
# STT Arena Mode
st.sidebar.radio("Mode", [
    "🎙️ STT Arena",
    "🤖 QA Arena",
    "📊 Analytics"
])
```

---

## 5. IMPLEMENTATION PLAN

### Phase 1: Fix WER + Database (Priority 1)
- [x] Identify WER issue
- [ ] Add Ground Truth upload
- [ ] Update WER calculation logic
- [ ] Create SQLite schema
- [ ] Implement database save/load functions
- [ ] Migrate existing session history to DB

### Phase 2: QA Arena Core (Priority 2)
- [ ] Create `qa_utils.py`
- [ ] Implement Mistral API integration
- [ ] Add question generation (Gemini)
- [ ] Build QA Arena UI tab
- [ ] Implement answer evaluation

### Phase 3: Analytics & Leaderboards (Priority 3)
- [ ] Provider comparison charts
- [ ] Cost-effectiveness analysis
- [ ] Historical trends
- [ ] Export functionality

### Phase 4: Advanced Features (Priority 4)
- [ ] Batch processing (multiple audio files)
- [ ] Custom evaluation prompts
- [ ] A/B testing framework
- [ ] API endpoints (FastAPI wrapper)

---

## 6. FILE STRUCTURE
```
/home/user/STT-arena/
├── app.py                    # Main Streamlit app (UPDATED)
├── utils.py                  # STT utilities (UPDATED)
├── qa_utils.py               # QA Arena utilities (NEW)
├── database.py               # SQLite operations (NEW)
├── evaluation.py             # Scoring algorithms (NEW)
├── requirements.txt          # Dependencies (UPDATED)
├── stt_arena.db              # SQLite database (AUTO-CREATED)
├── .streamlit/
│   └── secrets.toml          # API keys
└── DESIGN_PROPOSAL.md        # This file
```

---

## 7. NEXT STEPS

**Immediate (Dnes):**
1. ✅ Create this design document
2. ⏳ Review and approve with user
3. ⏳ Implement Ground Truth support
4. ⏳ Fix WER calculation
5. ⏳ Create database schema

**Short-term (Tento týždeň):**
1. Implement SQLite integration
2. Build QA Arena MVP
3. Add Mistral API support
4. Create basic analytics tab

**Long-term (Budúci týždeň):**
1. Advanced cost-effectiveness scoring
2. Historical trend analysis
3. Batch processing support
4. Documentation and deployment

---

## 8. PRICING COMPARISON (Updated 2026)

### STT Providers (Cost per minute)
| Provider | Model | Cost/min | Free Tier | Notes |
|----------|-------|----------|-----------|-------|
| Groq | Whisper Large v3 | FREE | ✅ 25MB files | Best free option |
| OpenAI | Whisper | $0.006 | ❌ | Industry standard |
| Deepgram | Whisper Large | $0.0125 | 12K min/year | Good for long audio |
| Gladia | v2 | $0.0183 | 300 min | Diarization included |
| ElevenLabs | Scribe v2 | $0.10 | ❌ | Premium quality |

### LLM Providers (Cost per 1K tokens)
| Provider | Model | Input | Output | Free Tier |
|----------|-------|-------|--------|-----------|
| Gemini | Flash Lite | $0.00001 | $0.00004 | ✅ 10 RPM |
| Groq | Llama 3.3 70B | FREE | FREE | ✅ 30 RPM |
| Mistral | Small | €0.001 | €0.001 | ✅ 25 RPM |
| OpenAI | GPT-4o | $0.0025 | $0.010 | ❌ Pay only |

---

## QUESTIONS FOR USER

1. **Ground Truth**: Chceš ručne nahrávať Ground Truth pre každý audio súbor, alebo stačí optional?
2. **QA Questions**: Auto-generate z Gemini, alebo preferuješ manuálny vstup?
3. **Database location**: Local SQLite súbor, alebo cloudová DB (PostgreSQL na Supabase)?
4. **Export format**: Stačí CSV, alebo potrebuješ aj JSON/Excel?
5. **Batch processing**: Potrebuješ nahrávať viacero audio súborov naraz?

---

**Autor:** Claude (STT Arena Assistant)
**Dátum:** 2026-01-22
**Verzia:** 1.0
