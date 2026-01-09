# 🎙️ STT Arena - Speech-to-Text Model Comparison Tool

**STT Arena** je pokročilá Streamlit aplikácia na porovnávanie rôznych Speech-to-Text (STT) modelov s využitím AI-powered "Golden Transcript" generovaného pomocou Google Gemini.

## 🌟 Hlavné funkcie

- **Multi-Provider Comparison**: Porovnanie viacerých STT API naraz (Gladia, OpenAI Whisper, Behavioral Signals, Deepgram)
- **AI-Powered Golden Transcript**: Gemini analyzuje všetky transkripty a vytvorí najpresnejšiu verziu
- **Word Error Rate (WER)**: Automatický výpočet WER pre každý provider voči Golden Transcript
- **Metadata Analysis**: Zobrazenie diarizácie, emócií, sentimentu a ďalších metadát
- **Interaktívne UI**: Prehľadné Streamlit rozhranie s možnosťou editácie transkriptov
- **Export funkcionalita**: Stiahnutie transkriptov a výsledkov

## 📋 Požiadavky

- **OS**: Windows 10/11 (alebo Linux/macOS)
- **Python**: 3.9 alebo vyššia verzia
- **API Keys**: Aspoň jeden STT provider + Google Gemini API key

## 🚀 Inštalácia

### 1. Klonovanie repozitára

```bash
git clone <repository-url>
cd STT-arena
```

### 2. Vytvorenie virtuálneho prostredia

**Na Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Na Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Inštalácia závislostí

```bash
pip install -r requirements.txt
```

### 4. Konfigurácia API Keys

Skopírujte `.env.example` na `.env`:

```bash
copy .env.example .env  # Windows
# alebo
cp .env.example .env    # Linux/macOS
```

Upravte `.env` súbor a doplňte vaše API keys:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
GLADIA_API_KEY=your_actual_gladia_api_key
OPENAI_API_KEY=your_actual_openai_api_key
BEHAVIORAL_SIGNALS_API_KEY=your_actual_bs_api_key
BEHAVIORAL_SIGNALS_URL=https://api.behavioralsignals.com/...
DEEPGRAM_API_KEY=your_actual_deepgram_api_key
```

## 🎯 Získanie API Keys

### Google Gemini API
1. Navštívte [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Prihláste sa pomocou Google účtu
3. Vygenerujte nový API key
4. Skopírujte kľúč do `.env` súboru

### OpenAI (Whisper)
1. Navštívte [OpenAI Platform](https://platform.openai.com/api-keys)
2. Vytvorte nový API key
3. Skopírujte kľúč do `.env` súboru

### Gladia
1. Navštívte [Gladia](https://www.gladia.io/)
2. Zaregistrujte sa a vytvorte účet
3. V dashboarde nájdite API key
4. Skopírujte kľúč do `.env` súboru

### Deepgram
1. Navštívte [Deepgram Console](https://console.deepgram.com/)
2. Zaregistrujte sa a vytvorte projekt
3. Vytvorte nový API key
4. Skopírujte kľúč do `.env` súboru

### Behavioral Signals
1. Kontaktujte [Behavioral Signals](https://www.behavioralsignals.com/) pre prístup k API
2. Získajte API key a endpoint URL
3. Skopírujte údaje do `.env` súboru

## 🎮 Spustenie aplikácie

```bash
streamlit run app.py
```

Aplikácia sa automaticky otvorí v prehliadači na adrese `http://localhost:8501`

## 📖 Ako používať

### 1. Nastavenie v Sidebari

- **API Keys**: Zadajte vaše API keys (alebo použite keys z `.env`)
- **Výber Providerov**: Označte, ktoré STT služby chcete použiť
- **Upload Audio**: Nahrajte audio súbor (mp3, wav, m4a, ogg, flac)

### 2. Spustenie transkripcie

- Kliknite na tlačidlo **"🚀 Start Transcription Arena"**
- Aplikácia paralelne pošle audio všetkým vybraným providerom
- Gemini vytvorí Golden Transcript zo všetkých výsledkov

### 3. Analýza výsledkov

- **Golden Transcript**: Najpresnejšia verzia transkriptu (editovateľná)
- **Performance Metrics**: Tabuľka s WER skóre a časmi spracovania
- **Transcript Comparison**: Porovnanie jednotlivých transkriptov v taboch
- **Metadata**: Detailné informácie od každého providera

### 4. Export

- Stiahnite Golden Transcript ako TXT súbor
- Stiahnite všetky transkripty vrátane WER skóre

## 🏗️ Štruktúra projektu

```
STT-arena/
├── app.py                 # Hlavná Streamlit aplikácia
├── utils.py               # Pomocné funkcie a API integrácie
├── requirements.txt       # Python závislosti
├── .env.example          # Šablóna pre API keys
├── .env                  # Vaše API keys (nie je v git)
├── .gitignore           # Git ignore konfigurácia
└── README.md            # Tento súbor
```

## 🔧 Riešenie problémov

### Chyba: "Module not found"
```bash
pip install -r requirements.txt
```

### Chyba: "API key not found"
Skontrolujte, či:
- Máte vytvorený `.env` súbor
- API keys sú správne zadané (bez úvodzoviek)
- Súbor je v rovnakom adresári ako `app.py`

### Chyba: "File upload size exceeded"
Zvýšte limit v Streamlit konfigurácii:

Vytvorte `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 500
```

### Audio formát nie je podporovaný
Podporované formáty: WAV, MP3, M4A, OGG, FLAC
Použite online konvertor alebo FFmpeg na konverziu.

## 📊 O Word Error Rate (WER)

WER meria presnosť transkripcie:

- **0-10%**: Vynikajúca presnosť
- **10-20%**: Dobrá presnosť
- **20-30%**: Priemerná presnosť
- **30%+**: Nízka presnosť

WER sa počíta porovnaním každého providera s Golden Transcript vytvorený Gemini.

## 🎯 Príklad použitia

1. **Nahrávky Call Centra**: Porovnajte kvalitu transkripcie zákazníckych hovorov
2. **Meeting Recording**: Testujte rôzne STT modely na meetingoch
3. **Quality Assurance**: Automatické hodnotenie presnosti STT služieb
4. **Research**: Analyzujte výkon rôznych modelov na špecifických audio datasetoch

## 🔐 Bezpečnosť

- **Nikdy nezdieľajte** váš `.env` súbor
- API keys sú citlivé údaje - uchovávajte ich v bezpečí
- `.env` súbor je automaticky ignorovaný gitom (`.gitignore`)
- Audio súbory sú dočasné a automaticky sa mažú

## 🤝 Príspevky

Contributions sú vítané! Otvorte issue alebo pull request.

## 📝 Licencia

MIT License

## 🙏 Poďakovanie

- **Google Gemini**: Za AI-powered Golden Transcript generáciu
- **Streamlit**: Za skvelý framework na tvorbu web aplikácií
- **jiwer**: Za WER kalkuláciu
- Všetkým STT providerom za ich API

## 📞 Podpora

Ak máte otázky alebo problémy:
1. Skontrolujte túto dokumentáciu
2. Pozrite si [Streamlit dokumentáciu](https://docs.streamlit.io/)
3. Otvorte issue na GitHub

---

**Vyrobené s ❤️ pre lepšie porozumenie Speech-to-Text technológiám**
