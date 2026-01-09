# 🚀 Quick Start Guide - STT Arena

Tento návod vás prevedie základným nastavením a prvým použitím aplikácie.

## ⚡ Rýchla inštalácia (5 minút)

### Krok 1: Inštalácia Python balíčkov

```bash
# Aktivujte virtuálne prostredie (ak ešte nie je aktivované)
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

# Nainštalujte závislosti
pip install -r requirements.txt
```

### Krok 2: Nastavenie API Keys

**Minimálne potrebné keys:**
- ✅ **GEMINI_API_KEY** (povinné - na vytvorenie Golden Transcript)
- ✅ **Aspoň jeden STT provider** (napr. OPENAI_API_KEY pre Whisper)

**Vytvorte .env súbor:**

```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

**Upravte .env a pridajte vaše keys:**

```env
GEMINI_API_KEY=AIzaSy...  # Váš Gemini API key
OPENAI_API_KEY=sk-...     # Váš OpenAI API key
```

### Krok 3: Spustenie

```bash
streamlit run app.py
```

Aplikácia sa otvorí na: **http://localhost:8501**

## 🎯 Prvé použitie

### 1. Pripravte si audio súbor
- Formát: WAV, MP3, M4A, OGG, alebo FLAC
- Odporúčaná dĺžka: 30 sekúnd - 10 minút
- Ideálne: Nahrávky hovorov, meetingov, alebo podcastov

### 2. V aplikácii:

1. **Sidebar** (vľavo):
   - Skontrolujte, že sa načítali vaše API keys
   - Označte checkboxy pri provideroch, ktoré chcete použiť
   - Nahrajte audio súbor pomocou "Upload Audio File"

2. **Hlavná stránka**:
   - Prehrať audio (overíte, že sa správne nahralo)
   - Kliknite **"🚀 Start Transcription Arena"**
   - Počkajte na transkripciu (trvá 10-60 sekúnd podľa providera)

3. **Výsledky**:
   - Pozrite si **Golden Transcript** (najpresnejšia verzia)
   - Skontrolujte **WER skóre** v tabuľke (čím nižšie, tým lepšie)
   - Porovnajte jednotlivé transkripty v **taboch**
   - Pozrite si **metadata** (diarizácia, emócie, atď.)

### 3. Export
- Stiahnite Golden Transcript ako TXT
- Alebo stiahnite všetky výsledky naraz

## 🆓 Získanie FREE API Keys

### Google Gemini (FREE)
1. ➡️ https://makersuite.google.com/app/apikey
2. Prihláste sa Google účtom
3. Kliknite "Create API Key"
4. Skopírujte kľúč → pridajte do `.env`

**Free tier**: 60 requestov/minútu

### OpenAI Whisper (PAID - $0.006/minúta)
1. ➡️ https://platform.openai.com/api-keys
2. Zaregistrujte sa / prihláste
3. Kliknite "Create new secret key"
4. Skopírujte kľúč → pridajte do `.env`

**Credit**: Potrebujete pridať platobný spôsob (aj $5 stačí)

### Deepgram (FREE Trial - $200 credit)
1. ➡️ https://console.deepgram.com/signup
2. Zaregistrujte sa emailom
3. Dostanete $200 free credit
4. V Console → API Keys → Create new key
5. Skopírujte kľúč → pridajte do `.env`

### Gladia (FREE Trial)
1. ➡️ https://app.gladia.io/auth/signup
2. Zaregistrujte sa
3. Dostanete free credits na testovanie
4. Dashboard → API Keys
5. Skopírujte kľúč → pridajte do `.env`

## 🎓 Test s ukážkovým audio

Ak nemáte vlastný audio súbor, použite:
1. YouTube video → stiahnite audio pomocą: https://ytmp3.nu/
2. Nahrajte zvukovú nahrávku telefonátom na mobile
3. Text-to-Speech: https://ttsmp3.com/ → vytvorte test audio

## ❓ Časté otázky

**Q: Ktoré providery mám použiť?**
- Pre začiatok: **Gemini + OpenAI Whisper** (najľahšie získať API keys)
- Pre profesionálne použitie: **Všetky 4** (najlepšie porovnanie)

**Q: Prečo je WER skóre vysoké?**
- Nízka kvalita audia (šum, echo)
- Nízky sampling rate
- Náročný akcent alebo dialekt
- Technické termíny, ktoré model nepozná

**Q: Môžem upraviť Golden Transcript?**
- Áno! Kliknite do textového poľa "Golden Transcript"
- Upravte text
- Kliknite "🔄 Recalculate WER" → WER sa prepočíta

**Q: Ako dlhé audio môžem nahrať?**
- Maximum: 500 MB (nastavené v config.toml)
- Odporúčané: do 100 MB (lepšia rýchlosť)
- Pre veľké súbory: rozdeľte ich na menšie časti

**Q: Sú moje dáta v bezpečí?**
- Audio súbory sa posielajú priamo na API providerov
- Nič sa neukladá natrvalo na serveri
- Temporary files sa automaticky mažú

**Q: Koľko to stojí?**
- **Gemini**: FREE (60 req/min)
- **OpenAI Whisper**: $0.006/minúta audia
- **Deepgram**: $200 free credit, potom $0.0043/minúta
- **Gladia**: Free trial, potom custom pricing

**Príklad kalkulácia:**
- 10 minút audia × 3 providery
- OpenAI: $0.06
- Deepgram: $0.043
- Gemini: FREE
- **Spolu: ~$0.10**

## 🔥 Pro Tips

1. **Pre najlepšie výsledky:**
   - Použite audio s dobrým sample rate (16kHz+)
   - Minimalizujte background noise
   - Jasná reč, nie príliš rýchla

2. **Optimalizácia nákladov:**
   - Testujte najprv na kratších vzorkách
   - Použite len potrebných providerov
   - Využite free credits

3. **Workflow:**
   - Najprv nahrajte 30-sekundovú vzorku
   - Otestujte všetkých providerov
   - Zvoľte najlepšieho → použite ho na celé audio

4. **Debug režim:**
   - Pozrite si Metadata v expanderoch
   - Skontrolujte diarizáciu (kto hovorí)
   - Analyzujte sentiment/emócie

## 🎉 Hotovo!

Teraz ste pripravení porovnávať STT modely ako profík!

Pre viac informácií pozrite [README.md](README.md)

---
**Happy transcribing! 🎙️✨**
