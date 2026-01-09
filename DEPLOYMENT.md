# 🚀 Streamlit Cloud Deployment Guide

Návod na nasadenie STT Arena aplikácie na Streamlit Cloud pre zdieľanie s kolegami.

## 📋 Predpoklady

- GitHub repozitár s kódom (✅ máš ho)
- Streamlit Cloud účet (zadarmo na https://streamlit.io/cloud)
- API keys pre STT providerov

---

## 🎯 Krok 1: Priprav GitHub repozitár

Tvoj repozitár už obsahuje všetko potrebné:

```
STT-arena/
├── app.py                  ✅ Hlavná aplikácia
├── utils.py                ✅ Helper funkcie
├── requirements.txt        ✅ Python závislosti
├── .streamlit/config.toml  ✅ Konfigurácia
├── README.md               ✅ Dokumentácia
└── .gitignore             ✅ Ignoruje .env
```

**DÔLEŽITÉ:** `.env` súbor **NIE JE** v GitHube (je ignorovaný), čo je správne! ✅

---

## 🎯 Krok 2: Vytvor Streamlit Cloud účet

1. Choď na: **https://streamlit.io/cloud**
2. Klikni **"Sign up"**
3. Prihlás sa cez GitHub účet
4. Autorizuj Streamlit prístup k tvojim repozitárom

---

## 🎯 Krok 3: Nasaď aplikáciu

### A) V Streamlit Cloud dashboarde:

1. Klikni **"New app"** alebo **"Deploy an app"**
2. Vyber:
   - **Repository:** `MSGreenbluMe/STT-arena` (alebo tvoj GitHub username)
   - **Branch:** `claude/stt-model-comparison-app-H8nP9` (alebo `main`)
   - **Main file path:** `app.py`
3. Klikni **"Advanced settings"** (pred Deploy)

### B) Pridaj API Keys do Secrets:

V **Advanced settings** → **Secrets** sekcii zadaj:

```toml
# Skopíruj toto a nahraď skutočnými hodnotami:

GEMINI_API_KEY = "AIzaSyDGSUavOcVehEj8aQAnVz2Dv9PCD7PstBg"
GLADIA_API_KEY = "974dce7f-037f-425d-8790-e3504d324224"
DEEPGRAM_API_KEY = "4d93678ab6ba66050d08faddd354aeb001b14c3b"
BEHAVIORAL_SIGNALS_API_KEY = "0db7c878cf91bebe30dea65cf588121d"
BEHAVIORAL_SIGNALS_URL = "https://api.behavioralsignals.com/v1/transcribe"
```

**Formát je TOML** - použite `=` a úvodzovky!

### C) Deploy:

4. Klikni **"Deploy!"**
5. Počkaj 2-5 minút na build a deployment
6. Aplikácia sa spustí automaticky

---

## 🎯 Krok 4: Zdieľaj s kolegami

Po úspešnom deploymente dostaneš URL, napr.:

```
https://stt-arena-msgreenblueme.streamlit.app
```

Túto URL môžeš zdieľať s kýmkoľvek:
- ✅ Funguje v každom browseri
- ✅ Nepotrebujú nič inštalovať
- ✅ API keys sú bezpečné (nie sú viditeľné)
- ✅ Zadarmo na Streamlit Community Cloud

---

## ⚙️ Správa aplikácie

### Aktualizácia kódu:

1. Push zmeny do GitHub repozitára:
   ```bash
   git add .
   git commit -m "Update feature X"
   git push
   ```

2. Streamlit Cloud automaticky redeployuje aplikáciu (cca 2 min)

### Zmena API Keys:

1. V Streamlit Cloud dashboarde
2. Klikni na aplikáciu → **"Settings"** (⚙️)
3. Klikni **"Secrets"**
4. Uprav hodnoty
5. Klikni **"Save"**
6. Aplikácia sa reštartuje automaticky

### Zobrazenie logov:

- V Streamlit Cloud dashboarde → **"Manage app"** → **"Logs"**
- Vidíš tu chyby a debug info

---

## 🔒 Bezpečnosť

### ✅ SPRÁVNE (čo robíš):
- API keys sú v Streamlit Secrets (nie v kóde)
- `.env` súbor nie je v GitHube
- Secrets nie sú viditeľné pre používateľov aplikácie

### ❌ NIKDY nerob:
- Nechoď API keys do kódu
- Necommituj `.env` súbor
- Nezdieľaj secrets v README alebo komentároch

---

## 🎛️ Nastavenia výkonu

Ak aplikácia beží pomaly alebo crashuje:

### 1. Zväčši resources (pre platené účty):

V **Advanced settings**:
```
Python version: 3.11
Resources: Standard (alebo Large)
```

### 2. Optimalizuj veľkosť audio súborov:

V `app.py` je už nastavené na max 500 MB, ale odporúčame max 50 MB pre rýchlejšie spracovanie.

---

## 🐛 Riešenie problémov

### Aplikácia sa nezapne:

1. **Skontroluj logy** v Streamlit Cloud dashboarde
2. Chyba "Module not found" → Pridaj chýbajúci balíček do `requirements.txt`
3. Chyba "Secret not found" → Skontroluj názvy v Secrets sekcii

### API keys nefungujú:

1. Overte, že názvy v Secrets sú **PRESNE** ako v kóde:
   ```
   GEMINI_API_KEY (nie gemini_api_key ani GeminiApiKey)
   ```
2. Overte, že keys majú správny formát (bez medzier na začiatku/konci)

### Aplikácia je pomalá:

1. Zmenšite audio súbory pred nahraním (odporúčané: max 10 MB)
2. Použite len potrebných STT providerov (nie všetkých naraz)
3. Zvážte upgrade na Streamlit Team/Enterprise

---

## 💰 Ceny

**Streamlit Community Cloud:**
- ✅ **ZADARMO** pre verejné aplikácie
- ✅ 1 GB RAM, 1 CPU core
- ✅ Neobmedzený počet používateľov
- ✅ Vhodné pre STT Arena

**Streamlit Team/Enterprise:**
- 💲 Od $20/mesiac
- Viac resources, private apps, SSO

**API Costs (odhadované):**
- Gemini: FREE (60 req/min)
- Gladia: Od $0.002/min
- Deepgram: Od $0.0043/min
- Behavioral Signals: Custom pricing

---

## 📊 Monitoring využitia

Sledujte API costs:
1. **Gemini:** https://console.cloud.google.com/
2. **Deepgram:** https://console.deepgram.com/
3. **Gladia:** https://app.gladia.io/

Nastavte si billing alerts, aby ste sa vyhli prekvapeniu!

---

## 🎉 Hotovo!

Teraz máš:
- ✅ Aplikáciu na Streamlit Cloud
- ✅ Bezpečné API keys
- ✅ Zdieľateľnú URL pre kolegov
- ✅ Automatické updates pri git push

**Užívaj si porovnávanie STT modelov! 🎙️**

---

## 📞 Podpora

- **Streamlit Docs:** https://docs.streamlit.io/streamlit-community-cloud
- **Community Forum:** https://discuss.streamlit.io/
- **GitHub Issues:** https://github.com/MSGreenbluMe/STT-arena/issues
