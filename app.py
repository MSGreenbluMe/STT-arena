"""
STT Arena - Main Streamlit Application
Compare multiple STT providers with Gemini-powered Golden Transcript generation
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import time

from utils import (
    transcribe_with_all_providers,
    GeminiGoldenTranscript,
    calculate_wer_scores,
    calculate_cer_scores,
    format_metadata_for_display
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="STT Arena - Model Comparison",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .golden-transcript {
        background-color: #fff9e6;
        border-left: 5px solid #ffd700;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .provider-transcript {
        background-color: #f0f2f6;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffe6e6;
        border-left: 5px solid #ff4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'transcription_results' not in st.session_state:
        st.session_state.transcription_results = None
    if 'golden_transcript' not in st.session_state:
        st.session_state.golden_transcript = None
    if 'wer_scores' not in st.session_state:
        st.session_state.wer_scores = None
    if 'cer_scores' not in st.session_state:
        st.session_state.cer_scores = None
    if 'audio_file_path' not in st.session_state:
        st.session_state.audio_file_path = None
    if 'transcription_history' not in st.session_state:
        st.session_state.transcription_history = []


def sidebar_config():
    """Sidebar configuration and API key inputs"""
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 API Keys")

    # API Key inputs (with environment variable fallback)
    # Try multiple environment variable names for compatibility
    api_keys = {}

    api_keys['gemini'] = st.sidebar.text_input(
        "Google Gemini API Key",
        value=os.getenv('GEMINI_API_KEY', os.getenv('gemini', '')),
        type="password",
        help="Required for Golden Transcript generation"
    )

    api_keys['elevenlabs'] = st.sidebar.text_input(
        "ElevenLabs API Key",
        value=os.getenv('ELEVENLABS_API_KEY', os.getenv('elevenlabs', '')),
        type="password"
    )

    api_keys['gladia'] = st.sidebar.text_input(
        "Gladia API Key",
        value=os.getenv('GLADIA_API_KEY', os.getenv('gladia', '')),
        type="password"
    )

    api_keys['openai'] = st.sidebar.text_input(
        "OpenAI API Key",
        value=os.getenv('OPENAI_API_KEY', os.getenv('openai', '')),
        type="password"
    )

    api_keys['groq'] = st.sidebar.text_input(
        "Groq API Key",
        value=os.getenv('GROQ_API_KEY', os.getenv('groq', '')),
        type="password",
        help="Groq provides free access to Whisper Large v3"
    )

    api_keys['behavioral'] = st.sidebar.text_input(
        "Behavioral Signals API Key",
        value=os.getenv('BEHAVIORAL_SIGNALS_API_KEY', os.getenv('Behavioral signals', '')),
        type="password"
    )

    api_keys['behavioral_url'] = st.sidebar.text_input(
        "Behavioral Signals URL",
        value=os.getenv('BEHAVIORAL_SIGNALS_URL', 'https://api.behavioralsignals.com/v5/clients/10000215/processes/audio'),
        help="API endpoint for Behavioral Signals (format: /v5/clients/{CID}/processes/audio)"
    )

    api_keys['deepgram'] = st.sidebar.text_input(
        "Deepgram API Key",
        value=os.getenv('DEEPGRAM_API_KEY', os.getenv('Deepgram', '')),
        type="password"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Select STT Providers")

    # Provider selection
    enabled_providers = {}
    enabled_providers['elevenlabs'] = st.sidebar.checkbox(
        "ElevenLabs",
        value=bool(api_keys['elevenlabs']),
        disabled=not bool(api_keys['elevenlabs'])
    )
    enabled_providers['gladia'] = st.sidebar.checkbox(
        "Gladia",
        value=bool(api_keys['gladia']),
        disabled=not bool(api_keys['gladia'])
    )
    enabled_providers['openai'] = st.sidebar.checkbox(
        "OpenAI Whisper",
        value=bool(api_keys['openai']),
        disabled=not bool(api_keys['openai'])
    )
    enabled_providers['groq'] = st.sidebar.checkbox(
        "Groq Whisper Large v3",
        value=bool(api_keys['groq']),
        disabled=not bool(api_keys['groq'])
    )
    enabled_providers['behavioral'] = st.sidebar.checkbox(
        "Behavioral Signals",
        value=bool(api_keys['behavioral']),
        disabled=not bool(api_keys['behavioral'])
    )
    enabled_providers['deepgram'] = st.sidebar.checkbox(
        "Deepgram",
        value=bool(api_keys['deepgram']),
        disabled=not bool(api_keys['deepgram'])
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 Language Settings")

    language = st.sidebar.selectbox(
        "Audio Language",
        options=['auto', 'cs', 'sk'],
        format_func=lambda x: {
            'auto': '🔍 Auto-detect',
            'cs': '🇨🇿 Czech (čeština)',
            'sk': '🇸🇰 Slovak (slovenčina)'
        }[x],
        index=0,
        help="Select the language of your audio file"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Audio Upload")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Audio File",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
        help="Upload a call center audio recording"
    )

    return api_keys, enabled_providers, uploaded_file, language


def main():
    """Main application logic"""
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">🎙️ STT Arena</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Compare Speech-to-Text models with AI-powered Golden Transcript</div>',
        unsafe_allow_html=True
    )

    # Sidebar configuration
    api_keys, enabled_providers, uploaded_file, language = sidebar_config()

    # Check if Gemini API key is provided
    if not api_keys['gemini']:
        st.warning("⚠️ Please provide a Google Gemini API Key in the sidebar to generate Golden Transcripts.")
        st.info("💡 Tip: Create a `.env` file with your API keys or enter them in the sidebar.")
        st.stop()

    # Check if at least one provider is enabled
    if not any(enabled_providers.values()):
        st.info("👈 Please select at least one STT provider from the sidebar and upload an audio file.")
        st.stop()

    # Audio file handling
    if uploaded_file is not None:
        # Display audio player with proper MIME type
        file_ext = uploaded_file.name.split(".")[-1].lower()
        mime_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'm4a': 'audio/mp4',
            'ogg': 'audio/ogg',
            'flac': 'audio/flac'
        }
        audio_format = mime_types.get(file_ext, f'audio/{file_ext}')
        st.audio(uploaded_file, format=audio_format)

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_file_path = tmp_file.name
            st.session_state.audio_file_path = audio_file_path

        # Transcribe button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Transcription Arena", use_container_width=True, type="primary"):
                with st.spinner("🔄 Transcribing with multiple providers..."):
                    # Transcribe with all enabled providers
                    results = transcribe_with_all_providers(
                        audio_file_path,
                        enabled_providers,
                        api_keys,
                        language
                    )

                    st.session_state.transcription_results = results

                    # Generate Golden Transcript
                    if results:
                        with st.spinner("✨ Generating Golden Transcript with Gemini..."):
                            # Extract successful transcripts
                            transcripts = {
                                provider: data['transcript']
                                for provider, data in results.items()
                                if data.get('success') and data.get('transcript')
                            }

                            if transcripts:
                                # Initialize Gemini with Groq fallback
                                gemini = GeminiGoldenTranscript(
                                    api_keys['gemini'],
                                    groq_api_key=api_keys.get('groq')
                                )
                                golden_transcript, gen_metadata = gemini.generate_golden_transcript(transcripts)
                                st.session_state.golden_transcript = golden_transcript

                                # Show which generator was used
                                if gen_metadata.get('generator'):
                                    if 'fallback' in gen_metadata.get('generator', '').lower():
                                        st.info(f"ℹ️ Generated using: {gen_metadata['generator']}")

                                # Calculate WER and CER scores
                                wer_scores = calculate_wer_scores(transcripts, golden_transcript)
                                cer_scores = calculate_cer_scores(transcripts, golden_transcript)
                                st.session_state.wer_scores = wer_scores
                                st.session_state.cer_scores = cer_scores

                                # Add to history
                                history_entry = {
                                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'audio_file': uploaded_file.name if uploaded_file else 'Unknown',
                                    'language': language,
                                    'providers': list(transcripts.keys()),
                                    'wer_scores': wer_scores.copy(),
                                    'cer_scores': cer_scores.copy(),
                                    'generator': gen_metadata.get('generator', 'Unknown')
                                }
                                st.session_state.transcription_history.append(history_entry)
                            else:
                                st.error("❌ No successful transcriptions to generate Golden Transcript.")

        # Display results
        if st.session_state.transcription_results:
            st.markdown("---")
            st.markdown("## 🏆 The Arena Results")

            # Golden Transcript Section
            if st.session_state.golden_transcript:
                st.markdown("### ✨ Golden Transcript (Source of Truth)")
                st.markdown('<div class="golden-transcript">', unsafe_allow_html=True)

                # Editable golden transcript
                edited_golden = st.text_area(
                    "You can edit the Golden Transcript if needed:",
                    value=st.session_state.golden_transcript,
                    height=200,
                    key="golden_edit"
                )

                if edited_golden != st.session_state.golden_transcript:
                    if st.button("🔄 Recalculate Metrics with Edited Transcript"):
                        st.session_state.golden_transcript = edited_golden
                        # Recalculate WER and CER
                        transcripts = {
                            provider: data['transcript']
                            for provider, data in st.session_state.transcription_results.items()
                            if data.get('success') and data.get('transcript')
                        }
                        st.session_state.wer_scores = calculate_wer_scores(transcripts, edited_golden)
                        st.session_state.cer_scores = calculate_cer_scores(transcripts, edited_golden)
                        st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

            # Performance Metrics
            st.markdown("### 📊 Performance Metrics")

            metrics_data = []
            for provider, result in st.session_state.transcription_results.items():
                if result.get('success'):
                    wer_score = st.session_state.wer_scores.get(provider, 'N/A') if st.session_state.wer_scores else 'N/A'
                    wer_display = f"{wer_score}%" if wer_score != 'N/A' and wer_score is not None else 'N/A'

                    cer_score = st.session_state.cer_scores.get(provider, 'N/A') if st.session_state.cer_scores else 'N/A'
                    cer_display = f"{cer_score}%" if cer_score != 'N/A' and cer_score is not None else 'N/A'

                    metrics_data.append({
                        'Provider': provider,
                        'WER': wer_display,
                        'CER': cer_display,
                        'Processing Time (s)': f"{result.get('processing_time', 0):.2f}",
                        'Status': '✅ Success'
                    })
                else:
                    metrics_data.append({
                        'Provider': provider,
                        'WER': 'N/A',
                        'CER': 'N/A',
                        'Processing Time (s)': 'N/A',
                        'Status': f"❌ {result.get('error', 'Failed')}"
                    })

            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # Transcript Comparison
            st.markdown("### 📝 Transcript Comparison")

            # Create tabs for each provider
            provider_names = list(st.session_state.transcription_results.keys())
            tabs = st.tabs(provider_names)

            for tab, provider in zip(tabs, provider_names):
                with tab:
                    result = st.session_state.transcription_results[provider]

                    if result.get('success'):
                        st.markdown(f'<div class="provider-transcript">', unsafe_allow_html=True)
                        st.markdown(f"**Transcript:**")
                        st.write(result.get('transcript', 'No transcript available'))
                        st.markdown('</div>', unsafe_allow_html=True)

                        # WER Score
                        wer_score = st.session_state.wer_scores.get(provider, 'N/A') if st.session_state.wer_scores else 'N/A'
                        if wer_score != 'N/A' and wer_score is not None:
                            st.metric("Word Error Rate", f"{wer_score}%")

                        # Metadata
                        with st.expander("🔍 View Metadata & Details"):
                            st.json(result.get('metadata', {}))
                    else:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
                        st.markdown('</div>', unsafe_allow_html=True)

            # Download section
            st.markdown("### 💾 Export Results")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📥 Download Golden Transcript"):
                    st.download_button(
                        label="Download as TXT",
                        data=st.session_state.golden_transcript,
                        file_name="golden_transcript.txt",
                        mime="text/plain"
                    )

            with col2:
                if st.button("📥 Download All Transcripts"):
                    all_transcripts = "=" * 60 + "\n"
                    all_transcripts += "STT ARENA - ALL TRANSCRIPTS\n"
                    all_transcripts += "=" * 60 + "\n\n"

                    all_transcripts += "GOLDEN TRANSCRIPT:\n"
                    all_transcripts += "-" * 60 + "\n"
                    all_transcripts += st.session_state.golden_transcript + "\n\n"

                    for provider, result in st.session_state.transcription_results.items():
                        if result.get('success'):
                            all_transcripts += f"\n{provider.upper()}:\n"
                            all_transcripts += "-" * 60 + "\n"
                            all_transcripts += result.get('transcript', '') + "\n"
                            wer = st.session_state.wer_scores.get(provider, 'N/A') if st.session_state.wer_scores else 'N/A'
                            cer = st.session_state.cer_scores.get(provider, 'N/A') if st.session_state.cer_scores else 'N/A'
                            all_transcripts += f"WER: {wer}%\n" if wer != 'N/A' else "WER: N/A\n"
                            all_transcripts += f"CER: {cer}%\n" if cer != 'N/A' else "CER: N/A\n"

                    st.download_button(
                        label="Download All as TXT",
                        data=all_transcripts,
                        file_name="all_transcripts.txt",
                        mime="text/plain"
                    )

    else:
        st.info("👈 Please upload an audio file from the sidebar to begin.")

    # Transcription History
    if st.session_state.transcription_history:
        st.markdown("---")
        st.markdown("## 📜 Transcription History")

        # Add clear history button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.transcription_history = []
                st.rerun()

        # Display history table
        history_data = []
        for idx, entry in enumerate(reversed(st.session_state.transcription_history), 1):
            # Calculate average WER and CER
            wer_values = [v for v in entry['wer_scores'].values() if v is not None]
            cer_values = [v for v in entry['cer_scores'].values() if v is not None]
            avg_wer = round(sum(wer_values) / len(wer_values), 2) if wer_values else 'N/A'
            avg_cer = round(sum(cer_values) / len(cer_values), 2) if cer_values else 'N/A'

            history_data.append({
                '#': idx,
                'Timestamp': entry['timestamp'],
                'Audio File': entry['audio_file'],
                'Language': entry['language'].upper(),
                'Providers': len(entry['providers']),
                'Avg WER': f"{avg_wer}%" if avg_wer != 'N/A' else 'N/A',
                'Avg CER': f"{avg_cer}%" if avg_cer != 'N/A' else 'N/A',
                'Generator': entry['generator']
            })

        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

        # Export history button
        if st.button("📥 Download History as CSV"):
            csv_data = df_history.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"stt_arena_history_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "STT Arena | Powered by Google Gemini | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
