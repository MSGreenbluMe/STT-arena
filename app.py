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
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    transcribe_with_all_providers,
    GeminiGoldenTranscript,
    calculate_wer_scores,
    calculate_cer_scores,
    calculate_costs,
    calculate_segment_wer,
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
    if 'costs' not in st.session_state:
        st.session_state.costs = None
    if 'audio_file_path' not in st.session_state:
        st.session_state.audio_file_path = None
    if 'transcription_history' not in st.session_state:
        st.session_state.transcription_history = []


def sidebar_config():
    """Sidebar configuration and API key inputs"""
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.markdown("---")

    # API keys are loaded from environment/secrets only
    api_keys = {
        'gemini': os.getenv('GEMINI_API_KEY', os.getenv('gemini', '')),
        'elevenlabs': os.getenv('ELEVENLABS_API_KEY', os.getenv('elevenlabs', '')),
        'gladia': os.getenv('GLADIA_API_KEY', os.getenv('gladia', '')),
        'openai': os.getenv('OPENAI_API_KEY', os.getenv('openai', '')),
        'groq': os.getenv('GROQ_API_KEY', os.getenv('groq', '')),
        'behavioral': os.getenv('BEHAVIORAL_SIGNALS_API_KEY', os.getenv('Behavioral signals', '')),
        'behavioral_url': os.getenv('BEHAVIORAL_SIGNALS_URL', 'https://api.behavioralsignals.com/v5/clients/10000215/processes/audio'),
        'deepgram': os.getenv('DEEPGRAM_API_KEY', os.getenv('Deepgram', ''))
    }

    st.sidebar.info("🔐 API Keys are loaded from Streamlit secrets. Configure them in your Streamlit Cloud dashboard.")

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
    # Behavioral Signals temporarily disabled
    st.sidebar.checkbox(
        "Behavioral Signals ⚠️ (Temporarily Disabled)",
        value=False,
        disabled=True,
        help="Temporarily disabled due to performance issues and API errors"
    )
    enabled_providers['behavioral'] = False  # Force disabled

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

    # Manual transcript upload option
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Manual Transcript (Optional)")

    st.sidebar.markdown("Upload an existing transcript to compare:")

    manual_transcript = st.sidebar.text_area(
        "Paste transcript here",
        height=150,
        placeholder="Example:\nSpeaker 1: Hello, how are you?\nSpeaker 2: I'm fine, thanks!",
        help="Paste a transcript you already have. Format: plain text or with speaker labels"
    )

    manual_provider_name = None
    if manual_transcript and manual_transcript.strip():
        manual_provider_name = st.sidebar.text_input(
            "Provider name",
            value="Manual Transcript",
            help="Name to display for your transcript"
        )

    return api_keys, enabled_providers, uploaded_file, language, manual_transcript, manual_provider_name


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
    api_keys, enabled_providers, uploaded_file, language, manual_transcript, manual_provider_name = sidebar_config()

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

        # Special handling for WAV files (Streamlit has issues with some WAV formats)
        if file_ext == 'wav':
            # For WAV, try reading the bytes and providing them directly
            try:
                audio_bytes = uploaded_file.getvalue()
                st.audio(audio_bytes, format='audio/wav')
            except Exception as e:
                st.warning(f"⚠️ WAV playback may not work in browser. File uploaded successfully. Error: {e}")
        else:
            # For other formats, use standard approach
            mime_types = {
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
                with st.status("🎙️ STT Arena Processing...", expanded=True) as status:
                    # Step 1: Transcribe with all providers
                    st.write("🔄 Transcribing audio with enabled providers...")
                    enabled_count = sum(1 for v in enabled_providers.values() if v)
                    st.write(f"📊 {enabled_count} providers selected")

                    results = transcribe_with_all_providers(
                        audio_file_path,
                        enabled_providers,
                        api_keys,
                        language
                    )

                    # Add manual transcript if provided
                    if manual_transcript and manual_transcript.strip():
                        st.write(f"📝 Adding manual transcript: {manual_provider_name}")
                        results[manual_provider_name] = {
                            'success': True,
                            'transcript': manual_transcript.strip(),
                            'metadata': {'source': 'manual_upload'},
                            'processing_time': 0.0
                        }

                    st.session_state.transcription_results = results

                    # Count successful transcriptions
                    success_count = sum(1 for r in results.values() if r.get('success'))
                    st.write(f"✅ {success_count}/{len(results)} providers successful")

                    # Step 2: Calculate costs
                    st.write("💰 Calculating costs...")
                    costs = calculate_costs(results)
                    st.session_state.costs = costs
                    total_cost = sum(c for c in costs.values() if c is not None)
                    st.write(f"💵 Total cost: ${total_cost:.4f}")

                    # Step 3: Generate Golden Transcript
                    if results:
                        st.write("✨ Generating Golden Transcript...")

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

                            generator = gen_metadata.get('generator', 'Unknown')
                            st.write(f"🤖 Generator used: {generator}")

                            # Show fallback info if used
                            if 'fallback' in generator.lower():
                                gemini_error = gen_metadata.get('gemini_error', 'Unknown error')
                                st.error(f"❌ Gemini API Error: {gemini_error}")
                                st.info(f"ℹ️ Using fallback: {generator}")

                            # Step 4: Calculate metrics
                            st.write("📊 Calculating quality metrics (WER, CER)...")
                            wer_scores = calculate_wer_scores(transcripts, golden_transcript)
                            cer_scores = calculate_cer_scores(transcripts, golden_transcript)
                            st.session_state.wer_scores = wer_scores
                            st.session_state.cer_scores = cer_scores

                            # Step 5: Save to history
                            st.write("💾 Saving to history...")
                            history_entry = {
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'audio_file': uploaded_file.name if uploaded_file else 'Unknown',
                                'language': language,
                                'providers': list(transcripts.keys()),
                                'wer_scores': wer_scores.copy(),
                                'cer_scores': cer_scores.copy(),
                                'costs': costs.copy(),
                                'total_cost': total_cost,
                                'generator': generator
                            }
                            st.session_state.transcription_history.append(history_entry)

                            status.update(label="✅ Transcription Complete!", state="complete")
                        else:
                            st.error("❌ No successful transcriptions to generate Golden Transcript.")
                            status.update(label="❌ Failed", state="error")

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

            # Add metric explanations
            with st.expander("ℹ️ What do these metrics mean?"):
                st.markdown("""
                - **WER (Word Error Rate)**: Percentage of words that are incorrect compared to the reference transcript. Lower is better.
                  - 0% = Perfect match
                  - <10% = Excellent
                  - 10-20% = Good
                  - >20% = Needs improvement

                - **CER (Character Error Rate)**: Percentage of characters that are incorrect. More sensitive than WER for languages with diacritics.
                  - 0% = Perfect match
                  - <5% = Excellent
                  - 5-15% = Good
                  - >15% = Needs improvement

                - **Cost**: Estimated cost in USD based on provider pricing (as of Jan 2025)
                  - Groq: FREE
                  - OpenAI: $0.006/min
                  - Deepgram: $0.0125/min
                  - Gladia: $0.0183/min
                  - ElevenLabs: $0.10/min
                """)

            metrics_data = []
            for provider, result in st.session_state.transcription_results.items():
                if result.get('success'):
                    wer_score = st.session_state.wer_scores.get(provider, 'N/A') if st.session_state.wer_scores else 'N/A'
                    wer_display = f"{wer_score}%" if wer_score != 'N/A' and wer_score is not None else 'N/A'

                    cer_score = st.session_state.cer_scores.get(provider, 'N/A') if st.session_state.cer_scores else 'N/A'
                    cer_display = f"{cer_score}%" if cer_score != 'N/A' and cer_score is not None else 'N/A'

                    cost = st.session_state.costs.get(provider, 'N/A') if st.session_state.costs else 'N/A'
                    cost_display = f"${cost:.4f}" if cost != 'N/A' and cost is not None else 'FREE' if cost == 0 else 'N/A'

                    metrics_data.append({
                        'Provider': provider,
                        'WER': wer_display,
                        'CER': cer_display,
                        'Cost': cost_display,
                        'Time (s)': f"{result.get('processing_time', 0):.2f}",
                        'Status': '✅ Success'
                    })
                else:
                    metrics_data.append({
                        'Provider': provider,
                        'WER': 'N/A',
                        'CER': 'N/A',
                        'Cost': 'N/A',
                        'Time (s)': 'N/A',
                        'Status': f"❌ {result.get('error', 'Failed')[:50]}..."
                    })

            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # Show total cost
            if st.session_state.costs:
                total_cost = sum(c for c in st.session_state.costs.values() if c is not None)
                st.metric("💰 Total Cost", f"${total_cost:.4f}")

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

                        # Try to format transcript with diarization if available
                        metadata = result.get('metadata', {})
                        utterances = metadata.get('utterances', [])

                        if utterances and isinstance(utterances, list) and len(utterances) > 0:
                            # Format with speakers
                            formatted_text = ""
                            for item in utterances:
                                if isinstance(item, dict):
                                    speaker = item.get('speaker', item.get('channel', 'Unknown'))
                                    text = item.get('text', item.get('transcript', ''))
                                    if text:
                                        formatted_text += f"**Speaker {speaker}:** {text}\n\n"

                            if formatted_text:
                                st.markdown(formatted_text)
                            else:
                                st.write(result.get('transcript', 'No transcript available'))
                        else:
                            st.write(result.get('transcript', 'No transcript available'))

                        st.markdown('</div>', unsafe_allow_html=True)

                        # Metrics row
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            wer_score = st.session_state.wer_scores.get(provider, 'N/A') if st.session_state.wer_scores else 'N/A'
                            if wer_score != 'N/A' and wer_score is not None:
                                st.metric("WER", f"{wer_score}%")
                        with col2:
                            cer_score = st.session_state.cer_scores.get(provider, 'N/A') if st.session_state.cer_scores else 'N/A'
                            if cer_score != 'N/A' and cer_score is not None:
                                st.metric("CER", f"{cer_score}%")
                        with col3:
                            cost = st.session_state.costs.get(provider, 'N/A') if st.session_state.costs else 'N/A'
                            if cost != 'N/A' and cost is not None:
                                st.metric("Cost", f"${cost:.4f}")

                        # Diarization (Speaker Identification)
                        metadata = result.get('metadata', {})

                        # Extract diarization info based on provider
                        diarization_info = None
                        if 'utterances' in metadata and metadata['utterances']:  # Deepgram/Gladia
                            diarization_info = metadata['utterances']
                        elif 'diarization' in metadata and metadata['diarization']:  # Gladia
                            diarization_info = metadata['diarization']
                        elif 'words' in metadata and metadata['words']:  # Check if words have speaker info
                            words = metadata['words']
                            if isinstance(words, list) and len(words) > 0:
                                if 'speaker' in words[0] or 'speaker_id' in words[0]:
                                    diarization_info = words

                        if diarization_info:
                            with st.expander("👥 Speaker Diarization"):
                                if isinstance(diarization_info, list):
                                    # Group by speaker
                                    speakers = {}
                                    for item in diarization_info:
                                        if isinstance(item, dict):
                                            speaker_id = item.get('speaker', item.get('speaker_id', item.get('channel', 'Unknown')))
                                            text = item.get('text', item.get('transcript', ''))
                                            if speaker_id not in speakers:
                                                speakers[speaker_id] = []
                                            if text:
                                                speakers[speaker_id].append(text)

                                    # Display speaker summaries
                                    for speaker_id, texts in speakers.items():
                                        st.markdown(f"**Speaker {speaker_id}:** {len(texts)} segments")
                                        with st.expander(f"View Speaker {speaker_id} segments"):
                                            for idx, text in enumerate(texts, 1):
                                                st.write(f"{idx}. {text}")
                                else:
                                    st.json(diarization_info)

                        # Timeline WER Visualization
                        if st.session_state.golden_transcript:
                            segments = metadata.get('segments', metadata.get('utterances', []))
                            if segments and isinstance(segments, list) and len(segments) > 0:
                                segment_wers = calculate_segment_wer(segments, st.session_state.golden_transcript)

                                if segment_wers:
                                    with st.expander("📊 WER Over Time (Timeline)"):
                                        # Create dataframe for plotly
                                        df_timeline = pd.DataFrame(segment_wers)

                                        # Create timeline chart
                                        fig = go.Figure()

                                        # Add WER line
                                        fig.add_trace(go.Scatter(
                                            x=df_timeline['start'],
                                            y=df_timeline['wer'],
                                            mode='lines+markers',
                                            name='WER',
                                            line=dict(color='#ff4444', width=2),
                                            marker=dict(size=8),
                                            hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>WER:</b> %{y:.1f}%<br><extra></extra>'
                                        ))

                                        # Add threshold lines
                                        fig.add_hline(y=10, line_dash="dash", line_color="green",
                                                     annotation_text="Excellent (<10%)")
                                        fig.add_hline(y=20, line_dash="dash", line_color="orange",
                                                     annotation_text="Good (<20%)")

                                        fig.update_layout(
                                            title=f"{provider} - Word Error Rate Over Time",
                                            xaxis_title="Time (seconds)",
                                            yaxis_title="WER (%)",
                                            height=400,
                                            hovermode='x unified'
                                        )

                                        st.plotly_chart(fig, use_container_width=True)

                                        # Show worst segments
                                        worst_segments = sorted(segment_wers, key=lambda x: x.get('wer', 0) if x.get('wer') is not None else 0, reverse=True)[:3]
                                        if worst_segments and worst_segments[0].get('wer', 0) > 0:
                                            st.markdown("**🔴 Highest Error Segments:**")
                                            for idx, seg in enumerate(worst_segments, 1):
                                                if seg.get('wer', 0) > 0:
                                                    st.write(f"{idx}. Time {seg['start']:.1f}s - {seg['end']:.1f}s: WER {seg['wer']}%")
                                                    st.caption(f"   Text: {seg['text']}")

                        # Metadata
                        with st.expander("🔍 View Full Metadata & Details"):
                            st.json(metadata)
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

            total_cost = entry.get('total_cost', 0)

            history_data.append({
                '#': idx,
                'Timestamp': entry['timestamp'],
                'Audio File': entry['audio_file'],
                'Language': entry['language'].upper(),
                'Providers': len(entry['providers']),
                'Avg WER': f"{avg_wer}%" if avg_wer != 'N/A' else 'N/A',
                'Avg CER': f"{avg_cer}%" if avg_cer != 'N/A' else 'N/A',
                'Cost': f"${total_cost:.4f}",
                'Generator': entry['generator']
            })

        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

        # Visualizations
        if len(st.session_state.transcription_history) > 1:
            st.markdown("### 📈 History Visualizations")

            # Prepare data for charts
            viz_data = []
            for idx, entry in enumerate(st.session_state.transcription_history):
                timestamp = entry['timestamp']
                for provider, wer_score in entry['wer_scores'].items():
                    if wer_score is not None:
                        viz_data.append({
                            'Timestamp': timestamp,
                            'Provider': provider,
                            'WER': wer_score,
                            'CER': entry['cer_scores'].get(provider, None),
                            'Cost': entry.get('costs', {}).get(provider, 0)
                        })

            df_viz = pd.DataFrame(viz_data)

            if not df_viz.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### WER Over Time by Provider")
                    fig_wer = px.line(
                        df_viz,
                        x='Timestamp',
                        y='WER',
                        color='Provider',
                        markers=True,
                        title='Word Error Rate Trends'
                    )
                    fig_wer.update_layout(yaxis_title="WER (%)", height=400)
                    st.plotly_chart(fig_wer, use_container_width=True)

                with col2:
                    st.markdown("#### CER Over Time by Provider")
                    fig_cer = px.line(
                        df_viz,
                        x='Timestamp',
                        y='CER',
                        color='Provider',
                        markers=True,
                        title='Character Error Rate Trends'
                    )
                    fig_cer.update_layout(yaxis_title="CER (%)", height=400)
                    st.plotly_chart(fig_cer, use_container_width=True)

                # Cost analysis
                st.markdown("#### 💰 Cost Analysis by Provider")
                cost_by_provider = df_viz.groupby('Provider')['Cost'].sum().reset_index()
                fig_cost = px.bar(
                    cost_by_provider,
                    x='Provider',
                    y='Cost',
                    title='Total Cost by Provider',
                    color='Provider'
                )
                fig_cost.update_layout(yaxis_title="Total Cost ($)", height=400, showlegend=False)
                st.plotly_chart(fig_cost, use_container_width=True)

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
