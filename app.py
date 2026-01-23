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
from database import STTArenaDB
from qa_utils import (
    GeminiLLM,
    GroqLLM,
    MistralLLM,
    QuestionGenerator,
    calculate_llm_costs
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
    if 'db' not in st.session_state:
        st.session_state.db = STTArenaDB()
    if 'current_audio_id' not in st.session_state:
        st.session_state.current_audio_id = None
    if 'audio_tags' not in st.session_state:
        st.session_state.audio_tags = []


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

    # Tags for test scenarios
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏷️ Tags (Optional)")
    st.sidebar.markdown("Add tags to organize test scenarios:")

    # Get existing tags from database
    existing_tags = st.session_state.db.get_all_tags() if 'db' in st.session_state else []

    # Tag input
    tags_input = st.sidebar.text_input(
        "Enter tags (comma-separated)",
        placeholder="e.g., test-call-1, customer-support",
        help="Tags help you find and organize repeated test calls"
    )

    tags = []
    if tags_input:
        tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]

    # Show existing tags
    if existing_tags:
        st.sidebar.caption(f"Existing tags: {', '.join(existing_tags[:5])}")

    notes = st.sidebar.text_area(
        "Notes (Optional)",
        placeholder="Any notes about this test...",
        height=80
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

    return api_keys, enabled_providers, uploaded_file, language, manual_transcript, manual_provider_name, tags, notes


def stt_arena_tab(api_keys, enabled_providers, uploaded_file, language, manual_transcript, manual_provider_name, tags, notes):
    """STT Arena tab - transcription and evaluation"""

    # Check if Gemini API key is provided
    if not api_keys['gemini']:
        st.warning("⚠️ Please provide a Google Gemini API Key in the sidebar to generate Golden Transcripts.")
        st.info("💡 Tip: Create a `.env` file with your API keys or enter them in the sidebar.")
        return

    # Check if at least one provider is enabled
    if not any(enabled_providers.values()):
        st.info("👈 Please select at least one STT provider from the sidebar and upload an audio file.")
        return

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
            if st.button("🚀 Start Transcription Arena", type="primary"):
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
                            # Using gemini-2.5-flash for better quality (balanced speed/quality)
                            gemini = GeminiGoldenTranscript(
                                api_keys['gemini'],
                                groq_api_key=api_keys.get('groq'),
                                model_name='gemini-2.5-flash'
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

                            # Step 5: Save to history and database
                            st.write("💾 Saving to history and database...")

                            # Save to database
                            try:
                                # Save audio file
                                audio_bytes = uploaded_file.getvalue()
                                audio_id = st.session_state.db.save_audio_file(
                                    filename=uploaded_file.name,
                                    file_bytes=audio_bytes,
                                    duration_seconds=None,  # TODO: extract from metadata
                                    language=language,
                                    tags=tags if tags else None,
                                    notes=notes if notes else None
                                )
                                st.session_state.current_audio_id = audio_id

                                # Save transcriptions
                                transcription_ids = {}
                                for provider_name, result in results.items():
                                    trans_id = st.session_state.db.save_transcription(
                                        audio_file_id=audio_id,
                                        provider_name=provider_name,
                                        result=result
                                    )
                                    transcription_ids[provider_name] = trans_id

                                # Save evaluation (golden transcript)
                                eval_id = st.session_state.db.save_evaluation(
                                    audio_file_id=audio_id,
                                    golden_transcript=golden_transcript,
                                    golden_generator=generator
                                )

                                # Save evaluation scores
                                for provider_name in transcripts.keys():
                                    st.session_state.db.save_evaluation_scores(
                                        evaluation_id=eval_id,
                                        transcription_id=transcription_ids.get(provider_name),
                                        provider_name=provider_name,
                                        wer_score=wer_scores.get(provider_name),
                                        cer_score=cer_scores.get(provider_name),
                                        cost_usd=costs.get(provider_name)
                                    )

                                st.write(f"✅ Saved to database (ID: {audio_id})")
                                if tags:
                                    st.write(f"🏷️ Tags: {', '.join(tags)}")

                            except Exception as db_error:
                                st.warning(f"⚠️ Database save failed: {db_error}")

                            # Save to session history (legacy)
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
            st.dataframe(df_metrics, width='stretch', hide_index=True)

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

                                        st.plotly_chart(fig)

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
        st.dataframe(df_history, width='stretch', hide_index=True)

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
                    st.plotly_chart(fig_wer)

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
                    st.plotly_chart(fig_cer)

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
                st.plotly_chart(fig_cost)

        # Export history button
        if st.button("📥 Download History as CSV"):
            csv_data = df_history.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"stt_arena_history_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def qa_arena_tab(api_keys):
    """QA Arena tab - LLM question answering evaluation"""
    st.markdown("## 🤖 QA Arena - LLM Testing")

    db = st.session_state.db

    # Check API keys
    if not api_keys['gemini']:
        st.warning("⚠️ Gemini API key required for QA Arena")
        return

    # Step 1: Select transcript source
    st.markdown("### 📝 Step 1: Select Transcript")

    source_type = st.radio(
        "Transcript source:",
        ["From Database (by tag)", "Paste manually"],
        horizontal=True
    )

    transcript_text = ""
    transcript_source = ""

    if source_type == "From Database (by tag)":
        all_tags = db.get_all_tags()

        if all_tags:
            selected_tag = st.selectbox("Select tag:", all_tags)

            if selected_tag:
                tagged_files = db.get_audio_files_by_tag(selected_tag)

                if tagged_files:
                    file_options = {f"{f['filename']} (ID: {f['id']})": f['id'] for f in tagged_files}
                    selected_file_label = st.selectbox("Select audio file:", list(file_options.keys()))
                    selected_file_id = file_options[selected_file_label]

                    # Get transcript from database
                    # For now, use the most recent transcription
                    # TODO: Add provider selection
                    st.info(f"📂 Loading transcripts for audio file ID: {selected_file_id}")

                    # Get best transcript (lowest WER)
                    cursor = db.conn.cursor()
                    cursor.execute("""
                        SELECT t.provider_name, t.transcript_text, es.wer_score
                        FROM transcriptions t
                        LEFT JOIN evaluation_scores es ON t.id = es.transcription_id
                        WHERE t.audio_file_id = ? AND t.success = 1
                        ORDER BY es.wer_score ASC
                        LIMIT 1
                    """, (selected_file_id,))

                    result = cursor.fetchone()

                    if result:
                        provider_name, transcript_text, wer = result
                        transcript_source = f"{provider_name} (WER: {wer}%)"
                        st.success(f"✅ Using transcript from: {transcript_source}")
                    else:
                        st.error("No transcripts found for this file.")
                else:
                    st.info(f"No files found with tag '{selected_tag}'")
        else:
            st.info("No tags in database. Run STT Arena first to generate transcripts.")

    else:  # Paste manually
        transcript_text = st.text_area(
            "Paste transcript here:",
            height=200,
            placeholder="Enter the transcript you want to test..."
        )
        transcript_source = "Manual input"

    if not transcript_text:
        st.info("👈 Select or paste a transcript to continue")
        return

    # Show transcript preview
    with st.expander("📄 View Transcript"):
        st.text_area("Transcript preview:", transcript_text, height=150, disabled=True)

    st.markdown("---")

    # Step 2: Questions
    st.markdown("### ❓ Step 2: Questions")

    question_mode = st.radio(
        "Question source:",
        ["Auto-generate (Gemini)", "Manual input"],
        horizontal=True
    )

    questions = []

    if question_mode == "Auto-generate (Gemini)":
        num_questions = st.slider("Number of questions:", 1, 10, 5)

        if st.button("🎲 Generate Questions"):
            with st.spinner("Generating questions..."):
                try:
                    generator = QuestionGenerator(api_keys['gemini'])
                    questions = generator.generate_questions(transcript_text, num_questions)
                    st.session_state.qa_questions = questions
                    st.success(f"✅ Generated {len(questions)} questions")
                except Exception as e:
                    st.error(f"Question generation failed: {e}")

        if 'qa_questions' in st.session_state:
            questions = st.session_state.qa_questions

    else:  # Manual input
        st.markdown("Enter questions (one per line):")
        questions_text = st.text_area(
            "Questions:",
            height=150,
            placeholder="What was the customer's main issue?\nHow was it resolved?\nWhat was the customer's sentiment?"
        )

        if questions_text:
            questions = [
                {"question": q.strip(), "type": "manual"}
                for q in questions_text.split('\n')
                if q.strip()
            ]

    if questions:
        st.markdown(f"**📋 Questions to test ({len(questions)}):**")
        for idx, q in enumerate(questions, 1):
            st.write(f"{idx}. {q['question']} *({q['type']})*")

        st.markdown("---")

        # Step 3: Select LLM providers
        st.markdown("### 🤖 Step 3: Select LLM Providers")

        col1, col2, col3 = st.columns(3)

        with col1:
            use_gemini = st.checkbox("Gemini", value=True, disabled=not api_keys['gemini'])
            if use_gemini:
                gemini_model = st.selectbox(
                    "Gemini model:",
                    ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
                )

        with col2:
            use_groq = st.checkbox("Groq (FREE)", value=bool(api_keys.get('groq')), disabled=not api_keys.get('groq'))
            if use_groq:
                groq_model = st.selectbox(
                    "Groq model:",
                    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
                )

        with col3:
            mistral_key = os.getenv('MISTRAL_API_KEY', '')
            use_mistral = st.checkbox("Mistral", value=bool(mistral_key), disabled=not bool(mistral_key))
            if use_mistral:
                mistral_model = st.selectbox(
                    "Mistral model:",
                    ["mistral-small-latest", "mistral-large-latest", "open-mistral-7b"]
                )

        # Step 4: Run QA Arena
        if st.button("🚀 Start QA Arena", type="primary"):
            with st.status("🤖 QA Arena Processing...", expanded=True) as status:
                all_results = {}

                # Test each LLM provider
                if use_gemini:
                    st.write(f"🔄 Testing {gemini_model}...")
                    llm = GeminiLLM(api_keys['gemini'], gemini_model)
                    provider_results = []

                    for q in questions:
                        result = llm.ask_question(transcript_text, q['question'])
                        provider_results.append(result)

                    all_results[f"Gemini ({gemini_model})"] = provider_results

                if use_groq:
                    st.write(f"🔄 Testing {groq_model}...")
                    llm = GroqLLM(api_keys['groq'], groq_model)
                    provider_results = []

                    for q in questions:
                        result = llm.ask_question(transcript_text, q['question'])
                        provider_results.append(result)

                    all_results[f"Groq ({groq_model})"] = provider_results

                if use_mistral:
                    st.write(f"🔄 Testing {mistral_model}...")
                    llm = MistralLLM(mistral_key, mistral_model)
                    provider_results = []

                    for q in questions:
                        result = llm.ask_question(transcript_text, q['question'])
                        provider_results.append(result)

                    all_results[f"Mistral ({mistral_model})"] = provider_results

                st.session_state.qa_results = all_results
                status.update(label="✅ QA Arena Complete!", state="complete")

        # Display results
        if 'qa_results' in st.session_state:
            st.markdown("---")
            st.markdown("## 🏆 QA Arena Results")

            all_results = st.session_state.qa_results

            # Summary metrics
            st.markdown("### 📊 Performance Metrics")

            metrics_data = []
            for provider_name, results in all_results.items():
                success_count = sum(1 for r in results if r.get('success'))
                avg_time = sum(r.get('processing_time', 0) for r in results) / len(results) if results else 0
                total_tokens = sum(r.get('tokens_used', 0) for r in results if r.get('success'))

                # Calculate cost
                # Create temporary dict for cost calculation
                cost_results = {}
                if results:
                    cost_results[provider_name] = {
                        'success': True,
                        'model': results[0].get('model', ''),
                        'tokens_used': total_tokens
                    }
                costs = calculate_llm_costs(cost_results)
                total_cost = costs.get(provider_name, 0)

                metrics_data.append({
                    'Provider': provider_name,
                    'Success Rate': f"{success_count}/{len(results)}",
                    'Avg Time (s)': f"{avg_time:.2f}",
                    'Total Tokens': total_tokens,
                    'Cost': f"${total_cost:.6f}" if total_cost else "FREE"
                })

            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, width='stretch', hide_index=True)

            # Detailed answers
            st.markdown("### 📝 Detailed Answers")

            for idx, question_data in enumerate(questions, 1):
                st.markdown(f"#### Question {idx}: {question_data['question']}")

                # Create tabs for each provider
                provider_names = list(all_results.keys())
                if provider_names:
                    answer_tabs = st.tabs(provider_names)

                    for tab, provider_name in zip(answer_tabs, provider_names):
                        with tab:
                            result = all_results[provider_name][idx - 1]

                            if result.get('success'):
                                st.markdown(f"**Answer:**")
                                st.write(result['answer'])

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Time", f"{result['processing_time']:.2f}s")
                                with col2:
                                    st.metric("Tokens", result.get('tokens_used', 'N/A'))
                                with col3:
                                    # Calculate cost for this single answer
                                    single_cost_results = {
                                        provider_name: {
                                            'success': True,
                                            'model': result.get('model', ''),
                                            'tokens_used': result.get('tokens_used', 0)
                                        }
                                    }
                                    single_costs = calculate_llm_costs(single_cost_results)
                                    cost = single_costs.get(provider_name, 0)
                                    st.metric("Cost", f"${cost:.6f}" if cost else "FREE")
                            else:
                                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

                st.markdown("---")
    else:
        st.info("Generate or enter questions to continue")


def analytics_tab():
    """Analytics tab - provider leaderboards and historical trends"""
    st.markdown("## 📊 Analytics & Leaderboards")

    db = st.session_state.db

    # Provider Statistics
    st.markdown("### 🏆 STT Provider Leaderboard")
    provider_stats = db.get_provider_stats()

    if provider_stats:
        df_stats = pd.DataFrame(provider_stats)
        # Sort by cost effectiveness (higher is better)
        df_stats = df_stats.sort_values('avg_cost_effectiveness', ascending=False)

        st.dataframe(df_stats, width='stretch', hide_index=True)

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Average WER by Provider")
            fig_wer = px.bar(
                df_stats,
                x='provider_name',
                y='avg_wer',
                title='Average Word Error Rate',
                color='avg_wer',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_wer)

        with col2:
            st.markdown("#### Cost Effectiveness")
            fig_cost_eff = px.bar(
                df_stats,
                x='provider_name',
                y='avg_cost_effectiveness',
                title='Cost Effectiveness Score (Higher = Better)',
                color='avg_cost_effectiveness',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_cost_eff)
    else:
        st.info("No evaluations in database yet. Run STT Arena to generate data.")

    # Recent Evaluations
    st.markdown("### 📜 Recent Evaluations")
    recent_evals = db.get_recent_evaluations(limit=20)

    if recent_evals:
        df_recent = pd.DataFrame(recent_evals)
        st.dataframe(df_recent, width='stretch', hide_index=True)

        # Export button
        if st.button("📥 Export All Data to CSV"):
            try:
                db.export_to_csv('evaluations', 'evaluations_export.csv')
                db.export_to_csv('evaluation_scores', 'scores_export.csv')
                st.success("✅ Exported to evaluations_export.csv and scores_export.csv")
            except Exception as e:
                st.error(f"Export failed: {e}")
    else:
        st.info("No recent evaluations found.")

    # Tag filter
    st.markdown("### 🏷️ Filter by Tags")
    all_tags = db.get_all_tags()

    if all_tags:
        selected_tag = st.selectbox("Select tag", ["All"] + all_tags)

        if selected_tag != "All":
            tagged_files = db.get_audio_files_by_tag(selected_tag)
            st.write(f"Found {len(tagged_files)} files with tag '{selected_tag}'")
            if tagged_files:
                df_tagged = pd.DataFrame(tagged_files)
                st.dataframe(df_tagged, width='stretch', hide_index=True)
    else:
        st.info("No tags found. Add tags when uploading audio files.")


def main():
    """Main application with tab-based navigation"""
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">🎙️ STT Arena</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Compare Speech-to-Text models with AI-powered Golden Transcript</div>',
        unsafe_allow_html=True
    )

    # Sidebar configuration
    api_keys, enabled_providers, uploaded_file, language, manual_transcript, manual_provider_name, tags, notes = sidebar_config()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎙️ STT Arena", "🤖 QA Arena", "📊 Analytics"])

    with tab1:
        stt_arena_tab(api_keys, enabled_providers, uploaded_file, language, manual_transcript, manual_provider_name, tags, notes)

    with tab2:
        qa_arena_tab(api_keys)

    with tab3:
        analytics_tab()

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
