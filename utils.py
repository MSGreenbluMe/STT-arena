"""
STT Arena - Utility Functions
Handles API integrations for multiple STT providers and Gemini-based Golden Transcript generation
"""

import os
import time
import requests
import mimetypes
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from jiwer import wer
import json


class STTProvider:
    """Base class for STT providers"""

    def __init__(self, name: str, api_key: str):
        self.name = name
        self.api_key = api_key
        self.transcript = ""
        self.metadata = {}
        self.processing_time = 0
        self.error = None

    def transcribe(self, audio_file_path: str) -> Dict:
        """Override this method in subclasses"""
        raise NotImplementedError


class ElevenLabsSTT(STTProvider):
    """ElevenLabs Speech-to-Text Provider"""

    def __init__(self, api_key: str):
        super().__init__("ElevenLabs", api_key)
        self.base_url = "https://api.elevenlabs.io/v1/speech-to-text"

    def transcribe(self, audio_file_path: str, language: str = 'auto') -> Dict:
        """Transcribe audio using ElevenLabs Scribe API"""
        start_time = time.time()

        try:
            headers = {
                "xi-api-key": self.api_key
            }

            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()

            files = {'file': ('audio.mp3', audio_data, 'audio/mpeg')}
            data = {
                'model_id': 'scribe_v2',
                'diarize': 'true',
                'tag_audio_events': 'true'
            }

            # Add language if specified
            if language != 'auto':
                data['language'] = language

            response = requests.post(
                self.base_url,
                headers=headers,
                files=files,
                data=data,
                timeout=600  # 10 minutes for long files
            )

            if response.status_code == 200:
                result = response.json()
                self.transcript = result.get('text', '')
                self.metadata = {
                    'language': result.get('language_code', 'unknown'),
                    'confidence': result.get('language_probability', 0),
                    'words': result.get('words', []),
                    'transcription_id': result.get('transcription_id', '')
                }
                self.processing_time = time.time() - start_time

                return {
                    'success': True,
                    'transcript': self.transcript,
                    'metadata': self.metadata,
                    'processing_time': self.processing_time
                }
            else:
                self.error = f"API Error: {response.status_code} - {response.text}"
                return {'success': False, 'error': self.error}

        except Exception as e:
            self.error = str(e)
            self.processing_time = time.time() - start_time
            return {'success': False, 'error': self.error}


class GladiaSTT(STTProvider):
    """Gladia STT Provider"""

    def __init__(self, api_key: str):
        super().__init__("Gladia", api_key)
        self.upload_url = "https://api.gladia.io/v2/upload"
        self.transcribe_url = "https://api.gladia.io/v2/transcription"  # Fixed endpoint

    def transcribe(self, audio_file_path: str, language: str = 'auto') -> Dict:
        """Transcribe audio using Gladia API (2-step: upload + transcribe)"""
        start_time = time.time()

        try:
            headers = {
                "x-gladia-key": self.api_key
            }

            # Step 1: Upload audio file
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': (os.path.basename(audio_file_path), audio_file, 'audio/mpeg')}

                upload_response = requests.post(
                    self.upload_url,
                    headers=headers,
                    files=files,
                    timeout=600
                )

            if upload_response.status_code not in [200, 201]:
                self.error = f"Upload Error: {upload_response.status_code} - {upload_response.text}"
                return {'success': False, 'error': self.error}

            upload_result = upload_response.json()
            audio_url = upload_result.get('audio_url')

            if not audio_url:
                self.error = "No audio_url returned from upload"
                return {'success': False, 'error': self.error}

            # Step 2: Request transcription with proper payload format
            transcribe_payload = {
                "audio_url": audio_url,
                "diarization": True,
                "diarization_config": {
                    "number_of_speakers": 2,
                    "min_speakers": 1,
                    "max_speakers": 10
                }
            }

            # Add language if not auto-detect
            if language != 'auto':
                transcribe_payload["language"] = language

            transcribe_response = requests.post(
                self.transcribe_url,
                headers={**headers, "Content-Type": "application/json"},
                json=transcribe_payload,
                timeout=600
            )

            if transcribe_response.status_code not in [200, 201]:
                self.error = f"Transcription Error: {transcribe_response.status_code} - {transcribe_response.text}"
                return {'success': False, 'error': self.error}

            initial_result = transcribe_response.json()

            # Get transcription ID and result URL for polling
            transcription_id = initial_result.get('id')
            result_url = initial_result.get('result_url')

            if not transcription_id:
                self.error = "No transcription ID returned"
                return {'success': False, 'error': self.error}

            # Step 3: Poll for results (Gladia is async)
            max_polls = 120  # Max 2 minutes polling (1s intervals)
            poll_count = 0

            while poll_count < max_polls:
                time.sleep(1)  # Wait 1 second between polls
                poll_count += 1

                # Get transcription status
                status_url = f"https://api.gladia.io/v2/pre-recorded/{transcription_id}"
                status_response = requests.get(
                    status_url,
                    headers=headers,
                    timeout=30
                )

                if status_response.status_code != 200:
                    continue

                result = status_response.json()
                status = result.get('status')

                if status == 'done':
                    # Extract transcript from completed result
                    if 'result' in result:
                        result_data = result['result']
                        if 'transcription' in result_data:
                            transcription = result_data['transcription']
                            self.transcript = transcription.get('full_transcript', '')
                            self.metadata = {
                                'diarization': transcription.get('utterances', []),
                                'language': transcription.get('language', 'unknown'),
                                'confidence': transcription.get('confidence', 0)
                            }
                        else:
                            self.transcript = result_data.get('text', '')
                            self.metadata = result_data

                    self.processing_time = time.time() - start_time
                    break
                elif status == 'error':
                    self.error = f"Gladia transcription error: {result.get('error', 'Unknown error')}"
                    return {'success': False, 'error': self.error}

            if not self.transcript:
                self.error = "Transcription timeout or empty result"
                return {'success': False, 'error': self.error}

            self.processing_time = time.time() - start_time

            return {
                'success': True,
                'transcript': self.transcript,
                'metadata': self.metadata,
                'processing_time': self.processing_time
            }

        except Exception as e:
            self.error = str(e)
            self.processing_time = time.time() - start_time
            return {'success': False, 'error': self.error}


class OpenAIWhisperSTT(STTProvider):
    """OpenAI Whisper STT Provider"""

    def __init__(self, api_key: str):
        super().__init__("OpenAI Whisper", api_key)
        self.base_url = "https://api.openai.com/v1/audio/transcriptions"

    def transcribe(self, audio_file_path: str, language: str = 'auto') -> Dict:
        """Transcribe audio using OpenAI Whisper API"""
        start_time = time.time()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }

            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()

            files = {'file': ('audio.mp3', audio_data)}
            data = {
                'model': 'whisper-1',
                'response_format': 'verbose_json',
                'timestamp_granularities': ['word']
            }

            # Add language if specified (OpenAI uses ISO-639-1 codes)
            if language != 'auto':
                data['language'] = language

            response = requests.post(
                self.base_url,
                headers=headers,
                files=files,
                data=data,
                timeout=600
            )

            if response.status_code == 200:
                result = response.json()
                self.transcript = result.get('text', '')
                self.metadata = {
                    'language': result.get('language', 'unknown'),
                    'duration': result.get('duration', 0),
                    'words': result.get('words', [])
                }
                self.processing_time = time.time() - start_time

                return {
                    'success': True,
                    'transcript': self.transcript,
                    'metadata': self.metadata,
                    'processing_time': self.processing_time
                }
            else:
                self.error = f"API Error: {response.status_code} - {response.text}"
                return {'success': False, 'error': self.error}

        except Exception as e:
            self.error = str(e)
            self.processing_time = time.time() - start_time
            return {'success': False, 'error': self.error}


class BehavioralSignalsSTT(STTProvider):
    """Behavioral Signals STT Provider with emotion analysis"""

    def __init__(self, api_key: str, api_url: str):
        super().__init__("Behavioral Signals", api_key)
        self.api_url = api_url if api_url else "https://api.behavioralsignals.com/v5/clients/10000215/processes/audio"

    def transcribe(self, audio_file_path: str, language: str = 'auto') -> Dict:
        """Transcribe audio using Behavioral Signals API v5"""
        start_time = time.time()

        try:
            headers = {
                "X-Auth-Token": self.api_key,
                "Accept": "application/json"
            }

            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()

            # Detect file extension for proper MIME type
            ext = os.path.splitext(audio_file_path)[1].lower()
            mime_map = {
                '.wav': 'audio/wav',
                '.mp3': 'audio/mpeg',
                '.m4a': 'audio/mp4',
                '.ogg': 'audio/ogg',
                '.flac': 'audio/flac'
            }
            mime_type = mime_map.get(ext, 'audio/wav')

            files = {'file': (f'audio{ext}', audio_data, mime_type)}

            # Build metadata
            meta = {'source': 'stt-arena'}
            if language != 'auto':
                meta['language'] = language

            data = {
                'name': 'stt-arena-transcription',
                'meta': json.dumps(meta)
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                files=files,
                data=data,
                timeout=600
            )

            if response.status_code in [200, 201]:
                result = response.json()

                # Check if this is an async response (contains pid for polling)
                if 'pid' in result and 'transcript' not in result:
                    self.error = f"Async processing not yet implemented. Process ID: {result.get('pid')}"
                    return {'success': False, 'error': self.error}

                # Extract transcript from various possible fields
                self.transcript = result.get('transcript', result.get('text', result.get('transcription', '')))

                # Extract metadata
                self.metadata = {
                    'pid': result.get('pid', ''),
                    'status': result.get('status', ''),
                    'duration': result.get('duration', 0),
                    'emotion': result.get('emotion', {}),
                    'sentiment': result.get('sentiment', {}),
                    'tone': result.get('tone', {}),
                    'conversation_metrics': result.get('metrics', {}),
                    'raw_response': result  # Keep full response for debugging
                }
                self.processing_time = time.time() - start_time

                if not self.transcript:
                    self.error = f"No transcript in response. Full response: {json.dumps(result, indent=2)}"
                    return {'success': False, 'error': self.error}

                return {
                    'success': True,
                    'transcript': self.transcript,
                    'metadata': self.metadata,
                    'processing_time': self.processing_time
                }
            else:
                self.error = f"API Error: {response.status_code} - {response.text}"
                return {'success': False, 'error': self.error}

        except Exception as e:
            self.error = str(e)
            self.processing_time = time.time() - start_time
            return {'success': False, 'error': self.error}


class DeepgramSTT(STTProvider):
    """Deepgram STT Provider"""

    def __init__(self, api_key: str):
        super().__init__("Deepgram", api_key)
        self.base_url = "https://api.deepgram.com/v1/listen"

    def transcribe(self, audio_file_path: str, language: str = 'auto') -> Dict:
        """Transcribe audio using Deepgram API"""
        start_time = time.time()

        try:
            # Detect correct Content-Type from file extension
            mime_type, _ = mimetypes.guess_type(audio_file_path)
            if not mime_type or not mime_type.startswith('audio'):
                # Fallback to common audio types
                ext = os.path.splitext(audio_file_path)[1].lower()
                mime_map = {
                    '.mp3': 'audio/mpeg',
                    '.wav': 'audio/wav',
                    '.m4a': 'audio/mp4',
                    '.ogg': 'audio/ogg',
                    '.flac': 'audio/flac'
                }
                mime_type = mime_map.get(ext, 'application/octet-stream')

            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": mime_type
            }

            params = {
                'punctuate': 'true',
                'diarize': 'true',
                'utterances': 'true',
                'smart_format': 'true',
                'model': 'whisper-large'
            }

            # Add language parameter based on selection
            if language != 'auto':
                params['language'] = language
                params['detect_language'] = 'false'
            else:
                params['detect_language'] = 'true'

            with open(audio_file_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    params=params,
                    data=audio_data,
                    timeout=600
                )

            if response.status_code == 200:
                result = response.json()

                # Extract transcript
                if 'results' in result and 'channels' in result['results']:
                    channel = result['results']['channels'][0]
                    alternatives = channel.get('alternatives', [])
                    if alternatives:
                        self.transcript = alternatives[0].get('transcript', '')
                        self.metadata = {
                            'confidence': alternatives[0].get('confidence', 0),
                            'words': alternatives[0].get('words', []),
                            'utterances': result['results'].get('utterances', [])
                        }

                self.processing_time = time.time() - start_time

                return {
                    'success': True,
                    'transcript': self.transcript,
                    'metadata': self.metadata,
                    'processing_time': self.processing_time
                }
            else:
                self.error = f"API Error: {response.status_code} - {response.text}"
                return {'success': False, 'error': self.error}

        except Exception as e:
            self.error = str(e)
            self.processing_time = time.time() - start_time
            return {'success': False, 'error': self.error}


class GeminiGoldenTranscript:
    """Generate Golden Transcript using Google Gemini"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Use gemini-1.5-flash for stability (gemini-2.5-flash may not be available yet)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_golden_transcript(self, transcripts: Dict[str, str]) -> Tuple[str, Dict]:
        """
        Generate a "Golden Transcript" by analyzing multiple STT outputs

        Args:
            transcripts: Dictionary of provider_name: transcript_text

        Returns:
            Tuple of (golden_transcript, metadata)
        """
        try:
            # Build prompt for Gemini
            prompt = self._build_prompt(transcripts)

            # Generate golden transcript
            response = self.model.generate_content(prompt)
            golden_transcript = response.text.strip()

            metadata = {
                'sources_count': len(transcripts),
                'sources': list(transcripts.keys()),
                'generation_successful': True
            }

            return golden_transcript, metadata

        except Exception as e:
            # Fallback: return the longest transcript if Gemini fails
            fallback_transcript = max(transcripts.values(), key=len) if transcripts else ""
            metadata = {
                'sources_count': len(transcripts),
                'sources': list(transcripts.keys()),
                'generation_successful': False,
                'error': str(e),
                'fallback_used': True
            }
            return fallback_transcript, metadata

    def _build_prompt(self, transcripts: Dict[str, str]) -> str:
        """Build the prompt for Gemini to generate golden transcript"""

        prompt = """You are an expert transcriptionist with years of experience in call center audio analysis.

Your task is to create the MOST ACCURATE transcription possible by analyzing multiple Speech-to-Text outputs from different providers. These transcripts were generated from the same audio file but contain errors and inconsistencies.

Instructions:
1. Carefully analyze all provided transcripts
2. Look for consensus across multiple sources
3. Use context clues to resolve ambiguities
4. Consider phonetic similarities when outputs differ
5. Maintain proper grammar, punctuation, and formatting
6. Output ONLY the corrected transcript text - no explanations or commentary

Here are the transcripts from different STT providers:

"""

        for i, (provider, transcript) in enumerate(transcripts.items(), 1):
            prompt += f"\n{'='*60}\n"
            prompt += f"TRANSCRIPT {i} (from {provider}):\n"
            prompt += f"{'='*60}\n"
            prompt += f"{transcript}\n"

        prompt += f"\n{'='*60}\n"
        prompt += "\nNow, synthesize these transcripts into ONE highly accurate Golden Transcript:"

        return prompt


def calculate_wer_scores(transcripts: Dict[str, str], reference: str) -> Dict[str, float]:
    """
    Calculate Word Error Rate (WER) for each transcript against the reference

    Args:
        transcripts: Dictionary of provider_name: transcript_text
        reference: The golden/reference transcript

    Returns:
        Dictionary of provider_name: wer_score
    """
    wer_scores = {}

    for provider, transcript in transcripts.items():
        try:
            if transcript and reference:
                # Calculate WER
                error_rate = wer(reference, transcript)
                wer_scores[provider] = round(error_rate * 100, 2)  # Convert to percentage
            else:
                wer_scores[provider] = None
        except Exception as e:
            wer_scores[provider] = None

    return wer_scores


def transcribe_with_all_providers(
    audio_file_path: str,
    enabled_providers: Dict[str, bool],
    api_keys: Dict[str, str],
    language: str = 'auto'
) -> Dict[str, Dict]:
    """
    Transcribe audio file with all enabled providers

    Args:
        audio_file_path: Path to audio file
        enabled_providers: Dict of provider_name: is_enabled
        api_keys: Dict of provider API keys
        language: Language code ('auto', 'cs', 'sk', etc.)

    Returns:
        Dictionary of provider results
    """
    results = {}

    # ElevenLabs
    if enabled_providers.get('elevenlabs', False) and api_keys.get('elevenlabs'):
        elevenlabs = ElevenLabsSTT(api_keys['elevenlabs'])
        results['ElevenLabs'] = elevenlabs.transcribe(audio_file_path, language)

    # Gladia
    if enabled_providers.get('gladia', False) and api_keys.get('gladia'):
        gladia = GladiaSTT(api_keys['gladia'])
        results['Gladia'] = gladia.transcribe(audio_file_path, language)

    # OpenAI Whisper
    if enabled_providers.get('openai', False) and api_keys.get('openai'):
        whisper = OpenAIWhisperSTT(api_keys['openai'])
        results['OpenAI Whisper'] = whisper.transcribe(audio_file_path, language)

    # Behavioral Signals
    if enabled_providers.get('behavioral', False) and api_keys.get('behavioral'):
        behavioral = BehavioralSignalsSTT(
            api_keys['behavioral'],
            api_keys.get('behavioral_url', '')
        )
        results['Behavioral Signals'] = behavioral.transcribe(audio_file_path, language)

    # Deepgram
    if enabled_providers.get('deepgram', False) and api_keys.get('deepgram'):
        deepgram = DeepgramSTT(api_keys['deepgram'])
        results['Deepgram'] = deepgram.transcribe(audio_file_path, language)

    return results


def format_metadata_for_display(metadata: Dict) -> str:
    """Format metadata dictionary for nice display"""
    return json.dumps(metadata, indent=2, ensure_ascii=False)
