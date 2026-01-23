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
from jiwer import wer, cer, Compose, RemovePunctuation, RemoveMultipleSpaces, Strip, ToLowerCase
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
            elif response.status_code == 401:
                # Specific handling for 401 errors (common with free tier)
                error_detail = response.text
                if 'Free Tier' in error_detail or 'unusual activity' in error_detail.lower():
                    self.error = f"ElevenLabs Free Tier blocked. Try: 1) New API key 2) Paid plan 3) Contact support"
                else:
                    self.error = f"Authentication failed: Check API key format (should start with 'xi-api-key')"
                return {'success': False, 'error': self.error}
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


class GroqWhisperSTT(STTProvider):
    """Groq Whisper Large v3 STT Provider (Free & Fast)"""

    def __init__(self, api_key: str):
        super().__init__("Groq Whisper", api_key)
        self.base_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    def transcribe(self, audio_file_path: str, language: str = 'auto') -> Dict:
        """Transcribe audio using Groq's Whisper Large v3 API"""
        start_time = time.time()

        try:
            # Check file size (Groq has 25MB limit)
            file_size = os.path.getsize(audio_file_path)
            file_size_mb = file_size / (1024 * 1024)

            if file_size > 25 * 1024 * 1024:  # 25MB in bytes
                self.error = f"File too large: {file_size_mb:.1f}MB (Groq limit: 25MB). Try shorter audio or use Deepgram/Gladia for long files."
                return {'success': False, 'error': self.error}

            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }

            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()

            files = {'file': ('audio.mp3', audio_data)}
            data = {
                'model': 'whisper-large-v3',  # Groq's SOTA model
                'response_format': 'verbose_json',  # Get segments with confidence
                'temperature': 0.0  # Deterministic output
            }

            # Add language if specified
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

                # Extract rich metadata from verbose_json
                segments = result.get('segments', [])
                avg_confidence = 0
                if segments:
                    # Calculate average confidence from avg_logprob
                    # avg_logprob closer to 0 = higher confidence
                    avg_logprob = sum(s.get('avg_logprob', -1) for s in segments) / len(segments)
                    avg_confidence = round((1 + avg_logprob) * 100, 2)  # Convert to percentage

                self.metadata = {
                    'language': result.get('language', 'unknown'),
                    'duration': result.get('duration', 0),
                    'segments': segments,
                    'avg_confidence': avg_confidence,
                    'model': 'whisper-large-v3',
                    'provider': 'Groq (Free)',
                    'task': result.get('task', 'transcribe')
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
                if 'pid' in result:
                    pid = result.get('pid')

                    # If no transcript yet, poll for results
                    if not result.get('transcript') and not result.get('text'):
                        # Polling loop (max 3 minutes for 11-minute audio)
                        max_polls = 180  # 3 minutes
                        poll_count = 0

                        # Extract base URL and construct status URL
                        # URL format: /v5/clients/{cid}/processes/{pid}
                        base_url = self.api_url.replace('/processes/audio', '')
                        status_url = f"{base_url}/processes/{pid}"

                        while poll_count < max_polls:
                            time.sleep(2)  # Wait 2 seconds between polls
                            poll_count += 1

                            status_response = requests.get(
                                status_url,
                                headers=headers,
                                timeout=30
                            )

                            if status_response.status_code == 200:
                                result = status_response.json()
                                status = result.get('status', '')
                                # Convert to string and lowercase safely
                                if isinstance(status, int):
                                    status = str(status)
                                status = status.lower() if status else ''

                                # Check for completion
                                if status in ['done', 'completed', 'success']:
                                    break
                                elif status in ['error', 'failed']:
                                    self.error = f"Processing failed: {result.get('statusmsg', 'Unknown error')}"
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
    """Generate Golden Transcript using Google Gemini (with Groq fallback)"""

    def __init__(self, api_key: str, groq_api_key: str = None, model_name: str = 'gemini-2.5-flash'):
        self.gemini_api_key = api_key
        self.groq_api_key = groq_api_key
        genai.configure(api_key=api_key)
        # Use gemini-2.5-flash for better quality (5 RPM free tier)
        # Alternative models:
        # - gemini-2.5-flash-lite: 10 RPM (faster, lower quality)
        # - gemini-2.5-flash: 5 RPM (balanced, recommended)
        # - gemini-2.5-pro: 2 RPM (best quality, slowest)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

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
                'generation_successful': True,
                'generator': f'Gemini ({self.model_name})'
            }

            return golden_transcript, metadata

        except Exception as e:
            # Log the detailed error
            error_msg = f"Gemini Error: {type(e).__name__}: {str(e)}"
            print(f"[ERROR] {error_msg}")  # Server-side log
            # Fallback to Groq if available
            if self.groq_api_key:
                try:
                    fallback_transcript = self._generate_with_groq(transcripts)
                    metadata = {
                        'sources_count': len(transcripts),
                        'sources': list(transcripts.keys()),
                        'generation_successful': True,
                        'generator': 'Groq (Gemini fallback)',
                        'gemini_error': str(e)
                    }
                    return fallback_transcript, metadata
                except Exception as groq_error:
                    # Both failed, use longest transcript
                    fallback_transcript = max(transcripts.values(), key=len) if transcripts else ""
                    metadata = {
                        'sources_count': len(transcripts),
                        'sources': list(transcripts.keys()),
                        'generation_successful': False,
                        'generator': 'Longest transcript (both AI failed)',
                        'gemini_error': str(e),
                        'groq_error': str(groq_error),
                        'fallback_used': True
                    }
                    return fallback_transcript, metadata
            else:
                # No Groq fallback, use longest transcript
                fallback_transcript = max(transcripts.values(), key=len) if transcripts else ""
                metadata = {
                    'sources_count': len(transcripts),
                    'sources': list(transcripts.keys()),
                    'generation_successful': False,
                    'generator': 'Longest transcript (Gemini failed)',
                    'error': str(e),
                    'fallback_used': True
                }
                return fallback_transcript, metadata

    def _generate_with_groq(self, transcripts: Dict[str, str]) -> str:
        """Fallback generation using Groq LLM"""
        prompt = self._build_prompt(transcripts)

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.3-70b-versatile",  # Groq's best model
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert transcriptionist. Analyze multiple STT outputs and generate the most accurate transcript."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4096
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")

    def _build_prompt(self, transcripts: Dict[str, str]) -> str:
        """Build the prompt for Gemini to generate golden transcript"""

        prompt = """You are an expert transcriptionist specializing in Czech and Slovak languages with years of experience in call center audio analysis.

Your task is to create the MOST ACCURATE transcription possible by analyzing multiple Speech-to-Text outputs from different providers. These transcripts were generated from the same audio file but contain errors and inconsistencies.

CRITICAL INSTRUCTIONS:
1. Carefully analyze ALL provided transcripts word-by-word
2. Look for CONSENSUS across multiple sources - if 3+ providers agree, that's likely correct
3. Preserve DIACRITICS correctly (ľščťžýáíéúôäň etc.) - this is crucial for Slovak/Czech
4. Use context clues to resolve ambiguities
5. Consider phonetic similarities when outputs differ
6. Maintain proper grammar, punctuation, and formatting
7. For Slovak/Czech: Pay special attention to:
   - Diacritics (č vs c, š vs s, ž vs z, ý vs y, etc.)
   - Similar sounding words (deň/den, máte/mate, dobrý/dobry)
   - Proper names and company names
8. Output ONLY the corrected transcript text - NO explanations, NO commentary, NO metadata

QUALITY CRITERIA:
- Every word should match the actual spoken audio
- Diacritics must be 100% accurate
- Natural Slovak/Czech grammar and syntax
- Professional formatting (proper capitalization, punctuation)

Here are the transcripts from different STT providers:

"""

        for i, (provider, transcript) in enumerate(transcripts.items(), 1):
            prompt += f"\n{'='*60}\n"
            prompt += f"TRANSCRIPT {i} (from {provider}):\n"
            prompt += f"{'='*60}\n"
            prompt += f"{transcript}\n"

        prompt += f"\n{'='*60}\n"
        prompt += "\nNow, synthesize these transcripts into ONE highly accurate Golden Transcript.\n"
        prompt += "Remember: ONLY the transcript text, NO explanations:"

        return prompt


def calculate_wer_scores(transcripts: Dict[str, str], reference: str) -> Dict[str, float]:
    """
    Calculate Word Error Rate (WER) for each transcript against the reference
    Uses text normalization: lowercase, remove punctuation, strip whitespace

    Args:
        transcripts: Dictionary of provider_name: transcript_text
        reference: The golden/reference transcript

    Returns:
        Dictionary of provider_name: wer_score
    """
    wer_scores = {}

    # Text normalization pipeline for fair comparison
    # Remove punctuation, convert to lowercase, normalize whitespace
    transformation = Compose([
        ToLowerCase(),
        RemovePunctuation(),
        RemoveMultipleSpaces(),
        Strip()
    ])

    for provider, transcript in transcripts.items():
        try:
            if transcript and reference:
                # Normalize both texts before comparison
                normalized_ref = transformation(reference)
                normalized_trans = transformation(transcript)

                # Calculate WER on normalized text
                error_rate = wer(normalized_ref, normalized_trans)
                wer_scores[provider] = round(error_rate * 100, 2)  # Convert to percentage
            else:
                wer_scores[provider] = None
        except Exception as e:
            print(f"[ERROR] WER calculation for {provider}: {e}")
            wer_scores[provider] = None

    return wer_scores


def calculate_cer_scores(transcripts: Dict[str, str], reference: str) -> Dict[str, float]:
    """
    Calculate Character Error Rate (CER) for each transcript against the reference
    Uses text normalization: lowercase, remove punctuation, strip whitespace

    Args:
        transcripts: Dictionary of provider_name: transcript_text
        reference: The golden/reference transcript

    Returns:
        Dictionary of provider_name: cer_score
    """
    cer_scores = {}

    # Text normalization pipeline for fair comparison
    transformation = Compose([
        ToLowerCase(),
        RemovePunctuation(),
        RemoveMultipleSpaces(),
        Strip()
    ])

    for provider, transcript in transcripts.items():
        try:
            if transcript and reference:
                # Normalize both texts before comparison
                normalized_ref = transformation(reference)
                normalized_trans = transformation(transcript)

                # Calculate CER on normalized text
                error_rate = cer(normalized_ref, normalized_trans)
                cer_scores[provider] = round(error_rate * 100, 2)  # Convert to percentage
            else:
                cer_scores[provider] = None
        except Exception as e:
            print(f"[ERROR] CER calculation for {provider}: {e}")
            cer_scores[provider] = None

    return cer_scores


def calculate_segment_wer(provider_segments: List[Dict], reference_text: str) -> List[Dict]:
    """
    Calculate WER for each segment/timestamp by comparing segment text with corresponding
    portion of reference text (estimated by word position)

    Args:
        provider_segments: List of segments with 'start', 'end', 'text' fields
        reference_text: The golden/reference transcript

    Returns:
        List of dicts with 'start', 'end', 'wer', 'text' for visualization
    """
    segment_wers = []

    if not provider_segments or not reference_text:
        return segment_wers

    try:
        from jiwer import wer as calculate_wer

        # Split reference into words
        ref_words = reference_text.split()
        total_words = len(ref_words)

        if total_words == 0:
            return segment_wers

        # Calculate cumulative word counts for segments
        segment_word_counts = []
        for segment in provider_segments:
            segment_text = segment.get('text', segment.get('transcript', ''))
            word_count = len(segment_text.split())
            segment_word_counts.append(word_count)

        # Map each segment to approximate portion of reference
        cumulative_words = 0
        for idx, segment in enumerate(provider_segments):
            segment_text = segment.get('text', segment.get('transcript', ''))
            start_time = segment.get('start', segment.get('start_time', 0))
            end_time = segment.get('end', segment.get('end_time', 0))

            if not segment_text:
                continue

            # Calculate which portion of reference corresponds to this segment
            segment_words = segment_word_counts[idx]

            # Get corresponding words from reference (proportional mapping)
            start_word_idx = cumulative_words
            end_word_idx = min(cumulative_words + segment_words, total_words)

            # Extract reference portion for this segment
            ref_portion = ' '.join(ref_words[start_word_idx:end_word_idx])

            # Calculate WER for this segment
            try:
                if ref_portion and segment_text:
                    error_rate = calculate_wer(ref_portion, segment_text)
                    wer_score = min(round(error_rate * 100, 2), 100.0)  # Cap at 100%
                else:
                    wer_score = 0.0
            except:
                wer_score = None

            segment_wers.append({
                'start': start_time,
                'end': end_time,
                'wer': wer_score,
                'text': segment_text[:50] + '...' if len(segment_text) > 50 else segment_text,
                'ref_text': ref_portion[:50] + '...' if len(ref_portion) > 50 else ref_portion
            })

            cumulative_words += segment_words

    except Exception as e:
        print(f"[ERROR] calculate_segment_wer: {e}")
        pass

    return segment_wers


def calculate_costs(results: Dict[str, Dict]) -> Dict[str, float]:
    """
    Calculate estimated cost for each transcription provider

    Args:
        results: Dictionary of provider results with metadata

    Returns:
        Dictionary of provider_name: cost_in_usd
    """
    # Pricing per minute (as of Jan 2025)
    pricing = {
        'ElevenLabs': 0.10,  # $0.10/min (Scribe v2)
        'Gladia': 0.01830,   # $0.01830/min ($0.000305/second)
        'OpenAI Whisper': 0.006,  # $0.006/min
        'Groq Whisper': 0.0,  # FREE!
        'Behavioral Signals': 0.0,  # Custom pricing
        'Deepgram': 0.0125,  # $0.0125/min (Whisper model)
        'Manual Transcript': 0.0  # Manual upload (no cost)
    }

    costs = {}

    for provider, result in results.items():
        if not result.get('success'):
            costs[provider] = None
            continue

        # Get duration from metadata (in seconds)
        metadata = result.get('metadata', {})
        duration_seconds = metadata.get('duration', 0)

        # If no duration in metadata, estimate from processing time (not accurate but better than nothing)
        if not duration_seconds:
            duration_seconds = result.get('processing_time', 0)

        # Convert to minutes
        duration_minutes = duration_seconds / 60.0

        # Calculate cost
        price_per_min = pricing.get(provider, 0)
        cost = duration_minutes * price_per_min

        costs[provider] = round(cost, 4)  # Round to 4 decimal places

    return costs


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

    # Groq Whisper Large v3
    if enabled_providers.get('groq', False) and api_keys.get('groq'):
        groq = GroqWhisperSTT(api_keys['groq'])
        results['Groq Whisper'] = groq.transcribe(audio_file_path, language)

    # Behavioral Signals - TEMPORARILY DISABLED (slow performance, API issues)
    # if enabled_providers.get('behavioral', False) and api_keys.get('behavioral'):
    #     behavioral = BehavioralSignalsSTT(
    #         api_keys['behavioral'],
    #         api_keys.get('behavioral_url', '')
    #     )
    #     results['Behavioral Signals'] = behavioral.transcribe(audio_file_path, language)

    # Deepgram
    if enabled_providers.get('deepgram', False) and api_keys.get('deepgram'):
        deepgram = DeepgramSTT(api_keys['deepgram'])
        results['Deepgram'] = deepgram.transcribe(audio_file_path, language)

    return results


def format_metadata_for_display(metadata: Dict) -> str:
    """Format metadata dictionary for nice display"""
    return json.dumps(metadata, indent=2, ensure_ascii=False)
