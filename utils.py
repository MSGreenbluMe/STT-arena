"""
STT Arena - Utility Functions
Handles API integrations for multiple STT providers and Gemini-based Golden Transcript generation
"""

import os
import time
import asyncio
import aiohttp
import requests
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


class GladiaSTT(STTProvider):
    """Gladia STT Provider"""

    def __init__(self, api_key: str):
        super().__init__("Gladia", api_key)
        self.base_url = "https://api.gladia.io/v2/transcription"

    def transcribe(self, audio_file_path: str) -> Dict:
        """Transcribe audio using Gladia API"""
        start_time = time.time()

        try:
            headers = {
                "x-gladia-key": self.api_key,
            }

            # Upload audio file
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                data = {
                    'toggle_diarization': True,
                    'language_behaviour': 'automatic single language'
                }

                response = requests.post(
                    self.base_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=300
                )

            if response.status_code == 200 or response.status_code == 201:
                result = response.json()

                # Extract transcript text
                if 'result' in result and 'transcription' in result['result']:
                    self.transcript = result['result']['transcription'].get('full_transcript', '')
                    self.metadata = {
                        'diarization': result['result']['transcription'].get('utterances', []),
                        'language': result['result']['transcription'].get('language', 'unknown'),
                        'confidence': result['result']['transcription'].get('confidence', 0)
                    }
                else:
                    self.transcript = result.get('transcription', {}).get('text', '')
                    self.metadata = result

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


class OpenAIWhisperSTT(STTProvider):
    """OpenAI Whisper STT Provider"""

    def __init__(self, api_key: str):
        super().__init__("OpenAI Whisper", api_key)
        self.base_url = "https://api.openai.com/v1/audio/transcriptions"

    def transcribe(self, audio_file_path: str) -> Dict:
        """Transcribe audio using OpenAI Whisper API"""
        start_time = time.time()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }

            with open(audio_file_path, 'rb') as audio_file:
                files = {'file': audio_file}
                data = {
                    'model': 'whisper-1',
                    'response_format': 'verbose_json',
                    'timestamp_granularities': ['word']
                }

                response = requests.post(
                    self.base_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=300
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
        self.api_url = api_url

    def transcribe(self, audio_file_path: str) -> Dict:
        """Transcribe audio using Behavioral Signals API"""
        start_time = time.time()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Note: This is a placeholder implementation
            # You'll need to adjust based on actual Behavioral Signals API documentation
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': audio_file}

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    timeout=300
                )

            if response.status_code == 200:
                result = response.json()

                # Extract transcript and emotion data
                self.transcript = result.get('transcript', result.get('text', ''))
                self.metadata = {
                    'emotion': result.get('emotion', {}),
                    'sentiment': result.get('sentiment', {}),
                    'tone': result.get('tone', {}),
                    'conversation_metrics': result.get('metrics', {})
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


class DeepgramSTT(STTProvider):
    """Deepgram STT Provider"""

    def __init__(self, api_key: str):
        super().__init__("Deepgram", api_key)
        self.base_url = "https://api.deepgram.com/v1/listen"

    def transcribe(self, audio_file_path: str) -> Dict:
        """Transcribe audio using Deepgram API"""
        start_time = time.time()

        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav"
            }

            params = {
                'punctuate': 'true',
                'diarize': 'true',
                'utterances': 'true',
                'smart_format': 'true'
            }

            with open(audio_file_path, 'rb') as audio_file:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    params=params,
                    data=audio_file,
                    timeout=300
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
        self.model = genai.GenerativeModel('gemini-pro')

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
    api_keys: Dict[str, str]
) -> Dict[str, Dict]:
    """
    Transcribe audio file with all enabled providers

    Args:
        audio_file_path: Path to audio file
        enabled_providers: Dict of provider_name: is_enabled
        api_keys: Dict of provider API keys

    Returns:
        Dictionary of provider results
    """
    results = {}

    # Gladia
    if enabled_providers.get('gladia', False) and api_keys.get('gladia'):
        gladia = GladiaSTT(api_keys['gladia'])
        results['Gladia'] = gladia.transcribe(audio_file_path)

    # OpenAI Whisper
    if enabled_providers.get('openai', False) and api_keys.get('openai'):
        whisper = OpenAIWhisperSTT(api_keys['openai'])
        results['OpenAI Whisper'] = whisper.transcribe(audio_file_path)

    # Behavioral Signals
    if enabled_providers.get('behavioral', False) and api_keys.get('behavioral'):
        behavioral = BehavioralSignalsSTT(
            api_keys['behavioral'],
            api_keys.get('behavioral_url', '')
        )
        results['Behavioral Signals'] = behavioral.transcribe(audio_file_path)

    # Deepgram
    if enabled_providers.get('deepgram', False) and api_keys.get('deepgram'):
        deepgram = DeepgramSTT(api_keys['deepgram'])
        results['Deepgram'] = deepgram.transcribe(audio_file_path)

    return results


def format_metadata_for_display(metadata: Dict) -> str:
    """Format metadata dictionary for nice display"""
    return json.dumps(metadata, indent=2, ensure_ascii=False)
