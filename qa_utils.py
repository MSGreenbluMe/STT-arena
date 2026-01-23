"""
QA Arena - Utility Functions
Handles LLM integrations for question answering and evaluation
"""

import os
import time
import requests
import json
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai


class LLMProvider:
    """Base class for LLM providers"""

    def __init__(self, name: str, api_key: str):
        self.name = name
        self.api_key = api_key

    def ask_question(self, context: str, question: str) -> Dict:
        """Override this method in subclasses"""
        raise NotImplementedError


class GeminiLLM(LLMProvider):
    """Google Gemini LLM Provider"""

    def __init__(self, api_key: str, model_name: str = 'gemini-2.5-flash'):
        super().__init__(f"Gemini ({model_name})", api_key)
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def ask_question(self, context: str, question: str) -> Dict:
        """Ask a question based on the provided context"""
        start_time = time.time()

        try:
            prompt = f"""Based on the following transcript, answer the question accurately and concisely.

TRANSCRIPT:
{context}

QUESTION:
{question}

ANSWER (be specific and factual):"""

            response = self.model.generate_content(prompt)
            answer = response.text.strip()

            processing_time = time.time() - start_time

            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            input_tokens = len(prompt) // 4
            output_tokens = len(answer) // 4

            return {
                'success': True,
                'answer': answer,
                'processing_time': processing_time,
                'tokens_used': input_tokens + output_tokens,
                'model': self.model_name
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }


class GroqLLM(LLMProvider):
    """Groq LLM Provider (Llama, Mixtral models)"""

    def __init__(self, api_key: str, model_name: str = 'llama-3.3-70b-versatile'):
        super().__init__(f"Groq ({model_name})", api_key)
        self.model_name = model_name
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def ask_question(self, context: str, question: str) -> Dict:
        """Ask a question based on the provided context"""
        start_time = time.time()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided transcripts. Be accurate, concise, and factual."
                    },
                    {
                        "role": "user",
                        "content": f"Based on this transcript:\n\n{context}\n\nAnswer this question: {question}"
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1024
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content'].strip()
                tokens_used = result['usage']['total_tokens']

                return {
                    'success': True,
                    'answer': answer,
                    'processing_time': time.time() - start_time,
                    'tokens_used': tokens_used,
                    'model': self.model_name
                }
            else:
                return {
                    'success': False,
                    'error': f"API Error: {response.status_code} - {response.text}",
                    'processing_time': time.time() - start_time
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }


class MistralLLM(LLMProvider):
    """Mistral AI LLM Provider"""

    def __init__(self, api_key: str, model_name: str = 'mistral-small-latest'):
        super().__init__(f"Mistral ({model_name})", api_key)
        self.model_name = model_name
        self.base_url = "https://api.mistral.ai/v1/chat/completions"

    def ask_question(self, context: str, question: str) -> Dict:
        """Ask a question based on the provided context"""
        start_time = time.time()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided transcripts. Be accurate, concise, and factual."
                    },
                    {
                        "role": "user",
                        "content": f"Based on this transcript:\n\n{context}\n\nAnswer this question: {question}"
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1024
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content'].strip()
                tokens_used = result['usage']['total_tokens']

                return {
                    'success': True,
                    'answer': answer,
                    'processing_time': time.time() - start_time,
                    'tokens_used': tokens_used,
                    'model': self.model_name
                }
            else:
                return {
                    'success': False,
                    'error': f"API Error: {response.status_code} - {response.text}",
                    'processing_time': time.time() - start_time
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }


class QuestionGenerator:
    """Generate questions from transcript using Gemini"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_questions(self, transcript: str, num_questions: int = 5) -> List[Dict]:
        """
        Generate questions from transcript

        Args:
            transcript: The transcript text
            num_questions: Number of questions to generate

        Returns:
            List of question dictionaries with 'question' and 'type' fields
        """
        try:
            prompt = f"""Based on the following call center transcript, generate {num_questions} diverse questions that would test a Q&A system's understanding.

TRANSCRIPT:
{transcript}

Generate exactly {num_questions} questions covering different aspects:
1. Factual questions (who, what, when, where)
2. Summarization (main topic, key points)
3. Sentiment analysis (customer mood, satisfaction)
4. Entity extraction (names, companies, products)
5. Problem resolution (what was the issue, how was it resolved)

Format your response as a JSON array like this:
[
    {{"question": "What was the customer's main complaint?", "type": "factual"}},
    {{"question": "Summarize the conversation in 2-3 sentences.", "type": "summary"}},
    {{"question": "What was the customer's sentiment?", "type": "sentiment"}}
]

Return ONLY the JSON array, no additional text."""

            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            questions = json.loads(response_text)

            return questions[:num_questions]  # Ensure we return exactly num_questions

        except Exception as e:
            print(f"[ERROR] Question generation failed: {e}")
            # Return default questions
            return [
                {"question": "What is the main topic of this conversation?", "type": "summary"},
                {"question": "Who are the speakers in this conversation?", "type": "factual"},
                {"question": "What problem was discussed?", "type": "factual"},
                {"question": "How was the issue resolved?", "type": "resolution"},
                {"question": "What was the overall sentiment?", "type": "sentiment"}
            ][:num_questions]


def calculate_llm_costs(results: Dict[str, Dict]) -> Dict[str, float]:
    """
    Calculate costs for LLM providers

    Pricing (as of Jan 2026):
    - Gemini 2.5-flash: $0.00001/1K input, $0.00004/1K output (avg: $0.000025/1K)
    - Groq: FREE
    - Mistral small: €0.001/1K (~$0.0011/1K)
    - Mistral large: €0.008/1K (~$0.0088/1K)
    """
    pricing = {
        'gemini-2.5-flash': 0.000025,  # Average of input/output
        'gemini-2.5-flash-lite': 0.000025,
        'gemini-2.5-pro': 0.003125,  # Average of input/output
        'llama-3.3-70b-versatile': 0.0,  # FREE
        'llama-3.1-8b-instant': 0.0,  # FREE
        'mixtral-8x7b-32768': 0.0,  # FREE
        'mistral-small-latest': 0.0011,
        'mistral-large-latest': 0.0088,
        'open-mistral-7b': 0.0  # FREE tier
    }

    costs = {}

    for provider_name, result in results.items():
        if not result.get('success'):
            costs[provider_name] = None
            continue

        model = result.get('model', '')
        tokens_used = result.get('tokens_used', 0)

        # Find pricing for model
        price_per_1k = 0.0
        for model_key, price in pricing.items():
            if model_key in model:
                price_per_1k = price
                break

        # Calculate cost
        cost = (tokens_used / 1000.0) * price_per_1k
        costs[provider_name] = round(cost, 6)

    return costs
