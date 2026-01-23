"""
STT Arena - Database Module
SQLite database for storing audio files, transcriptions, evaluations, and QA sessions
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class STTArenaDB:
    """SQLite database manager for STT Arena"""

    def __init__(self, db_path: str = "stt_arena.db"):
        self.db_path = db_path
        self.conn = None
        self.initialize_db()

    def initialize_db(self):
        """Initialize database connection and create tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.create_tables()

    def create_tables(self):
        """Create all necessary tables"""
        cursor = self.conn.cursor()

        # Table: audio_files
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                audio_hash TEXT UNIQUE,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                duration_seconds REAL,
                language TEXT,
                file_size_mb REAL,
                tags TEXT,
                notes TEXT,
                metadata JSON
            )
        """)

        # Table: transcriptions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id INTEGER,
                provider_name TEXT NOT NULL,
                transcript_text TEXT,
                processing_time_seconds REAL,
                success BOOLEAN,
                error_message TEXT,
                metadata JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files(id) ON DELETE CASCADE
            )
        """)

        # Table: evaluations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id INTEGER,
                evaluation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ground_truth_text TEXT,
                golden_transcript TEXT,
                golden_generator TEXT,
                notes TEXT,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files(id) ON DELETE CASCADE
            )
        """)

        # Table: evaluation_scores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER,
                transcription_id INTEGER,
                provider_name TEXT,
                wer_score REAL,
                cer_score REAL,
                cost_usd REAL,
                cost_effectiveness_score REAL,
                FOREIGN KEY (evaluation_id) REFERENCES evaluations(id) ON DELETE CASCADE,
                FOREIGN KEY (transcription_id) REFERENCES transcriptions(id) ON DELETE CASCADE
            )
        """)

        # Table: qa_sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id INTEGER,
                transcript_source TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files(id) ON DELETE CASCADE
            )
        """)

        # Table: qa_questions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                qa_session_id INTEGER,
                question_text TEXT NOT NULL,
                question_type TEXT,
                ground_truth_answer TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (qa_session_id) REFERENCES qa_sessions(id) ON DELETE CASCADE
            )
        """)

        # Table: qa_answers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER,
                llm_provider TEXT,
                llm_model TEXT,
                answer_text TEXT,
                processing_time_seconds REAL,
                cost_usd REAL,
                tokens_used INTEGER,
                quality_score REAL,
                metadata JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES qa_questions(id) ON DELETE CASCADE
            )
        """)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audio_hash ON audio_files(audio_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audio_tags ON audio_files(tags)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcriptions_audio ON transcriptions(audio_file_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_audio ON evaluations(audio_file_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_qa_sessions_audio ON qa_sessions(audio_file_id)")

        self.conn.commit()

    def calculate_file_hash(self, file_bytes: bytes) -> str:
        """Calculate SHA256 hash of file for deduplication"""
        return hashlib.sha256(file_bytes).hexdigest()

    def save_audio_file(
        self,
        filename: str,
        file_bytes: bytes,
        duration_seconds: float = None,
        language: str = None,
        tags: List[str] = None,
        notes: str = None,
        metadata: Dict = None
    ) -> int:
        """
        Save audio file metadata to database
        Returns audio_file_id
        """
        audio_hash = self.calculate_file_hash(file_bytes)
        file_size_mb = len(file_bytes) / (1024 * 1024)

        # Convert tags list to JSON string
        tags_json = json.dumps(tags) if tags else None

        cursor = self.conn.cursor()

        # Check if file already exists (by hash)
        cursor.execute("SELECT id FROM audio_files WHERE audio_hash = ?", (audio_hash,))
        existing = cursor.fetchone()

        if existing:
            return existing[0]  # Return existing file ID

        # Insert new file
        cursor.execute("""
            INSERT INTO audio_files (filename, audio_hash, duration_seconds, language, file_size_mb, tags, notes, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (filename, audio_hash, duration_seconds, language, file_size_mb, tags_json, notes, json.dumps(metadata) if metadata else None))

        self.conn.commit()
        return cursor.lastrowid

    def save_transcription(
        self,
        audio_file_id: int,
        provider_name: str,
        result: Dict
    ) -> int:
        """
        Save transcription result to database
        Returns transcription_id
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO transcriptions (
                audio_file_id, provider_name, transcript_text, processing_time_seconds,
                success, error_message, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            audio_file_id,
            provider_name,
            result.get('transcript', ''),
            result.get('processing_time', 0),
            result.get('success', False),
            result.get('error', None),
            json.dumps(result.get('metadata', {}))
        ))

        self.conn.commit()
        return cursor.lastrowid

    def save_evaluation(
        self,
        audio_file_id: int,
        golden_transcript: str,
        golden_generator: str,
        ground_truth_text: str = None,
        notes: str = None
    ) -> int:
        """
        Save evaluation (golden transcript + ground truth)
        Returns evaluation_id
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO evaluations (
                audio_file_id, ground_truth_text, golden_transcript, golden_generator, notes
            ) VALUES (?, ?, ?, ?, ?)
        """, (audio_file_id, ground_truth_text, golden_transcript, golden_generator, notes))

        self.conn.commit()
        return cursor.lastrowid

    def save_evaluation_scores(
        self,
        evaluation_id: int,
        transcription_id: int,
        provider_name: str,
        wer_score: float,
        cer_score: float,
        cost_usd: float
    ):
        """Save WER/CER/Cost scores for a provider"""
        # Calculate cost-effectiveness score
        if wer_score is not None and cost_usd is not None and cost_usd > 0:
            # Score = (100 - WER) / cost
            # Higher is better: low error rate, low cost
            accuracy = 100 - wer_score
            cost_effectiveness = accuracy / cost_usd
        else:
            cost_effectiveness = None

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO evaluation_scores (
                evaluation_id, transcription_id, provider_name,
                wer_score, cer_score, cost_usd, cost_effectiveness_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (evaluation_id, transcription_id, provider_name, wer_score, cer_score, cost_usd, cost_effectiveness))

        self.conn.commit()

    def get_audio_files_by_tag(self, tag: str) -> List[Dict]:
        """Get all audio files with a specific tag"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM audio_files
            WHERE tags LIKE ?
            ORDER BY upload_timestamp DESC
        """, (f'%"{tag}"%',))

        return [dict(row) for row in cursor.fetchall()]

    def get_all_tags(self) -> List[str]:
        """Get all unique tags from audio files"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT tags FROM audio_files WHERE tags IS NOT NULL")

        all_tags = set()
        for row in cursor.fetchall():
            if row[0]:
                tags = json.loads(row[0])
                all_tags.update(tags)

        return sorted(list(all_tags))

    def get_provider_stats(self) -> Dict:
        """Get aggregated statistics for each provider"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                es.provider_name,
                COUNT(*) as total_evaluations,
                AVG(es.wer_score) as avg_wer,
                AVG(es.cer_score) as avg_cer,
                AVG(es.cost_usd) as avg_cost,
                AVG(es.cost_effectiveness_score) as avg_cost_effectiveness
            FROM evaluation_scores es
            GROUP BY es.provider_name
            ORDER BY avg_cost_effectiveness DESC
        """)

        return [dict(row) for row in cursor.fetchall()]

    def get_recent_evaluations(self, limit: int = 10) -> List[Dict]:
        """Get recent evaluations with scores"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                e.id,
                e.evaluation_timestamp,
                a.filename,
                a.tags,
                e.golden_generator,
                COUNT(es.id) as num_providers,
                AVG(es.wer_score) as avg_wer,
                AVG(es.cer_score) as avg_cer,
                SUM(es.cost_usd) as total_cost
            FROM evaluations e
            JOIN audio_files a ON e.audio_file_id = a.id
            LEFT JOIN evaluation_scores es ON e.id = es.evaluation_id
            GROUP BY e.id
            ORDER BY e.evaluation_timestamp DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def export_to_csv(self, table_name: str, output_path: str):
        """Export table to CSV"""
        import csv

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")

        rows = cursor.fetchall()
        if not rows:
            return

        # Get column names
        column_names = [description[0] for description in cursor.description]

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)
            writer.writerows(rows)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
