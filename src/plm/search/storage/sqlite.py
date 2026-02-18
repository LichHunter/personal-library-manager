import sqlite3
from contextlib import contextmanager

import numpy as np


class SQLiteStorage:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_tables(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(id),
                    content TEXT NOT NULL,
                    enriched_content TEXT,
                    embedding BLOB,
                    heading TEXT,
                    start_char INTEGER,
                    end_char INTEGER
                );
            """)

    def add_document(self, doc_id: str, source_file: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents (id, source_file) VALUES (?, ?)",
                (doc_id, source_file),
            )

    def add_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        embedding: np.ndarray,
        enriched_content: str,
        **kwargs,
    ) -> None:
        embedding_blob = embedding.tobytes() if embedding is not None else None
        heading = kwargs.get("heading")
        start_char = kwargs.get("start_char")
        end_char = kwargs.get("end_char")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                    (id, doc_id, content, enriched_content, embedding, heading, start_char, end_char)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, doc_id, content, enriched_content, embedding_blob, heading, start_char, end_char),
            )

    def get_all_chunks(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_document_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0]

    def get_chunk_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0]

    def get_chunk_by_id(self, chunk_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def delete_documents_by_content_hash(self, content_hash: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT id FROM documents WHERE id LIKE ?",
                (f"%_{content_hash}",),
            )
            doc_ids = [row[0] for row in cursor.fetchall()]
            if not doc_ids:
                return 0
            placeholders = ",".join("?" * len(doc_ids))
            conn.execute(f"DELETE FROM chunks WHERE doc_id IN ({placeholders})", doc_ids)
            conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", doc_ids)
            return len(doc_ids)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        if d.get("embedding") is not None:
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
        return d
