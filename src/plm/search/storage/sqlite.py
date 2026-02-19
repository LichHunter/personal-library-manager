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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding BLOB,
                    keywords_json TEXT,
                    entities_json TEXT
                );

                CREATE TABLE IF NOT EXISTS headings (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(id),
                    heading_text TEXT NOT NULL,
                    heading_level INTEGER,
                    start_char INTEGER,
                    end_char INTEGER,
                    embedding BLOB,
                    keywords_json TEXT,
                    entities_json TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(id),
                    heading_id TEXT REFERENCES headings(id),
                    content TEXT NOT NULL,
                    enriched_content TEXT,
                    embedding BLOB,
                    heading TEXT,
                    start_char INTEGER,
                    end_char INTEGER,
                    chunk_index INTEGER,
                    keywords_json TEXT,
                    entities_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_headings_doc_id ON headings(doc_id);
            """)
            self._run_migrations(conn)
            self._create_indexes_after_migration(conn)

    def _create_indexes_after_migration(self, conn) -> None:
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_heading_id ON chunks(heading_id)")
        except sqlite3.OperationalError:
            pass

    def _run_migrations(self, conn) -> None:
        self._migrate_add_column(conn, "documents", "embedding", "BLOB")
        self._migrate_add_column(conn, "documents", "keywords_json", "TEXT")
        self._migrate_add_column(conn, "documents", "entities_json", "TEXT")
        self._migrate_add_column(conn, "chunks", "heading_id", "TEXT")
        self._migrate_add_column(conn, "chunks", "chunk_index", "INTEGER")
        self._migrate_add_column(conn, "chunks", "keywords_json", "TEXT")
        self._migrate_add_column(conn, "chunks", "entities_json", "TEXT")

    def _migrate_add_column(self, conn, table: str, column: str, col_type: str) -> None:
        """Add column to table if it doesn't exist (idempotent migration)."""
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

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
        heading_id = kwargs.get("heading_id")
        start_char = kwargs.get("start_char")
        end_char = kwargs.get("end_char")
        chunk_index = kwargs.get("chunk_index")
        keywords_json = kwargs.get("keywords_json")
        entities_json = kwargs.get("entities_json")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                    (id, doc_id, heading_id, content, enriched_content, embedding, heading,
                     start_char, end_char, chunk_index, keywords_json, entities_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, doc_id, heading_id, content, enriched_content, embedding_blob, heading,
                 start_char, end_char, chunk_index, keywords_json, entities_json),
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
            conn.execute(f"DELETE FROM headings WHERE doc_id IN ({placeholders})", doc_ids)
            conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", doc_ids)
            return len(doc_ids)

    def add_heading(
        self,
        heading_id: str,
        doc_id: str,
        heading_text: str,
        heading_level: int | None = None,
        start_char: int | None = None,
        end_char: int | None = None,
        embedding: np.ndarray | None = None,
        keywords_json: str | None = None,
        entities_json: str | None = None,
    ) -> None:
        embedding_blob = embedding.tobytes() if embedding is not None else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO headings
                    (id, doc_id, heading_text, heading_level, start_char, end_char,
                     embedding, keywords_json, entities_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (heading_id, doc_id, heading_text, heading_level, start_char, end_char,
                 embedding_blob, keywords_json, entities_json),
            )

    def get_headings_by_doc(self, doc_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM headings WHERE doc_id = ? ORDER BY start_char",
                (doc_id,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_all_headings(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM headings ORDER BY rowid").fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_chunks_by_doc(self, doc_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index, start_char",
                (doc_id,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_chunks_by_heading(self, heading_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE heading_id = ? ORDER BY chunk_index",
                (heading_id,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_all_documents(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY rowid").fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_document_aggregates(
        self,
        doc_id: str,
        embedding: np.ndarray | None = None,
        keywords_json: str | None = None,
        entities_json: str | None = None,
    ) -> None:
        embedding_blob = embedding.tobytes() if embedding is not None else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET embedding = ?, keywords_json = ?, entities_json = ?
                WHERE id = ?
                """,
                (embedding_blob, keywords_json, entities_json, doc_id),
            )

    def update_heading_aggregates(
        self,
        heading_id: str,
        embedding: np.ndarray | None = None,
        keywords_json: str | None = None,
        entities_json: str | None = None,
    ) -> None:
        embedding_blob = embedding.tobytes() if embedding is not None else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE headings
                SET embedding = ?, keywords_json = ?, entities_json = ?
                WHERE id = ?
                """,
                (embedding_blob, keywords_json, entities_json, heading_id),
            )

    def update_chunk_heading(self, chunk_id: str, heading_id: str, chunk_index: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chunks SET heading_id = ?, chunk_index = ? WHERE id = ?",
                (heading_id, chunk_index, chunk_id),
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        if d.get("embedding") is not None:
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
        return d
