import sqlite3
import json
import time
from typing import Optional, List, Dict, Any

DB_PATH = "adaptive.db"


class AdaptiveDB:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            arch_key TEXT NOT NULL,
            chunk_ids TEXT NOT NULL,
            rating INTEGER NOT NULL,
            ts REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS query_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arch_key TEXT NOT NULL,
            query_text TEXT NOT NULL,
            query_embedding TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT NOT NULL,
            ts REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arch_key TEXT NOT NULL,
            query TEXT NOT NULL,
            elapsed REAL NOT NULL DEFAULT 0,
            faithfulness REAL DEFAULT 0,
            relevance REAL DEFAULT 0,
            context_precision REAL DEFAULT 0,
            context_recall REAL DEFAULT 0,
            cached INTEGER DEFAULT 0,
            ts REAL NOT NULL
        );
        """)
        self.conn.commit()

    # ── Feedback ──────────────────────────────────────────────────────────────

    def store_feedback(self, query: str, arch_key: str, chunk_ids: List[str], rating: int):
        self.conn.execute(
            "INSERT INTO feedback (query, arch_key, chunk_ids, rating, ts) VALUES (?, ?, ?, ?, ?)",
            (query, arch_key, json.dumps(chunk_ids), rating, time.time()),
        )
        self.conn.commit()

    def get_positive_sources(self, arch_key: str) -> List[str]:
        """Returns list of chunk text snippets that received positive feedback for this arch."""
        rows = self.conn.execute(
            "SELECT chunk_ids FROM feedback WHERE arch_key = ? AND rating > 0 ORDER BY ts DESC LIMIT 100",
            (arch_key,),
        ).fetchall()
        result = []
        for (chunk_ids_json,) in rows:
            try:
                result.extend(json.loads(chunk_ids_json))
            except Exception as e:
                print(f"[adaptive_db] chunk_ids parse error: {e}")
        return result

    # ── Semantic cache ────────────────────────────────────────────────────────

    def find_similar_query(
        self, query_embedding: List[float], arch_key: str, threshold: float = 0.92
    ) -> Optional[Dict]:
        try:
            import numpy as np
        except ImportError:
            return None

        rows = self.conn.execute(
            "SELECT query_text, query_embedding, answer, sources FROM query_cache "
            "WHERE arch_key = ? ORDER BY ts DESC LIMIT 100",
            (arch_key,),
        ).fetchall()
        if not rows:
            return None

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm == 0:
            return None

        for query_text, emb_json, answer, sources_json in rows:
            try:
                c_vec = np.array(json.loads(emb_json), dtype=np.float32)
                c_norm = float(np.linalg.norm(c_vec))
                if c_norm == 0:
                    continue
                sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
                if sim >= threshold:
                    return {
                        "query": query_text,
                        "answer": answer,
                        "sources": json.loads(sources_json),
                        "similarity": round(sim, 3),
                    }
            except Exception as e:
                print(f"[adaptive_db] similarity calc error: {e}")
                continue
        return None

    def store_query_cache(
        self,
        arch_key: str,
        query: str,
        query_embedding: List[float],
        answer: str,
        sources: List[Dict],
    ):
        self.conn.execute(
            "DELETE FROM query_cache WHERE arch_key = ? AND query_text = ?",
            (arch_key, query),
        )
        self.conn.execute(
            "INSERT INTO query_cache (arch_key, query_text, query_embedding, answer, sources, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                arch_key,
                query,
                json.dumps(query_embedding),
                answer,
                json.dumps(sources),
                time.time(),
            ),
        )
        self.conn.commit()

    def get_cache_count(self, arch_key: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM query_cache WHERE arch_key = ?", (arch_key,)
        ).fetchone()
        return row[0] if row else 0

    # ── Analytics ─────────────────────────────────────────────────────────────

    def store_query_analytics(
        self, arch_key: str, query: str, elapsed: float, cached: bool = False
    ):
        self.conn.execute(
            "INSERT INTO analytics (arch_key, query, elapsed, cached, ts) VALUES (?, ?, ?, ?, ?)",
            (arch_key, query, elapsed, 1 if cached else 0, time.time()),
        )
        self.conn.commit()

    def store_eval_analytics(
        self, arch_key: str, query: str, eval_scores: Dict[str, float]
    ):
        self.conn.execute(
            "INSERT INTO analytics (arch_key, query, elapsed, faithfulness, relevance, "
            "context_precision, context_recall, ts) VALUES (?, ?, 0, ?, ?, ?, ?, ?)",
            (
                arch_key,
                query,
                eval_scores.get("faithfulness", 0),
                eval_scores.get("relevance", 0),
                eval_scores.get("context_precision", 0),
                eval_scores.get("context_recall", 0),
                time.time(),
            ),
        )
        self.conn.commit()

    def get_analytics(self) -> Dict:
        rows = self.conn.execute("""
            SELECT arch_key,
                SUM(CASE WHEN elapsed > 0 THEN 1 ELSE 0 END) as qcount,
                AVG(CASE WHEN elapsed > 0 THEN elapsed END) as avg_elapsed,
                AVG(CASE WHEN faithfulness > 0 THEN faithfulness END) as avg_faith,
                AVG(CASE WHEN relevance > 0 THEN relevance END) as avg_rel,
                AVG(CASE WHEN context_precision > 0 THEN context_precision END) as avg_cp,
                AVG(CASE WHEN context_recall > 0 THEN context_recall END) as avg_cr,
                SUM(cached) as cache_hits
            FROM analytics GROUP BY arch_key
        """).fetchall()

        fb_rows = self.conn.execute("""
            SELECT arch_key,
                SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) as pos,
                COUNT(*) as total
            FROM feedback GROUP BY arch_key
        """).fetchall()

        fb_map = {r[0]: {"positive": r[1], "total": r[2]} for r in fb_rows}

        result = {}
        for r in rows:
            fb = fb_map.get(r[0], {"positive": 0, "total": 0})
            result[r[0]] = {
                "query_count": r[1] or 0,
                "avg_elapsed": round(r[2] or 0, 2),
                "avg_faithfulness": round(r[3] or 0, 1) if r[3] else None,
                "avg_relevance": round(r[4] or 0, 1) if r[4] else None,
                "avg_context_precision": round(r[5] or 0, 1) if r[5] else None,
                "avg_context_recall": round(r[6] or 0, 1) if r[6] else None,
                "cache_hits": r[7] or 0,
                "feedback_positive": fb["positive"],
                "feedback_total": fb["total"],
            }
        return result

    def get_feedback_docs(self, arch_key: str):
        """Returns (positive_snippets, negative_snippets) as sets of lowercased text prefixes."""
        rows = self.conn.execute(
            "SELECT chunk_ids, rating FROM feedback WHERE arch_key = ? ORDER BY ts DESC LIMIT 500",
            (arch_key,),
        ).fetchall()
        positive, negative = set(), set()
        for (chunk_ids_json, rating) in rows:
            try:
                snippets = json.loads(chunk_ids_json)
                if rating > 0:
                    positive.update(s.lower() for s in snippets if s)
                else:
                    negative.update(s.lower() for s in snippets if s)
            except Exception as e:
                print(f"[adaptive_db] feedback_docs parse error: {e}")
        return positive, negative

    def apply_feedback_boost(self, texts: List[str], metas: List[Any], arch_key: str):
        """Reorders (texts, metas) so positively-rated chunks surface first, negative last.
        Matching is done on the first 80 chars of each text against stored feedback snippets."""
        positive, negative = self.get_feedback_docs(arch_key)
        if not positive and not negative:
            return texts, metas

        def score(text: str) -> int:
            prefix = text[:80].lower()
            for p in positive:
                if p and (p in prefix or prefix in p):
                    return 1
            for n in negative:
                if n and (n in prefix or prefix in n):
                    return -1
            return 0

        pairs = list(zip(texts, metas))
        if not pairs:
            return texts, metas
        pairs.sort(key=lambda x: score(x[0]), reverse=True)
        new_texts, new_metas = zip(*pairs)
        return list(new_texts), list(new_metas)

    def get_recent_queries(self, limit: int = 20) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT arch_key, query, elapsed, ts FROM analytics "
            "WHERE elapsed > 0 ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"arch_key": r[0], "query": r[1], "elapsed": round(r[2], 2), "ts": r[3]}
            for r in rows
        ]


adaptive_db = AdaptiveDB()
