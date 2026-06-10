from typing import Dict, Any, List, Set

from architectures.hybrid_rag import HybridRAGPipeline
from architectures.graph_rag import GraphRAGPipeline
from architectures.agentic_rag import AgenticRAGPipeline
from architectures.corrective_rag import CorrectiveRAGPipeline
from architectures.multimodal_rag import MultimodalRAGPipeline
from architectures.multilingual_rag import MultilingualRAGPipeline
from architectures.rag_fusion import RAGFusionPipeline
from architectures.hyde_rag import HyDERAGPipeline
from architectures.structured_rag import StructuredRAGPipeline

ARCH_KEYS: List[str] = [
    "01 Hybrid RAG (Dense + Sparse)",
    "02 Graph RAG (Knowledge Graphs)",
    "03 Agentic RAG (LangGraph)",
    "04 Corrective RAG (CRAG)",
    "05 Multimodal RAG (Vision + Text)",
    "06 Multilingual RAG (BGE-M3)",
    "07 RAG-Fusion (Query Expansion)",
    "08 HyDE RAG (Hypothetical Document)",
    "09 Structured RAG (CSV/Excel)",
]

STATE_KEY_MAP: Dict[str, str] = {
    "01 Hybrid RAG (Dense + Sparse)":       "hybrid_pipeline",
    "02 Graph RAG (Knowledge Graphs)":       "graph_pipeline",
    "03 Agentic RAG (LangGraph)":            "agentic_pipeline",
    "04 Corrective RAG (CRAG)":              "crag_pipeline",
    "05 Multimodal RAG (Vision + Text)":     "multimodal_pipeline",
    "06 Multilingual RAG (BGE-M3)":          "multilingual_pipeline",
    "07 RAG-Fusion (Query Expansion)":       "rag_fusion_pipeline",
    "08 HyDE RAG (Hypothetical Document)":   "hyde_pipeline",
    "09 Structured RAG (CSV/Excel)":         "structured_pipeline",
}

ARCH_INFO: Dict[str, Dict] = {
    "01 Hybrid RAG (Dense + Sparse)": {
        "key": "01 Hybrid RAG (Dense + Sparse)",
        "icon": "🔀", "label": "Hybrid RAG (Dense + Sparse + Re-ranking)",
        "tagline": "Dense vector search + BM25 keyword search fused via Reciprocal Rank Fusion",
        "how": "Runs two retrievers in parallel: ChromaDB (semantic) and BM25 (keyword). Their ranked lists are merged with RRF. A cross-encoder then re-ranks the fused results for maximum precision.",
        "best_for": "General-purpose documents — best accuracy across mixed query types",
        "state_key": "hybrid_pipeline",
    },
    "02 Graph RAG (Knowledge Graphs)": {
        "key": "02 Graph RAG (Knowledge Graphs)",
        "icon": "🕸️", "label": "Graph RAG (Entities + Knowledge Graph)",
        "tagline": "LLM-extracted entity/relationship graph + vector fallback",
        "how": "Uses Gemini to extract (entity → relationship → entity) triples and builds a NetworkX knowledge graph. At query time it walks the graph for matching entities and combines it with dense vector results.",
        "best_for": "Documents rich in named entities and relationships (research papers, reports)",
        "state_key": "graph_pipeline",
    },
    "03 Agentic RAG (LangGraph)": {
        "key": "03 Agentic RAG (LangGraph)",
        "icon": "🤖", "label": "Agentic RAG (Planner → Tools → Reasoner)",
        "tagline": "LangGraph planner routes queries to vector search, web search, or direct answer",
        "how": "A 3-node LangGraph state machine: Planner → Tool Executor → Reasoner. The Planner decides whether to use VECTOR_SEARCH, WEB_SEARCH (DuckDuckGo), or answer directly.",
        "best_for": "Queries that may need web context or multi-step reasoning",
        "state_key": "agentic_pipeline",
    },
    "04 Corrective RAG (CRAG)": {
        "key": "04 Corrective RAG (CRAG)",
        "icon": "✅", "label": "Corrective RAG (Retrieve → Evaluate → Correct)",
        "tagline": "Evaluator grades retrieved docs; rewrites query and falls back to web if needed",
        "how": "A 5-node LangGraph workflow: Retrieve → Evaluate → Route → Generate. The Evaluator grades docs as CORRECT, AMBIGUOUS, or INCORRECT. INCORRECT triggers a web search fallback.",
        "best_for": "When retrieval quality is uncertain or documents may not cover the query",
        "state_key": "crag_pipeline",
    },
    "05 Multimodal RAG (Vision + Text)": {
        "key": "05 Multimodal RAG (Vision + Text)",
        "icon": "🖼️", "label": "Multimodal RAG (Vision + Text)",
        "tagline": "Stores images in metadata; sends text + image to Gemini vision for answers",
        "how": "When an image is uploaded, Gemini generates a text summary for embedding. The base64 image is stored in ChromaDB metadata. At query time retrieved chunks include both text and raw image.",
        "best_for": "Documents with figures, charts, screenshots, or mixed image/text content",
        "state_key": "multimodal_pipeline",
    },
    "06 Multilingual RAG (BGE-M3)": {
        "key": "06 Multilingual RAG (BGE-M3)",
        "icon": "🌍", "label": "Multilingual RAG (Cross-lingual + Re-ranking)",
        "tagline": "Cross-lingual embedding space — query in any language, retrieve from any language",
        "how": "Uses a multilingual sentence-transformer so all languages share the same vector space. A cross-encoder re-ranks results before Gemini answers in the same language as the query.",
        "best_for": "Multilingual documents or when users may query in different languages",
        "state_key": "multilingual_pipeline",
    },
    "07 RAG-Fusion (Query Expansion)": {
        "key": "07 RAG-Fusion (Query Expansion)",
        "icon": "🔮", "label": "RAG-Fusion (Multi-Query + RRF)",
        "tagline": "Expands your query into 4 sub-queries, retrieves for each, fuses with RRF",
        "how": "Gemini generates 4 different phrasings of your query. Each sub-query retrieves its own ranked list. All four are merged with Reciprocal Rank Fusion — documents appearing in multiple lists get boosted.",
        "best_for": "Ambiguous or broad queries where a single phrasing might miss relevant docs",
        "state_key": "rag_fusion_pipeline",
    },
    "08 HyDE RAG (Hypothetical Document)": {
        "key": "08 HyDE RAG (Hypothetical Document)",
        "icon": "💡", "label": "HyDE RAG (Hypothetical Document Embeddings)",
        "tagline": "Generates a hypothetical answer first, uses it as the search query",
        "how": "Gemini generates a hypothetical ideal answer as if it were in your document. That hypothetical is embedded and used to retrieve real chunks closest to it — bridging the vocabulary gap between questions and documents.",
        "best_for": "Short or keyword-style queries worded very differently from the source text",
        "state_key": "hyde_pipeline",
    },
    "09 Structured RAG (CSV/Excel)": {
        "key": "09 Structured RAG (CSV/Excel)",
        "icon": "📊", "label": "Structured RAG (Text-to-Pandas + Vector)",
        "tagline": "Converts natural language queries into pandas operations on tabular data",
        "how": "When CSV or Excel files are ingested, Gemini generates pandas code from the natural language query and executes it directly on the DataFrame. Falls back to vector search for narrative questions. Both results are combined for the final answer.",
        "best_for": "Spreadsheets, CSV datasets, numerical analysis, filtering, aggregation, and statistics",
        "state_key": "structured_pipeline",
    },
}


class GlobalSession:
    def __init__(self):
        self.pipelines: Dict[str, Any] = {
            "hybrid_pipeline":       HybridRAGPipeline(),
            "graph_pipeline":        GraphRAGPipeline(),
            "agentic_pipeline":      AgenticRAGPipeline(),
            "crag_pipeline":         CorrectiveRAGPipeline(),
            "multimodal_pipeline":   MultimodalRAGPipeline(),
            "multilingual_pipeline": MultilingualRAGPipeline(),
            "rag_fusion_pipeline":   RAGFusionPipeline(),
            "hyde_pipeline":         HyDERAGPipeline(),
            "structured_pipeline":   StructuredRAGPipeline(),
        }
        self.history: List[Dict] = []
        self.ingested_archs: Set[str] = set()
        self.doc_library: List[Dict] = []

    def get_pipeline(self, state_key: str) -> Any:
        return self.pipelines.get(state_key)

    def reset(self):
        for pipeline in self.pipelines.values():
            pipeline.reset()
        self.history.clear()
        self.ingested_archs.clear()
        self.doc_library.clear()


# Singleton — one shared session for the API server lifetime
session = GlobalSession()
