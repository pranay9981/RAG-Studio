import networkx as nx
import json
import uuid
from typing import List, Dict
from langchain_core.documents import Document
from core.shared_services import services

class GraphRAGPipeline:
    def __init__(self):
        self.graph = nx.Graph()
        self.collection_name = "graph_rag_collection"
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)

    def _extract_entities_and_relationships(self, text: str) -> List[Dict]:
        prompt = f"""Extract key entities and relationships from the following text.
Output ONLY a valid JSON array of objects. Each object must have exactly three keys: "source", "target", "relationship".
Do not include markdown, backticks, or any explanation — just the raw JSON array.

Text:
{text}
"""
        try:
            response = services.llm.invoke(prompt)
            content = services.extract_response_text(response).strip()

            # Strip markdown code fences if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Find the JSON array bounds robustly
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                content = content[start:end]

            triples = json.loads(content)
            return triples if isinstance(triples, list) else []
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []

    def reset(self):
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self.graph.clear()

    def ingest(self, documents: List[Document]):
        if not documents:
            return

        # Clear stale data from any previous ingest
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self.graph.clear()

        texts = [doc.page_content for doc in documents]
        ids = [f"graph_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]

        embeddings = services.embeddings.embed_documents(texts)
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

        # Build knowledge graph from entity/relationship triples
        for doc in documents:
            triples = self._extract_entities_and_relationships(doc.page_content)
            for triple in triples:
                if all(k in triple for k in ("source", "target", "relationship")):
                    self.graph.add_edge(
                        triple["source"],
                        triple["target"],
                        relationship=triple["relationship"],
                    )

    def _extract_query_entities(self, query: str) -> List[str]:
        prompt = f"""Extract the main entities from this query. Output ONLY a comma-separated list of entity names, nothing else.
Query: {query}"""
        response = services.llm.invoke(prompt)
        text = services.extract_response_text(response)
        return [e.strip() for e in text.split(",") if e.strip()]

    def render_graph_html(self, max_nodes: int = 80) -> str:
        """Returns a PyVis interactive graph as an HTML string. Empty string if no graph."""
        from pyvis.network import Network
        import tempfile, os

        if not self.graph.nodes():
            return ""

        net = Network(
            height="520px", width="100%",
            bgcolor="#0e1117", font_color="white",
            directed=True,
            notebook=False,
        )
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "springLength": 120
            },
            "solver": "forceAtlas2Based",
            "stabilization": { "iterations": 100 }
          },
          "edges": {
            "color": { "color": "#4a9eff" },
            "smooth": { "type": "curvedCW", "roundness": 0.15 }
          }
        }
        """)

        # Pick top-N nodes by degree so huge graphs don't crash the browser
        sorted_nodes = sorted(self.graph.nodes(), key=lambda n: self.graph.degree(n), reverse=True)
        nodes_to_show = set(sorted_nodes[:max_nodes])

        for node in nodes_to_show:
            degree = self.graph.degree(node)
            size = max(12, min(40, 12 + degree * 4))
            color = "#4a9eff" if degree > 2 else "#a0c4ff"
            net.add_node(str(node), label=str(node), title=f"{node}\n{degree} connection(s)", size=size, color=color)

        for u, v, data in self.graph.edges(data=True):
            if u in nodes_to_show and v in nodes_to_show:
                rel = data.get("relationship", "")
                net.add_edge(str(u), str(v), label=rel, title=rel, arrows="to")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
            net.write_html(f.name, notebook=False)
            tmp = f.name
        with open(tmp, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tmp)
        return html

    def query(self, query: str, on_step=None) -> str:
        def step(msg):
            if on_step:
                on_step(("step", msg))

        if not self.collection.count():
            return "Please ingest a document first!"

        # 1. Retrieve relevant subgraph via entity matching
        step("Extracting query entities…")
        query_entities = self._extract_query_entities(query)

        step("Searching knowledge graph for matching nodes…")
        subgraph_lines = []
        for entity in query_entities:
            for node in self.graph.nodes():
                if entity.lower() in str(node).lower():
                    for neighbor in self.graph.neighbors(node):
                        rel = self.graph[node][neighbor].get("relationship", "associated with")
                        subgraph_lines.append(f"{node} --[{rel}]--> {neighbor}")
        graph_text = "\n".join(set(subgraph_lines))

        # 2. Dense retrieval as semantic fallback
        step("Dense retrieval from ChromaDB (fallback)…")
        query_embedding = services.embeddings.embed_query(query)
        n = min(3, self.collection.count())
        dense_response = self.collection.query(query_embeddings=[query_embedding], n_results=n)
        vector_text = "\n\n".join(dense_response["documents"][0]) if dense_response["documents"][0] else ""

        # 3. Synthesize answer from both sources
        prompt = f"""You are GraphRAG. Answer the user's query using the Knowledge Graph relationships and semantic text below.

Knowledge Graph Subgraph:
{graph_text if graph_text else "No specific relationships found."}

Semantic Text:
{vector_text if vector_text else "No semantic context available."}

Query: {query}

Answer:"""

        step("Generating answer with Gemini…")
        return services.stream_llm(prompt, on_token=lambda t: on_step and on_step(("token", t)))
