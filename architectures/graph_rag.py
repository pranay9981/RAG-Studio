import networkx as nx
import json
import uuid
from typing import List, Dict, Any
from langchain_core.documents import Document
from core.shared_services import services

class GraphRAGPipeline:
    def __init__(self):
        self.graph = nx.Graph()
        self.collection_name = "graph_rag_collection"
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        
    def _extract_entities_and_relationships(self, text: str) -> List[Dict]:
        """Uses Gemini to extract entities and relationships from text."""
        prompt = f"""
        Extract key entities and relationships from the following text.
        Format the output EXACTLY as a JSON array of objects with keys: "source", "target", "relationship".
        Do not include any markdown formatting, just the raw JSON.
        
        Text:
        {text}
        """
        
        try:
            # We enforce JSON output in the prompt
            response = services.llm.invoke(prompt)
            content = services.extract_response_text(response).strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            triples = json.loads(content)
            return triples
        except Exception as e:
            print(f"Extraction error: {e}")
            return []

    def ingest(self, documents: List[Document]):
        """Builds the Knowledge Graph and Vector Index."""
        if not documents:
            return
            
        texts = [doc.page_content for doc in documents]
        ids = [f"graph_chunk_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]
        
        # 1. Store in ChromaDB as fallback
        embeddings = services.embeddings.embed_documents(texts)
        
        self.collection.add(documents=texts, embeddings=embeddings, ids=ids)
        
        # 2. Build Knowledge Graph
        self.graph.clear()
        for doc in documents:
            triples = self._extract_entities_and_relationships(doc.page_content)
            for triple in triples:
                if "source" in triple and "target" in triple and "relationship" in triple:
                    self.graph.add_edge(
                        triple["source"], 
                        triple["target"], 
                        relationship=triple["relationship"]
                    )

    def _extract_query_entities(self, query: str) -> List[str]:
        prompt = f"""Extract main entities from this query. Output ONLY a comma-separated list of entity names.
        Query: {query}"""
        response = services.llm.invoke(prompt)
        text = services.extract_response_text(response)
        return [e.strip() for e in text.split(",")]

    def query(self, query: str) -> str:
        """Retrieves from Graph + Vectors and generates an answer."""
        # 1. Retrieve relevant Subgraph
        query_entities = self._extract_query_entities(query)
        subgraph_context = []
        
        for entity in query_entities:
            # Simple exact match or fallback iteration
            for node in self.graph.nodes():
                if entity.lower() in str(node).lower():
                    # Get immediate neighbors
                    neighbors = self.graph.neighbors(node)
                    for n in neighbors:
                        rel = self.graph[node][n].get("relationship", "associated with")
                        subgraph_context.append(f"{node} --[{rel}]--> {n}")
        
        graph_text = "\n".join(set(subgraph_context))
        
        # 2. Retrieve semantic chunks (Dense retrieval)
        query_embedding = services.embeddings.embed_query(query)
        dense_response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        vector_text = "\n\n".join(dense_response['documents'][0]) if dense_response['documents'] else ""
        
        # 3. Synthesize Final Answer
        prompt = f"""You are GraphRAG. Answer the user's query using the following Knowledge Graph relationships and semantic text.
        
        Knowledge Graph Subgraph:
        {graph_text if graph_text else "No specific relationships found."}
        
        Semantic Text:
        {vector_text}
        
        Query: {query}
        
        Answer:"""
        
        response = services.llm.invoke(prompt)
        return services.extract_response_text(response)
