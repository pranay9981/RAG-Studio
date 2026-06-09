import streamlit as st
import tempfile
import os
from core.shared_services import services
from architectures.hybrid_rag import HybridRAGPipeline
from architectures.graph_rag import GraphRAGPipeline
from architectures.agentic_rag import AgenticRAGPipeline
from architectures.corrective_rag import CorrectiveRAGPipeline
from architectures.multimodal_rag import MultimodalRAGPipeline
from architectures.multilingual_rag import MultilingualRAGPipeline
import tempfile
import base64
import time
from langchain_core.messages import HumanMessage

# Initialize pipelines in session state so they persist
if "hybrid_pipeline" not in st.session_state:
    st.session_state.hybrid_pipeline = HybridRAGPipeline()
if "graph_pipeline" not in st.session_state:
    st.session_state.graph_pipeline = GraphRAGPipeline()
if "agentic_pipeline" not in st.session_state:
    st.session_state.agentic_pipeline = AgenticRAGPipeline()
if "crag_pipeline" not in st.session_state:
    st.session_state.crag_pipeline = CorrectiveRAGPipeline()
if "multimodal_pipeline" not in st.session_state:
    st.session_state.multimodal_pipeline = MultimodalRAGPipeline()
if "multilingual_pipeline" not in st.session_state:
    st.session_state.multilingual_pipeline = MultilingualRAGPipeline()


st.set_page_config(page_title="Top 6 RAG Architectures", page_icon="🤖", layout="wide")

st.title("Top RAG Architectures in 2026")
st.markdown("Select an architecture, upload a document, and test it out!")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    selected_arch = st.selectbox(
        "Choose RAG Architecture:",
        [
            "01 Hybrid RAG (Dense + Sparse)",
            "02 Graph RAG (Knowledge Graphs)",
            "03 Agentic RAG (LangGraph)",
            "04 Corrective RAG (CRAG)",
            "05 Multimodal RAG (Vision + Text)",
            "06 Multilingual RAG (BGE-M3)"
        ]
    )
    
    st.divider()
    
    compare_mode = st.checkbox("🔍 Compare All Architectures")
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload a Document (PDF/Images)", type=["pdf", "png", "jpg", "jpeg", "txt", "docx"])
    
    if st.button("Ingest Document") and uploaded_file:
        with st.spinner("Processing document..."):
            if uploaded_file.name.lower().endswith('.pdf'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                docs = services.load_pdf(tmp_file_path)
                os.remove(tmp_file_path)
            else:
                # Handle Image Uploads
                b64_img = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
                # Generate a quick summary for text embeddings using Gemini
                msg = HumanMessage(content=[
                    {"type": "text", "text": "Describe this image in detail. What is it about?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ])
                summary = services.extract_response_text(services.llm.invoke([msg]))
                
                docs = [Document(
                    page_content=f"Image Description: {summary}",
                    metadata={"source": uploaded_file.name, "type": "image", "image_base64": b64_img}
                )]
            
            # Ingest into selected architecture
            if "01 Hybrid RAG" in selected_arch:
                st.session_state.hybrid_pipeline.ingest(docs)
                st.success("Hybrid RAG: Ingested successfully!")
            elif "02 Graph RAG" in selected_arch:
                st.session_state.graph_pipeline.ingest(docs)
                st.success("Graph RAG: Ingested successfully!")
            elif "03 Agentic RAG" in selected_arch:
                st.session_state.agentic_pipeline.ingest(docs)
                st.success("Agentic RAG: Ingested successfully!")
            elif "04 Corrective RAG" in selected_arch:
                st.session_state.crag_pipeline.ingest(docs)
                st.success("Corrective RAG: Ingested successfully!")
            elif "05 Multimodal RAG" in selected_arch:
                st.session_state.multimodal_pipeline.ingest(docs)
                st.success("Multimodal RAG: Ingested successfully!")
            elif "06 Multilingual RAG" in selected_arch:
                st.session_state.multilingual_pipeline.ingest(docs)
                st.success("Multilingual RAG: Ingested successfully!")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
        
    with st.chat_message("assistant", avatar="🤖"):
        if compare_mode:
            st.markdown("### 🔍 RAG Architecture Comparison")
            pipelines = {
                "01 Hybrid RAG": st.session_state.hybrid_pipeline,
                "02 Graph RAG": st.session_state.graph_pipeline,
                "03 Agentic RAG": st.session_state.agentic_pipeline,
                "04 Corrective RAG": st.session_state.crag_pipeline,
                "05 Multimodal RAG": st.session_state.multimodal_pipeline,
                "06 Multilingual RAG": st.session_state.multilingual_pipeline
            }
            for name, pipeline in pipelines.items():
                with st.expander(f"**{name}**", expanded=False):
                    start_time = time.time()
                    try:
                        ans = pipeline.query(prompt)
                    except Exception as e:
                        ans = f"Error: {str(e)}"
                    elapsed = time.time() - start_time
                    st.markdown(f"*{elapsed:.2f} seconds*")
                    st.markdown(ans)
            response = "Comparison complete. Expand the tabs above to see each RAG's output!"
        else:
            with st.spinner(f"Running {selected_arch}..."):
                if "01 Hybrid RAG" in selected_arch:
                    answer = st.session_state.hybrid_pipeline.query(prompt)
                    st.markdown(f"**[Hybrid RAG (Dense + Sparse Fusion)]**\n\n{answer}")
                    response = answer
                elif "02 Graph RAG" in selected_arch:
                    answer = st.session_state.graph_pipeline.query(prompt)
                    st.markdown(f"**[Graph RAG (Entities + Vectors)]**\n\n{answer}")
                    response = answer
                elif "03 Agentic RAG" in selected_arch:
                    answer = st.session_state.agentic_pipeline.query(prompt)
                    st.markdown(f"**[Agentic RAG (LangGraph Planner -> Tools -> Reasoner)]**\n\n{answer}")
                    response = answer
                elif "04 Corrective RAG" in selected_arch:
                    answer = st.session_state.crag_pipeline.query(prompt)
                    st.markdown(f"**[Corrective RAG (Retrieve -> Evaluate -> Rewrite/Fallback)]**\n\n{answer}")
                    response = answer
                elif "05 Multimodal RAG" in selected_arch:
                    answer = st.session_state.multimodal_pipeline.query(prompt)
                    st.markdown(f"**[Multimodal RAG (Vision + Text)]**\n\n{answer}")
                    response = answer
                elif "06 Multilingual RAG" in selected_arch:
                    answer = st.session_state.multilingual_pipeline.query(prompt)
                    st.markdown(f"**[Multilingual RAG (BGE-M3 Cross-lingual)]**\n\n{answer}")
                    response = answer
                else:
                    response = f"I am ready to run the **{selected_arch}** pipeline for your query: '{prompt}', but it is not implemented yet!"
                    st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
