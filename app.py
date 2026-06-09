import streamlit as st

# MUST be the first Streamlit command in the script
st.set_page_config(page_title="Top 6 RAG Architectures", page_icon="🤖", layout="wide")

import tempfile
import os
import base64
import time
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from core.shared_services import services
from architectures.hybrid_rag import HybridRAGPipeline
from architectures.graph_rag import GraphRAGPipeline
from architectures.agentic_rag import AgenticRAGPipeline
from architectures.corrective_rag import CorrectiveRAGPipeline
from architectures.multimodal_rag import MultimodalRAGPipeline
from architectures.multilingual_rag import MultilingualRAGPipeline

# Initialize pipelines in session state so they persist across reruns
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
            try:
                if uploaded_file.name.lower().endswith('.pdf'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    docs = services.load_pdf(tmp_file_path)
                    os.remove(tmp_file_path)
                elif uploaded_file.name.lower().endswith('.txt'):
                    text = uploaded_file.getvalue().decode("utf-8", errors="replace")
                    chunks = services.text_splitter.split_text(text)
                    docs = [Document(page_content=chunk, metadata={"source": uploaded_file.name, "type": "txt"}) for chunk in chunks]
                elif uploaded_file.name.lower().endswith('.docx'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    from docx import Document as DocxDocument
                    docx = DocxDocument(tmp_file_path)
                    text = "\n".join([para.text for para in docx.paragraphs if para.text.strip()])
                    os.remove(tmp_file_path)
                    chunks = services.text_splitter.split_text(text)
                    docs = [Document(page_content=chunk, metadata={"source": uploaded_file.name, "type": "docx"}) for chunk in chunks]
                else:
                    # Handle image uploads (PNG, JPG, JPEG)
                    b64_img = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
                    msg = HumanMessage(content=[
                        {"type": "text", "text": "Describe this image in detail. What is it about?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ])
                    summary = services.extract_response_text(services.llm.invoke([msg]))
                    docs = [Document(
                        page_content=f"Image Description: {summary}",
                        metadata={"source": uploaded_file.name, "type": "image", "image_base64": b64_img}
                    )]

                if not docs:
                    st.error("Could not extract any content from the document.")
                else:
                    # Ingest into selected or all architectures
                    arch_map = {
                        "01 Hybrid RAG": ("hybrid_pipeline", "Hybrid RAG"),
                        "02 Graph RAG": ("graph_pipeline", "Graph RAG"),
                        "03 Agentic RAG": ("agentic_pipeline", "Agentic RAG"),
                        "04 Corrective RAG": ("crag_pipeline", "Corrective RAG"),
                        "05 Multimodal RAG": ("multimodal_pipeline", "Multimodal RAG"),
                        "06 Multilingual RAG": ("multilingual_pipeline", "Multilingual RAG"),
                    }

                    if compare_mode:
                        # Ingest into ALL architectures when compare mode is on
                        for key, (state_key, name) in arch_map.items():
                            st.session_state[state_key].ingest(docs)
                        st.success(f"Ingested {len(docs)} chunks into all 6 architectures!")
                    else:
                        for key, (state_key, name) in arch_map.items():
                            if key in selected_arch:
                                st.session_state[state_key].ingest(docs)
                                st.success(f"{name}: Ingested {len(docs)} chunks successfully!")
                                break

            except Exception as e:
                st.error(f"Ingestion failed: {str(e)}")

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
                "06 Multilingual RAG": st.session_state.multilingual_pipeline,
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
                try:
                    arch_to_pipeline = {
                        "01 Hybrid RAG": ("hybrid_pipeline", "Hybrid RAG (Dense + Sparse Fusion)"),
                        "02 Graph RAG": ("graph_pipeline", "Graph RAG (Entities + Vectors)"),
                        "03 Agentic RAG": ("agentic_pipeline", "Agentic RAG (LangGraph Planner -> Tools -> Reasoner)"),
                        "04 Corrective RAG": ("crag_pipeline", "Corrective RAG (Retrieve -> Evaluate -> Rewrite/Fallback)"),
                        "05 Multimodal RAG": ("multimodal_pipeline", "Multimodal RAG (Vision + Text)"),
                        "06 Multilingual RAG": ("multilingual_pipeline", "Multilingual RAG (BGE-M3 Cross-lingual)"),
                    }
                    response = "Unknown architecture selected."
                    for key, (state_key, label) in arch_to_pipeline.items():
                        if key in selected_arch:
                            answer = st.session_state[state_key].query(prompt)
                            st.markdown(f"**[{label}]**\n\n{answer}")
                            response = answer
                            break
                except Exception as e:
                    response = f"Error running pipeline: {str(e)}"
                    st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
