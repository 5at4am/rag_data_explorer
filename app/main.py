import streamlit as st
import pandas as pd

from app.config.settings import settings
from app.utils.logger import setup_logger
from app.utils.validators import validate_file, FileValidationError
from app.loaders.loader import DataLoader
from app.profiling.profiler import DataProfiler
from app.services.summary_service import SummaryService
from app.services.query_service import QueryService
from app.services.hybrid_engine import HybridQueryEngine
from app.rag.document_creator import DocumentCreator
from app.rag.chunking import ChunkingService
from app.rag.embeddings import EmbeddingService
from app.rag.vectorstore import VectorStoreService
from app.chat.manager import ChatManager
from app.memory.memory import ConversationMemory

setup_logger(settings.debug)


def init_session():
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if "df" not in st.session_state:
        st.session_state.df = None
        st.session_state.filename = None
        st.session_state.profile = None
        st.session_state.summary = None
        st.session_state.vectorstore = None


def load_and_process(uploaded_file) -> str | None:
    try:
        temp_path = f"uploads/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        validate_file(temp_path)

        loader = DataLoader()
        df = loader.load(temp_path)

        profiler = DataProfiler(df)
        profile = profiler.profile()

        doc_creator = DocumentCreator()
        docs = doc_creator.create(df, uploaded_file.name)

        chunker = ChunkingService()
        chunks = chunker.split(docs)

        embedder = EmbeddingService()
        vectorstore = VectorStoreService(embedder)
        vectorstore.build(chunks)

        summary_service = SummaryService()
        summary = summary_service.generate(profile)

        st.session_state.df = df
        st.session_state.filename = uploaded_file.name
        st.session_state.profile = profile
        st.session_state.summary = summary
        st.session_state.vectorstore = vectorstore
        st.session_state.chat_manager = ChatManager()

        return summary

    except FileValidationError as e:
        return f"**Validation error:** {e}"
    except Exception as e:
        return f"**Error:** {e}"


def handle_question(question: str) -> str:
    df = st.session_state.df
    profile = st.session_state.profile
    summary = st.session_state.summary
    vectorstore = st.session_state.vectorstore
    chat_manager = st.session_state.chat_manager

    engine = HybridQueryEngine(
        df=df,
        profile=profile,
        summary=summary,
        vectorstore=vectorstore,
        memory=chat_manager.memory,
    )

    response = engine.answer(question)
    return response


st.set_page_config(page_title=settings.app_name, layout="wide")
init_session()

st.title("chat with your dataset")

with st.sidebar:
    st.header("upload")
    uploaded = st.file_uploader(
        "choose a file",
        type=["csv", "xlsx", "json", "parquet"],
    )

    if uploaded:
        if st.session_state.filename != uploaded.name:
            with st.spinner("processing..."):
                result = load_and_process(uploaded)
            if result.startswith("**Error") or result.startswith("**Validation"):
                st.error(result)
            else:
                st.success(f"loaded **{uploaded.name}**")

        if st.session_state.profile:
            profile = st.session_state.profile
            cols = st.columns(3)
            cols[0].metric("rows", profile.shape[0])
            cols[1].metric("columns", profile.shape[1])
            cols[2].metric("file", st.session_state.filename)

            with st.expander("preview"):
                st.dataframe(st.session_state.df.head(10), use_container_width=True)

            with st.expander("data profile"):
                st.json(profile.model_dump())

if uploaded is None:
    st.info("upload a dataset to start")
    st.stop()

if st.session_state.summary:
    with st.chat_message("assistant"):
        st.markdown(st.session_state.summary)

for msg in st.session_state.chat_manager.memory.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("ask about your data..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            answer = handle_question(prompt)
        st.markdown(answer)

    st.session_state.chat_manager.memory.add_message("user", prompt)
    st.session_state.chat_manager.memory.add_message("assistant", answer)
    st.rerun()
