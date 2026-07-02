from operator import itemgetter
from typing import Any

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from app.config.settings import settings
from app.prompts.templates import qa_template
from app.rag.vectorstore import VectorStoreService
from app.memory.memory import ConversationMemory


class QueryService:

    def __init__(
        self,
        dataset_summary: str,
        vectorstore: VectorStoreService,
        memory: ConversationMemory,
    ):
        self._summary = dataset_summary
        self._vectorstore = vectorstore
        self._memory = memory
        self._chain = None

    def _get_llm(self):
        if settings.llm_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=settings.google_api_key,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        elif settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        raise ValueError(f"Unsupported LLM: {settings.llm_provider}")

    def _build_chain(self):
        if self._chain is not None:
            return self._chain

        llm = self._get_llm()
        retriever = self._vectorstore.get_retriever()

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        def get_history(_):
            return self._memory.get_langchain_history()

        def pass_summary(_):
            return self._summary

        self._chain = (
            RunnableParallel(
                {
                    "context": itemgetter("question") | retriever | format_docs,
                    "question": itemgetter("question"),
                    "chat_history": itemgetter("question") | RunnablePassthrough() | get_history,
                    "dataset_summary": itemgetter("question") | RunnablePassthrough() | pass_summary,
                }
            )
            | qa_template
            | llm
            | StrOutputParser()
        )

        return self._chain

    def answer(self, question: str) -> str:
        logger.info("Running RAG query")
        chain = self._build_chain()
        response = chain.invoke({"question": question})
        text = response if isinstance(response, str) else response.content
        return text.strip()
