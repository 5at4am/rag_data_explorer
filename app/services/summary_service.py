from loguru import logger

from app.config.settings import settings
from app.profiling.profiler import DatasetProfile
from app.prompts.templates import summary_template


class SummaryService:

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        if settings.llm_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=settings.google_api_key,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        elif settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

        return self._llm

    def generate(self, profile: DatasetProfile) -> str:
        logger.info("Generating AI dataset summary")
        llm = self._get_llm()

        col_details = []
        for c in profile.columns:
            details = f"  - {c.name} ({c.inferred_type}, {c.dtype}): {c.null_pct}% null, {c.unique_count} unique"
            if c.mean is not None:
                details += f", mean={c.mean}, min={c.min}, max={c.max}"
            if c.top_value:
                details += f", top='{c.top_value}' ({c.top_pct}%)"
            col_details.append(details)

        prompt = summary_template.format(
            filename=profile.filename,
            rows=profile.shape[0],
            columns=profile.shape[1],
            column_details="\n".join(col_details),
        )

        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        logger.debug(f"Summary generated ({len(text)} chars)")
        return text.strip()
