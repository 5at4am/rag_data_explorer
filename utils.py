import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

_client = None
MODEL = "gemini-2.0-flash-lite"


def get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


def build_context(df):
    cols = df.columns.tolist()
    dtypes = {c: str(df[c].dtype) for c in cols}
    sample = df.head(5).to_string(index=False)
    nulls = df.isnull().sum().to_dict()
    stats = df.describe(include="all").to_string()
    return f"""Dataset:
- {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(cols)}
- Dtypes: {dtypes}
- Nulls: {nulls}
- Sample:
{sample}

Stats:
{stats}"""


def generate_summary(df):
    context = build_context(df)
    prompt = f"""You are a data analyst. Explain this dataset in simple plain language — what it contains, key columns, any patterns you notice.

{context}

Write 3-4 short lines. No markdown tables. No technical jargon."""
    client = get_client()
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    return resp.text.strip()


def answer_question(df, question, history):
    context = build_context(df)
    chat_history = "\n".join(
        [f"User: {m['content']}\nAssistant: {m['response']}" for m in history[-6:]]
    )
    prompt = f"""You are a data assistant. You have this dataset:

{context}

Previous conversation:
{chat_history}

Current question: {question}

Answer using the data. Be precise. Use numbers. If user asks for a table, format as markdown table. If they ask to count, filter, or compare — do it based on the data above."""
    client = get_client()
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    return resp.text.strip()
