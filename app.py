import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq

# Ensure key is only from env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set in environment.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

AITOPTOOLS_BASE = "https://aitoptools.com"

def fetch_tool_page(url: str) -> str:
    """Fetch and extract main text from a single tool or category page."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching {url}: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    # Simple extraction: main content text
    # (You can refine selectors to target tool description blocks)
    texts = [t.get_text(" ", strip=True) for t in soup.find_all(["h1", "h2", "p", "li"])]
    return "\n".join(texts[:80])  # limit tokens

def ask_groq(prompt: str, context: str) -> str:
    """Call Groq LLM with given context and question."""
    system_msg = (
        "You are an AI tools recommendation assistant.\n"
        "You answer questions using ONLY the provided context about tools.\n"
        "Mention tool names, key features, price type (free, freemium, paid), and the source URL.\n"
        "If you are not sure, say so instead of inventing tools."
    )

    completion = client.chat.completions.create(
        model="llama3-70b-8192",  # or other Groq model
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": (
                    f"Context from tool pages:\n\n{context}\n\n"
                    f"User question: {prompt}\n\n"
                    "Answer in a concise, developer-friendly way."
                ),
            },
        ],
        temperature=0.3,
    )

    return completion.choices[0].message.content

def main():
    st.set_page_config(page_title="AI Tools Assistant", page_icon="🤖", layout="wide")

    st.title("🤖 AI Tools Assistant (Groq-powered)")
    st.write(
        "Ask about AI tools (e.g., best tools for a use case, features, pricing). "
        "This assistant uses open data and live tool pages as context."
    )

    user_question = st.text_input("What do you want AI to do?", "")
    default_urls = [
        "https://aitoptools.com/tool/thelibrarian/",
        "https://aitoptools.com/tool/flowith/",
    ]
    url_list = st.text_area(
        "Tool or category URLs to use as context "
        "(one per line, ideally specific tool pages from aitoptools.com or others):",
        value="\n".join(default_urls),
        height=120,
    )

    if st.button("Get answer") and user_question.strip():
        urls = [u.strip() for u in url_list.splitlines() if u.strip()]
        with st.spinner("Fetching pages and asking Groq..."):
            contexts = []
            for u in urls:
                txt = fetch_tool_page(u)
                contexts.append(f"URL: {u}\n\n{txt}\n\n---\n")
            full_context = "\n".join(contexts)
            answer = ask_groq(user_question, full_context)
        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources")
        for u in urls:
            st.markdown(f"- {u}")

if __name__ == "__main__":
    main()
