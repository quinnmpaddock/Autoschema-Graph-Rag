import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="KG Chat", page_icon="🧠")

# Config
BASE_URL = "http://localhost:10085/v1/"  # same as 0.0.0.0 for local
API_KEY = "EMPTY"
MODEL = "llama"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

st.title("Knowledge Graph Chat")

# Sidebar controls
with st.sidebar:
    st.markdown("### Settings")
    default_system = "You are a helpful assistant that answers questions based on the knowledge graph."
    system_prompt = st.text_area("System prompt", value=default_system, height=100)
    temperature = st.slider("Temperature", 0.0, 0.1, 0.5, 1.5)
    max_tokens = st.number_input("Max tokens", min_value=64, max_value=4096, value=2048, step=64)
    enable_stream = st.checkbox("Stream response", value=False)  # start with non-streaming for compatibility
    show_raw = st.checkbox("Show raw response JSON", value=False)
    reset = st.button("Reset conversation", use_container_width=True)

# Initialize chat history once
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# If user changed system prompt mid-session and wants a clean reset, use the reset button
if reset:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
    st.rerun()

# Render chat
for m in st.session_state.messages:
    if m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(m["content"])
    elif m["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(m["content"])

# Input
user_input = st.chat_input("Ask a question about the knowledge graph...")
if user_input:
    # Ensure backend can extract the question
    question = user_input.strip()
    if not question.lower().startswith("question:"):
        question = f"Question: {question}"

    # Append user message and render it
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get assistant reply
    assistant_text = ""
    error_text = None

    try:
        if enable_stream:
            # Try streaming first
            with st.chat_message("assistant"):
                placeholder = st.empty()
                chunks_collected = []
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    # Some OpenAI-compatible servers may not send delta.content the same way
                    try:
                        delta = chunk.choices[0].delta.content
                    except Exception:
                        delta = None
                    if delta:
                        chunks_collected.append(delta)
                        placeholder.markdown("".join(chunks_collected))
                assistant_text = "".join(chunks_collected)

            # If streaming yielded nothing, fall back to non-streaming
            if not assistant_text:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if show_raw:
                    st.caption("Raw response (non-streaming fallback):")
                    st.json(resp.model_dump())
                assistant_text = resp.choices[0].message.content or ""
        else:
            # Non-streaming path (more compatible)
            resp = client.chat.completions.create(
                model=MODEL,
                messages=st.session_state.messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if show_raw:
                st.caption("Raw response:")
                st.json(resp.model_dump())
            assistant_text = resp.choices[0].message.content or ""
    except Exception as e:
        error_text = f"Error: {e}"

    # Render assistant reply or error
    with st.chat_message("assistant"):
        if error_text:
            st.error(error_text)
        elif assistant_text.strip():
            st.markdown(assistant_text)
        else:
            st.warning("No content returned by the API.")

    # Persist assistant message if we got one (or an error placeholder)
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_text if assistant_text else (error_text or "No content.")
    })
