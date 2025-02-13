from openai import OpenAI
import streamlit as st

st.title("ChatGPT-like clone")

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "local-model"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: Could not connect to LM Studio. Make sure it's running on port 1234. Details: {str(e)}")



