import os

import dotenv
import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate

from chatui.utils import get_project_root


def set_model() -> str:
    cola, colb, colc = st.columns(3)

    with colb:
        model_version = st.toggle(
            "Use GPT-4",
            value=False,
            key="model_version",
            help="Use GPT-4 instead of GPT-3",
        )

    if model_version:
        return "gpt4"
    else:
        return "gpt3"


def set_streamlit_config():
    st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
    st.title("📖 StreamlitChatMessageHistory")


def _run_example(
    llm_chain, key, user_input: str = "Tell me a fun fact about the roman empire"
):
    st.caption(user_input)
    st.button("▶️", on_click=lambda: llm_chain.run(user_input), key=key)


def get_and_show_examples(llm_chain):
    """show examples of what the model can do"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("__Tell me a fun fact__")
        example = "Tell me a fun fact about the roman empire"
        _run_example(llm_chain, key="col1", user_input=example)

    with col2:
        st.write("__Tell me a fun fact__")
        example = "Tell me a fun fact about the micky mouse"
        _run_example(llm_chain, key="col2", user_input=example)

    with col3:
        st.write("__Tell me a fun fact__")
        example = "Tell me a fun fact about the the universe"
        _run_example(llm_chain, key="col3", user_input=example)


def display_intro_message():
    """
    Display the introductory message and source code link.
    """
    st.markdown(
        """
    A basic example of using StreamlitChatMessageHistory to help LLMChain remember messages in a conversation.
    The messages are stored in Session State across re-runs automatically. View the
    [source code for this app](https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py).
    """
    )


def initialize_memory():
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")
    return msgs, memory


def fetch_openai_key():
    dotenv.load_dotenv(get_project_root() / ".env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise Exception(
            "OpenAI API-Key wurde nicht gefunden. Bitte überprüfen Sie Ihre .env-Datei."
        )
    return openai_api_key


def get_llm_chain(api_key, memory):
    template = """You are an AI chatbot having a conversation with a human.

    {history}
    Human: {human_input}
    AI: """
    prompt = PromptTemplate(
        input_variables=["history", "human_input"], template=template
    )
    return LLMChain(llm=OpenAI(openai_api_key=api_key, ), prompt=prompt, memory=memory)


def render_chat_messages(msgs):
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)


def handle_user_input(llm_chain, msgs):
    user_input = st.chat_input()
    if user_input:
        # Prüfung auf Magic Commands

        if user_input.startswith("/qa"):
            msgs.add_ai_message("`/qa:` command detected")

        elif user_input.startswith("/img"):
            msgs.add_ai_message("`/img:` command detected")

        elif user_input.startswith("/"):
            msgs.add_ai_message(
                "List of commands: <br>\
                                - `/img:` command detected <br>\
                                - `/qa:` command detected"
            )

        else:
            st.chat_message("human").write(user_input)
            response = llm_chain.run(user_input)
            st.chat_message("ai").write(response)


def display_memory_contents():
    with st.expander("View the message contents in session state"):
        st.markdown(
            """
        Memory initialized with:
        ```python
        msgs = StreamlitChatMessageHistory(key="langchain_messages")
        memory = ConversationBufferMemory(chat_memory=msgs)
        ```

        Contents of `st.session_state.langchain_messages`:
        """
        )
        st.json(st.session_state.langchain_messages)


def main():
    set_streamlit_config()
    model = set_model()
    msgs, memory = initialize_memory()
    api_key = fetch_openai_key()
    llm_chain = get_llm_chain(api_key, memory)
    get_and_show_examples(llm_chain)
    handle_user_input(llm_chain, msgs)
    display_memory_contents()
    render_chat_messages(msgs)


if __name__ == "__main__":
    main()
