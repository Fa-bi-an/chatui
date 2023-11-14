import logging
import os
import time

import dotenv
import streamlit as st
from langchain.llms import OpenAI
from openai.error import InvalidRequestError

from chatui.utils import get_project_root

logger = logging.getLogger(__name__)


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


def get_and_show_examples():
    """show examples of what the model can do"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("__Tell me a fun fact__")
        st.caption("Tell me a fun fact about the roman empire")
        st.button(
            "▶️",
            on_click=lambda: st.session_state.messages.append(
                "Tell me a fun fact about the roman empire"
            ),
            key="run_example_one",
        )

    with col2:
        st.write("__Tell me a fun fact__")
        st.caption("Tell me a fun fact about the roman empire")
        st.button(
            "▶️",
            on_click=lambda: st.session_state.messages.append(
                "Tell me a fun fact about the roman empire"
            ),
            key="run_example_two",
        )

    with col3:
        st.write("__Tell me a fun fact__")
        st.caption("Tell me a fun fact about the roman empire")
        st.button(
            "▶️",
            on_click=lambda: st.session_state.messages.append(
                "Tell me a fun fact about the roman empire"
            ),
            key="run_example_three",
        )


def generate_response(input_text):
    dotenv.load_dotenv(get_project_root() / ".env")
    # Stellen Sie sicher, dass der API-Schlüssel geladen wurde
    if os.getenv("OPENAI_API_KEY") is None:
        raise Exception(
            "OpenAI API-Key wurde nicht gefunden. Bitte überprüfen Sie Ihre .env-Datei."
        )
    llm = OpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
    st.info(llm(input_text))
    return llm(input_text)


def Chat_UI():
    with st.chat_message(
        "assistant",
        avatar="https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/1024px-ChatGPT_logo.svg.png",
    ):
        message_placeholder = st.empty()
        full_response = ""
        st.write(
            "Hi, I'm panda. I'm here to help you with your questions about the GPT-3 API. Ask me anything!"
        )

        for message in st.session_state.messages:
            message_placeholder.write(message)
            time.sleep(0.5)

    get_and_show_examples()

    # Chat Input
    prompt = st.chat_input("What are you thinking about?", key="prompt")

    if str(prompt).startswith("/qa"):
        st.write("`/qa:` command detected")
        pass

        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        # TODO: Add PDF parser from Langchanin and refactor to function

    elif str(prompt).startswith("/img"):
        st.write("`/img:` command detected")
        pass

    elif str(prompt).startswith("/"):
        st.write("List of commands:")
        st.write("`/qa`: Ask a question")
        st.write("`/img`: Generate an image")
        pass

    else:
        st.write(f"User has sent the following prompt: {prompt}")
        response = generate_response(str(prompt))
        st.session_state.messages.append(prompt)
        st.session_state.messages.append(response)


def main():
    """
    Entrypoint for the streamlit application
    """

    st.set_page_config(
        page_title="panda",
        page_icon="🐼",
        menu_items={"About": "# This is a header. This is an *extremely* cool app!"},
    )

    st.title("panda")
    st.markdown(
        "Welcome to your new streamlit application. Get started by adding your code in the `app.py` file."
    )

    model_version = set_model()

    st.session_state.messages = st.session_state.get("messages", [])
    Chat_UI()


if __name__ == "__main__":
    main()
