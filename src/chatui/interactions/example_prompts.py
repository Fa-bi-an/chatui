import random

import streamlit as st


def generate_random_prompt(n: int = 3):
    """
    Generates a random prompt from a predefined list of sample prompts.

    This function selects a prompt at random from a list of various sample prompts.
    Each call to this function returns a different prompt.

    Returns:
        str: A randomly selected prompt.
    """
    sample_prompts = [
        "__Write__ a story about a lost dog finding its way home",
        "__Describe__ a day in the life of a space explorer",
        "__Imagine__ a world where cars can fly",
        "__Create__ a dialogue between two historical figures",
        "__Explain__ how photosynthesis works in a fun and engaging way",
        "__Craft__ a mystery story set in an ancient castle",
        "__Describe__ the perfect vacation spot",
        "__Write__ a poem about the changing seasons",
        "__Tell__ a story from the perspective of a cat",
        "__Explain__ the importance of recycling in a futuristic world",
        "__Tell__ me a fun fact about the roman empire",
        "__Tell__ me a fun fact about the micky mouse",
        "__Tell__ me a fun fact about the the universe",
        "__Translate__ this python function to javascript '''print('hello world'))''' ",
    ]

    if n > len(sample_prompts):
        raise ValueError(
            "Requested number of prompts exceeds the available sample prompts"
        )

    return random.sample(sample_prompts, n)


def _run_example(
    llm_chain, key, user_input: str = "Tell me a fun fact about the roman empire"
):
    st.write(user_input)
    st.button("▶️", on_click=lambda: llm_chain.run(user_input), key=key)


if __name__ == "__main__":
    example = generate_random_prompt()
    print(example)
