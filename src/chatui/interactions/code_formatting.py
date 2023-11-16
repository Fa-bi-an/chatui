import re

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name


def format_source_code(text, language="python"):
    """
    Detects and formats source code in the provided text.

    Args:
        text (str): The text that may contain source code.
        language (str, optional): The programming language of the source code. Defaults to 'python'.

    Returns:
        str: Formatted source code in HTML, or the original text if no source code is detected.
    """
    # Simple source code detection (can be refined)
    if re.search(r"(def |class |import |from |#|# |//|/\*|\*/)", text):
        lexer = get_lexer_by_name(language)
        formatter = HtmlFormatter(linenos=True, full=True)
        return highlight(text, lexer, formatter)
    else:
        return text


if __name__ == "__main__":
    code = """
    def hello_world():
        print("Hello, world!")
    """

    formatted_code = format_source_code(code)
    print(formatted_code)
