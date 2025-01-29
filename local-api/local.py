import time
import threading
import numpy
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

console = Console()

template = """
Provide responses of less than 100 words.

The conversation transcript as follows:
{history}

User's follow-up: {input}

Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

# Ensure your Ollama server is running with the model
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(base_url="http://localhost:11434", model="deepseek-r1:1.5b"),
)


def get_response(text: str) -> str:
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response


if __name__ == "__main__":
    console.print("[cyan]Let's talk! Press Ctrl+C to exit.")

    try:
        while True:
            user_input = console.input("[yellow]You: ")  # Fix input handling
            with console.status("Waiting assistant...", spinner="earth"):
                response = get_response(user_input)  # Pass user input
            console.print(f"[cyan]Assistant: {response}")

    except KeyboardInterrupt:  # Fix typo
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
