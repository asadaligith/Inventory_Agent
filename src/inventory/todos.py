from agents import Agent , Runner, function_tool , OpenAIChatCompletionsModel , set_tracing_disabled , enable_verbose_stdout_logging
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio

enable_verbose_stdout_logging()

# Load environment variables from a .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

set_tracing_disabled(disabled=True)

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

@function_tool
async def list_todos() -> str:
    """List all todo items."""
    with open("src/inventory/todos.txt", "r") as file:
        todos = file.readlines()

    if not todos:
        return "No todos found."
    return "".join(todos).strip()

@function_tool
async def add_todo(item: str) -> str:
    """Add a todo item."""
    with open("src/inventory/todos.txt", "a") as file:
        file.write(item + "\n")
    return f"Added todo: {item}"

@function_tool
async def delete_todo(item: str) -> str:
    """Delete a todo item."""
    with open("src/inventory/todos.txt", "r") as file:
        todos = file.readlines()

    if item + "\n" in todos:
        todos.remove(item + "\n")
        with open("src/inventory/todos.txt", "w") as file:
            file.writelines(todos)
        return f"Deleted todo: {item}"
    return f"Todo '{item}' not found."


async def main():
    todos_agent = Agent(
        name="Todo Manager",
        instructions="You are a todo manager. Use tools to add, delete, or list todo items.",
        tools=[add_todo, delete_todo, list_todos],
        model=model,
    )

    query = input("Enter a Query: ")

    result = await Runner.run(starting_agent=todos_agent, input=query)
    print(result.final_output)

def start():
    asyncio.run(main())
