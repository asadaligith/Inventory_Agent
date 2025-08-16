from agents import Agent, Runner,  function_tool , OpenAIChatCompletionsModel , set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio

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

inventory = {}

@function_tool
async def add_item(name: str, quantity: int) -> str:
    """Add item with given quantity to inventory."""
    inventory[name] = inventory.get(name, 0) + quantity
    return f"added {quantity} x {name}"

@function_tool
async def delete_item(name: str) -> str:
    """Delete item from inventory."""
    if name in inventory:
        del inventory[name]
        return f"deleted item {name}"
    return f"{name} not found!"

@function_tool
async def list_inventory() -> str:
    """List all inventory items."""
    if not inventory:
        return "inventory is empty"
    return "\n".join(f"{item} : {qty}" for item, qty in inventory.items())

agent = Agent(
    name="Inventory Manager",
    instructions="Use tools to add, delete, or list inventory items.",
    tools=[add_item, delete_item, list_inventory],
    model=model,
)

async def main():
    
   result = await Runner.run(
       agent, 
       input="Add five laptops to the inventory. and tell me how many laptops are availbale in the inventory, given the list items names.",
   )
   print("Result:", result.final_output)


def start():
    asyncio.run(main())