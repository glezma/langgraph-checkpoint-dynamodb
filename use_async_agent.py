from typing import Literal
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from a_dynamodb_checkpointer import AsyncDynamoDBSaver

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
table_name = "checkpoints_table"
region_name = "us-east-1"
writes_table_name = "checkpoint_writes_table"



thread_id = "thread-2"
config = {"configurable": {"thread_id": thread_id}}

import asyncio

async def main():
    async with AsyncDynamoDBSaver(
        table_name=table_name,
        writes_table_name=writes_table_name,
        region_name= "us-east-1",
    ) as checkpointer:
        config = {
            "configurable": {
                "thread_id": "3",
                "checkpoint_ns": "2",
            }
        }
        @tool
        def get_weather(city: Literal["nyc", "sf"]):
            """Use this to get weather information."""
            if city == "nyc":
                return "It might be cloudy in nyc"
            elif city == "sf":
                return "It's always sunny in sf"
            else:
                raise AssertionError("Unknown city")


        tools = [get_weather]
        model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        # %

        graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
        thread_id = "session-as2"
        config = {"configurable": {"thread_id": thread_id}}
        
        input_message = HumanMessage(content="Hi I am Gonzalo")
        res = await graph.ainvoke({"messages": [input_message]}, config)

        input_message = HumanMessage(content="Do you know my name?")
        res = await graph.ainvoke({"messages": [input_message]}, config)

        input_message = HumanMessage(content="What is the wheater in ny")
        res = await graph.ainvoke({"messages": [input_message]}, config)
        
        for m in res['messages']:
            m.pretty_print()

# Run the main function in an asyncio event loop
asyncio.run(main())