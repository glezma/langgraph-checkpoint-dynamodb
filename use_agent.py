# %%
from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dynamodb_checkpointer import DynamoDBSaver

from langchain_core.messages import  HumanMessage
table_name = "checkpoints_table"
region_name = "us-east-1"
writes_table_name = "checkpoint_writes_table"

checkpointer = DynamoDBSaver(
    table_name=table_name,
    writes_table_name=writes_table_name,
    region_name=region_name
)

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
# # %

graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
thread_id = "thread-1"
config = {"configurable": {"thread_id": thread_id}}
input_message = HumanMessage(content="Hi I am Gonzalo")
res = graph.invoke({"messages": [input_message]}, config)
# for m in res['messages']:
#     m.pretty_print()

input_message = HumanMessage(content="Do you know my name?")
res = graph.invoke({"messages": [input_message]}, config)
# for m in res['messages']:
#     m.pretty_print()
    
input_message = HumanMessage(content="Can you tell me the weather in ny?")
res = graph.invoke({"messages": [input_message]}, config)
for m in res['messages']:
    m.pretty_print()
    
    
latest_checkpoint = checkpointer.get(config)
latest_checkpoint_tuple = checkpointer.get_tuple(config)
checkpoint_tuples = list(checkpointer.list(config))
