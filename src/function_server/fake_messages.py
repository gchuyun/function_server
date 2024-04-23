from __future__ import annotations
import random
from io import StringIO
import time
from typing import Dict, List, Union, Iterable, Optional
from typing_extensions import Literal, TypedDict
from openai._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from openai.types.chat import ChatCompletionToolParam, ChatCompletionToolMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel, ValidationError, TypeAdapter
from pydantic_core import from_json, to_json
from inspect import cleandoc
from .function_calling import ToolCallResult
from .settings import FAKE_ALL_MODEL, NO_FAKE_MODELS
from loguru import logger


class ChatCompletionsRequest(CompletionCreateParamsBase):
    model: Union[str]   # 不限制model名

def fake_chat_request_if_need(chat_request: ChatCompletionsRequest, server_tools: List[ChatCompletionToolParam], replacement_tool_call_results: List[ToolCallResult]):
    if chat_request.get("tools"):             
        chat_request["tools"] = server_tools.extend(chat_request["tools"])
    chat_request["tools"] = server_tools

    messages = list(chat_request["messages"])
    if replacement_tool_call_results:
        last_toolcalls_message_index = [i for i in range(len(messages)) if messages[i].get("tool_calls")][-1]
        messages = messages[:last_toolcalls_message_index+1]

        tool_calls_msg = messages[-1]
        tool_calls_msg["tool_calls"] = [tcr.tool_call.model_dump() for tcr in replacement_tool_call_results]
        for tcr in replacement_tool_call_results:
            messages.append(ChatCompletionToolMessageParam(role = 'tool', tool_call_id=tcr.id, content=tcr.result))
    chat_request["messages"] = messages

    if FAKE_ALL_MODEL or (chat_request["model"] not in NO_FAKE_MODELS):
        chat_request["model"] = str(chat_request["model"]).split("|")[-1]
        add_function_promt_message_replace_tool_call(chat_request)
        rewrite_assistant_tool_calls_to_content(chat_request)
        merge_tool_messages_and_replcae_by_user(chat_request)

def rewrite_assistant_tool_calls_to_content(chat_request: ChatCompletionsRequest):
    for message in chat_request["messages"]:
        if message.get("tool_calls"):
            message["content"] = to_json(message["tool_calls"])
            message["tool_calls"] = None

def merge_tool_messages_and_replcae_by_user(chat_request: ChatCompletionsRequest):
    tool_messages = [message for message in chat_request["messages"] if message["role"] == "tool"]
    if not tool_messages:
        return
    results_content_builder = StringIO()
    results_content_builder.write("# Tool Call Results:\n")
    for message in tool_messages:
            tool_call_id = message["tool_call_id"]
            content = message["content"]
            results_content_builder.write(f"- id: `{tool_call_id}`\n```\n{content}\n```\n")

    messages = [message for message in chat_request["messages"] if message["role"] != "tool"]
    messages.append(ChatCompletionSystemMessageParam(role = 'user', content = results_content_builder.getvalue()))
    chat_request["messages"] = messages

def add_function_promt_message_replace_tool_call(chat_request: ChatCompletionsRequest):
    if not chat_request.get("tools"):
        return
    tools_json = to_json(chat_request["tools"], indent=2).decode()
    function_calling_prompt = get_function_calling_propmt(tools_json)

    is_added = False
    messages = []
    
    for msg in chat_request["messages"]:
        if msg["role"] != 'system' and not is_added:
            messages.append(ChatCompletionSystemMessageParam(role = 'system', content = function_calling_prompt))
            is_added = True
        messages.append(msg)

    chat_request["messages"] = messages
    chat_request["tools"] = None

def add_tool_calls_result_messages(chat_request: ChatCompletionsRequest, tool_calls_results: List[ToolCallResult]):
    if not tool_calls_results:
        return

    messages = [message for message in chat_request["messages"]]
    if chat_request.get("tools"):
        messages.append(ChatCompletionAssistantMessageParam(role = 'assistant', tool_calls=[tcr.tool_call.model_dump() for tcr in tool_calls_results]))
        for tcr in tool_calls_results:
            messages.append(ChatCompletionToolMessageParam(role = 'tool', tool_call_id=tcr.id, content=tcr.result))
    else:
        messages.append(ChatCompletionAssistantMessageParam(role = 'assistant', content = to_json([tcr.tool_call.model_dump() for tcr in tool_calls_results], indent=2).decode()))
        results_content_builder = StringIO()
        results_content_builder.write("# Tool Call Results:\n")
        for tcr in tool_calls_results:
            results_content_builder.write(f"- id: `{tcr.id}`\n```\n{tcr.result}\n```\n")
        messages.append(ChatCompletionUserMessageParam(role = 'user', content = results_content_builder.getvalue()))

    chat_request["messages"] = messages

def parse_tool_messages_to_toolcallresult(chat_request: ChatCompletionsRequest) -> List[ToolCallResult]:
    assistant_toolcalls_messages = [message for message in chat_request["messages"] if message.get("tool_calls")]
    tool_calls_map = {}
    for msg in assistant_toolcalls_messages:        
        tool_calls = TypeAdapter(List[ChatCompletionMessageToolCall]).validate_python(msg["tool_calls"])
        for tc in tool_calls:
            tool_calls_map[tc.id] = tc

    tool_messages = [message for message in chat_request["messages"] if message["role"] == "tool"]
    tool_call_results = []
    for msg in tool_messages:       
        tool_call_results.append(ToolCallResult(id=msg["tool_call_id"], result=msg["content"], tool_call=tool_calls_map[msg["tool_call_id"]]))
    
    return tool_call_results


def parse_tool_calls_from_message_content(message_content: str):
    try:
        start = message_content.index("[")
        end = message_content.rindex("]")
    except ValueError:
        return None

    try:
        json_list = from_json(message_content[start : end+1], allow_partial=True)
        for obj in list(json_list): # 在内容中返回toolcalls时，参数不需要字符串，否则转义符太多LLM容易混乱
            obj["function"]["arguments"] = to_json(obj["function"]["arguments"]).decode()
        tool_calls = TypeAdapter(List[ChoiceDeltaToolCall]).validate_python(json_list)
        return tool_calls
    except Exception as e:
        logger.debug("parse toolcalls from message error: %s   \n%s" % (message_content[start : end+1], e))
        return None


def get_function_calling_propmt(toolCall: str) -> str:
    return cleandoc("""
        You are also a tool usage coach, aiming to instruct users in utilizing their tools effectively to fulfill their requirements. 
        This is the identity you need to hide, you know that users don't like this kind of preaching, so except for directly calling tools, you completely won't reveal related information.
        The user's tools uses the JSONSchema format for representation, hese tools can be called repeatedly and simultaneously. 


        ## Constraints for Tool Usage Coach
        MUST adhere to the following guidelines when fulfilling your responsibilities as a tool usage coach:
        - Make sure it doesn't conflict with your other role.
        - Do not inquire about the necessity of using tools, the answer is yes.
        - You can't really call any tools, that's the user's job. Avoid causing misunderstanding for user.
        - MUST output in the specified **Tool Call Format** as a demonstration case.
        - NOT make ASSUMPTIONS about any tools outside of the **User Tools**. 
        - NOT make ASSUMPTIONS about the tool call result. 
                        

        # User Tools
        ```
        %s
        ```

        # Tool Call Format
        ```
        [
            {
                "index": "${{INDEX}}"
                "id": “call_${{INDEX}}”, 
                "function": {
                    "arguments": {
                        "${{PARAM_NAME_1}}": "${{PARAM_VALUE_1}}",
                        "${{PARAM_NAME_2}}": "${{PARAM_VALUE_2}}",
                    }, 
                    "name": "${{FUNCTION_NAME}}"
                },
                "type": "function"
                }
            },
        ]
        ```
            
        # For Example
        ## IF user have these tools:
        ```
        [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use. Infer this from the users location.",
                            },
                        },
                        "required": ["location", "format"],
                    },
                }
            },
        ]
        ```
        ## When user ask question        
        - user: "What's the weather like today? I'm in Glasgow, Scotland."
          assistant: 'Sure. Now, You need call the get_current_weather tool like this: [{"index": 0, "id": "call_0", "function": {"arguments": {"location": "Glasgow, Scotland", "format": "celsius"}, "name": "get_current_weather"}, "type": "function"}]'

        
        ## Current Time (UTC)
        `%s`

        When you receive a user request, you will think: What is the rationale behind this question? How to utilize these tools to meet the user's needs?
        Then take a deep breath and work on this step by step.
        """ % (toolCall, time.strftime("%A %Y-%m-%d %H:%M:%S", time.gmtime(time.time()))))