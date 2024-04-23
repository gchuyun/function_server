import os
import glob
import ast
import importlib
from pip._internal import main as pip
from inspect import getmembers, isfunction
from typing import Union, Callable

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat import ChatCompletionToolParam
from openai_function_calling.tool_helpers import ToolHelpers
from pydantic import BaseModel
from pydantic_core import from_json, to_json
from loguru import logger


FUNCTION_CALLING_TOOLS: dict[str, (Callable, ChatCompletionToolParam)] = {}

def tool(func):
    '''tool装饰器'''
    func.is_function_calling_tool = True
    return func

class ToolCallResult(BaseModel):
    id: str
    result: str
    tool_call: Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCall]    

def calling(tool_call: Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCall]) -> ToolCallResult:
    id = tool_call.id
    tool_name = tool_call.function.name
    args = from_json(tool_call.function.arguments)
    if FUNCTION_CALLING_TOOLS.get(tool_name):
        func, _  = FUNCTION_CALLING_TOOLS[tool_name]
        try:
            result = func(**args)
        except Exception as e:
            if isinstance(args, str):
                try:
                    args = from_json(args)
                    result = func(**args)
                except:
                    result = "call [%s] error" % tool_name
            else:            
                result = "call [%s] error" % tool_name

        if isinstance(result, bytes):
            result = bytes(result).decode()
        elif not isinstance(result, str):
            result = to_json(result, indent=2).decode()
    else:
        result = ""
    
    return ToolCallResult(id=id, result=result, tool_call=tool_call)    

def load_tools():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tools_dir = os.path.join(current_dir, "tools")

    py_files = glob.glob(os.path.join(tools_dir, "*.py"))
    for py_file in py_files:
        if py_file.endswith("/__init__.py"):
            continue
        try:
            dependencies = parse_requirements(py_file)
            setup_requirements(dependencies)
            module = importlib.import_module(".tools.%s" % py_file[py_file.rindex("/")+1 : -3], "function_server")
            for name, func in getmembers(module, isfunction):
                if getattr(func, 'is_function_calling_tool', False):
                    tool: ChatCompletionToolParam = ToolHelpers.infer_from_function_refs([func])[0]
                    FUNCTION_CALLING_TOOLS[name] = (func, tool)
        except Exception as e:
            print(f"Failed to load module {py_file}: {e}")


def parse_requirements(py_file_path):
    with open(py_file_path) as pyfile:
        tree = ast.parse(pyfile.read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'requirements':
                if isinstance(node.value, ast.Constant):
                    requirements = node.value.s
                    return requirements.split('\n')
    return []

def setup_requirements(dependencies):
    for dependency in dependencies:
        if dependency:
            pip(['install', dependency])