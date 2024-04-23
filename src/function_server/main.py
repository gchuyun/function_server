import httpx
import asyncio
from io import StringIO
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, Response
from starlette.responses import StreamingResponse
from contextlib import asynccontextmanager
from .fake_messages import ChatCompletionsRequest
from .fake_messages import fake_chat_request_if_need, add_tool_calls_result_messages, parse_tool_calls_from_message_content, parse_tool_messages_to_toolcallresult
from .function_calling import calling, load_tools, FUNCTION_CALLING_TOOLS, ToolCallResult
from openai._types import NOT_GIVEN, Body, Query, Headers
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDeltaToolCall
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from pydantic_core import from_json, to_json
from pydantic import TypeAdapter, ValidationError
from typing import List, Union
from starlette.background import BackgroundTask
import hashlib
from .utils import init_logger, Cache, ReReadbleHttpxSuccessfulResponse
from loguru import logger


init_logger()
load_tools()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.httpx_client = httpx.AsyncClient(timeout=600)
    app.function_executor = ThreadPoolExecutor(max_workers=5)
    app.chat_proxy_cache = Cache(expire_milliseconds = 5*60*1000)
    app.toolcalls_in_process = Cache(expire_milliseconds = 60000)
    yield
    await app.httpx_client.aclose()
    app.function_executor.shutdown(wait=False, cancel_futures=True)
    app.chat_proxy_cache.clear()
    app.toolcalls_in_process.clear()

MAX_TOOL_CALL_ITERATIONS_NUMBER = 10
app = FastAPI(lifespan=lifespan)

@app.get("/tools")
async def get_tools():
    return [v[1] for v in FUNCTION_CALLING_TOOLS.values()]

@app.post("/toolcalls")
async def call_tools(request: Request, tool_calls: List[Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCall]]):
    tool_call_results = []
    unknown_tool_calls = []
    loop = asyncio.get_running_loop()
    for tc in tool_calls:
        if tc.function.name in FUNCTION_CALLING_TOOLS.keys():
            tc_result = await loop.run_in_executor(request.app.function_executor, calling, tc)
            tool_call_results.append(tc_result)
        else:
            unknown_tool_calls.append(tc)    
    return {"results": tool_call_results, "unknown_tool_calls": unknown_tool_calls}

@app.api_route("/{target_url:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"])
async def proxy(request: Request, target_url: str):
    headers = {}
    for key, value in request.headers.items():
        if key.lower() not in ['host', 'connection', 'content-length']:
            headers[key.lower()] = value    

    target_url = urllib.parse.unquote(target_url)
    logger.info(target_url)

    if target_url.lower().endswith("/v1/chat/completions") and request.method.lower() == "post": 
        body = await request.body()
        request_hash = hashlib.md5(body).hexdigest()
        chat_proxy_task = request.app.chat_proxy_cache.get(request_hash)
        if chat_proxy_task is None:            
            chat_proxy_coroutine = _proxy_and_call_function_if_need(target_url, headers, body, request.app.httpx_client, request.app.function_executor)
            chat_proxy_task = asyncio.get_running_loop().create_task(chat_proxy_coroutine)            
            request.app.chat_proxy_cache.put(request_hash, chat_proxy_task)

        resp, tool_call_results = await chat_proxy_task
        if tool_call_results:
            for tc in tool_call_results:
                if isinstance(tc, str):
                    request.app.toolcalls_in_process.put(tc, tool_call_results)
        if resp.status_code != 200:            
            request.app.chat_proxy_cache.pop(request_hash)
        
        logger.debug("========= FINNAL REQUEST:\n%s" % resp.content.decode())
    else:
        req = request.app.httpx_client.build_request(request.method, target_url, headers=headers, content=request.stream())
        resp = await request.app.httpx_client.send(req)

    if resp.is_stream_consumed:
        return Response(content=resp.content, status_code=resp.status_code, headers=resp.headers, background=BackgroundTask(resp.aclose))
    else:
        return StreamingResponse(content=resp.aiter_raw(), status_code=resp.status_code, headers=resp.headers, background=BackgroundTask(resp.aclose))

async def _proxy_and_call_function_if_need(target_url: str, headers: Headers, body: bytes, http_client: httpx.AsyncClient, function_executor: ThreadPoolExecutor) -> tuple[ReReadbleHttpxSuccessfulResponse, List]:
    chat_request = None
    try:        
        chat_request = from_json(body.decode())
        TypeAdapter(ChatCompletionsRequest).validate_python(chat_request)
    except ValidationError as e:
        return ReReadbleHttpxSuccessfulResponse(httpx.Response(status_code=400, text="%s" % e)), None
    
    client_tools_names = [t["function"]["name"] for t in list(chat_request["tools"])] if chat_request.get("tools") else []
    server_tools = [v[1] for k,v in FUNCTION_CALLING_TOOLS.items() if k not in client_tools_names]

    tool_call_results = parse_tool_messages_to_toolcallresult(chat_request)
    tool_call_results = await merge_toolcallresult_from_cache(tool_call_results)

    logger.debug("========= ORIGIN REQUEST:\n %s" % to_json(chat_request, indent=2).decode())
    fake_chat_request_if_need(chat_request, server_tools, tool_call_results)
    
    for i in range(MAX_TOOL_CALL_ITERATIONS_NUMBER):
        tool_calls, chat_response = await get_tool_calls_from_openai_response(target_url, headers, chat_request, http_client)
        if not tool_calls or i == 9:
            return ReReadbleHttpxSuccessfulResponse(chat_response), None
        
        client_tool_calls = []
        tool_call_results = []
        loop = asyncio.get_running_loop()
        for tc in tool_calls:
            if tc.function.name in client_tools_names:
                tool_call_results.append(tc.id)
                client_tool_calls.append(tc)
            else:
                tc_result = loop.run_in_executor(function_executor, calling, tc)
                tool_call_results.append(tc_result)                                
        if client_tool_calls:            
            client_tool_call_resp = await create_response_for_toolcalls(chat_response, client_tool_calls)
            return ReReadbleHttpxSuccessfulResponse(client_tool_call_resp), tool_call_results
        else:
            await chat_response.aclose()
            add_tool_calls_result_messages(chat_request, [await r for r in tool_call_results])

async def merge_toolcallresult_from_cache(client_results: List[ToolCallResult]) -> List[ToolCallResult]:
    cached_results = []
    new_results = []
    for r in client_results:
        tmp_cached_results = app.toolcalls_in_process.pop(r.id)
        if tmp_cached_results:            
            cached_results = list(tmp_cached_results)
            i = cached_results.index(r.id)
            cached_results[i] = r
        else:
            new_results.append(r)
    
    tool_call_results = new_results
    for r in cached_results:
        if isinstance(r, str):
            continue
        elif isinstance(r, ToolCallResult):
            tool_call_results.append(r)
        else:
            tool_call_results.append(await r)

    return tool_call_results

async def create_response_for_toolcalls(chat_response: httpx.Response, client_tool_calls: List[Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCall]]) -> httpx.Response:
    body_text = None
    if not chat_response.headers["content-type"].lower().startswith("text/event-stream"):
        chat_completion_json = from_json(chat_response.text, allow_partial=True)
        if not chat_completion_json["choices"][0]["finish_reason"]: # github copilot
            chat_completion_json["choices"][0]["finish_reason"] = "stop"
        chat_completion = ChatCompletion.model_validate(chat_completion_json)
        chat_completion.choices[0].finish_reason = "tool_calls"
        chat_completion.choices[0].message.tool_calls = client_tool_calls
        chat_completion.choices[0].message.content = ""
        body_text = to_json(chat_completion).decode()
    else:
        chunk = None
        for line in chat_response.iter_lines():
            if len(line) < 6:
                continue                
            if line[:6] != "data: ":
                continue
            data = line[6:].removesuffix("\r")
            if data.startswith("[DONE]"):
                break
            
            chunk_dict = from_json(data, allow_partial=True)
            if chunk_dict["choices"][0].get("index") is None:   # for some proxy miss the index attribute
                chunk_dict["choices"][0]["index"] = 0
            if chunk_dict["choices"][0].get("delta") is not None and chunk_dict["choices"][0]["delta"].get("tool_calls") is not None and chunk_dict["choices"][0]["delta"]["tool_calls"][0].get("index") is None:
                chunk_dict["choices"][0]["delta"]["tool_calls"][0]["index"] = 0

            chunk = ChatCompletionChunk.model_validate(chunk_dict)
            break

        chunk.choices[0].finish_reason = "tool_calls"
        chunk.choices[0].delta.tool_calls = client_tool_calls
        chunk.choices[0].delta.content = ""
        chunk.choices[0].delta.role = "assistant"
        body_text = "data: %s\n\ndata: [DONE]" % to_json(chunk).decode()

    await chat_response.aclose()
    headers = {}
    for key, value in chat_response.headers.items():
        if key.lower() not in ['connection', 'content-length']:
            headers[key] = value  
    resp = httpx.Response(status_code=chat_response.status_code, headers=headers, text=body_text)
    return resp

async def get_tool_calls_from_openai_response(target_url: str, headers: Headers, chat_request: ChatCompletionsRequest, httpx_client: httpx.AsyncClient) -> tuple[List[Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCall]], httpx.Response]:   
    logger.debug("========= REQUEST:\n%s" % to_json(chat_request, indent=2).decode())
    chat_response = await httpx_client.post(target_url, content=to_json(chat_request), headers=headers)
    
    if chat_response.status_code != 200:
        return None, chat_response

    await chat_response.aread()
    logger.debug("========= RESPONSE:\n%s" % chat_response.text)

    content_builder = StringIO()                        
    tool_calls: List[Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCall]] = []
    if not chat_response.headers["content-type"].lower().startswith("text/event-stream"):
        chat_completion_json = from_json(chat_response.text, allow_partial=True)
        if not chat_completion_json["choices"][0]["finish_reason"]: # github copilot
            chat_completion_json["choices"][0]["finish_reason"] = "stop"
        chat_completion = ChatCompletion.model_validate(chat_completion_json)
        tool_calls = chat_completion.choices[0].message.tool_calls 
        if chat_completion.choices[0].message.content:
            content_builder.write(chat_completion.choices[0].message.content) 
    else:
        for line in chat_response.iter_lines():
            if len(line) < 6:
                continue                
            if line[:6] != "data: ":
                continue
            data = line[6:].removesuffix("\r")
            if data.startswith("[DONE]"):
                break
            
            chunk_dict = from_json(data, allow_partial=True)
            if chunk_dict["choices"][0].get("index") is None:   # for some proxy miss the index attribute
                chunk_dict["choices"][0]["index"] = 0
            if chunk_dict["choices"][0].get("delta") is not None and chunk_dict["choices"][0]["delta"].get("tool_calls") is not None and chunk_dict["choices"][0]["delta"]["tool_calls"][0].get("index") is None:
                chunk_dict["choices"][0]["delta"]["tool_calls"][0]["index"] = 0

            chunk = ChatCompletionChunk.model_validate(chunk_dict)
            delta = chunk.choices[0].delta
            if delta and delta.content:
                content_builder.write(delta.content)
            elif delta and delta.tool_calls:
                tcchunklist = delta.tool_calls
                for tcchunk in tcchunklist:
                    if len(tool_calls) <= tcchunk.index:
                        tool_calls.append(tcchunk)
                    else:
                        tc = tool_calls[tcchunk.index]
                        if tcchunk.id:
                            tc.id += tcchunk.id
                        if tcchunk.function.name:
                            tc.function.name += tcchunk.function.name
                        if tcchunk.function.arguments:
                            tc.function.arguments += tcchunk.function.arguments
    if not tool_calls:
        tool_calls = parse_tool_calls_from_message_content(content_builder.getvalue())

    return tool_calls, chat_response
    

def main():
    import uvicorn    
    uvicorn.run("function_server.main:app", host="0.0.0.0")    

if __name__ == '__main__':
    main()