import argparse
import asyncio
import json
import time
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union
import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from packaging import version
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice, CompletionResponseStreamChoice,
    CompletionStreamResponse, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse,
    ChatMessage, DeltaMessage, ErrorResponse, LogProbs, ModelCard, ModelList, ModelPermission,
    UsageInfo
)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid
import torch
import dataclasses
from dataclasses import dataclass, field

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)
served_model = None
app = fastapi.FastAPI()
engine = None


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(),
        status_code=status_code.value
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


@dataclasses.dataclass
class Conversation:
    # simplier and more flexible
    system_template: str = "{system_message}\n\n"
    utterance_template: str = '{role}: {message}\n\n'
    query_template: str = '{role}: '
    #
    system_message: str = ""
    roles: Dict[str, str] = field(default_factory=lambda: {"user": "USER", "assistant": "ASSISTANT"})
    # All messages. Each item is (role, message).
    utterances: List[str] = ()
    # The number of few shot examples
    offset: int = 0

    def get_prompt(self) -> str:
        ret = self.system_template.format(system_message=self.system_message)
        for utter in self.utterances:
            ret += utter
        return ret

    def append_message(self, role, message, is_query=False):
        if is_query:
            self.utterances.append(self.query_template.format(role=role, message=message))
        else:
            self.utterances.append(self.utterance_template.format(role=role, message=message))

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.utterances[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.utterances[self.offset:]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret


async def get_gen_prompt(request, conv_conf) -> str:

    conv = Conversation(**conv_conf)
    # avoid fastchat API
    for message in request.messages:
        msg_role = message["role"]
        if msg_role == "system":
            if message["content"] != '':
                conv.system_message = message["content"]
        elif msg_role in ["user", "assistant"]:
            conv.append_message(conv.roles[msg_role], message["content"])
        else:
            raise ValueError(f"Unknown role: {msg_role}")

    # Add a blank message for the assistant.
    conv.append_message(conv.roles['assistant'], '', is_query=True)
    prompt = conv.get_prompt()
    return prompt


async def check_length(
    request: Union[ChatCompletionRequest, CompletionRequest],
    prompt: Optional[str] = None,
    prompt_ids: Optional[List[int]] = None
) -> Tuple[List[int], Optional[JSONResponse]]:
    assert (
        not (prompt is None and prompt_ids is None)
        and not (prompt is not None and prompt_ids is not None)
    ), "Either prompt or prompt_ids should be provided."
    input_ids = prompt_ids if prompt_ids is not None else tokenizer(prompt).input_ids
    token_num = len(input_ids)

    if request.max_tokens is None:
        request.max_tokens = max_model_len - token_num
    if token_num + request.max_tokens > max_model_len:
        return input_ids, create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return input_ids, None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [ModelCard(id=served_model, root=served_model, permission=[ModelPermission()])]
    return ModelList(data=model_cards)


def create_logprobs(
    token_ids: List[int],
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None,
    num_output_top_logprobs: Optional[int] = None,
    initial_text_offset: int = 0,
) -> LogProbs:
    """Create OpenAI-style logprobs."""
    logprobs = LogProbs()
    last_token_len = 0
    if num_output_top_logprobs:
        logprobs.top_logprobs = []
    for i, token_id in enumerate(token_ids):
        step_top_logprobs = top_logprobs[i]
        if step_top_logprobs is not None:
            token_logprob = step_top_logprobs[token_id]
        else:
            token_logprob = None
        token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(token_logprob)
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
        last_token_len = len(token)

        if num_output_top_logprobs:
            logprobs.top_logprobs.append(
                {tokenizer.convert_ids_to_tokens(i): p
                 for i, p in step_top_logprobs.items()} if step_top_logprobs else None
            )
    return logprobs


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """

    # big-agi: ChatCompletionRequest(model='Qwen-openhermes', messages=[{'role': 'system', 'content': 'below is a instuction\n'}, {'role': 'user', 'content': xxx}, {}], temperature=0.5, top_p=1.0, n=1, max_tokens=512, stop=[], stream=True, presence_penalty=0.0, frequency_penalty=0.0, logit_bias=None, user=None, best_of=None, top_k=-1, ignore_eos=False, use_beam_search=False, stop_token_ids=[], skip_special_tokens=True, spaces_between_special_tokens=True
    # TODO: big-agi 会带入error localai: Internal Server Error (500)

    conf = {
        'conv_conf':{
            'system_template': "{system_message}\n\n",
            'system_message': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.',
            'utterance_template': '{role}: {message}\n\n',
            'roles': {
                'user': "### Instruction",
                'assistant': "### Response"
            },
            'utterances': [],
            'offset': 0,
        },
        'sampling_conf': {
            'stop': ["Instruction:", "Response:", "</s>", "<|endoftext|>"]
        }
    }

    # error_check_ret = await check_model(request)
    error_check_ret = None
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST, "logit_bias is not currently supported")

    prompt = await get_gen_prompt(request, conf['conv_conf'])
    print(prompt)

    token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())

    spaces_between_special_tokens = request.spaces_between_special_tokens
    sampling_conf = conf['sampling_conf']
    sampling_params = SamplingParams(
        n=request.n,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=sampling_conf['stop'] if len(request.stop)==0 else request.stop,
        stop_token_ids=request.stop_token_ids,
        max_tokens=request.max_tokens,
        best_of=request.best_of,
        top_k=request.top_k,
        ignore_eos=request.ignore_eos,
        use_beam_search=request.use_beam_search,
        skip_special_tokens=request.skip_special_tokens,
        spaces_between_special_tokens=spaces_between_special_tokens,
    )
    # except ValueError as e:
    # return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = engine.generate(prompt, sampling_params, request_id, token_ids)

    def create_stream_response_json(
        index: int,
        text: str,
        finish_reason: Optional[str] = None,
        usage: Optional[UsageInfo] = None,
    ) -> str:
        choice_data = ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(content=text),
            finish_reason=finish_reason,
        )
        response = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        if usage is not None:
            response.usage = usage
        # exclude unset to leave details out of each sse
        response_json = response.json(exclude_unset=True, ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id, choices=[choice_data], model=model_name
            )
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                completion_tokens = len(output.token_ids)
                previous_num_tokens[i] = completion_tokens
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    prompt_tokens = len(res.prompt_token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        finish_reason=output.finish_reason,
                        usage=final_usage,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(), media_type="text/event-stream")

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return create_error_response(HTTPStatus.BAD_REQUEST, "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        choice_data = ChatCompletionResponseChoice(
            index=output.index,
            message=ChatMessage(role="assistant", content=output.text),
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )
    print(response)
    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(), media_type="text/event-stream")

    return response


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following features:
        - suffix (the language models we currently support do not support
          suffix)
        - logit_bias (to be supported by vLLM engine)
    """

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    # OpenAI API supports echoing the prompt when max_tokens is 0.
    echo_without_generation = request.echo and request.max_tokens == 0

    if request.suffix is not None:
        # The language models we currently support do not support suffix.
        return create_error_response(HTTPStatus.BAD_REQUEST, "suffix is not currently supported")

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST, "logit_bias is not currently supported")

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"

    use_token_ids = False
    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return create_error_response(HTTPStatus.BAD_REQUEST, "please provide at least one prompt")
        first_element = request.prompt[0]
        if isinstance(first_element, int):
            use_token_ids = True
            prompt = request.prompt
        elif isinstance(first_element, (str, list)):
            # TODO: handles multiple prompt case in list[list[int]]
            if len(request.prompt) > 1:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST, "multiple prompts in a batch is not currently supported"
                )
            use_token_ids = not isinstance(first_element, str)
            prompt = request.prompt[0]
    else:
        prompt = request.prompt

    if use_token_ids:
        _, error_check_ret = await check_length(request, prompt_ids=prompt)
    else:
        token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    created_time = int(time.monotonic())
    try:
        spaces_between_special_tokens = request.spaces_between_special_tokens
        sampling_params = SamplingParams(
            n=request.n,
            best_of=request.best_of,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            ignore_eos=request.ignore_eos,
            max_tokens=request.max_tokens if not echo_without_generation else 1,
            logprobs=request.logprobs,
            use_beam_search=request.use_beam_search,
            prompt_logprobs=request.logprobs if request.echo else None,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    if use_token_ids:
        result_generator = engine.generate(None, sampling_params, request_id, prompt_token_ids=prompt)
    else:
        result_generator = engine.generate(prompt, sampling_params, request_id, token_ids)

    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. In addition, we do not stream the results when use beam search.
    stream = (
        request.stream and (request.best_of is None or request.n == request.best_of)
        and not request.use_beam_search
    )

    def create_stream_response_json(
        index: int,
        text: str,
        logprobs: Optional[LogProbs] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[UsageInfo] = None,
    ) -> str:
        choice_data = CompletionResponseStreamChoice(
            index=index,
            text=text,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
        response = CompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        if usage is not None:
            response.usage = usage
        response_json = response.json(exclude_unset=True, ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        has_echoed = [False] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                token_ids = output.token_ids[previous_num_tokens[i]:]
                top_logprobs = output.logprobs[previous_num_tokens[i]:]
                offsets = len(previous_texts[i])
                if request.echo and not has_echoed[i]:
                    if not echo_without_generation:
                        delta_text = res.prompt + delta_text
                        token_ids = res.prompt_token_ids + token_ids
                        top_logprobs = res.prompt_logprobs + top_logprobs
                    else:
                        delta_text = res.prompt
                        token_ids = res.prompt_token_ids
                        top_logprobs = res.prompt_logprobs
                    has_echoed[i] = True
                if request.logprobs is not None:
                    logprobs = create_logprobs(
                        token_ids=token_ids,
                        top_logprobs=top_logprobs,
                        num_output_top_logprobs=request.logprobs,
                        initial_text_offset=offsets,
                    )
                else:
                    logprobs = None
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                finish_reason = output.finish_reason
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                    logprobs=logprobs,
                    finish_reason=finish_reason,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    logprobs = (LogProbs() if request.logprobs is not None else None)
                    prompt_tokens = len(res.prompt_token_ids)
                    completion_tokens = len(output.token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        logprobs=logprobs,
                        finish_reason=output.finish_reason,
                        usage=final_usage,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if stream:
        return StreamingResponse(completion_stream_generator(), media_type="text/event-stream")

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return create_error_response(HTTPStatus.BAD_REQUEST, "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    prompt_token_ids = final_res.prompt_token_ids
    prompt_logprobs = final_res.prompt_logprobs
    prompt_text = final_res.prompt
    for output in final_res.outputs:
        if request.logprobs is not None:
            if not echo_without_generation:
                token_ids = output.token_ids
                top_logprobs = output.logprobs
                if request.echo:
                    token_ids = prompt_token_ids + token_ids
                    top_logprobs = prompt_logprobs + top_logprobs
            else:
                token_ids = prompt_token_ids
                top_logprobs = prompt_logprobs
            logprobs = create_logprobs(
                token_ids=token_ids,
                top_logprobs=top_logprobs,
                num_output_top_logprobs=request.logprobs,
            )
        else:
            logprobs = None
        if not echo_without_generation:
            output_text = output.text
            if request.echo:
                output_text = prompt_text + output_text
        else:
            output_text = prompt_text
        choice_data = CompletionResponseChoice(
            index=output.index,
            text=output_text,
            logprobs=logprobs,
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(), media_type="text/event-stream")

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--allow-credentials", action="store_true", help="allow credentials")
    parser.add_argument("--allowed-origins", type=json.loads, default=["*"], help="allowed origins")
    parser.add_argument("--allowed-methods", type=json.loads, default=["*"], help="allowed methods")
    parser.add_argument("--allowed-headers", type=json.loads, default=["*"], help="allowed headers")
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not "
        "specified, the model name will be the same as "
        "the huggingface name."
    )
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    # monkey patch cannot work for multi-process
    # if args.lora:

    #     from transformers import AutoModelForCausalLM
    #     from peft import PeftModel

    #     def _direct_load_model_iterator(
    #         model_name_or_path: str,
    #         cache_dir: Optional[str] = None,
    #         load_format: str = "auto",
    #         revision: Optional[str] = None,
    #     ):
    #         print('>>> use hack version loader')
    #         # need to merge
    #         model = AutoModelForCausalLM.from_pretrained(
    #             model_name_or_path,
    #             trust_remote_code=True,
    #             device_map='auto',
    #             torch_dtype=torch.bfloat16,
    #         )
    #         model = PeftModel.from_pretrained(
    #             model,
    #             args.lora,
    #         )
    #         model = model.merge_and_unload(progressbar=True,safe_merge=True)
    #         for name, param in model.state_dict().items():
    #             yield name, param
    #         torch.cuda.empty_cache()

    #     # to support lora / merge dynamic loading
    #     # import vllm.model_executor.models.qwen as qwen
    #     import vllm.model_executor.models.qwen
    #     vllm.model_executor.models.qwen.hf_model_weights_iterator = _direct_load_model_iterator

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())
    max_model_len = engine_model_config.max_model_len

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_model_config.tokenizer,
        tokenizer_mode=engine_model_config.tokenizer_mode,
        trust_remote_code=engine_model_config.trust_remote_code
    )

    uvicorn.run(
        app, host=args.host, port=args.port, log_level="info", timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )