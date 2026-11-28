"""The agentic loop - same shape for every backend (Ollama, vLLM, NVIDIA NIM)."""

import json
import logging
import time

import httpx
from ollama import Client, ResponseError
from ollama._types import ChatResponse
from openai import BadRequestError, OpenAI

from . import runtime
from .config import (ollama_keep_alive, ollama_num_ctx, ollama_think, ollama_url,
                     openai_api_key, openai_base_url, openai_enable_thinking)
from .episode import resolve_task
from .prompt import build_system_message, tool_arguments
from .rendering import state_to_text
from .tools import available_tools

logger = logging.getLogger(__name__)

#: Transport failures are retried before a turn - and with it the episode - is
#: given up. Ollama unloads an idle model after five minutes, and loading a 17 GB
#: 27B takes longer than one turn may take, so the reload lands inside
#: `--llm-timeout` and the whole episode dies as `llm_error` on its first turn.
LLM_RETRIES = 2
#: Seconds before the first retry, doubled per retry (5 s, then 10 s). The wait is
#: cheap next to the episode a failed turn throws away.
LLM_RETRY_BACKOFF = 5.0

#: History cap of both loops: the system prompt, the task and this many recent
#: messages. A step appends one assistant turn plus one tool result, and every tool
#: result repeats the state block, so a 60-step run outgrows the context window of
#: the server mid-episode - the prompts of the 27B sweep reached 16-25k tokens and
#: then died on a read timeout (`llm_error`) rather than on a clean 400.
MAX_HISTORY_MESSAGES = 42


def trim_history(messages):
    """Drop the oldest turns, never splitting a `tool_calls` turn from its results.

    vLLM rejects a `tool` message that answers no preceding `tool_calls`, so a
    window that starts in the middle of a step loses those orphan results with it.
    Ollama accepts orphans, but both loops keep the same window so that an episode
    of one backend stays comparable with the other. The world is not lost with
    them: every tool result carries `current_state`.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    tail = messages[-(MAX_HISTORY_MESSAGES - 2):]
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]
    return messages[:2] + tail


def llm_transient_error(error):
    """True when resending the same request can plausibly succeed.

    Everything `httpx` raises on the way to the server is transient - timeouts
    above all, but also a reset connection or a proxy error - and so is the
    builtin `ConnectionError` that `ollama.Client` re-raises a refused connection
    as. A `ResponseError` only is when the server itself says "not now" (429,
    5xx); a 400/404 is answered the same way forever.
    """
    if isinstance(error, (httpx.TransportError, ConnectionError)):
        return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and status in (429, 500, 502, 503, 504)


def ollama_request_extras():
    """The optional `/api/chat` fields of the `ollama:` block of config.yaml.

    Three fields the harness never sent, each optional (`None` = send nothing):

    * `keep_alive` - Ollama's default unloads the weights after five idle minutes,
      and the reload is charged to the *next* turn's `--llm-timeout`;
    * `think` - `false` keeps a reasoning model from spending the turn (and the
      timeout) on hidden tokens;
    * `num_ctx` - pins the KV cache instead of leaving the size to the server.
    """
    extras = {}
    keep_alive = ollama_keep_alive()
    if keep_alive is not None:
        extras["keep_alive"] = keep_alive
    think = ollama_think()
    if think is not None:
        extras["think"] = bool(think)
    num_ctx = ollama_num_ctx()
    if num_ctx:
        extras["options"] = {"num_ctx": int(num_ctx)}
    return extras


def _ollama_chat(client, model, messages, schemas, step, extras):
    """`client.chat` with `LLM_RETRIES` retries and exponential backoff.

    `llm_calls` is incremented *before* the request on purpose: a turn that times
    out is one LLM request, and a first-turn timeout used to write
    `no_of_llm_calls: 0`. For the same reason the wait of a failed attempt is
    recorded in `llm_time_taken` - a 60 s timeout is a latency fact, not a missing
    value - while the backoff sleep is not: that is the harness waiting, not the
    model.
    """
    delay = LLM_RETRY_BACKOFF
    attempt = 0
    while True:
        attempt += 1
        runtime.llm_calls += 1
        started = time.time()
        try:
            response = client.chat(model=model, messages=messages, tools=schemas,
                                   **extras)
        except Exception as exc:
            runtime.llm_time_taken.append(time.time() - started)
            if attempt > LLM_RETRIES or not llm_transient_error(exc):
                raise
            logger.warning(f"step {step}: LLM request {attempt} failed "
                           f"({type(exc).__name__}: {exc}) - retrying in {delay:.0f}s")
            print(f"llm request {attempt} failed ({exc}) - retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
            continue
        runtime.llm_time_taken.append(time.time() - started)
        return response


def chat_completion_ollama(client, model, messages, schemas, step):
    """One completion, with the `ollama:` extras and a fallback for older servers.

    `keep_alive`, `think` and `options.num_ctx` are optional fields of `/api/chat`,
    so a server that predates one of them answers 400 - and that is not a failed
    step: the same prompt is resent without it, the graceful degradation
    `chat_completion` below already gives the vLLM/NIM path for
    `chat_template_kwargs`.
    """
    extras = ollama_request_extras()
    # Most specific request first, then one field less at a time, then none:
    # `think` is the field an old server is likeliest to reject, `keep_alive` the
    # one it must have known longest, so the order only decides how much is
    # dropped per attempt. Without an `ollama:` block there is exactly one attempt.
    variants = []
    for candidate in (extras,
                      {key: value for key, value in extras.items() if key != "think"},
                      {key: value for key, value in extras.items() if key != "options"},
                      {}):
        if candidate not in variants:
            variants.append(candidate)

    last_error = None
    for index, variant in enumerate(variants):
        try:
            return _ollama_chat(client, model, messages, schemas, step, variant)
        except ResponseError as exc:
            if exc.status_code != 400:
                raise
            last_error = exc
            if index + 1 < len(variants):
                dropped = sorted(set(variant) - set(variants[index + 1]))
                logger.info(f"step {step}: Ollama answered 400 ({exc}) - retrying "
                            f"without {', '.join(dropped) or 'every extra'}")
    raise last_error


def run_agent(args, schemas):
    # The prompt and the first user turn both carry the task; `--task` overrides
    # it, otherwise it is the objective this preset assigns (never empty, which is
    # what used to leave the model without a task to obey).
    task_text = resolve_task(args)
    system_message2 = build_system_message(runtime.state, schemas, task_text)
    messages = [
        {'role': 'system', 'content': system_message2},
        {'role': 'user', 'content': task_text},
    ]
    print(f"task: {task_text}")
    logger.info(f"task: {task_text}")
    logger.info(f"system message: {system_message2}")

    client = Client(host=ollama_url(), timeout=args.llm_timeout)

    silent_steps = 0  # consecutive model turns without a tool call
    stuck_repeats = 0  # consecutive identical tool calls
    last_signature = None
    try:
        i = 0
        while i < args.max_steps:
            i += 1
            # Kept at module level so `budget_to_text` can report what is left in the
            # observation the model reads after this step's action.
            runtime.STEPS_USED = i
            # Client-side clock around the completion: `total_duration` below is what
            # the server measured (prefill + decode), the one `llm_time_taken` records
            # also carries queueing and the network - and, since `chat_completion_ollama`
            # retries, it is the time of the request that answered (a failed attempt is
            # recorded with its own entry, the backoff sleep is not).
            # The window is capped like the vLLM loop's: a 60-step history of state
            # blocks outgrows the context Ollama serves and the turn dies on a timeout.
            messages = trim_history(messages)
            response: ChatResponse = chat_completion_ollama(client, args.model, messages,
                                                            schemas, i)
        
            print(f"input_tokens: {response['prompt_eval_count']}")
            print("Prompt evaluation time:", response["prompt_eval_duration"] / 1e9, "seconds")
            logger.info(f"Prompt evaluation time:{response['prompt_eval_duration'] / 1e9:.2f}")
            print("Generation time:", response["eval_duration"] / 1e9, "seconds")
            logger.info(f"Generation time: {response['eval_duration'] / 1e9:.2f}")
            print(f"output_tokens: {response['eval_count']}")
            print(f"response time: {runtime.llm_time_taken[-1]:.2f}s "
                  f"(server {response['total_duration'] / 1e9:.2f}s)")
            logger.info(f"step {i} response time: {runtime.llm_time_taken[-1]:.2f}s "
                        f"(server {response['total_duration'] / 1e9:.2f}s)")

            runtime.total_output_tokens += response['eval_count'] or 0
            runtime.total_input_tokens += response['prompt_eval_count'] or 0

            print("response:")
            print(response.message)
            if response.message.content:
                print('Content: ')
                print(response.message.content + '\n')
                logger.info(f"content : {response.message.content}")

            if getattr(response.message, "thinking", None):
                print('Thinking: ')
                print(response.message.thinking + '\n')
                logger.info(f"thinking : {response.message.thinking}")

            messages.append(response.message)

            if response.message.tool_calls:
                silent_steps = 0
                print("llm tool_calls:")
                print(len(response.message.tool_calls))
                runtime.tool_calls_sent += len(response.message.tool_calls)
                # Every call is a request of its own, in the order the model asked for
                # them: the wrapper sends, book-keeps and answers in one step, so the
                # reply goes straight into the `tool` message below.
                answers = []
                for tool_call in response.message.tool_calls:
                    # The LLM decides which wrapper to call, we never repair its plan.
                    name = tool_call.function.name
                    arguments = tool_arguments(tool_call)
                    # A model that repeats one call forever burns the whole step
                    # budget: remember the call so the loop can nudge it once.
                    signature = (name, json.dumps(arguments, sort_keys=True, default=str))
                    if signature == last_signature:
                        stuck_repeats += 1
                    else:
                        last_signature, stuck_repeats = signature, 1
                    function_to_call = available_tools.get(name)
                    logger.info(f"tool call: {name} args={json.dumps(arguments)}")
                    if function_to_call:
                        try:
                            real_result_json = function_to_call(**arguments)
                        except TypeError as e:
                            runtime.invalid_moves += 1
                            key = "bad tool arguments"
                            runtime.invalid_move_object[key] = (
                                runtime.invalid_move_object.get(key, 0) + 1)
                            real_result_json = json.dumps({"status": "false", "action": name,
                                                           "message": f"Error in tool call: {str(e)}"})
                    else:
                        runtime.invalid_moves += 1
                        key = "unknown tool"
                        runtime.invalid_move_object[key] = (
                            runtime.invalid_move_object.get(key, 0) + 1)
                        print(f'Tool {name} not found')
                        real_result_json = json.dumps({"status": "false", "action": name,
                                                       "message": f'Tool {name} not found'})
                    answers.append((name, json.loads(real_result_json)))

                for name, reply in answers:
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({
                            "action_result": reply,
                            "current_state": state_to_text(runtime.state),
                        }),
                        "tool_name": name,
                    })

                if stuck_repeats >= runtime.STUCK_REPEATS:
                    # Naming the repeat is the cheapest way out of a loop: the model
                    # sees the same arguments it just sent and a list of alternatives.
                    print(f"stuck: {last_signature[0]} called {stuck_repeats}x with the same arguments")
                    logger.info(f"stuck: {last_signature[0]} called {stuck_repeats}x with the same "
                                f"arguments {last_signature[1]} - hinting the model")
                    messages.append({
                        'role': 'user',
                        'content': (f"You have called `{last_signature[0]}` with identical arguments "
                                    f"{stuck_repeats} times and the state did not change. Read "
                                    "`reason` and `note` in the last action_result and change the plan: "
                                    "`rest` when energy is low, `move` one tile when the path is "
                                    "blocked, `inspect` the tile when you are unsure."),
                    })
                    last_signature, stuck_repeats = None, 0
            elif runtime.state.get("objective", {}).get("is_complete"):
                print("goal completed")
                logger.info("objective completed")
                runtime.stop_reason = "objective_complete"
                break
            else:
                # A turn without a tool call used to end the episode, which reads as
                # "the agent ignored the task". Nudge it back, then give up.
                silent_steps += 1
                print(f"LLM did not call tools but goal is not complete (nudge {silent_steps}/2).")
                logger.info(f"LLM did not call tools but goal is not complete (nudge {silent_steps}/2)")
                if silent_steps >= 2:
                    logger.info("no tool call after 2 nudges - ending the episode")
                    runtime.stop_reason = "silent_steps"
                    break
                messages.append({
                    'role': 'user',
                    'content': ("You did not call a tool and the objective is still open: "
                                f"{task_text} Call exactly one tool now."),
                })

            if (runtime.state.get("objective", {}).get("is_complete")
                    or runtime.state.get("objective", {}).get("is_failed")):
                runtime.stop_reason = ("objective_complete"
                                       if runtime.state["objective"].get("is_complete")
                                       else "objective_failed")
                print("objective finished - ending the episode")
                logger.info(f"objective finished: {runtime.state['objective']['id']} "
                            f"complete={runtime.state['objective']['is_complete']} "
                            f"failed={runtime.state['objective']['is_failed']}")
                break
        if not runtime.stop_reason:
            # The loop ran out of steps with the goal still open: that is a result,
            # not a silent end - the summary has to say so.
            runtime.stop_reason = "max_steps"
            print(f"stopped at max_steps ({args.max_steps}) - the objective is still open")
            logger.info(f"stopped at max_steps ({args.max_steps}) "
                        f"objective={json.dumps(runtime.state.get('objective'))}")
    except Exception as e:
        runtime.stop_reason = "llm_error"
        logger.error(f"LLm failed due to error: {str(e)}")
        print(f"llm failed due to error: {e}")
        # The usual cause: the Ollama host in config.yaml is unreachable, so no
        # tool call was ever sent and the game looks untouched.
        print(f"model server: {ollama_url()} "
              f"(config.yaml server_urls.ollama_url) - is it up and is '{args.model}' pulled?")
        print("nothing was posted to the game after the last tool call above")
        logger.error(f"model server: {ollama_url()}")


def tool_call_id(tool_call, step, index):
    """The `id` the follow-up `tool` message has to quote (synthesised if absent)."""
    return getattr(tool_call, "id", None) or f"call_{step}_{index}"


def assistant_message(message, step):
    """The assistant turn as a plain dict - what an OpenAI request expects back."""
    echoed = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        echoed["tool_calls"] = [
            {
                "id": tool_call_id(tool_call, step, index),
                "type": "function",
                "function": {"name": tool_call.function.name,
                             "arguments": tool_call.function.arguments or "{}"},
            }
            for index, tool_call in enumerate(message.tool_calls)
        ]
    return echoed


def message_get(message, field):
    """`field` of an OpenAI message, including the extra (non-schema) ones."""
    value = getattr(message, field, None)
    if value is None:
        value = (getattr(message, "model_extra", None) or {}).get(field)
    return value


def chat_completion(client, model, messages, schemas, step, thinking=None):
    """One completion, with the template's `enable_thinking` switch when configured.

    The server renders the model's own chat template, so `enable_thinking` travels as
    a template kwarg instead of a request field - the same wire format for vLLM and
    for NVIDIA NIM. A model whose template has no such switch (gemma, ...) answers
    400, and that is not a failed step: the same prompt is resent without the kwarg.
    """
    request = {"model": model, "messages": messages, "tools": schemas}
    if thinking is not None:
        request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    try:
        return client.chat.completions.create(**request)
    except BadRequestError as exc:
        if "extra_body" not in request:
            raise
        logger.info(f"step {step}: chat_template_kwargs rejected ({exc}) - "
                    "retrying without it")
        request.pop("extra_body")
        return client.chat.completions.create(**request)


def run_agent_openai(args, schemas):
    """The loop of `run_agent`, against an OpenAI-compatible server.

    `args.backend` picks the block of config.yaml: `vllm` (the DGX proxy) or `nvidia`
    (the hosted NIM endpoint). Only the transport differs: both answer
    `/v1/chat/completions` with `choices[0].message`, want the assistant turn and every
    tool result echoed as plain dicts (`tool_calls` / `tool_call_id`), and report tokens
    in `usage` instead of Ollama's `prompt_eval_count` / `eval_count`. Prompt, tools,
    nudges and stop reasons are the same, so such an episode stays comparable with an
    Ollama one.
    """
    backend = getattr(args, "backend", None) or "vllm"
    base_url = openai_base_url(backend)
    if not base_url:
        example = ("https://integrate.api.nvidia.com/v1" if backend == "nvidia"
                   else "https://llm-proxy-dgx.skim.th-owl.de")
        print(f"--{backend} needs `{backend}.base_url` in config.yaml, e.g. {example}")
        logger.error(f"--{backend} without {backend}.base_url in config.yaml")
        runtime.stop_reason = "llm_error"
        return

    thinking = openai_enable_thinking(backend)

    # The prompt and the first user turn both carry the task; `--task` overrides
    # it, otherwise it is the objective this preset assigns (never empty, which is
    # what used to leave the model without a task to obey).
    task_text = resolve_task(args)
    system_message2 = build_system_message(runtime.state, schemas, task_text)
    messages = [
        {'role': 'system', 'content': system_message2},
        {'role': 'user', 'content': task_text},
    ]
    print(f"task: {task_text}")
    logger.info(f"task: {task_text}")
    logger.info(f"system message: {system_message2}")

    client = OpenAI(base_url=base_url, api_key=openai_api_key(backend),
                    timeout=args.llm_timeout)
    print(f"{backend}: {base_url} model={args.model} thinking={thinking}")
    logger.info(f"{backend}: {base_url} model={args.model} thinking={thinking}")

    silent_steps = 0  # consecutive model turns without a tool call
    stuck_repeats = 0  # consecutive identical tool calls
    last_signature = None
    try:
        i = 0
        while i < args.max_steps:
            i += 1
            # Kept at module level so `budget_to_text` can report what is left in the
            # observation the model reads after this step's action.
            runtime.STEPS_USED = i
            st_time = time.time()
            messages = trim_history(messages)
            # Counted before the request, like the Ollama loop: a step that dies on a
            # 429/timeout is still an LLM request, and the summary must not report 0.
            # The clock is closed in `finally` for the same reason: a failed request
            # has a latency too, and `len(llm_time_taken) == no_of_llm_calls` holds.
            runtime.llm_calls += 1
            try:
                response = chat_completion(client, args.model, messages, schemas, i, thinking)
            finally:
                runtime.llm_time_taken.append(time.time() - st_time)
            step_time = runtime.llm_time_taken[-1]
            message = response.choices[0].message

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            runtime.total_input_tokens += prompt_tokens
            runtime.total_output_tokens += completion_tokens
            print(f"input_tokens: {prompt_tokens}")
            print(f"output_tokens: {completion_tokens}")
            print(f"response time: {step_time:.2f}s")
            logger.info(f"step {i}: {prompt_tokens} input tokens, "
                        f"{completion_tokens} output tokens, {step_time:.2f}s")

            if message.content:
                print('Content: ')
                print(message.content + '\n')
                logger.info(f"content : {message.content}")

            # A thinking model ships its reasoning as an extra (non-schema) field,
            # which is why it is read through `message_get` and not as an attribute.
            reasoning = message_get(message, "reasoning_content")
            if reasoning:
                print('Thinking: ')
                print(reasoning + '\n')
                logger.info(f"thinking : {reasoning}")

            # The assistant turn goes back into the history as a dict: the SDK object
            # is not what the next request accepts, and its tool calls lose their
            # ids on the way - the `tool` messages below have to quote exactly them.
            messages.append(assistant_message(message, i))

            if message.tool_calls:
                silent_steps = 0
                print("llm tool_calls:")
                print(len(message.tool_calls))
                runtime.tool_calls_sent += len(message.tool_calls)
                # Every call is a request of its own, in the order the model asked for
                # them: the wrapper sends, book-keeps and answers in one step, so the
                # reply goes straight into the `tool` message below.
                answers = []
                for index, tool_call in enumerate(message.tool_calls):
                    # The LLM decides which wrapper to call, we never repair its plan.
                    name = tool_call.function.name
                    arguments = tool_arguments(tool_call)
                    call_id = tool_call_id(tool_call, i, index)
                    # A model that repeats one call forever burns the whole step
                    # budget: remember the call so the loop can nudge it once.
                    signature = (name, json.dumps(
                        arguments, sort_keys=True, default=str))
                    if signature == last_signature:
                        stuck_repeats += 1
                    else:
                        last_signature, stuck_repeats = signature, 1
                    function_to_call = available_tools.get(name)
                    logger.info(
                        f"tool call: {name} args={json.dumps(arguments)}")
                    if function_to_call:
                        try:
                            real_result_json = function_to_call(**arguments)
                        except TypeError as e:
                            runtime.invalid_moves += 1
                            key = "bad tool arguments"
                            runtime.invalid_move_object[key] = (
                                runtime.invalid_move_object.get(key, 0) + 1)
                            real_result_json = json.dumps({"status": "false", "action": name,
                                                           "message": f"Error in tool call: {str(e)}"})
                    else:
                        runtime.invalid_moves += 1
                        key = "unknown tool"
                        runtime.invalid_move_object[key] = (
                            runtime.invalid_move_object.get(key, 0) + 1)
                        print(f'Tool {name} not found')
                        real_result_json = json.dumps({"status": "false", "action": name,
                                                       "message": f'Tool {name} not found'})
                    answers.append((call_id, json.loads(real_result_json)))

                for call_id, reply in answers:
                    # `tool_call_id` is what ties the answer to the call above; a
                    # result without it is a 400 on the next request.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps({
                            "action_result": reply,
                            "current_state": state_to_text(runtime.state),
                        }),
                    })

                if stuck_repeats >= runtime.STUCK_REPEATS:
                    # Naming the repeat is the cheapest way out of a loop: the model
                    # sees the same arguments it just sent and a list of alternatives.
                    print(
                        f"stuck: {last_signature[0]} called {stuck_repeats}x with the same arguments")
                    logger.info(f"stuck: {last_signature[0]} called {stuck_repeats}x with the same "
                                f"arguments {last_signature[1]} - hinting the model")
                    messages.append({
                        'role': 'user',
                        'content': (f"You have called `{last_signature[0]}` with identical arguments "
                                    f"{stuck_repeats} times and the state did not change. Read "
                                    "`reason` and `note` in the last action_result and change the plan: "
                                    "`rest` when energy is low, `move` one tile when the path is "
                                    "blocked, `inspect` the tile when you are unsure."),
                    })
                    last_signature, stuck_repeats = None, 0
            elif runtime.state.get("objective", {}).get("is_complete"):
                print("goal completed")
                logger.info("objective completed")
                runtime.stop_reason = "objective_complete"
                break
            else:
                # A turn without a tool call used to end the episode, which reads as
                # "the agent ignored the task". Nudge it back, then give up.
                silent_steps += 1
                print(
                    f"LLM did not call tools but goal is not complete (nudge {silent_steps}/2).")
                logger.info(
                    f"LLM did not call tools but goal is not complete (nudge {silent_steps}/2)")
                if silent_steps >= 2:
                    logger.info(
                        "no tool call after 2 nudges - ending the episode")
                    runtime.stop_reason = "silent_steps"
                    break
                messages.append({
                    'role': 'user',
                    'content': ("You did not call a tool and the objective is still open: "
                                f"{task_text} Call exactly one tool now."),
                })

            if (runtime.state.get("objective", {}).get("is_complete")
                    or runtime.state.get("objective", {}).get("is_failed")):
                runtime.stop_reason = ("objective_complete"
                                       if runtime.state["objective"].get("is_complete")
                                       else "objective_failed")
                print("objective finished - ending the episode")
                logger.info(f"objective finished: {runtime.state['objective']['id']} "
                            f"complete={runtime.state['objective']['is_complete']} "
                            f"failed={runtime.state['objective']['is_failed']}")
                break
        if not runtime.stop_reason:
            # The loop ran out of steps with the goal still open: that is a result,
            # not a silent end - the summary has to say so.
            runtime.stop_reason = "max_steps"
            print(
                f"stopped at max_steps ({args.max_steps}) - the objective is still open")
            logger.info(f"stopped at max_steps ({args.max_steps}) "
                        f"objective={json.dumps(runtime.state.get('objective'))}")
    except Exception as e:
        runtime.stop_reason = "llm_error"
        logger.error(f"LLm failed due to error: {str(e)}")
        print(f"llm failed due to error: {e}")
        # Two usual causes: `<backend>.base_url` is unreachable or `<backend>.model`
        # is not served, or the deployment has no tool calling enabled (vLLM needs
        # `--enable-auto-tool-choice --tool-call-parser <name>`, and NIM only accepts
        # `tools` for models whose card lists tool calling). Either way no tool call
        # was ever sent, so the game looks untouched.
        print(f"model server: {base_url} (config.yaml {backend}.base_url) - is it up, "
              f"does /v1/models list '{args.model}', and does that deployment serve tools?")
        print("nothing was posted to the game after the last tool call above")
        logger.error(f"model server: {base_url} model={args.model}")


#: `--vllm` entry point kept by name (the experiments drive the loops directly). Both
#: OpenAI-compatible backends are the one loop above, told apart by `args.backend`.
def run_agent_vllm(args, schemas):
    return run_agent_openai(args, schemas)
