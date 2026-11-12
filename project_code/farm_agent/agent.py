"""The agentic loop - same shape for both backends (Ollama and vLLM)."""

import json
import logging
import time

from ollama import Client
from ollama._types import ChatResponse
from openai import BadRequestError, OpenAI

from . import runtime
from .config import ollama_url, vllm_api_key, vllm_base_url, vllm_enable_thinking
from .episode import resolve_task
from .prompt import build_system_message, tool_arguments
from .rendering import state_to_text
from .tools import available_tools

logger = logging.getLogger(__name__)


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
            st_time = time.time()
            response: ChatResponse = client.chat(model=args.model, messages=messages, tools=schemas)

        
            print(f"input_tokens: {response['prompt_eval_count']}")
            print("Prompt evaluation time:", response["prompt_eval_duration"] / 1e9, "seconds")
            logger.info(f"Prompt evaluation time:{response['prompt_eval_duration'] / 1e9:.2f}")
            print("Generation time:", response["eval_duration"] / 1e9, "seconds")
            logger.info(f"Generation time: {response['eval_duration'] / 1e9:.2f}")
            print(f"output_tokens: {response['eval_count']}")
            print(f"response time: {(response['total_duration'] / 1e9)}")
            logger.info(f"step {i} response time: {(response['total_duration'] / 1e9):.2f}s")

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
                        messages.append({'role': 'tool', 'content': f'Tool {name} not found',
                                         'tool_name': name})
                        continue

                    print(f"time for tool {name}: {str(time.time() - st_time)}")
                    if json.loads(real_result_json).get("status") == "true":
                        runtime.tool_calls.append(name)
                        runtime.time_taken.append(time.time() - st_time)
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({
                            "action_result": json.loads(real_result_json),
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


#: History cap of the vLLM loop: the system prompt, the task and this many recent
#: messages. A step appends one assistant turn plus one tool result and every tool
#: result repeats the state block, so a 60-step run outgrows the context window of
#: the server mid-episode - and then every following step answers 400.
MAX_HISTORY_MESSAGES = 42


def trim_history(messages):
    """Drop the oldest turns, never splitting a `tool_calls` turn from its results.

    vLLM rejects a `tool` message that answers no preceding `tool_calls`, so a
    window that starts in the middle of a step loses those orphan results with it.
    The world is not lost with them: every tool result carries `current_state`.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    tail = messages[-(MAX_HISTORY_MESSAGES - 2):]
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]
    return messages[:2] + tail


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


def chat_completion(client, model, messages, schemas, step):
    """One completion, with the Qwen3 `enable_thinking` switch when configured.

    vLLM renders the model's own chat template, so `enable_thinking` travels as a
    template kwarg instead of a request field. A model whose template has no such
    switch (gemma, ...) answers 400, and that is not a failed step: the same prompt
    is resent without the kwarg.
    """
    request = {"model": model, "messages": messages, "tools": schemas}
    thinking = vllm_enable_thinking()
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


def run_agent_vllm(args, schemas):
    """The loop of `run_agent`, against an OpenAI-compatible server (`--vllm`).

    Only the transport differs: vLLM answers `/v1/chat/completions` with
    `choices[0].message`, wants the assistant turn and every tool result echoed as
    plain dicts (`tool_calls` / `tool_call_id`), and reports tokens in `usage`
    instead of Ollama's `prompt_eval_count` / `eval_count`. Prompt, tools, nudges
    and stop reasons are the same, so a vLLM episode stays comparable with an
    Ollama one.
    """
    if not vllm_base_url():
        print("--vllm needs `vllm.base_url` in config.yaml, e.g. "
              "https://llm-proxy-dgx.skim.th-owl.de")
        logger.error("--vllm without vllm.base_url in config.yaml")
        runtime.stop_reason = "llm_error"
        return

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

    client = OpenAI(base_url=vllm_base_url(), api_key=vllm_api_key(),
                    timeout=args.llm_timeout)
    print(f"vllm: {vllm_base_url()} model={args.model} "
          f"thinking={vllm_enable_thinking()}")
    logger.info(f"vllm: {vllm_base_url()} model={args.model} "
                f"thinking={vllm_enable_thinking()}")

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
            response = chat_completion(client, args.model, messages, schemas, i)
            step_time = time.time() - st_time
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
                        messages.append({'role': 'tool', 'tool_call_id': call_id,
                                         'content': f'Tool {name} not found'})
                        continue

                    print(
                        f"time for tool {name}: {str(time.time() - st_time)}")
                    if json.loads(real_result_json).get("status") == "true":
                        runtime.tool_calls.append(name)
                        runtime.time_taken.append(time.time() - st_time)
                    # `tool_call_id` is what ties the answer to the call above; a
                    # result without it is a 400 on the next request.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps({
                            "action_result": json.loads(real_result_json),
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
        # Two usual causes: `vllm.base_url` is unreachable or `vllm.model` is not
        # served, or the deployment has no tool calling enabled (vLLM needs
        # `--enable-auto-tool-choice --tool-call-parser <name>`, and the proxy then
        # answers 400 "auto tool choice requires ..."). Either way no tool call was
        # ever sent, so the game looks untouched.
        print(f"model server: {vllm_base_url()} (config.yaml vllm.base_url) - is it up, "
              f"does /v1/models list '{args.model}', and does that deployment serve tools?")
        print("nothing was posted to the game after the last tool call above")
        logger.error(f"model server: {vllm_base_url()} model={args.model}")
