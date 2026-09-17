#!/usr/bin/env python3
"""Smoke-test the local text server with the official Python SDKs."""

import argparse
import json

try:
    from anthropic import Anthropic
    from openai import OpenAI
    from agents import Agent, ModelSettings, OpenAIProvider, RunConfig, Runner, function_tool
except ImportError as error:
    raise SystemExit(
        "Install the acceptance dependencies with: "
        "python -m pip install openai anthropic openai-agents"
    ) from error


def text(value: str | None, label: str) -> None:
    assert isinstance(value, str) and value.strip(), f"{label} returned no text"
    print(f"{label}: ok")


def function_schema(protocol: str) -> dict:
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    if protocol == "chat":
        return {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Look up weather for a city.",
                "parameters": schema,
                "strict": True,
            },
        }
    if protocol == "anthropic":
        return {
            "name": "lookup_weather",
            "description": "Look up weather for a city.",
            "input_schema": schema,
        }
    return {
        "type": "function",
        "name": "lookup_weather",
        "description": "Look up weather for a city.",
        "parameters": schema,
        "strict": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    root = args.base_url.rstrip("/")
    openai = OpenAI(
        api_key="local", base_url=f"{root}/v1", timeout=args.timeout, max_retries=0
    )
    anthropic = Anthropic(
        api_key="local", base_url=root, timeout=args.timeout, max_retries=0
    )

    chat = openai.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Answer exactly OK."}],
        max_tokens=32,
    )
    text(chat.choices[0].message.content, "chat JSON")
    chunks = list(
        openai.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Answer exactly OK."}],
            max_tokens=32,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    text(
        "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices),
        "chat SSE",
    )
    assert chunks[-1].usage is not None

    chat_tool = function_schema("chat")
    first = openai.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Use the tool for Hangzhou weather."}],
        tools=[chat_tool],
        tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
        max_tokens=128,
        temperature=0,
    )
    call = first.choices[0].message.tool_calls[0]
    assert json.loads(call.function.arguments)["city"]
    second = openai.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "user", "content": "Use the tool for Hangzhou weather."},
            first.choices[0].message.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": call.id, "content": "sunny"},
        ],
        tools=[chat_tool],
        tool_choice="none",
        max_tokens=64,
        temperature=0,
    )
    text(second.choices[0].message.content, "chat tool loop")

    message = anthropic.messages.create(
        model=args.model,
        messages=[{"role": "user", "content": "Answer exactly OK."}],
        max_tokens=32,
    )
    text(next(block.text for block in message.content if block.type == "text"), "Anthropic JSON")
    with anthropic.messages.stream(
        model=args.model,
        messages=[{"role": "user", "content": "Answer exactly OK."}],
        max_tokens=32,
    ) as stream:
        text(stream.get_final_text(), "Anthropic SSE")
    count = anthropic.messages.count_tokens(
        model=args.model, messages=[{"role": "user", "content": "Count me."}]
    )
    assert count.input_tokens > 0

    anthropic_tool = function_schema("anthropic")
    first = anthropic.messages.create(
        model=args.model,
        messages=[{"role": "user", "content": "Use the tool for Hangzhou weather."}],
        tools=[anthropic_tool],
        tool_choice={"type": "tool", "name": "lookup_weather"},
        max_tokens=128,
    )
    call = next(block for block in first.content if block.type == "tool_use")
    second = anthropic.messages.create(
        model=args.model,
        messages=[
            {"role": "user", "content": "Use the tool for Hangzhou weather."},
            {"role": "assistant", "content": [block.model_dump() for block in first.content]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call.id, "content": "sunny"}
                ],
            },
        ],
        max_tokens=64,
    )
    text(next(block.text for block in second.content if block.type == "text"), "Anthropic tool loop")

    response = openai.responses.create(
        model=args.model, input="Answer exactly OK.", max_output_tokens=32, temperature=0, store=True
    )
    text(response.output_text, "Responses JSON")
    continued = openai.responses.create(
        model=args.model,
        input="Repeat the previous answer.",
        previous_response_id=response.id,
        max_output_tokens=32,
        temperature=0,
        store=False,
    )
    text(continued.output_text, "Responses continuation")
    with openai.responses.stream(
        model=args.model, input="Answer exactly OK.", max_output_tokens=32, temperature=0, store=False
    ) as stream:
        for _ in stream:
            pass
        text(stream.get_final_response().output_text, "Responses SSE")
    incomplete = openai.responses.create(
        model=args.model,
        input="Write twenty numbered sentences.",
        max_output_tokens=1,
        temperature=0,
        store=False,
    )
    assert incomplete.status == "incomplete"
    assert incomplete.incomplete_details.reason == "max_output_tokens"

    response_tool = function_schema("responses")
    with openai.responses.stream(
        model=args.model,
        input="Use the tool for Hangzhou weather.",
        tools=[response_tool],
        tool_choice={"type": "function", "name": "lookup_weather"},
        max_output_tokens=128,
        temperature=0,
        store=True,
    ) as stream:
        for _ in stream:
            pass
        first = stream.get_final_response()
    call = next(item for item in first.output if item.type == "function_call")
    second = openai.responses.create(
        model=args.model,
        previous_response_id=first.id,
        input=[{"type": "function_call_output", "call_id": call.call_id, "output": "sunny"}],
        tool_choice="none",
        max_output_tokens=64,
        temperature=0,
        store=False,
    )
    text(second.output_text, "Responses tool loop")
    assert openai.responses.retrieve(response.id).id == response.id
    openai.responses.delete(response.id)

    calls = []

    @function_tool
    def lookup_weather(city: str) -> str:
        """Look up weather for a city."""
        calls.append(city)
        return "sunny"

    agent = Agent(
        name="local-weather",
        model=args.model,
        instructions="Call lookup_weather once, then answer the user with the result.",
        tools=[lookup_weather],
    )
    result = Runner.run_sync(
        agent,
        "What is the weather in Hangzhou?",
        max_turns=4,
        run_config=RunConfig(
            model_provider=OpenAIProvider(api_key="local", base_url=f"{root}/v1", use_responses=True),
            model_settings=ModelSettings(temperature=0, max_tokens=128, tool_choice="lookup_weather"),
            tracing_disabled=True,
        ),
    )
    assert calls
    text(str(result.final_output), "OpenAI Agents tool loop")
    print("all SDK checks passed")


if __name__ == "__main__":
    main()
