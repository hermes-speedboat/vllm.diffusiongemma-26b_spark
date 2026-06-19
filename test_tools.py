"""Tool-calling test for DiffusionGemma 26B A4B-it."""
import json
import os
import time

import requests

PORT = os.environ.get("PORT", "8001")
HOST = os.environ.get("HOST", "127.0.0.1")
BASE_URL = f"http://{HOST}:{PORT}/v1"
MODEL = "google/diffusiongemma-26B-A4B-it"


def chat_with_tools(messages, tools=None):
    """Send a chat completion with tool definitions."""
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.7,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    resp = requests.post(f"{BASE_URL}/chat/completions", json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main():
    print(f"=== DiffusionGemma Tool-Calling Test at {BASE_URL} ===")
    print()

    # Define a simple weather tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country, e.g. 'Zurich, Switzerland'",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What's the weather like in Zurich?"},
    ]

    print("--- Request ---")
    print(f"Messages: {messages}")
    print(f"Tools: {[t['function']['name'] for t in tools]}")
    print()

    t0 = time.time()
    result = chat_with_tools(messages, tools)
    elapsed = time.time() - t0

    choice = result["choices"][0]
    msg = choice["message"]

    print(f"--- Response ({elapsed:.1f}s) ---")
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc["function"]
            args = json.loads(fn["arguments"])
            print(f" Tool call: {fn['name']}({json.dumps(args)})")
            print(f"   (would call with location={args.get('location')})")
    elif msg.get("content"):
        print(f" Text response: {msg['content'][:200]}")
    else:
        print(f" Full message: {json.dumps(msg, indent=2)}")

    print()
    print("=== Done ===")


if __name__ == "__main__":
    main()