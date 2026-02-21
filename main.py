
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
import concurrent.futures
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }

#Hitika idea due to the Paris query
def generate_hybrid(messages, tools, confidence_threshold=0.85):
    """Threshold 0.99 + validation + fix negatives locally."""

    local = generate_cactus(messages, tools)
    user_query = messages[-1]["content"]
    print(f"\n[DEBUG] Query: {user_query}")
    print(f"[DEBUG] Confidence: {local['confidence']}")
    print(f"[DEBUG] Function calls: {json.dumps(local['function_calls'], indent=2)}")
    print(f"[DEBUG] Tools available: {[t['name'] for t in tools]}")
    calls = local.get("function_calls", [])

    # --- Baseline confidence check ---
    if local["confidence"] < confidence_threshold:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local["confidence"]
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    # --- Validation 1: No empty function calls ---
    if not calls:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local["confidence"]
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    # --- Fix: Convert negative numeric values to positive ---
    # FunctionGemma sometimes negates numbers. The right function
    # and right magnitude, just wrong sign. Fix it locally.
    for call in calls:
        args = call.get("arguments", {})
        for key in args:
            if isinstance(args[key], (int, float)) and args[key] < 0:
                args[key] = abs(args[key])

    # --- Validation 2: Multi-tool queries need multiple calls ---
    user_query = ""
    for m in messages:
        if m["role"] == "user":
            user_query = m["content"].lower()

    multi_keywords = [" and ", " also ", " then ", " plus "]
    likely_multi = any(kw in f" {user_query} " for kw in multi_keywords)

    if likely_multi and len(calls) < 2:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local["confidence"]
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    # --- All checks passed, trust local ---
    local["source"] = "on-device"
    return local

# Original GENERATE HYBRID function
# def generate_hybrid(messages, tools, confidence_threshold=0.99):
#     """Baseline hybrid inference strategy; fall back to cloud if Cactus Confidence is below threshold."""
#         local = generate_cactus(messages, tools)
#
#         if local["confidence"] >= confidence_threshold:
#             local["source"] = "on-device"
#             return local
#
#         cloud = generate_cloud(messages, tools)
#         cloud["source"] = "cloud (fallback)"
#         cloud["local_confidence"] = local["confidence"]
#         cloud["total_time_ms"] += local["total_time_ms"]
#         return cloud



def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
