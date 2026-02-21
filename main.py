
import re
import sys
import spacy

# Load spaCy model once at module level
_nlp = spacy.load("en_core_web_md")
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
        if candidate.content and candidate.content.parts:    # ← added safety check
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

def generate_hybrid(messages, tools, confidence_threshold=0.93):
    """
    Three-layer hybrid routing (fully generic, no hardcoded patterns):
    1. FunctionGemma with all tools (fast, works when confident)
    2. Smart routing: score tools → FunctionGemma with 1 tool each
    3. Gemini Cloud fallback (handles complex/unknown queries)
    """

    user_query = ""
    for m in messages:
        if m["role"] == "user":
            user_query = m["content"]

    query_lower = user_query.lower()

    # Detect multi-tool
    multi_keywords = [" and ", " also ", " then ", " plus "]
    likely_multi = any(kw in f" {query_lower} " for kw in multi_keywords)
    if query_lower.count(",") >= 2:
        likely_multi = True

    # =========================================================
    # LAYER 1: FunctionGemma with ALL tools
    # =========================================================
    local = generate_cactus(messages, tools)
    calls = local.get("function_calls", [])

    if calls:
        _fix_negatives(calls)

    layer1_ok = (
        local.get("confidence", 0) >= confidence_threshold
        and len(calls) > 0
        and (not likely_multi or len(calls) >= 2)
    )

    if layer1_ok:
        local["source"] = "on-device"
        return local

    # =========================================================
    # LAYER 2: Smart routing → FunctionGemma with 1 tool each
    # =========================================================

    # How many tools do we need?
    if likely_multi:
        and_count = query_lower.count(" and ")
        comma_count = query_lower.count(",")
        expected_count = max(and_count + 1, comma_count + 1) if (and_count or comma_count) else 2
    else:
        expected_count = 1

    # Score all tools
    scored_tools = []
    for tool in tools:
        score = _score_tool(query_lower, tool)
        if score > 0:
            scored_tools.append((score, tool))

    scored_tools.sort(key=lambda x: x[0], reverse=True)

    if scored_tools:
        min_score = max(3, scored_tools[0][0] * 0.3)
        selected_tools = []
        for score, tool in scored_tools:
            if score >= min_score and len(selected_tools) < expected_count:
                selected_tools.append(tool)

        if selected_tools:
            all_calls = []
            total_time = local.get("total_time_ms", 0)

            for tool in selected_tools:
                result = generate_cactus(messages, [tool])
                total_time += result.get("total_time_ms", 0)
                result_calls = result.get("function_calls", [])

                if result_calls:
                    _fix_negatives(result_calls)
                    all_calls.extend(result_calls)

            if all_calls and (not likely_multi or len(all_calls) >= 2):
                return {
                    "function_calls": all_calls,
                    "total_time_ms": total_time,
                    "confidence": local.get("confidence", 0),
                    "source": "on-device",
                }

    # =========================================================
    # LAYER 3: Cloud fallback
    # =========================================================
    return _cloud_fallback(messages, tools, local)


def _cloud_fallback(messages, tools, local):
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] += local.get("total_time_ms", 0)
    return cloud


def _fix_negatives(calls):
    for call in calls:
        args = call.get("arguments", {})
        for key in args:
            if isinstance(args[key], (int, float)) and args[key] < 0:
                args[key] = abs(args[key])


def _score_tool(query_lower, tool):
    """Score how well a query matches a tool. Reads tool definitions only."""
    score = 0
    tool_name = tool["name"]
    tool_desc = tool.get("description", "").lower()

    # Signal 1: Tool name keywords (weight: 2)
    name_keywords = tool_name.lower().replace("_", " ").split()
    for kw in name_keywords:
        if len(kw) > 2 and kw in query_lower:
            score += 2

    # Signal 2: Description keyword overlap (weight: 1)
    stop_words = {"a", "an", "the", "for", "to", "of", "is", "in", "at",
                  "by", "or", "and", "with", "get", "set", "create"}
    desc_keywords = [w for w in tool_desc.split() if w not in stop_words and len(w) > 2]
    for kw in desc_keywords:
        if kw in query_lower:
            score += 1

    # Signal 3: Action verb synonyms (weight: 3)
    action_synonyms = _get_action_synonyms(tool_name, tool_desc)
    query_words = query_lower.split()
    primary_words = query_words[:5]
    for synonym in action_synonyms:
        if synonym in primary_words:
            score += 3
            break

    return score


def _get_action_synonyms(tool_name, tool_desc):
    """Derive action verbs from tool name/description. Concept-based."""
    synonyms = set()
    for part in tool_name.lower().replace("_", " ").split():
        synonyms.add(part)

    synonym_map = {
        "alarm": ["wake", "alarm"],
        "timer": ["timer", "countdown"],
        "remind": ["remind", "reminder"],
        "message": ["message", "text", "send"],
        "search": ["search", "find", "look", "lookup", "show"],
        "play": ["play", "listen"],
        "weather": ["weather", "forecast", "temperature", "rain", "pack"],
        "call": ["call", "dial", "phone"],
        "book": ["book", "reserve", "reservation"],
        "create": ["create", "make", "add", "new"],
        "delete": ["delete", "remove", "cancel"],
        "set": ["set"],
        "get": ["get", "check", "fetch", "show", "find"],
        "translate": ["translate", "say"],
        "convert": ["convert", "much", "exchange"],
        "direction": ["direction", "directions", "route", "navigate"],
        "emergency": ["emergency", "nearest", "hospital", "police", "pharmacy", "need"],
        "hotel": ["hotel", "stay", "accommodation", "hostel", "place"],
        "flight": ["flight", "fly", "flying", "flights", "airline"],
        "activity": ["activity", "activities", "tour", "experience", "stuff", "things"],
        "currency": ["currency", "dollars", "rupiah", "bucks", "money"],
    }

    combined = tool_name.lower() + " " + tool_desc
    for key, syns in synonym_map.items():
        if key in combined:
            synonyms.update(syns)

    return synonyms

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
