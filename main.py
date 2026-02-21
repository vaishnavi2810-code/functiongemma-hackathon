
import re
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import math
import json, os, time
import concurrent.futures
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_embed
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

#Hitika's approach
def _cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _precompute_tool_embeddings(tools: list) -> dict:
    """
    Embed each tool's name + description once at startup.
    Returns a dict of {tool_name: embedding_vector}.
    """
    model = cactus_init(functiongemma_path)
    tool_vecs = {}
    for tool in tools:
        tool_text = f"{tool['name'].replace('_', ' ')} {tool['description']}"
        tool_vecs[tool["name"]] = cactus_embed(model, tool_text, normalize=True)
    cactus_destroy(model)
    return tool_vecs


def _detect_needed_tools(query: str, tool_vecs: dict, top_n: int = 1) -> list:
    """
    Embed the query and rank tools by cosine similarity against precomputed vectors.

    Args:
        query:     Original-case user query.
        tool_vecs: Precomputed tool embeddings from _precompute_tool_embeddings.
        top_n:     Number of tools to return.

    Returns:
        List of tool names ranked by similarity, length = top_n.
    """
    model = cactus_init(functiongemma_path)
    query_vec = cactus_embed(model, query, normalize=True)
    cactus_destroy(model)

    scored = [
        (_cosine_sim(query_vec, tool_vec), tool_name)
        for tool_name, tool_vec in tool_vecs.items()
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in scored[:top_n]]

#────────── Tool embedding cache — populated at first use ─────────────────────────────
_TOOL_VECS: dict = {}

def _fix_negatives(calls: list):
    """Fix negative numeric arguments in-place."""
    for call in calls:
        for k, v in call.get("arguments", {}).items():
            if isinstance(v, (int, float)) and v < 0:
                call["arguments"][k] = abs(v)


def generate_hybrid(messages, tools, confidence_threshold=0.85):
    """
    Strategy:
    1. Run FunctionGemma locally
    2. Validate result (confidence, empty calls, negatives, multi-tool)
    3. If empty calls: try regex-based construction before cloud
    4. Cloud fallback as last resort
    """
    global _TOOL_VECS

    # Precompute tool embeddings once per unique tool set
    tool_key = tuple(t["name"] for t in tools)
    if tool_key not in _TOOL_VECS:
        _TOOL_VECS[tool_key] = _precompute_tool_embeddings(tools)
    tool_vecs = _TOOL_VECS[tool_key]

    user_query = ""
    for m in messages:
        if m["role"] == "user":
            user_query = m["content"]

    query_lower = user_query.lower()
    tool_names = {t["name"] for t in tools}

    # Detect multi-tool — consistent threshold: 1+ comma OR conjunction keyword
    multi_keywords = [" and ", " also ", " then ", " plus "]
    likely_multi = any(kw in f" {query_lower} " for kw in multi_keywords)
    if query_lower.count(",") >= 1:
        likely_multi = True

    # =========================================================
    # STEP 1: Run FunctionGemma
    # =========================================================
    local = generate_cactus(messages, tools)
    calls = local.get("function_calls", [])
    print(f"\n[DEBUG] Query: {user_query}")
    print(f"[DEBUG] Confidence: {local.get('confidence', 0)}")
    print(f"[DEBUG] FunctionGemma calls: {json.dumps(calls, indent=2)}")
    print(f"[DEBUG] Tools available: {[t['name'] for t in tools]}")
    print(f"[DEBUG] Likely multi: {likely_multi}")
    # =========================================================
    # STEP 2: Confidence check
    # =========================================================
    if local.get("confidence", 0) < confidence_threshold:
        if not calls:
            constructed = _try_construct_calls(query_lower, tool_names, likely_multi)
            if constructed:
                local["function_calls"] = constructed
                local["source"] = "on-device"
                return local
        return _cloud_fallback(messages, tools, local)

    # =========================================================
    # STEP 3: Empty calls — try regex construction
    # =========================================================
    if not calls:
        constructed = _try_construct_calls(query_lower, tool_names, likely_multi)
        if constructed:
            local["function_calls"] = constructed
            local["source"] = "on-device"
            return local
        return _cloud_fallback(messages, tools, local)

    # =========================================================
    # STEP 4: Fix negative numbers
    # =========================================================
    _fix_negatives(calls)

    # =========================================================
    # STEP 5: Multi-tool check — embed detection + per-tool loop
    # =========================================================
    if likely_multi and len(calls) < 2:

        # Count conjunctions to estimate how many tools are needed
        conjunction_count = query_lower.count(" and ") + query_lower.count(",")
        top_n = conjunction_count + 1

        # Rank tools by semantic similarity against precomputed vectors
        needed_tool_names = _detect_needed_tools(user_query, tool_vecs, top_n=top_n)
        print(f"[DEBUG-EMBED] Needed tools (top {top_n}): {needed_tool_names}")
        if not needed_tool_names:
            return _cloud_fallback(messages, tools, local)

        tool_index = {t["name"]: t for t in tools}
        all_calls = []
        total_time = local.get("total_time_ms", 0)
        min_confidence = local.get("confidence", 0)

        for tool_name in needed_tool_names:
            tool = tool_index.get(tool_name)
            if not tool:
                continue

            result = generate_cactus(messages, [tool])
            print(f"[DEBUG-EMBED] Tool {tool_name}: conf={result.get('confidence', 0)}, calls={result.get('function_calls', [])}")
            total_time += result.get("total_time_ms", 0)
            conf = result.get("confidence", 0)
            min_confidence = min(min_confidence, conf)

            if conf < confidence_threshold or not result.get("function_calls"):
                return _cloud_fallback(messages, tools, {
                    "function_calls": [],
                    "total_time_ms": total_time,
                    "confidence": min_confidence,
                })

            _fix_negatives(result["function_calls"])
            all_calls.extend(result["function_calls"])

        local["function_calls"] = all_calls
        local["total_time_ms"] = total_time
        local["confidence"] = min_confidence
        local["source"] = "on-device"
        return local

    # =========================================================
    # STEP 6: All checks passed
    # =========================================================
    local["source"] = "on-device"
    return local


def _cloud_fallback(messages, tools, local):
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] += local.get("total_time_ms", 0)
    return cloud


def _try_construct_calls(query, tool_names, likely_multi):
    """
    Try to construct function calls from the query using regex.
    Returns a list of call dicts, or None if we can't parse it.
    """
    calls = []

    # --- get_weather ---
    if "get_weather" in tool_names:
        # Match "weather in {location}"
        match = re.search(r'weather\s+(?:like\s+)?(?:in|for)\s+(.+?)(?:\.|,|\?|!|$| and )', query)
        if not match:
            match = re.search(r'(?:check|get)\s+(?:the\s+)?weather\s+(?:in|for)\s+(.+?)(?:\.|,|\?|!|$| and )', query)
        if match:
            location = match.group(1).strip().rstrip('.,?!')
            # Title case for city names
            location = ' '.join(w.capitalize() for w in location.split())
            calls.append({"name": "get_weather", "arguments": {"location": location}})

    # --- set_alarm ---
    if "set_alarm" in tool_names:
        # Match time like "6 AM", "10 AM", "7:30 AM", "8:15 AM"
        match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', query)
        if match and any(w in query for w in ["alarm", "wake"]):
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            if match.group(3) == "pm" and hour != 12:
                hour += 12
            if match.group(3) == "am" and hour == 12:
                hour = 0
            calls.append({"name": "set_alarm", "arguments": {"hour": hour, "minute": minute}})

    # --- set_timer ---
    if "set_timer" in tool_names:
        match = re.search(r'(\d+)\s*(?:-?\s*)?minute', query)
        if match and "timer" in query:
            minutes = int(match.group(1))
            calls.append({"name": "set_timer", "arguments": {"minutes": minutes}})

    # --- play_music ---
    if "play_music" in tool_names:
        match = re.search(r'play\s+(.+?)(?:\.|,|!|$| and )', query)
        if match:
            song = match.group(1).strip().rstrip('.,!?')
            # Clean up common prefixes
            song = re.sub(r'^(?:some|the)\s+', '', song)
            calls.append({"name": "play_music", "arguments": {"song": song}})

    # --- send_message ---
    if "send_message" in tool_names:
        # "message to X saying Y" or "text X saying Y"
        match = re.search(r'(?:message|text)\s+(?:to\s+)?(\w+)\s+saying\s+(.+?)(?:\.|,|!|$| and )', query)
        if not match:
            # "send him/her a message saying Y" — need to find name elsewhere
            match = re.search(r'(?:send|text)\s+(\w+)\s+(?:a\s+)?(?:message\s+)?saying\s+(.+?)(?:\.|,|!|$| and )', query)
        if match:
            recipient = match.group(1).strip().capitalize()
            message_text = match.group(2).strip().rstrip('.,!?')
            if recipient.lower() not in ["a", "the", "me", "him", "her"]:
                calls.append({"name": "send_message", "arguments": {"recipient": recipient, "message": message_text}})

    # --- create_reminder ---
    if "create_reminder" in tool_names:
        match = re.search(r'remind\s+(?:me\s+)?(?:about\s+|to\s+)?(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))', query)
        if match:
            title = match.group(1).strip()
            time_str = match.group(2).strip()
            # Normalize time format: "3:00 pm" -> "3:00 PM"
            time_str = time_str.upper()
            if ':' not in time_str:
                time_str = re.sub(r'(\d+)\s*(AM|PM)', r'\1:00 \2', time_str)
            else:
                time_str = re.sub(r'(\d+:\d+)\s*(AM|PM)', r'\1 \2', time_str)
            calls.append({"name": "create_reminder", "arguments": {"title": title, "time": time_str}})

    # --- search_contacts ---
    if "search_contacts" in tool_names:
        match = re.search(r'(?:find|look\s*up|search\s*for)\s+(\w+)', query)
        if match and "contact" in query:
            name = match.group(1).strip().capitalize()
            calls.append({"name": "search_contacts", "arguments": {"query": name}})

    # Return None if we couldn't construct anything
    if not calls:
        print(f"[DEBUG-REGEX] No calls constructed")
        return None

    print(f"[DEBUG-REGEX] Constructed: {json.dumps(calls, indent=2)}")

    # For single-tool queries, return what we found
    if not likely_multi:
        return [calls[0]] if calls else None

    # For multi-tool queries, only return if we got 2+
    if len(calls) >= 2:
        return calls
    print(f"[DEBUG-REGEX] Multi-tool but only {len(calls)} call, returning None")
    return None

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
