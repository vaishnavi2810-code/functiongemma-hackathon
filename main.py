
import re
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

def generate_hybrid(messages, tools, confidence_threshold=0.85):
    """
    Three-layer fallback:
    1. FunctionGemma (AI, ~200ms)
    2. Generic rule-based NLU (scoring + extraction, ~0ms)
    3. Gemini Cloud (AI, ~1000ms)
    """

    user_query = ""
    for m in messages:
        if m["role"] == "user":
            user_query = m["content"]

    query_lower = user_query.lower()
    tool_names = {t["name"] for t in tools}

    # Detect multi-tool
    multi_keywords = [" and ", " also ", " then ", " plus "]
    likely_multi = any(kw in f" {query_lower} " for kw in multi_keywords)
    if query_lower.count(",") >= 2:
        likely_multi = True

    # =========================================================
    # LAYER 1: FunctionGemma
    # =========================================================
    local = generate_cactus(messages, tools)
    calls = local.get("function_calls", [])

    # Confidence check
    if local.get("confidence", 0) < confidence_threshold:
        if not calls:
            constructed = _generic_construct(query_lower, user_query, tools, likely_multi)
            if constructed:
                local["function_calls"] = constructed
                local["source"] = "on-device"
                return local
        return _cloud_fallback(messages, tools, local)

    # Empty calls — try generic construction
    if not calls:
        constructed = _generic_construct(query_lower, user_query, tools, likely_multi)
        if constructed:
            local["function_calls"] = constructed
            local["source"] = "on-device"
            return local
        return _cloud_fallback(messages, tools, local)

    # Fix negative numbers
    for call in calls:
        args = call.get("arguments", {})
        for key in args:
            if isinstance(args[key], (int, float)) and args[key] < 0:
                args[key] = abs(args[key])

    # Multi-tool check
    if likely_multi and len(calls) < 2:
        constructed = _generic_construct(query_lower, user_query, tools, likely_multi)
        if constructed and len(constructed) >= 2:
            local["function_calls"] = constructed
            local["source"] = "on-device"
            return local
        return _cloud_fallback(messages, tools, local)

    local["source"] = "on-device"
    return local


def _cloud_fallback(messages, tools, local):
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] += local.get("total_time_ms", 0)
    return cloud


# =============================================================
# GENERIC TOOL MATCHING + PARAMETER EXTRACTION
# =============================================================

def _generic_construct(query_lower, original_query, tools, likely_multi):
    """
    Score each tool, pick the best match(es), extract parameters generically.
    All logic is driven by tool definitions — no hardcoded tool names.
    """
    # Score all tools
    scored_tools = []
    for tool in tools:
        score, extracted_args = _score_and_extract(query_lower, original_query, tool)
        if score > 0 and extracted_args is not None:
            scored_tools.append((score, tool, extracted_args))

    if not scored_tools:
        return None

    # Sort by score descending
    scored_tools.sort(key=lambda x: x[0], reverse=True)

    if likely_multi:
        # Return all tools that scored > 0 and have filled params
        calls = []
        used_tools = set()
        for score, tool, args in scored_tools:
            if tool["name"] not in used_tools:
                calls.append({"name": tool["name"], "arguments": args})
                used_tools.add(tool["name"])
        return calls if len(calls) >= 2 else None
    else:
        # Single tool — return highest scoring
        best_score, best_tool, best_args = scored_tools[0]
        return [{"name": best_tool["name"], "arguments": best_args}]


def _score_and_extract(query_lower, original_query, tool):
    """
    Score how well a query matches a tool, and extract parameter values.
    Returns (score, extracted_args) or (0, None) if no match.
    """
    score = 0
    tool_name = tool["name"]
    tool_desc = tool.get("description", "").lower()
    properties = tool.get("parameters", {}).get("properties", {})
    required = tool.get("parameters", {}).get("required", [])

    # --- Signal 1: Tool name keyword match (weight: 2 per keyword) ---
    # Split tool name like "set_alarm" → ["set", "alarm"]
    name_keywords = tool_name.lower().replace("_", " ").split()
    for kw in name_keywords:
        if kw in query_lower:
            score += 2

    # --- Signal 2: Description keyword overlap (weight: 1 per keyword) ---
    # Filter out generic words
    stop_words = {"a", "an", "the", "for", "to", "of", "is", "in", "at", "by", "or", "and", "with", "get", "set", "create"}
    desc_keywords = [w for w in tool_desc.split() if w not in stop_words and len(w) > 2]
    for kw in desc_keywords:
        if kw in query_lower:
            score += 1

    # --- Signal 3: Synonym/action verb expansion (weight: 3) ---
    # Derive related verbs from tool name and description
    action_synonyms = _get_action_synonyms(tool_name, tool_desc)
    # Check if the query's PRIMARY verb (first few words) matches
    query_words = query_lower.split()
    primary_words = query_words[:4]  # first 4 words usually contain the main verb
    for synonym in action_synonyms:
        if synonym in primary_words:
            score += 3
            break

    # --- Signal 4: Parameter fillability (weight: 2 per filled required param) ---
    extracted_args = {}
    for param_name, param_info in properties.items():
        value = _extract_param_value(
            query_lower, original_query, param_name,
            param_info.get("type", "string"),
            param_info.get("description", ""),
            tool_name, tool_desc
        )
        if value is not None:
            extracted_args[param_name] = value

    filled_required = sum(1 for r in required if r in extracted_args)
    score += filled_required * 2

    # If we can't fill ALL required params, this tool isn't viable
    if filled_required < len(required):
        return (0, None)

    return (score, extracted_args)


def _get_action_synonyms(tool_name, tool_desc):
    """
    Derive action verbs/synonyms from the tool name and description.
    These are words a user might say to trigger this tool.
    """
    synonyms = set()

    # From tool name: "set_alarm" → "set", "alarm"
    for part in tool_name.lower().replace("_", " ").split():
        synonyms.add(part)

    # Common synonym expansions based on description keywords
    # This is the only place with "semi-hardcoded" knowledge,
    # but it maps CONCEPTS not specific tools
    synonym_map = {
        "alarm": ["wake", "alarm"],
        "timer": ["timer", "countdown"],
        "remind": ["remind", "reminder"],
        "message": ["message", "text", "send"],
        "search": ["search", "find", "look", "lookup"],
        "play": ["play", "listen"],
        "weather": ["weather", "forecast", "temperature"],
        "call": ["call", "dial", "phone"],
        "book": ["book", "reserve", "reservation"],
        "create": ["create", "make", "add", "new"],
        "delete": ["delete", "remove", "cancel"],
        "set": ["set"],
        "get": ["get", "check", "fetch", "show"],
    }

    combined = tool_name.lower() + " " + tool_desc
    for key, syns in synonym_map.items():
        if key in combined:
            synonyms.update(syns)

    return synonyms


def _extract_param_value(query_lower, original_query, param_name, param_type, param_desc, tool_name, tool_desc):
    """
    Extract a parameter value from the query based on its type and description.
    Generic — works for any tool definition.
    """
    param_desc_lower = param_desc.lower()
    param_name_lower = param_name.lower()

    # =========================
    # INTEGER PARAMETERS
    # =========================
    if param_type in ("integer", "number"):

        # Hour extraction — look for time patterns
        if "hour" in param_name_lower:
            match = re.search(r'(\d{1,2})(?::\d{2})?\s*(am|pm)', query_lower)
            if match:
                hour = int(match.group(1))
                period = match.group(2)
                if period == "pm" and hour != 12:
                    hour += 12
                if period == "am" and hour == 12:
                    hour = 0
                return hour

        # Minute extraction — from time pattern (X:MM) or duration (X minutes)
        if "minute" in param_name_lower:
            # First check for time pattern like "8:15 AM"
            match = re.search(r'\d{1,2}:(\d{2})\s*(?:am|pm)', query_lower)
            if match:
                return int(match.group(1))
            # Then check for duration like "15 minutes"
            match = re.search(r'(\d+)\s*minute', query_lower)
            if match:
                return int(match.group(1))
            # If hour was found but no minute specified, default to 0
            if re.search(r'\d{1,2}\s*(?:am|pm)', query_lower):
                return 0

        # Generic number — "number of minutes", "count", etc.
        if "minute" in param_desc_lower or "duration" in param_desc_lower:
            match = re.search(r'(\d+)\s*minute', query_lower)
            if match:
                return int(match.group(1))

        # Fallback: find any number
        match = re.search(r'(\d+)', query_lower)
        if match:
            return int(match.group(1))

    # =========================
    # STRING PARAMETERS
    # =========================
    if param_type == "string":

        # --- Location/city/place ---
        if any(kw in param_desc_lower for kw in ["city", "location", "place", "where"]):
            # "weather in Paris" / "weather for London"
            match = re.search(r'(?:in|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', original_query)
            if match:
                return match.group(1).strip()
            # Fallback: look for capitalized words after "in"
            match = re.search(r'in\s+(.+?)(?:\.|,|\?|!|$| and )', query_lower)
            if match:
                location = match.group(1).strip().rstrip('.,?!')
                return ' '.join(w.capitalize() for w in location.split())

        # --- Person/recipient/contact name ---
        if any(kw in param_desc_lower for kw in ["name", "person", "recipient", "contact", "who"]):
            # Look for capitalized proper nouns in original query
            # Skip words that start sentences or are common words
            skip_words = {"set", "send", "play", "find", "look", "get", "check",
                         "remind", "text", "what", "how", "the", "and", "i'll",
                         "wake", "me", "my", "at", "for", "in", "a", "an"}

            # Try "to {Name}" pattern first
            match = re.search(r'(?:to|text|message)\s+([A-Z][a-z]+)', original_query)
            if match and match.group(1).lower() not in skip_words:
                return match.group(1)

            # Try "find/look up {Name}" pattern
            match = re.search(r'(?:find|look\s*up|search\s*for)\s+([A-Z][a-z]+)', original_query)
            if match and match.group(1).lower() not in skip_words:
                return match.group(1)

            # Fallback: any capitalized word that's not a city or common word
            for word in original_query.split():
                clean = word.strip('.,!?')
                if clean and clean[0].isupper() and clean.lower() not in skip_words:
                    # Skip if it looks like start of sentence
                    idx = original_query.index(clean)
                    if idx > 0:
                        return clean

        # --- Message content ---
        if any(kw in param_desc_lower for kw in ["message", "content", "body", "text"]):
            match = re.search(r'saying\s+(.+?)(?:\.|!|$|,\s*(?:and|check|set|remind|find|look|play))', query_lower)
            if match:
                return match.group(1).strip().rstrip('.,!?')

        # --- Song/music/playlist ---
        if any(kw in param_desc_lower for kw in ["song", "playlist", "music", "track"]):
            match = re.search(r'play\s+(.+?)(?:\.|,|!|$| and )', query_lower)
            if match:
                song = match.group(1).strip().rstrip('.,!?')
                # Only strip "some/the" prefix
                song = re.sub(r'^(?:some|the)\s+', '', song)
                return song

        # --- Title/subject ---
        if any(kw in param_desc_lower for kw in ["title", "subject", "about", "description"]):
            # "remind me about X at TIME" or "remind me to X at TIME"
            match = re.search(r'(?:about|to)\s+(.+?)\s+at\s+\d', query_lower)
            if match:
                title = match.group(1).strip()
                title = re.sub(r'^the\s+', '', title)
                return title
            # Fallback: text between the action verb and a time/end
            match = re.search(r'(?:remind\w*|create|add)\s+(?:me\s+)?(?:about\s+|to\s+)?(.+?)(?:\s+at\s+|\.|$)', query_lower)
            if match:
                title = match.group(1).strip()
                title = re.sub(r'^the\s+', '', title)
                return title

        # --- Time as string ---
        if "time" in param_desc_lower and "time" not in param_name_lower.replace("time", "x"):
            # Look for time patterns like "3:00 PM", "5 PM"
            # For tools with both hour(int) and time(str), this handles the string version
            match = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm))', query_lower)
            if match:
                time_str = match.group(1).strip().upper()
                if ':' not in time_str:
                    time_str = re.sub(r'(\d+)\s*(AM|PM)', r'\1:00 \2', time_str)
                else:
                    time_str = re.sub(r'(\d+:\d+)\s*(AM|PM)', r'\1 \2', time_str)
                return time_str

        # --- Search query ---
        if any(kw in param_desc_lower for kw in ["query", "search", "keyword"]):
            match = re.search(r'(?:find|look\s*up|search\s*for)\s+([A-Z][a-z]+)', original_query)
            if match:
                return match.group(1)

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
