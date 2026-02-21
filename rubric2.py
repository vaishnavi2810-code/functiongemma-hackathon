import sys, os
sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

import json
from main import generate_hybrid


############## Tool definitions ##############

TOOL_SEARCH_FLIGHTS = {
    "name": "search_flights",
    "description": "Search for available flights between two airports",
    "parameters": {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "description": "Departure airport code (e.g. BOS, JFK, LAX)"},
            "destination": {"type": "string", "description": "Arrival airport code (e.g. DPS for Bali)"},
            "departure_date": {"type": "string", "description": "Departure date in YYYY-MM-DD format"},
            "return_date": {"type": "string", "description": "Return date in YYYY-MM-DD format"},
            "passengers": {"type": "integer", "description": "Number of passengers"},
        },
        "required": ["origin", "destination", "departure_date"],
    },
}

TOOL_BOOK_FLIGHT = {
    "name": "book_flight",
    "description": "Book a specific flight by its flight ID",
    "parameters": {
        "type": "object",
        "properties": {
            "flight_id": {"type": "string", "description": "The flight ID to book"},
            "passengers": {"type": "integer", "description": "Number of passengers"},
        },
        "required": ["flight_id", "passengers"],
    },
}

TOOL_SEARCH_HOTELS = {
    "name": "search_hotels",
    "description": "Search for hotels or accommodations in a specific location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City, area, or neighborhood name"},
            "checkin": {"type": "string", "description": "Check-in date YYYY-MM-DD"},
            "checkout": {"type": "string", "description": "Check-out date YYYY-MM-DD"},
            "guests": {"type": "integer", "description": "Number of guests"},
            "max_price_per_night": {"type": "integer", "description": "Maximum price per night in USD"},
        },
        "required": ["location", "checkin", "checkout"],
    },
}

TOOL_BOOK_HOTEL = {
    "name": "book_hotel",
    "description": "Book a specific hotel by its hotel ID",
    "parameters": {
        "type": "object",
        "properties": {
            "hotel_id": {"type": "string", "description": "The hotel ID to book"},
            "checkin": {"type": "string", "description": "Check-in date YYYY-MM-DD"},
            "checkout": {"type": "string", "description": "Check-out date YYYY-MM-DD"},
            "guests": {"type": "integer", "description": "Number of guests"},
        },
        "required": ["hotel_id", "checkin", "checkout", "guests"],
    },
}

TOOL_SEARCH_ACTIVITIES = {
    "name": "search_activities",
    "description": "Search for tours, activities, and experiences at a destination",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City or area name"},
            "category": {"type": "string", "description": "Activity type: cultural, adventure, nature, food, wellness, nightlife"},
            "date": {"type": "string", "description": "Date for the activity YYYY-MM-DD"},
        },
        "required": ["location"],
    },
}

TOOL_BOOK_ACTIVITY = {
    "name": "book_activity",
    "description": "Book a specific activity or tour by its ID",
    "parameters": {
        "type": "object",
        "properties": {
            "activity_id": {"type": "string", "description": "The activity ID to book"},
            "date": {"type": "string", "description": "Date for the activity YYYY-MM-DD"},
            "participants": {"type": "integer", "description": "Number of participants"},
        },
        "required": ["activity_id", "date", "participants"],
    },
}

TOOL_GET_DIRECTIONS = {
    "name": "get_directions",
    "description": "Get directions and travel time between two locations",
    "parameters": {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "description": "Starting location"},
            "destination": {"type": "string", "description": "Ending location"},
            "mode": {"type": "string", "description": "Travel mode: driving, walking, transit, scooter"},
        },
        "required": ["origin", "destination"],
    },
}

TOOL_GET_WEATHER = {
    "name": "get_weather_forecast",
    "description": "Get weather forecast for a location on a specific date",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City or area name"},
            "date": {"type": "string", "description": "Date for forecast YYYY-MM-DD"},
        },
        "required": ["location"],
    },
}

TOOL_CURRENCY_CONVERT = {
    "name": "convert_currency",
    "description": "Convert an amount between two currencies",
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "Amount to convert"},
            "from_currency": {"type": "string", "description": "Source currency code (e.g. USD)"},
            "to_currency": {"type": "string", "description": "Target currency code (e.g. IDR)"},
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
}

TOOL_SET_REMINDER = {
    "name": "set_reminder",
    "description": "Set a reminder with a description and time",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Reminder description"},
            "datetime": {"type": "string", "description": "When to remind in YYYY-MM-DD HH:MM format"},
        },
        "required": ["title", "datetime"],
    },
}

TOOL_TRANSLATE = {
    "name": "translate_phrase",
    "description": "Translate a phrase to another language",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to translate"},
            "target_language": {"type": "string", "description": "Target language (e.g. Indonesian, Balinese)"},
        },
        "required": ["text", "target_language"],
    },
}

TOOL_EMERGENCY_INFO = {
    "name": "get_emergency_info",
    "description": "Get emergency contacts and nearest services for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "Current location"},
            "service_type": {"type": "string", "description": "Type: hospital, police, embassy, pharmacy"},
        },
        "required": ["location", "service_type"],
    },
}


############## Benchmark cases ##############

BENCHMARKS = [
    # =============================================================
    # EASY: 1 tool, direct request, clear parameters
    # =============================================================
    {
        "name": "flight_search_basic",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Search for flights from Boston to Bali on March 15th."}],
        "tools": [TOOL_SEARCH_FLIGHTS],
        "expected_calls": [{"name": "search_flights", "arguments": {"origin": "BOS", "destination": "DPS", "departure_date": "2026-03-15"}}],
    },
    {
        "name": "hotel_search_ubud",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Find hotels in Ubud checking in March 16 and checking out March 20."}],
        "tools": [TOOL_SEARCH_HOTELS],
        "expected_calls": [{"name": "search_hotels", "arguments": {"location": "Ubud", "checkin": "2026-03-16", "checkout": "2026-03-20"}}],
    },
    {
        "name": "weather_bali",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "What's the weather in Bali on March 16th?"}],
        "tools": [TOOL_GET_WEATHER],
        "expected_calls": [{"name": "get_weather_forecast", "arguments": {"location": "Bali", "date": "2026-03-16"}}],
    },
    {
        "name": "currency_usd_idr",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Convert 500 dollars to Indonesian rupiah."}],
        "tools": [TOOL_CURRENCY_CONVERT],
        "expected_calls": [{"name": "convert_currency", "arguments": {"amount": 500, "from_currency": "USD", "to_currency": "IDR"}}],
    },
    {
        "name": "translate_thankyou",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "How do you say thank you in Indonesian?"}],
        "tools": [TOOL_TRANSLATE],
        "expected_calls": [{"name": "translate_phrase", "arguments": {"text": "thank you", "target_language": "Indonesian"}}],
    },
    {
        "name": "activities_seminyak",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Show me activities in Seminyak."}],
        "tools": [TOOL_SEARCH_ACTIVITIES],
        "expected_calls": [{"name": "search_activities", "arguments": {"location": "Seminyak"}}],
    },
    {
        "name": "directions_basic",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Get directions from Ubud to Uluwatu Temple."}],
        "tools": [TOOL_GET_DIRECTIONS],
        "expected_calls": [{"name": "get_directions", "arguments": {"origin": "Ubud", "destination": "Uluwatu Temple"}}],
    },
    {
        "name": "reminder_checkout",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Remind me to check out at 2026-03-20 11:00."}],
        "tools": [TOOL_SET_REMINDER],
        "expected_calls": [{"name": "set_reminder", "arguments": {"title": "check out", "datetime": "2026-03-20 11:00"}}],
    },
    {
        "name": "emergency_hospital",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Find the nearest hospital in Ubud."}],
        "tools": [TOOL_EMERGENCY_INFO],
        "expected_calls": [{"name": "get_emergency_info", "arguments": {"location": "Ubud", "service_type": "hospital"}}],
    },
    {
        "name": "flight_search_roundtrip",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Find round trip flights from JFK to Bali leaving April 1st returning April 10th for 2 passengers."}],
        "tools": [TOOL_SEARCH_FLIGHTS],
        "expected_calls": [{"name": "search_flights", "arguments": {"origin": "JFK", "destination": "DPS", "departure_date": "2026-04-01", "return_date": "2026-04-10", "passengers": 2}}],
    },

    # =============================================================
    # MEDIUM: Pick correct tool among 2-5, or infer parameters
    # =============================================================
    {
        "name": "hotel_among_three",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Find a place to stay in Canggu from March 20 to March 23."}],
        "tools": [TOOL_SEARCH_FLIGHTS, TOOL_SEARCH_HOTELS, TOOL_SEARCH_ACTIVITIES],
        "expected_calls": [{"name": "search_hotels", "arguments": {"location": "Canggu", "checkin": "2026-03-20", "checkout": "2026-03-23"}}],
    },
    {
        "name": "activity_cultural_among_four",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Find cultural experiences in Ubud for March 17th."}],
        "tools": [TOOL_SEARCH_HOTELS, TOOL_SEARCH_ACTIVITIES, TOOL_GET_DIRECTIONS, TOOL_GET_WEATHER],
        "expected_calls": [{"name": "search_activities", "arguments": {"location": "Ubud", "category": "cultural", "date": "2026-03-17"}}],
    },
    {
        "name": "currency_among_three",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "How much is 200 bucks in rupiah?"}],
        "tools": [TOOL_CURRENCY_CONVERT, TOOL_SEARCH_FLIGHTS, TOOL_TRANSLATE],
        "expected_calls": [{"name": "convert_currency", "arguments": {"amount": 200, "from_currency": "USD", "to_currency": "IDR"}}],
    },
    {
        "name": "directions_scooter_among_four",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "How do I get from Canggu to Tanah Lot by scooter?"}],
        "tools": [TOOL_GET_DIRECTIONS, TOOL_SEARCH_ACTIVITIES, TOOL_SEARCH_HOTELS, TOOL_GET_WEATHER],
        "expected_calls": [{"name": "get_directions", "arguments": {"origin": "Canggu", "destination": "Tanah Lot", "mode": "scooter"}}],
    },
    {
        "name": "translate_among_five",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "How do I say 'where is the beach' in Indonesian?"}],
        "tools": [TOOL_TRANSLATE, TOOL_GET_DIRECTIONS, TOOL_SEARCH_ACTIVITIES, TOOL_CURRENCY_CONVERT, TOOL_GET_WEATHER],
        "expected_calls": [{"name": "translate_phrase", "arguments": {"text": "where is the beach", "target_language": "Indonesian"}}],
    },
    {
        "name": "flight_implicit_destination",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "I'm flying out of LAX on May 5th, find me flights to Bali coming back May 15th."}],
        "tools": [TOOL_SEARCH_FLIGHTS, TOOL_SEARCH_HOTELS, TOOL_GET_WEATHER],
        "expected_calls": [{"name": "search_flights", "arguments": {"origin": "LAX", "destination": "DPS", "departure_date": "2026-05-05", "return_date": "2026-05-15"}}],
    },
    {
        "name": "hotel_budget_among_four",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Find something cheap to stay in Amed under 50 dollars a night, March 22 to March 25."}],
        "tools": [TOOL_SEARCH_HOTELS, TOOL_SEARCH_FLIGHTS, TOOL_CURRENCY_CONVERT, TOOL_SEARCH_ACTIVITIES],
        "expected_calls": [{"name": "search_hotels", "arguments": {"location": "Amed", "checkin": "2026-03-22", "checkout": "2026-03-25", "max_price_per_night": 50}}],
    },
    {
        "name": "emergency_among_four",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "I need a pharmacy near Seminyak."}],
        "tools": [TOOL_EMERGENCY_INFO, TOOL_GET_DIRECTIONS, TOOL_SEARCH_HOTELS, TOOL_TRANSLATE],
        "expected_calls": [{"name": "get_emergency_info", "arguments": {"location": "Seminyak", "service_type": "pharmacy"}}],
    },
    {
        "name": "weather_among_five",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Should I pack a raincoat for Bali on March 18th?"}],
        "tools": [TOOL_GET_WEATHER, TOOL_SEARCH_FLIGHTS, TOOL_SEARCH_HOTELS, TOOL_CURRENCY_CONVERT, TOOL_TRANSLATE],
        "expected_calls": [{"name": "get_weather_forecast", "arguments": {"location": "Bali", "date": "2026-03-18"}}],
    },
    {
        "name": "activity_adventure_among_five",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "What adventure stuff can I do around Sidemen on March 19th?"}],
        "tools": [TOOL_SEARCH_ACTIVITIES, TOOL_SEARCH_HOTELS, TOOL_GET_DIRECTIONS, TOOL_TRANSLATE, TOOL_CURRENCY_CONVERT],
        "expected_calls": [{"name": "search_activities", "arguments": {"location": "Sidemen", "category": "adventure", "date": "2026-03-19"}}],
    },

    # =============================================================
    # HARD: Multi-tool chains, compound requests, inferred params
    # =============================================================
    {
        "name": "flight_and_hotel",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Search flights from Boston to Bali on March 15 and find hotels in Ubud from March 16 to March 20."}],
        "tools": [TOOL_SEARCH_FLIGHTS, TOOL_SEARCH_HOTELS, TOOL_SEARCH_ACTIVITIES, TOOL_GET_WEATHER],
        "expected_calls": [
            {"name": "search_flights", "arguments": {"origin": "BOS", "destination": "DPS", "departure_date": "2026-03-15"}},
            {"name": "search_hotels", "arguments": {"location": "Ubud", "checkin": "2026-03-16", "checkout": "2026-03-20"}},
        ],
    },
    {
        "name": "hotel_and_activities",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Find a hotel in Canggu March 20 to 23 and search for nature activities there on March 21."}],
        "tools": [TOOL_SEARCH_HOTELS, TOOL_SEARCH_ACTIVITIES, TOOL_GET_DIRECTIONS, TOOL_SEARCH_FLIGHTS],
        "expected_calls": [
            {"name": "search_hotels", "arguments": {"location": "Canggu", "checkin": "2026-03-20", "checkout": "2026-03-23"}},
            {"name": "search_activities", "arguments": {"location": "Canggu", "category": "nature", "date": "2026-03-21"}},
        ],
    },
    {
        "name": "weather_and_translate",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Check the weather in Ubud on March 17 and translate 'good morning' to Indonesian."}],
        "tools": [TOOL_GET_WEATHER, TOOL_TRANSLATE, TOOL_SEARCH_HOTELS, TOOL_CURRENCY_CONVERT],
        "expected_calls": [
            {"name": "get_weather_forecast", "arguments": {"location": "Ubud", "date": "2026-03-17"}},
            {"name": "translate_phrase", "arguments": {"text": "good morning", "target_language": "Indonesian"}},
        ],
    },
    {
        "name": "directions_and_weather",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Get directions from Ubud to Tegallalang Rice Terrace and check the weather there on March 18."}],
        "tools": [TOOL_GET_DIRECTIONS, TOOL_GET_WEATHER, TOOL_SEARCH_ACTIVITIES, TOOL_SEARCH_HOTELS],
        "expected_calls": [
            {"name": "get_directions", "arguments": {"origin": "Ubud", "destination": "Tegallalang Rice Terrace"}},
            {"name": "get_weather_forecast", "arguments": {"location": "Tegallalang", "date": "2026-03-18"}},
        ],
    },
    {
        "name": "currency_and_reminder",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Convert 1000 dollars to rupiah and remind me to withdraw cash at 2026-03-15 09:00."}],
        "tools": [TOOL_CURRENCY_CONVERT, TOOL_SET_REMINDER, TOOL_SEARCH_FLIGHTS, TOOL_TRANSLATE],
        "expected_calls": [
            {"name": "convert_currency", "arguments": {"amount": 1000, "from_currency": "USD", "to_currency": "IDR"}},
            {"name": "set_reminder", "arguments": {"title": "withdraw cash", "datetime": "2026-03-15 09:00"}},
        ],
    },
    {
        "name": "activity_and_directions",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Find wellness activities in Sidemen on March 19 and get directions from Ubud to Sidemen."}],
        "tools": [TOOL_SEARCH_ACTIVITIES, TOOL_GET_DIRECTIONS, TOOL_SEARCH_HOTELS, TOOL_GET_WEATHER, TOOL_TRANSLATE],
        "expected_calls": [
            {"name": "search_activities", "arguments": {"location": "Sidemen", "category": "wellness", "date": "2026-03-19"}},
            {"name": "get_directions", "arguments": {"origin": "Ubud", "destination": "Sidemen"}},
        ],
    },
    {
        "name": "emergency_and_translate",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Find the nearest hospital in Kuta and translate 'I need help' to Indonesian."}],
        "tools": [TOOL_EMERGENCY_INFO, TOOL_TRANSLATE, TOOL_GET_DIRECTIONS, TOOL_SEARCH_HOTELS, TOOL_CURRENCY_CONVERT],
        "expected_calls": [
            {"name": "get_emergency_info", "arguments": {"location": "Kuta", "service_type": "hospital"}},
            {"name": "translate_phrase", "arguments": {"text": "I need help", "target_language": "Indonesian"}},
        ],
    },
    {
        "name": "full_day_plan",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Check weather in Ubud March 17, find cultural activities there that day, and get directions from Canggu to Ubud."}],
        "tools": [TOOL_GET_WEATHER, TOOL_SEARCH_ACTIVITIES, TOOL_GET_DIRECTIONS, TOOL_SEARCH_HOTELS, TOOL_SEARCH_FLIGHTS, TOOL_CURRENCY_CONVERT],
        "expected_calls": [
            {"name": "get_weather_forecast", "arguments": {"location": "Ubud", "date": "2026-03-17"}},
            {"name": "search_activities", "arguments": {"location": "Ubud", "category": "cultural", "date": "2026-03-17"}},
            {"name": "get_directions", "arguments": {"origin": "Canggu", "destination": "Ubud"}},
        ],
    },
    {
        "name": "travel_prep_combo",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Convert 300 dollars to rupiah, translate 'excuse me where is the temple' to Indonesian, and remind me to pack sunscreen at 2026-03-14 20:00."}],
        "tools": [TOOL_CURRENCY_CONVERT, TOOL_TRANSLATE, TOOL_SET_REMINDER, TOOL_SEARCH_FLIGHTS, TOOL_GET_WEATHER, TOOL_SEARCH_ACTIVITIES],
        "expected_calls": [
            {"name": "convert_currency", "arguments": {"amount": 300, "from_currency": "USD", "to_currency": "IDR"}},
            {"name": "translate_phrase", "arguments": {"text": "excuse me where is the temple", "target_language": "Indonesian"}},
            {"name": "set_reminder", "arguments": {"title": "pack sunscreen", "datetime": "2026-03-14 20:00"}},
        ],
    },
    {
        "name": "arrival_day_full",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Search flights from BOS to Bali March 15, find hotels in Ubud March 16 to 20, and check the weather on March 16."}],
        "tools": [TOOL_SEARCH_FLIGHTS, TOOL_SEARCH_HOTELS, TOOL_GET_WEATHER, TOOL_SEARCH_ACTIVITIES, TOOL_GET_DIRECTIONS, TOOL_CURRENCY_CONVERT],
        "expected_calls": [
            {"name": "search_flights", "arguments": {"origin": "BOS", "destination": "DPS", "departure_date": "2026-03-15"}},
            {"name": "search_hotels", "arguments": {"location": "Ubud", "checkin": "2026-03-16", "checkout": "2026-03-20"}},
            {"name": "get_weather_forecast", "arguments": {"location": "Ubud", "date": "2026-03-16"}},
        ],
    },
]


############## Scoring (same as official benchmark.py) ##############

def _normalize(v):
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _call_matches(predicted, expected):
    if predicted["name"] != expected["name"]:
        return False
    pred_args = predicted.get("arguments", {})
    exp_args = expected.get("arguments", {})
    for key, exp_val in exp_args.items():
        if key not in pred_args:
            return False
        if _normalize(pred_args[key]) != _normalize(exp_val):
            return False
    return True


def compute_f1(predicted_calls, expected_calls):
    if not predicted_calls and not expected_calls:
        return 1.0
    if not predicted_calls or not expected_calls:
        return 0.0
    matched = 0
    used = set()
    for exp in expected_calls:
        for i, pred in enumerate(predicted_calls):
            if i not in used and _call_matches(pred, exp):
                matched += 1
                used.add(i)
                break
    precision = matched / len(predicted_calls)
    recall = matched / len(expected_calls)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_total_score(results):
    difficulty_weights = {"easy": 0.20, "medium": 0.30, "hard": 0.50}
    time_baseline_ms = 500

    total_score = 0
    for difficulty, weight in difficulty_weights.items():
        group = [r for r in results if r["difficulty"] == difficulty]
        if not group:
            continue
        avg_f1 = sum(r["f1"] for r in group) / len(group)
        avg_time = sum(r["total_time_ms"] for r in group) / len(group)
        on_device_ratio = sum(1 for r in group if r["source"] == "on-device") / len(group)
        time_score = max(0, 1 - avg_time / time_baseline_ms)
        level_score = (0.60 * avg_f1) + (0.15 * time_score) + (0.25 * on_device_ratio)
        total_score += weight * level_score

    return total_score * 100


def run_benchmark(benchmarks=None):
    if benchmarks is None:
        benchmarks = BENCHMARKS

    total = len(benchmarks)
    results = []
    for i, case in enumerate(benchmarks, 1):
        print(f"[{i}/{total}] Running: {case['name']} ({case['difficulty']})...", end=" ", flush=True)
        result = generate_hybrid(case["messages"], case["tools"])
        f1 = compute_f1(result["function_calls"], case["expected_calls"])
        source = result.get("source", "unknown")
        print(f"F1={f1:.2f} | {result['total_time_ms']:.0f}ms | {source}")
        results.append({
            "name": case["name"],
            "difficulty": case["difficulty"],
            "total_time_ms": result["total_time_ms"],
            "f1": f1,
            "source": source,
            "predicted": result["function_calls"],
            "expected": case["expected_calls"],
        })

    # Print results table
    print("\n=== Bali Travel Assistant — Benchmark Results ===\n")
    print(f"  {'#':>2} | {'Difficulty':<10} | {'Name':<30} | {'Time (ms)':>10} | {'F1':>5} | Source")
    print(f"  {'--':>2}-+-{'-'*10}-+-{'-'*30}-+-{'-'*10}-+-{'-'*5}-+-{'-'*20}")
    for i, r in enumerate(results, 1):
        print(f"  {i:>2} | {r['difficulty']:<10} | {r['name']:<30} | {r['total_time_ms']:>10.2f} | {r['f1']:>5.2f} | {r['source']}")

    # Print summary
    print(f"\n--- Summary ---")
    for difficulty in ["easy", "medium", "hard"]:
        group = [r for r in results if r["difficulty"] == difficulty]
        if not group:
            continue
        avg_f1 = sum(r["f1"] for r in group) / len(group)
        avg_time = sum(r["total_time_ms"] for r in group) / len(group)
        on_device = sum(1 for r in group if r["source"] == "on-device")
        cloud = len(group) - on_device
        print(f"  {difficulty:<8} avg F1={avg_f1:.2f}  avg time={avg_time:.2f}ms  on-device={on_device}/{len(group)} cloud={cloud}/{len(group)}")

    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_time = sum(r["total_time_ms"] for r in results) / len(results)
    total_time = sum(r["total_time_ms"] for r in results)
    on_device_total = sum(1 for r in results if r["source"] == "on-device")
    cloud_total = len(results) - on_device_total
    print(f"  {'overall':<8} avg F1={avg_f1:.2f}  avg time={avg_time:.2f}ms  total time={total_time:.2f}ms")
    print(f"           on-device={on_device_total}/{len(results)} ({100*on_device_total/len(results):.0f}%)  cloud={cloud_total}/{len(results)} ({100*cloud_total/len(results):.0f}%)")

    score = compute_total_score(results)
    print(f"\n{'='*50}")
    print(f"  TOTAL SCORE: {score:.1f}%")
    print(f"{'='*50}")

    # Print mismatches for debugging
    misses = [r for r in results if r["f1"] < 1.0]
    if misses:
        print(f"\n--- Mismatches ({len(misses)}) ---")
        for r in misses:
            print(f"\n  {r['name']} (F1={r['f1']:.2f})")
            print(f"    Expected:  {json.dumps(r['expected'], indent=None)}")
            print(f"    Predicted: {json.dumps(r['predicted'], indent=None)}")

    return results


if __name__ == "__main__":
    run_benchmark()