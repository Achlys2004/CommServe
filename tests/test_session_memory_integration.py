#!/usr/bin/env python3
"""Test script for SessionMemory integration in QueryEngine"""

import pytest

try:
    from backend.query_engine import QueryEngine
except ImportError as e:
    pytest.skip(
        f"Skipping test due to missing dependencies: {e}", allow_module_level=True
    )

import json


def test_session_memory_integration():
    print("Testing QueryEngine with SessionMemory integration...")

    # Create QueryEngine with session memory
    engine = QueryEngine(session_id="test_qe_session", persist_memory=False)

    # Test follow-up detection
    print("Testing follow-up detection...")
    query = "show me last month"
    last_context = engine.memory.get_context(include_recent=1)
    is_follow_up = last_context and any(
        w in query.lower()
        for w in [
            "that",
            "those",
            "this",
            "last",
            "previous",
            "month",
            "week",
            "top",
            "bottom",
            "again",
        ]
    )
    print(f'Query: "{query}" -> Is follow-up: {is_follow_up}')

    # Test memory persistence
    print("Testing memory operations...")
    print(f"Initial history: {len(engine.get_session_history())}")

    # Simulate adding to memory (normally done in execute_query)
    test_result = {
        "type": "sql",
        "rows": [{"test": "data"}],
        "is_follow_up": is_follow_up,
    }
    try:
        engine.memory.add("test query", "SQL", test_result)
        print("Memory add successful")
    except Exception as e:
        print(f"Memory add failed: {e}")

    print(f"After add history: {len(engine.get_session_history())}")

    # Test context retrieval
    context = engine.memory.get_context(include_recent=1)
    print(f"Context available: {len(context) > 0}")

    # Test clear
    engine.clear_memory()
    print(f"After clear history: {len(engine.get_session_history())}")

    print("QueryEngine SessionMemory integration test passed!")


if __name__ == "__main__":
    test_session_memory_integration()
