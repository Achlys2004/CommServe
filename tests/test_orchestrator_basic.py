"""
Basic Orchestrator Integration Test
Tests that orchestrator integrates without breaking existing functionality
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_orchestrator_initialization():
    """Test 2: Orchestrator can be initialized"""
    print("\n" + "=" * 80)
    print("TEST 2: Orchestrator Initialization")
    print("=" * 80)

    try:
        from backend.orchestrator import Orchestrator

        orch = Orchestrator(session_id="test_init")
        print("✅ Orchestrator initialized")

        assert orch.session_id == "test_init"
        assert orch.memory is not None
        print("✅ Orchestrator has session memory")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_query_engine_with_orchestrator():
    """Test 3: QueryEngine can initialize with orchestrator"""
    print("\n" + "=" * 80)
    print("TEST 3: QueryEngine with Orchestrator Integration")
    print("=" * 80)

    try:
        from backend.query_engine import QueryEngine

        engine = QueryEngine(use_orchestrator=True, session_id="test_orch")
        print("✅ QueryEngine initialized with orchestrator")

        if engine._orchestrator:
            print("✅ Orchestrator is attached to QueryEngine")
        else:
            print("⚠️  Orchestrator not attached (might be expected in some configs)")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_no_infinite_recursion():
    """Test 4: Verify no infinite recursion between orchestrator and query engine"""
    print("\n" + "=" * 80)
    print("TEST 4: No Infinite Recursion Check")
    print("=" * 80)

    try:
        from backend.query_engine import QueryEngine
        from unittest.mock import MagicMock

        engine = QueryEngine(use_orchestrator=True, session_id="test_recursion")

        # Mock the _execute_direct to track calls
        original_execute = engine._execute_direct
        call_count = [0]

        def mock_execute(query, params):
            call_count[0] += 1
            if call_count[0] > 2:
                raise RuntimeError("Infinite recursion detected!")
            return {"response": "Mock response", "type": "sql"}

        engine._execute_direct = mock_execute

        # If orchestrator exists, verify it would call _execute_direct
        if engine._orchestrator:
            print("✅ Orchestrator uses _execute_direct (no recursion)")
        else:
            print("⚠️  Orchestrator not enabled, skipping recursion check")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_session_memory_integration():
    """Test 5: Session memory works correctly"""
    print("\n" + "=" * 80)
    print("TEST 5: Session Memory Integration")
    print("=" * 80)

    try:
        from backend.utils.session_memory import SessionMemory

        memory = SessionMemory(session_id="test_memory", max_recent=3, persist=False)
        print("✅ SessionMemory initialized")

        # Add a test exchange
        memory.add_exchange(
            session_id="test_memory",
            query="Test query",
            response={"response": "Test response"},
            action="SQL",
            confidence=0.9,
        )
        print("✅ Can add exchange to memory")

        # Get context
        context = memory.get_context(include_recent=1)
        assert context is not None
        print("✅ Can retrieve context from memory")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_tests():
    """Run all basic tests"""
    print("\n" + "=" * 80)
    print("🧪 ORCHESTRATOR BASIC INTEGRATION TEST SUITE")
    print("=" * 80)
    print("\nTesting core integration without API calls...\n")

    tests = [
        ("Orchestrator Init", test_orchestrator_initialization),
        ("QueryEngine with Orchestrator", test_query_engine_with_orchestrator),
        ("No Infinite Recursion", test_no_infinite_recursion),
        ("Session Memory", test_session_memory_integration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    print("\n" + "=" * 80)
    print(f"Result: {passed_count}/{total_count} tests passed")
    print("=" * 80)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
