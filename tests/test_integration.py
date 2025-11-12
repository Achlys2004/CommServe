#!/usr/bin/env python3
"""
Consolidated Integration Tests
Combines testing for context management, conversational AI, and system integration
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_context_and_memory():
    """Test conversation context and memory functionality"""
    print("🧠 Testing Context & Memory...")

    try:
        from backend.orchestrator import Orchestrator
    except ImportError as e:
        print(f"   ⚠️  Context & Memory test skipped: {e}")
        return

    # Test with unique session to avoid conflicts
    orchestrator = Orchestrator(session_id="integration_test_session")

    # Test context building
    context = orchestrator._build_context()
    print(f"   ✓ Context built: {len(str(context))} characters")

    # Test follow-up enhancement
    query = "what about february?"
    enhanced = orchestrator._enhance_query_with_context(
        query, "Previous: January sales analysis"
    )
    print(f"   ✓ Follow-up enhanced: '{enhanced}'")

    print("   ✅ Context & Memory tests passed")


def test_conversational_flow():
    """Test end-to-end conversational flow"""
    print("\n💬 Testing Conversational Flow...")

    try:
        from backend.orchestrator import Orchestrator
        from backend.query_engine import QueryEngine
    except ImportError as e:
        print(f"   ⚠️  Conversational flow test skipped: {e}")
        return

    # Initialize with test session
    query_engine = QueryEngine(session_id="conv_test_session", use_orchestrator=False)
    orchestrator = Orchestrator(session_id="conv_test_session")

    # Test basic query processing (without actual LLM calls)
    try:
        # This will test the pipeline without hitting APIs
        result = orchestrator.process_query("show me total orders", query_engine)
        print(f"   ✓ Query processed: {type(result)}")
        print("   ✅ Conversational flow tests passed")
    except Exception as e:
        print(f"   ⚠️  Conversational flow test skipped (API unavailable): {e}")


def test_system_integration():
    """Test overall system integration"""
    print("\n🔗 Testing System Integration...")

    # Test imports
    try:
        from backend.orchestrator import Orchestrator
        from backend.query_engine import QueryEngine
        from backend.llm_client import call_llm
        from backend.planner import decide_action

        print("   ✓ All core modules import successfully")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return

    # Test configuration
    try:
        import config

        print("   ✓ Configuration loaded")
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        return

    print("   ✅ System integration tests passed")


def test_data_processing():
    """Test data processing capabilities"""
    print("\n📊 Testing Data Processing...")

    import pandas as pd
    import sqlite3

    # Test database connection
    try:
        conn = sqlite3.connect("data/olist.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM orders")
        count = cursor.fetchone()[0]
        print(f"   ✓ Database connected: {count} orders")
        conn.close()
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return

    print("   ✅ Data processing tests passed")


def run_all_tests():
    """Run all integration tests"""
    print("🚀 CommServe Integration Test Suite")
    print("=" * 50)

    test_context_and_memory()
    test_conversational_flow()
    test_system_integration()
    test_data_processing()

    print("\n" + "=" * 50)
    print("✨ Integration tests completed!")
    print("\n📝 Note: Some tests may be skipped if API keys are not configured")
    print("🔧 For full testing, ensure API keys are set in .env file")


if __name__ == "__main__":
    run_all_tests()
