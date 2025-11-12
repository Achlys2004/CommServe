#!/usr/bin/env python3
"""
Frontend Test Script
Tests the Streamlit frontend for basic functionality
"""

import subprocess
import sys
import time
import requests
import os


def test_frontend_launch():
    """Test if the frontend can be launched"""
    print("🧪 Testing Frontend Launch...")

    try:
        # Start Streamlit in background
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "frontend/streamlit_app.py",
                "--server.headless",
                "true",
                "--server.port",
                "8502",  # Use different port to avoid conflicts
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for startup
        time.sleep(5)

        # Check if process is still running
        if process.poll() is None:
            print("✅ Frontend launched successfully")
            process.terminate()
        else:
            stdout, stderr = process.communicate()
            print("❌ Frontend failed to launch")
            print("STDOUT:", stdout.decode())
            print("STDERR:", stderr.decode())
            raise AssertionError("Frontend failed to launch")

    except Exception as e:
        print(f"❌ Error launching frontend: {e}")
        raise


def test_frontend_imports():
    """Test if all required imports work"""
    print("🧪 Testing Frontend Imports...")

    try:
        # Test basic imports
        import streamlit as st
        import requests
        import pandas as pd
        from datetime import datetime, timedelta
        import base64
        import json
        import time

        print("✅ All imports successful")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise


def test_configuration():
    """Test configuration loading"""
    print("🧪 Testing Configuration...")

    try:
        # Test environment variables
        project_name = os.getenv("PROJECT_NAME", "Olist — GenAI Data Agent")
        api_url = os.getenv("API_URL", "http://127.0.0.1:8000")

        print(f"✅ Project Name: {project_name}")
        print(f"✅ API URL: {api_url}")

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def main():
    """Run all frontend tests"""
    print("🚀 Frontend Test Suite")
    print("=" * 50)

    tests = [test_frontend_imports, test_configuration, test_frontend_launch]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All frontend tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
