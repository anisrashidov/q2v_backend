"""Pytest configuration — must run before any app module is imported.

Sets a dummy OPENAI_API_KEY so pydantic-settings doesn't raise when
app.config.Settings() is instantiated at import time during test collection.
Real API calls are prevented by mocking run_pipeline / LLMPort in each test.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-do-not-use")
