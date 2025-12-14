#!/usr/bin/env python3
"""
Quick test to check if your API key works
"""

import os
from openai import OpenAI

# Set your key here TEMPORARILY to test
# Then DELETE this line and use environment variable!
TEST_KEY = "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"

print("Testing API key validity...")

try:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=TEST_KEY
    )

    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=10
    )

    print(f"✅ API key works! Response: {response.choices[0].message.content}")
    print("\n⚠️  Key is valid. Consider revoking if you want to be safe.")
    print("   Revoke at: https://build.nvidia.com")

except Exception as e:
    print(f"❌ API key invalid or revoked: {e}")
    print("   Generate new key at: https://build.nvidia.com")
