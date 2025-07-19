#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the robust JSON parser implementation.
This tests various JSON parsing scenarios to ensure the parser handles different edge cases.
"""

import json
from agent_team import parse_llm_json_response

def test_parser():
    """Test the robust JSON parser with various input formats"""
    print("Testing robust JSON parser...")
    
    # Test cases
    test_cases = [
        # 1. Valid JSON in markdown code block
        {
            "name": "Markdown code block",
            "input": """Here's the JSON output:
```json
{
  "expense_categories_identified": ["travel", "meals", "accommodation"],
  "total": 1250.75
}
```""",
            "expected_keys": ["expense_categories_identified", "total"]
        },
        
        # 2. Valid JSON without markdown formatting
        {
            "name": "Plain JSON",
            "input": """The parsed data is: 
{
  "expense_categories_identified": ["travel", "meals"],
  "total": 980.50
}""",
            "expected_keys": ["expense_categories_identified", "total"]
        },
        
        # 3. JSON with control characters
        {
            "name": "JSON with control characters",
            "input": "{\n\t\"expense_categories_identified\": [\"travel\"],\r\n\t\"total\": 500\n}",
            "expected_keys": ["expense_categories_identified", "total"]
        },
        
        # 4. JSON with single quotes instead of double quotes
        {
            "name": "JSON with single quotes",
            "input": "{'expense_categories_identified': ['travel'], 'total': 750}",
            "expected_keys": ["expense_categories_identified", "total"]
        },
        
        # 5. Invalid JSON (should use fallback)
        {
            "name": "Invalid JSON",
            "input": "This is not valid JSON at all",
            "expected_keys": ["error"]
        }
    ]
    
    # Run tests
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test['name']}")
        print(f"Input: {test['input'][:50]}..." if len(test['input']) > 50 else f"Input: {test['input']}")
        
        # Parse with our robust parser
        result = parse_llm_json_response(test["input"])
        
        # Check if expected keys are present
        keys_present = all(key in result for key in test["expected_keys"])
        print(f"Result: {result}")
        print(f"Expected keys present: {keys_present}")
        
        if not keys_present:
            print("❌ TEST FAILED")
        else:
            print("✅ TEST PASSED")
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    test_parser()
