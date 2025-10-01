#!/usr/bin/env python3
"""
Simple test script to verify the interview evaluation fixes.
"""
import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_evaluation():
    """Test the evaluation function with sample data."""
    try:
        # Import the evaluation function
        from resume_parser.clients.azure_openai import azure_client

        # Test data
        test_cases = [
            {
                "question_text": "Are you ready to relocate to Mumbai?",
                "expected_answer": "Yes",
                "candidate_response": "Yes",
                "description": "Yes/No question - should match"
            },
            {
                "question_text": "Are you ready to relocate to Mumbai?",
                "expected_answer": "Yes",
                "candidate_response": "No",
                "description": "Yes/No question - should not match"
            },
            {
                "question_text": "What is your expected salary?",
                "expected_answer": "80000-95000",
                "candidate_response": "I'm looking for a competitive salary that reflects my experience and the value I bring to the role. I'm more focused on the role fit and growth opportunities rather than a specific number right now.",
                "description": "Salary question - should accept detailed response"
            }
        ]

        # Mock the Azure OpenAI client for testing
        class MockClient:
            def chat_completions_create(self, **kwargs):
                # Mock response for testing
                return {
                    "choices": [{
                        "message": {
                            "content": '{"status": "ANSWERED_SUCCESS"}'
                        }
                    }]
                }

        # Replace the real client with mock for testing
        original_client = azure_client.get_sync_client
        azure_client.get_sync_client = lambda: MockClient()

        try:
            # Import and test the evaluation function
            from app import _generate_response_evaluation

            for i, test_case in enumerate(test_cases, 1):
                print(f"\nTest Case {i}: {test_case['description']}")
                print(f"Question: {test_case['question_text']}")
                print(f"Expected: {test_case['expected_answer']}")
                print(f"Response: {test_case['candidate_response']}")

                evaluation = await _generate_response_evaluation(
                    question_text=test_case['question_text'],
                    expected_answer=test_case['expected_answer'],
                    candidate_response=test_case['candidate_response']
                )

                print(f"Result - Status: {evaluation.get('status')}, Is Match: {evaluation.get('is_match')}")
                print(f"Assessment: {evaluation.get('overall_assessment')}")

        finally:
            # Restore original client
            azure_client.get_sync_client = original_client

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_evaluation())