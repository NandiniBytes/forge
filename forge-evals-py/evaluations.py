# Updated with better structure, more detailed logging, and readiness for future tests.

import tiktoken
from typing import Dict, Any

# We can import the interface here to avoid circular dependencies in other files
# and to use it for more advanced evaluations later.
from forge_interface import answer_question

class TestResult:
    """A simple class to hold the results of an evaluation."""
    def __init__(self, test_name: str, passed: bool, details: str):
        self.test_name = test_name
        self.passed = passed
        self.details = details

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"[{status}] {self.test_name}: {self.details}"

def _count_tokens(text: str) -> int:
    """Helper function to calculate token count."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Warning: Could not count tokens. Error: {e}")
        return 0

def evaluate_token_reduction(test_case: Dict[str, Any], compacted_text: str) -> TestResult:
    """
    Evaluation 1: Measures if compaction effectively reduces the token count.
    """
    original_text = test_case['conversation']
    original_tokens = _count_tokens(original_text)
    compacted_tokens = _count_tokens(compacted_text)

    passed = compacted_tokens < original_tokens
    
    details = (
        f"Original: {original_tokens} tokens, Compacted: {compacted_tokens} tokens."
    )
    if passed:
        reduction_percent = (1.0 - (compacted_tokens / original_tokens)) * 100.0 if original_tokens > 0 else 0
        details += f" Reduction of {reduction_percent:.2f}%."
    
    return TestResult(
        test_name="Token Reduction Test",
        passed=passed,
        details=details
    )

def evaluate_information_retrieval(test_case: Dict[str, Any], compacted_text: str) -> TestResult:
    """
    Evaluation 2: Checks if critical information is retained after compaction
    by asking a targeted question.
    """
    question = test_case['retrieval_test_question']
    expected_keyword = test_case['expected_answer_keyword']

    print(f"\n> Running retrieval test with question: '{question}'")
    
    # Ask the agent the question using ONLY the compacted context.
    agent_answer = answer_question(compacted_text, question)

    print(f"> Agent's Answer (from compacted context): '{agent_answer}'")

    passed = expected_keyword.lower() in agent_answer.lower()
    
    details = f"Expected keyword '{expected_keyword}' "
    details += "found in answer." if passed else f"NOT found in answer: '{agent_answer}'"
    
    return TestResult(
        test_name="Information Retrieval Test",
        passed=passed,
        details=details
    )

# This is the main function that will be called from main.py
def run_evaluations(test_case: Dict[str, Any], compacted_text: str) -> list[TestResult]:
    """
    Runs all defined evaluations for a given test case and returns the results.
    """
    results = []
    
    # Run each evaluation and append the result
    results.append(evaluate_token_reduction(test_case, compacted_text))
    results.append(evaluate_information_retrieval(test_case, compacted_text))
    
    # You can easily add more evaluation calls here in the future, e.g.:
    # results.append(evaluate_with_llm_judge(test_case, compacted_text))
    
    return results
