import subprocess

def run_forge_command(prompt: str) -> str:
    """
    Runs the 'forge' command-line tool with a given prompt
    and returns the output.
    """
    try:
        # We use '-p' for the prompt flag, which is common.
        # `capture_output=True` saves stdout and stderr.
        # `text=True` decodes the output as text.
        # `check=True` will raise an error if the command fails.
        result = subprocess.run(
            ['forge', '-p', prompt],
            capture_output=True,
            text=True,
            check=True
        )
        # Return the standard output, stripping any extra whitespace.
        return result.stdout.strip()
    except FileNotFoundError:
        # This error happens if 'forge' is not installed or not in the system's PATH.
        print("❌ ERROR: The 'forge' command was not found.")
        print("Please make sure you have installed it by running 'npx forgecode@latest'.")
        # Exit the program so the user can fix the installation.
        exit(1)
    except subprocess.CalledProcessError as e:
        # This error happens if the forge command returns a non-zero exit code.
        print(f"❌ ERROR: The 'forge' command failed with exit code {e.returncode}.")
        print(f"Stderr: {e.stderr.strip()}")
        # Return an error message instead of crashing.
        return f"Error executing forge: {e.stderr.strip()}"


def compact_conversation(conversation_text: str) -> str:
    """
    Uses the real Forge agent to compact a conversation.
    """
    print("--- Calling real Forge agent for compaction... ---")
    
    # We create a specific prompt that asks the agent to summarize.
    prompt = f"""
Summarize the key points and entities from the following conversation text. 
Be concise but do not lose critical information like file names or library names.

Conversation:
---
{conversation_text}
---
    """
    
    return run_forge_command(prompt)


def answer_question(context: str, question: str) -> str:
    """
    Uses the real Forge agent to answer a question based on a given context.
    """
    print("--- Calling real Forge agent to answer question... ---")
    
    # We structure the prompt to make it clear to the LLM that it should
    # only use the provided context to answer the question.
    prompt = f"""
Context:
---
{context}
---

Based ONLY on the context provided above, answer the following question.
Question: {question}
    """
    
    return run_forge_command(prompt)

# --- File: forge-evals-py/evaluations.py ---
# No changes needed here.

import tiktoken

def count_tokens(text: str) -> int:
    """Calculates the token count of a given text."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def token_reduction_test(test_id: str, original_text: str, compacted_text: str):
    """
    Evaluation 1: Measures how effectively compaction reduces the token count.
    """
    print(f"\n--- Running Token Reduction Test for '{test_id}' ---")
    original_tokens = count_tokens(original_text)
    compacted_tokens = count_tokens(compacted_text)
    
    print(f"Original Token Count: {original_tokens}")
    print(f"Compacted Token Count: {compacted_tokens}")

    if compacted_tokens < original_tokens:
        reduction_percent = (1.0 - (compacted_tokens / original_tokens)) * 100.0
        print(f"✅ PASS: Token count reduced by {reduction_percent:.2f}%.")
    else:
        print("❌ FAIL: Token count was not reduced.")

def information_retrieval_test(test_id: str, compacted_text: str, question: str, expected_keyword: str):
    """
    Evaluation 2: Checks if critical information is retained after compaction.
    """
    from forge_interface import answer_question

    print(f"\n--- Running Information Retrieval Test for '{test_id}' ---")
    print(f"Question: {question}")

    answer = answer_question(compacted_text, question)

    print(f"Agent's Answer (from compacted context): {answer}")

    if expected_keyword.lower() in answer.lower():
        print("✅ PASS: Agent correctly retrieved information from compacted context.")
    else:
        print("❌ FAIL: Agent failed to retrieve critical information.")
        print(f"Expected keyword: '{expected_keyword}'")
