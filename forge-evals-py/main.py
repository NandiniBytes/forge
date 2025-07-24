# --- File: forge-evals-py/main.py (Updated) ---

import os
from forge_interface import compact_conversation
# Import the main evaluation runner function
from evaluations import run_evaluations

def load_test_case(file_path: str) -> dict:
    """Loads a conversation dump and its metadata from a text file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    parts = content.split('---', 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid test case format in {file_path}")
        
    metadata_str, conversation_text = parts[1], parts[2]
    
    metadata = {}
    for line in metadata_str.strip().split('\n'):
        key, value = line.split(':', 1)
        metadata[key.strip()] = value.strip()
    
    metadata['conversation'] = conversation_text.strip()
    return metadata

def main():
    print("🚀 Starting Forge Agentic Evaluation Suite (Python)...")
    
    dataset_dir = "datasets"
    test_files = [f for f in os.listdir(dataset_dir) if f.endswith('.txt')]
    
    print(f"\nLoaded {len(test_files)} test cases from '{dataset_dir}' directory.")

    for test_file in test_files:
        file_path = os.path.join(dataset_dir, test_file)
        test_case = load_test_case(file_path)
        
        test_id = test_case['id']
        original_conversation = test_case['conversation']

        print("\n========================================================")
        print(f"Executing Test Case: {test_id}")
        print("========================================================")

        compacted_text = compact_conversation(original_conversation)

        # Run all evaluations and get a list of results
        results = run_evaluations(test_case, compacted_text)

        # Print each result
        print("\n--- Evaluation Results ---")
        for result in results:
            print(result)
        print("--------------------------")


if __name__ == "__main__":
    main()