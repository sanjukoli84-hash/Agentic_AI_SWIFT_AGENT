import sys
import json
import os
sys.path.append(os.path.dirname(__file__))

from graph import compiled_app

def run_test():
    # Read the valid sample message
    with open("samples/sample_valid.txt", "r") as f:
        raw_swift = f.read()
        
    initial_state = {
        "raw_message": raw_swift,
        "model_name": "llama3.2",
        "base_url": "http://localhost:11434",
        "extracted_fields": {},
        "reasoning_report": "",
        "errors": []
    }
    
    print("Running LangGraph workflow on valid SWIFT message...")
    try:
        final_state = compiled_app.invoke(initial_state)
        
        output_data = {
            "raw_message": raw_swift,
            "extracted_fields": final_state.get("extracted_fields", {}),
            "reasoning_report": final_state.get("reasoning_report", ""),
            "errors": final_state.get("errors", [])
        }
        
        # Save output to artifact directory
        artifact_path = r"C:\Users\Sanjiv\.gemini\antigravity\brain\46dfd21b-8136-4165-9dd2-fd03b0ea802c\test_valid_output.txt"
        with open(artifact_path, "w", encoding="utf-8") as out:
            out.write("=== EXTRACTED FIELDS ===\n")
            out.write(json.dumps(output_data["extracted_fields"], indent=2))
            out.write("\n\n=== REASONING REPORT ===\n")
            out.write(output_data["reasoning_report"])
            out.write("\n\n=== ERRORS ===\n")
            out.write(str(output_data["errors"]))
            
        print(f"Test completed. Output saved to {artifact_path}")
    except Exception as e:
        print(f"Error executing graph: {e}")
        # Save exception
        artifact_path = r"C:\Users\Sanjiv\.gemini\antigravity\brain\46dfd21b-8136-4165-9dd2-fd03b0ea802c\test_valid_output.txt"
        with open(artifact_path, "w", encoding="utf-8") as out:
            out.write(f"GRAPH EXECUTION ERROR: {str(e)}")

if __name__ == "__main__":
    run_test()
