from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from agents import get_llm, extract_swift_fields, reason_swift_failures

# Define the state structure for our multi-agent workflow
class AgentState(TypedDict):
    raw_message: str
    model_name: str
    base_url: str
    extracted_fields: Dict[str, Any]
    reasoning_report: str
    errors: List[str]

# Node 1: SWIFT message extractor agent
def extractor_node(state: AgentState) -> Dict[str, Any]:
    """Node that extracts fields from the raw SWIFT message."""
    raw_message = state.get("raw_message", "")
    model_name = state.get("model_name", "llama3.2")
    base_url = state.get("base_url", "http://localhost:11434")
    
    errors = state.get("errors", [])
    
    if not raw_message.strip():
        return {
            "errors": errors + ["Raw SWIFT message is empty."],
            "extracted_fields": {"error": "Input SWIFT message is empty."}
        }
        
    try:
        # Instantiate LLM
        llm = get_llm(model_name=model_name, base_url=base_url)
        # Perform extraction
        extracted = extract_swift_fields(raw_message, llm)
        
        if "error" in extracted:
            return {
                "extracted_fields": extracted,
                "errors": errors + [extracted["error"]]
            }
        
        return {
            "extracted_fields": extracted,
            "errors": errors
        }
    except Exception as e:
        error_msg = f"Extractor agent failed: {str(e)}"
        return {
            "extracted_fields": {"error": error_msg},
            "errors": errors + [error_msg]
        }

# Node 2: Reasoning agent
def reasoner_node(state: AgentState) -> Dict[str, Any]:
    """Node that evaluates parsed data and identifies omissions causing transaction failure."""
    raw_message = state.get("raw_message", "")
    extracted_fields = state.get("extracted_fields", {})
    model_name = state.get("model_name", "llama3.2")
    base_url = state.get("base_url", "http://localhost:11434")
    
    errors = state.get("errors", [])
    
    # If extractor ran into a critical error, skip reasoning LLM and return error report
    if "error" in extracted_fields:
        report = (
            "### Transaction Reasoning Interrupted\n\n"
            "Reasoning could not be completed because the Extractor Agent encountered a critical error:\n"
            f"**Error:** {extracted_fields['error']}\n\n"
            "**Remediation:** Please verify that Ollama is running locally (`ollama run llama3.2` or similar) "
            "and is accessible at the configured address."
        )
        return {
            "reasoning_report": report,
            "errors": errors
        }
        
    try:
        llm = get_llm(model_name=model_name, base_url=base_url)
        report = reason_swift_failures(extracted_fields, raw_message, llm)
        return {
            "reasoning_report": report,
            "errors": errors
        }
    except Exception as e:
        error_msg = f"Reasoner agent failed: {str(e)}"
        report = (
            "### Transaction Reasoning Failed\n\n"
            f"The Reasoning Agent failed with the following error: {str(e)}."
        )
        return {
            "reasoning_report": report,
            "errors": errors + [error_msg]
        }

# Router to decide next node
def router_edge(state: AgentState) -> str:
    """Decides whether to route to reasoning or stop early due to extraction issues."""
    extracted_fields = state.get("extracted_fields", {})
    # If we have a connection/critical error, we stop immediately.
    # Note: parsing failures are handled by the fallback regex, but connection failures have 'error'.
    if "error" in extracted_fields and "Ollama" in extracted_fields["error"]:
        return "stop"
    return "continue"

# Build and compile the LangGraph workflow
def create_workflow() -> StateGraph:
    # Initialize the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("reasoner", reasoner_node)
    
    # Set entry point
    workflow.set_entry_point("extractor")
    
    # Add conditional edge from extractor
    workflow.add_conditional_edges(
        "extractor",
        router_edge,
        {
            "continue": "reasoner",
            "stop": END
        }
    )
    
    # Add normal edge from reasoner to end
    workflow.add_edge("reasoner", END)
    
    # Compile
    return workflow.compile()

# Instantiated pre-compiled workflow
compiled_app = create_workflow()
