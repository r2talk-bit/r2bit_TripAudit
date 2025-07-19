"""
Multi-agent LangGraph workflow for expense report auditing.

This module implements a LangGraph workflow with four specialized agents:
1. ParsingAgent: Parses raw expense report text into structured JSON format
2. PolicyRetrievalAgent: Retrieves relevant company expense policies
3. ComplianceCheckAgent: Checks expenses against company policies
4. CommentarySynthesisAgent: Synthesizes information into a clear summary and email

The workflow maintains the same interface as f2_agentic_audit.py to ensure
compatibility with the audit_expenses.py module.
"""

import json
import os
import re
import yaml
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, Callable, TypeVar
from functools import wraps

# Using local clean_markdown_text implementation

# Import the policy management functions
from policy_management import get_relevant_policies
# Import utility functions from graph_utils
from graph_utils import parse_llm_json_response, parse_json_response, extract_pdf_text

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import langgraph
from langgraph.graph import END, StateGraph

from graph_utils import parse_llm_json_response, parse_json_response, extract_pdf_text, clean_markdown_text

# Load environment variables from .env file in the project root
from pathlib import Path
project_root = Path(__file__).parent.parent.absolute()
load_dotenv(os.path.join(project_root, '.env'))

# Load prompts from YAML file
prompts_path = os.path.join(Path(__file__).parent.absolute(), 'prompts.yaml')
with open(prompts_path, 'r', encoding='utf-8') as file:
    prompts = yaml.safe_load(file)
    

def remove_duplicate_policies(policies):
    """
    Remove duplicate policies based on their ID and chunk index.
    
    Args:
        policies: List of policy dictionaries returned from get_relevant_policies
        
    Returns:
        List of unique policies with duplicates removed
    """
    unique_policies = []
    seen_ids = set()  # Set to track unique policy IDs
    
    for policy in policies:
        # Create a unique identifier using policy_id and chunk_index from metadata
        if 'id' in policy and 'metadata' in policy and 'chunk_index' in policy['metadata']:
            unique_id = f"{policy['id']}_{policy['metadata']['chunk_index']}"
            if unique_id not in seen_ids:
                seen_ids.add(unique_id)
                unique_policies.append(policy)
        else:
            # If the policy doesn't have the expected structure, include it anyway
            unique_policies.append(policy)
            
    return unique_policies


#
# Define the workflow state structure
#
class WorkflowState(TypedDict):
    """
    State object for the expense audit workflow.
    Contains all data passed between workflow nodes.
    """
    # Input data
    pdf_path: str  # Path to the PDF file to process
    user_id: str  # User ID for policy management
    
    # Intermediate data
    structured_expenses: Dict[str, Any]
    relevant_policies: List[Dict[str, Any]]  # List of policy objects with policy_id, description, category, applies_to, applicability_reason, priority
    policy_analysis: Dict[str, Any]  # Analysis of expense data and policy queries including query_summary, expense_categories_identified, total_amount_analysis
    compliance_results: Dict[str, Any]
    
    # Output data
    email_content: Dict[str, str]
    error: str


# Initialize the OpenAI client
# client = ChatOpenAI(model="gpt-4o")
client = ChatOpenAI(model="gpt-3.5-turbo")

# Import OpenAI for direct API access if needed
import openai
openai_client = openai.OpenAI()

#
# LLM Call
#
def call_llm(system_message: str, user_message: str, fallback_response: str = None) -> str:
    try:
        response = client.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ])
        return response.content
    except Exception as e:
        print(f"LLM call failed: {e}")
        if fallback_response:
            return fallback_response
        raise

def handle_agent_errors(agent_name: str):
    """Decorator to handle common agent errors."""
    def decorator(func):
        def wrapper(state: WorkflowState) -> Dict[str, Any]:
            if state.get("error"):
                return {}
            try:
                return func(state)
            except Exception as e:
                return {"error": f"Error in {agent_name}: {str(e)}"}
        return wrapper
    return decorator

# Type variable for the decorator
T = TypeVar('T')

#
# Define helper functions
#
def format_policies(policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format policies for the compliance check agent.
    
    Args:
        policies: List of policy objects from vector database
        
    Returns:
        Formatted list of policies
    """
    formatted_policies = []
    
    for policy in policies:
        # Get the policy description and metadata
        description = policy.get("content", "")
        policy_title = policy.get("metadata", {}).get("policy_name", "Expense Policy")
        category = policy.get("metadata", {}).get("category", "General")
        
        # Extract policy_id from metadata or from the id field
        policy_id = policy.get("metadata", {}).get("policy_id", None)
        if not policy_id and "id" in policy:
            # If policy_id not in metadata, try to extract from the id field
            id_parts = policy.get("id", "")
            policy_id = id_parts  # Use the full ID
        
        formatted_policies.append({
            "policy_id": policy_id or f"policy-{len(formatted_policies)}",
            "policy_title": policy_title,
            "description": description,
            "content": description,  # Include full content for compliance checking
            "category": category,
            "applicability_reason": "Relevant to expense items in report",
            "priority": "medium"
        })
    
    return formatted_policies

def get_policy_type(policy: Dict[str, Any]) -> str:
    """
    Determine the type of policy based on its content and metadata.
    
    Args:
        policy: A policy dictionary
        
    Returns:
        Policy type identifier (e.g., 'meals', 'max_amount')
    """
    description = policy.get("description", "").lower()
    title = policy.get("policy_title", "").lower()
    policy_id = policy.get("policy_id", "").lower()
    
    if "meal" in description or "food" in description or "meal" in title or "meal" in policy_id:
        return "meals"
    elif "maximum" in description or "limit" in description or "5000" in description or "max" in policy_id:
        return "max_amount"
    else:
        return "other"


# Define the agent functions

@handle_agent_errors("parsing_agent")
def parsing_agent(state: WorkflowState) -> Dict[str, Any]:
    """Parse PDF expense report into structured JSON."""
    # Extract PDF text
    pdf_text = extract_pdf_text(state["pdf_path"])
    
    # Call LLM with unified function
    response = call_llm(
        prompts['parsing_agent']['system_message'],
        prompts['parsing_agent']['prompt'].format(pdf_text=pdf_text)
    )
    
    # Parse with unified parser
    structured_expenses = parse_llm_json_response(response, {
        "trip_purpose": "Unknown",
        "total_amount": 0,
        "currency": "BRL",
        "expense_items": []
    })
    
    # Debug output
    from graph_utils import generate_expense_report
    report = generate_expense_report(structured_expenses)
    print(report)
    
    return {"structured_expenses": structured_expenses}


@handle_agent_errors("policy_retrieval_agent")
def policy_retrieval_agent(state: WorkflowState) -> Dict[str, Any]:
    """Retrieve relevant expense policies based on structured data."""
    

    # Extract data from state
    structured_expenses = state["structured_expenses"]
    user_id = state.get("user_id", "default_user")
    
    # Use LLM to analyze expenses and generate queries
    system_message = prompts["policy_retrieval_agent"]["system_message"]
    
    # Convert the structured_expenses dictionary to a formatted JSON string
    import json
    structured_expenses_str = json.dumps(structured_expenses, indent=4)
    
    # Format the prompt with the JSON string
    user_message = prompts["policy_retrieval_agent"]["prompt"].format(structured_expenses=structured_expenses_str)
    
    # Call LLM to analyze expenses and generate queries
    llm_response = call_llm(system_message, user_message)
    
    search_queries = []

    # Process the plain text response from LLM
    if llm_response and llm_response.strip():
        # Split the response into lines and clean them
        lines = [line.strip() for line in llm_response.split('\n') if line.strip()]
        
        # Extract lines that look like queries (typically containing a question or keywords)
    
        for line in lines:
            # Remove bullet points, numbers, and other common prefixes
            clean_line = re.sub(r'^[-*•●■□▪▫◆◇◈⬤⚫⚪⦿⊙⊚⊛⊜⊝⊗⊘⊙⊚⊛⊜⊝0-9.\s"]*', '', line).strip()
            clean_line = clean_line.strip('"').strip("'").strip()
            
            # Skip empty lines or lines that are too short
            if clean_line and len(clean_line) > 5:
                search_queries.append(clean_line)
    
    # If no queries were extracted, use defaults
    if not search_queries:
        search_queries.append("list the refund rules")
    
    # Remove duplicates while preserving order
    seen = set()
    search_queries = [q for q in search_queries if not (q in seen or seen.add(q))]
    
    # Get and combine policies for all queries
    all_policies = []
    for query in search_queries:
        policies = get_relevant_policies(query, user_id)
        all_policies.extend(policies)
    
    # Remove duplicate policies
    all_policies = remove_duplicate_policies(all_policies)
        
    # Format policies for compliance checking
    formatted_policies = format_policies(all_policies)
    
    # Generate a readable report of the policies
    from graph_utils import generate_formatted_policies_report
    policy_report = generate_formatted_policies_report(formatted_policies)
    print("\n" + "=" * 80)
    print("FORMATTED POLICY REPORT:")
    print("=" * 80)
    print(policy_report)
    print("=" * 80)
    
    # Generate and print a human-readable report of the policies
    from graph_utils import generate_policies_report
    policy_data = {
        "relevant_policies": formatted_policies
    }
    report = generate_policies_report(policy_data)
    print("*" * 80)
    print(report)
    print("*" * 80)

    return policy_data
    
@handle_agent_errors("compliance_check_agent")
def compliance_check_agent(state: WorkflowState) -> Dict[str, Any]:
    """Checks expenses against company policies to determine compliance."""
    
    print("STARTING COMPLIANCE CHECK AGENT")
    # Extract data from state
    structured_expenses = state["structured_expenses"]
    relevant_policies = state["relevant_policies"]
    policy_analysis = state.get("policy_analysis", {})
    
    # Sort policies by priority
    sorted_policies = sorted(relevant_policies, key=lambda p: 0 if p.get("priority") == "high" else 1)
    
    print(f"RELEVANT POLICIES (sorted by priority):")
    for i, policy in enumerate(sorted_policies):
        print(f"{i+1}. [{policy.get('priority', 'medium')}] {policy.get('policy_id')}: {policy.get('category')}")
    
    # Create policy context
    policy_context = {
        "policies": sorted_policies,
        "analysis": {
            "expense_categories": policy_analysis.get("expense_categories_identified", []),
            "query_summary": policy_analysis.get("query_summary", ""),
            "total_amount_analysis": policy_analysis.get("total_amount_analysis", {})
        }
    }
    
    # Call LLM for compliance check
    prompt = prompts['compliance_check_agent']['prompt'].format(
        structured_expenses=json.dumps(structured_expenses, indent=2),
        relevant_policies=json.dumps(policy_context, indent=2)
    )
    
    # Define fallback response for LLM failures
    fallback_content = '{"compliant": false, "violations": [{"policy": "Error", "description": "Failed to evaluate compliance due to an error."}]}'
    
    content = call_llm(
        prompts['compliance_check_agent']['system_message'], 
        prompt, 
        fallback_content
    )
    
    # Parse results with fallback
    fallback = {
        "compliant": False,
        "violations": [{"policy": "Parsing Error", "description": "Failed to parse LLM response."}],
        "comments": ["An error occurred during compliance analysis. Please review manually."]
    }
    
    print("Using robust JSON parser for compliance check results")
    compliance_results = parse_llm_json_response(content, fallback)
    
    # Ensure required fields exist
    if "is_compliant" not in compliance_results:
        compliance_results["is_compliant"] = False
    if "total_amount" not in compliance_results:
        compliance_results["total_amount"] = structured_expenses.get("total_amount", 0)
    if "currency" not in compliance_results:
        compliance_results["currency"] = structured_expenses.get("currency", "BRL")
    if "violations" not in compliance_results:
        compliance_results["violations"] = []
    if "compliant_items" not in compliance_results:
        compliance_results["compliant_items"] = []
    if "non_compliant_items" not in compliance_results:
        compliance_results["non_compliant_items"] = []
    

    print("*" *50)
    print("CHECK COMPLIANCE RESULT")
    print(json.dumps(compliance_results, indent=4, ensure_ascii=False))
    print("*" *50)
    

    return {"compliance_results": compliance_results}

@handle_agent_errors("cleanup_agent")
def cleanup_agent(state: WorkflowState) -> Dict[str, Any]:
    """Cleans up user-specific data from the vector database after workflow completion."""
    # Get the user ID from the state
    user_id = state.get("user_id")
    
    if not user_id:
        print("No user ID found in state, skipping cleanup")
        return {}
    
    # Import the policy loader module
    from policy_management import delete_user_policies
    
    # Delete all policies for this user
    try:
        deleted_count = delete_user_policies(user_id)
        print(f"Cleanup complete: Deleted {deleted_count} policies for user {user_id}")
        return {"cleanup_status": f"Successfully deleted {deleted_count} policies for user {user_id}"}
    except Exception as e:
        error_msg = f"Error during policy cleanup: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

@handle_agent_errors("commentary_synthesis_agent")
def commentary_synthesis_agent(state: WorkflowState) -> Dict[str, Any]:
    """Generates a final human-readable summary of the expense audit results."""
    # Special case: if there's an error, generate a fallback email
    if state.get("error"):
        email_content = {
            "subject": "Error Processing Expense Report",
            "body": f"There was an error processing the expense report: {state['error']}",
            "recipient": "Finance Department",
            "approval_status": "Needs Review",
            "approval_comments": "An error occurred during processing. Please review manually."
        }
        return {"email_content": email_content}
    
    try:
        # Extract data from state
        structured_expenses = state["structured_expenses"]
        relevant_policies = state["relevant_policies"]
        policy_analysis = state.get("policy_analysis", {})
        compliance_results = state["compliance_results"]
        
        # Check if we have the necessary data to generate an email
        if not structured_expenses or not relevant_policies or not compliance_results:
            raise ValueError("Missing required data for email generation")
        
        # Sort policies by priority
        sorted_policies = sorted(relevant_policies, key=lambda p: 0 if p.get("priority") == "high" else 1)
        
        # Create enhanced context with safe defaults
        enhanced_context = {
            "policies": sorted_policies,
            "analysis": {
                "expense_categories": policy_analysis.get("expense_categories_identified", [])
            }
        }
        
        # Create a simplified version of the data for the LLM to reduce complexity
        simplified_expenses = {
            "trip_purpose": structured_expenses.get("trip_purpose", "Unknown"),
            "total_amount": structured_expenses.get("total_amount", 0),
            "currency": structured_expenses.get("currency", "BRL"),
            "expense_items": []
        }
        
        # Add only essential expense item data
        if "expense_items" in structured_expenses and isinstance(structured_expenses["expense_items"], list):
            for item in structured_expenses["expense_items"]:
                if isinstance(item, dict):
                    simplified_item = {
                        "description": item.get("description", "Unknown"),
                        "amount": item.get("amount", 0),
                        "category": item.get("category", "Unknown")
                    }
                    simplified_expenses["expense_items"].append(simplified_item)
        
        # Simplify compliance results
        simplified_compliance = {
            "compliant": compliance_results.get("compliant", False),
            "approval_recommendation": compliance_results.get("approval_recommendation", "Needs Review"),
            "violations": compliance_results.get("violations", [])
        }
        
        # Call LLM for synthesis with simplified data
        prompt = prompts['commentary_synthesis_agent']['prompt'].format(
            structured_expenses=json.dumps(simplified_expenses, indent=2),
            relevant_policies=json.dumps(enhanced_context, indent=2),
            compliance_results=json.dumps(simplified_compliance, indent=2)
        )
        
        # Define a well-structured fallback for the LLM call
        email_fallback = json.dumps({
            "subject": "Análise de Despesas de Viagem",
            "body": "Relatório de análise de despesas de viagem.",
            "recipient": "Finance Department",
            "approval_status": compliance_results.get("approval_recommendation", "Needs Review"),
            "approval_comments": "Please review the expense report.",
            "evaluated_policies": "Relevant policies were considered in this analysis."
        })
        
        # Call LLM with fallback
        content = call_llm(
            prompts['commentary_synthesis_agent']['system_message'],
            prompt,
            email_fallback
        )
        
        # Define parsing fallback
        fallback = {
            "subject": "Análise de Despesas de Viagem",
            "body": f"Análise de despesas no valor total de {structured_expenses.get('total_amount', 0)} {structured_expenses.get('currency', 'BRL')}.",
            "recipient": "Finance Department",
            "approval_status": compliance_results.get("approval_recommendation", "Needs Review"),
            "approval_comments": "Please review the expense report manually.",
            "evaluated_policies": "Relevant policies were considered in this analysis."
        }
        
        # Parse email content with better fallback
        email_content = parse_llm_json_response(content, fallback)
        
        # Clean up markdown text
        if 'body' in email_content and email_content['body']:
            email_content['body'] = clean_markdown_text(email_content['body'])
        
        if 'evaluated_policies' in email_content and email_content['evaluated_policies']:
            email_content['evaluated_policies'] = clean_markdown_text(email_content['evaluated_policies'])
            
        if 'approval_comments' in email_content and email_content['approval_comments']:
            email_content['approval_comments'] = clean_markdown_text(email_content['approval_comments'])
        
        # Ensure critical fields are present
        if "subject" not in email_content or not email_content["subject"]:
            email_content["subject"] = fallback["subject"]
        
        if "body" not in email_content or not email_content["body"]:
            email_content["body"] = fallback["body"]
            
        if "approval_status" not in email_content:
            email_content["approval_status"] = compliance_results.get("approval_recommendation", "Needs Review")
        
        return {"email_content": email_content}
        
    except Exception as e:
        # Generate a meaningful fallback email on error
        print(f"Error in commentary synthesis: {str(e)}")
        email_content = {
            "subject": "Análise de Despesas de Viagem",
            "body": "Foi realizada uma análise das despesas de viagem, mas ocorreu um erro na geração do relatório detalhado. Por favor, revise manualmente.",
            "recipient": "Finance Department",
            "approval_status": "Needs Review",
            "approval_comments": "An error occurred during email generation. Please review manually.",
            "evaluated_policies": "As políticas da empresa foram consideradas na análise."
        }
        return {"email_content": email_content}
    
# Note: We're using a try-except inside the decorator because we want to return a
# custom fallback email rather than propagating the error to the decorator

# Define the edge functions for conditional workflow routing

def should_continue(state: WorkflowState) -> str:
    """
    Determines whether the workflow should continue to the next step or end due to an error.
    
    Args:
        state: Current workflow state
        
    Returns:
        Name of the next node or END if an error occurred
    """
    # If an error has occurred, end the workflow
    if state.get("error"):
        return "commentary_synthesis_agent"
    
    # Otherwise, continue to the next step in the workflow
    return "continue"

# Define the main workflow function

def create_agent_workflow() -> StateGraph:
    """
    Builds the LangGraph workflow for expense auditing with four specialized agents.
    
    Returns:
        A configured StateGraph workflow
    """
    # Create a new workflow graph
    workflow = StateGraph(WorkflowState)
    
    # Add the nodes to the workflow
    workflow.add_node("parsing_agent", parsing_agent)
    workflow.add_node("policy_retrieval_agent", policy_retrieval_agent)
    workflow.add_node("compliance_check_agent", compliance_check_agent)
    workflow.add_node("commentary_synthesis_agent", commentary_synthesis_agent)
    workflow.add_node("cleanup_agent", cleanup_agent)  # Add the cleanup agent
    
    # Define the edges between nodes
    workflow.add_edge("parsing_agent", "policy_retrieval_agent")
    workflow.add_conditional_edges(
        "policy_retrieval_agent",
        should_continue,
        {
            "continue": "compliance_check_agent",
            "commentary_synthesis_agent": "commentary_synthesis_agent"
        }
    )
    workflow.add_conditional_edges(
        "compliance_check_agent",
        should_continue,
        {
            "continue": "commentary_synthesis_agent",
            "commentary_synthesis_agent": "commentary_synthesis_agent"
        }
    )
    
    # Add edge from commentary synthesis to cleanup agent
    workflow.add_edge("commentary_synthesis_agent", "cleanup_agent")
    
    # Add edge from cleanup agent to END
    workflow.add_edge("cleanup_agent", END)
    
    # Remove direct edge from commentary synthesis to END
    
    # Set the entry point
    workflow.set_entry_point("parsing_agent")
    
    return workflow

# Create the workflow
agent_team_workflow = create_agent_workflow().compile()

def run_agentic_auditor(pdf_path: str, user_id: str = None) -> Dict[str, Any]:
    """
    Main entry point for the expense audit workflow.
    
    Args:
        pdf_path: Path to the PDF expense report file
        user_id: Optional user ID for policy management
        
    Returns:
        Dictionary containing the generated email content and any errors
    """
    print("Starting agent team auditor with LangGraph workflow...")
    
    # Check if the OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        error_message = "The OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable."
        print(f"Error: {error_message}")
        return {
            "email_content": {
                "subject": "Error: API Key Missing",
                "body": error_message,
                "recipient": "System Administrator",
                "approval_status": "Error",
                "approval_comments": "Configuration error: API key missing."
            },
            "error": error_message
        }
    
    # Verify the PDF file exists
    if not os.path.exists(pdf_path):
        error_message = f"The PDF file does not exist at path: {pdf_path}"
        print(f"Error: {error_message}")
        return {
            "email_content": {
                "subject": "Error: PDF File Not Found",
                "body": error_message,
                "recipient": "System Administrator",
                "approval_status": "Error",
                "approval_comments": "File not found error."
            },
            "error": error_message
        }
    
    # If no user_id is provided, generate a default one
    if not user_id:
        import uuid
        user_id = str(uuid.uuid4())
        print(f"No user_id provided, generated: {user_id}")
    
    try:
        # Build and compile the workflow graph
        print("Building and compiling the LangGraph workflow...")
        workflow = create_agent_workflow()
        app = workflow.compile()
        
        # Initialize the workflow state with the PDF path and user_id
        initial_state = WorkflowState(
            pdf_path=pdf_path,
            user_id=user_id,  # Include user_id in the initial state
            structured_expenses={},
            relevant_policies=[],
            compliance_results={},
            email_content={},
            error=""
        )
        
        # Run the workflow with the initial state
        print("Executing the LangGraph workflow...")
        result = app.invoke(initial_state)
        
        # Extract and return the relevant results from the final state
        print("Workflow execution completed successfully.")
        return {
            "email_content": result.get("email_content", {}),
            "error": result.get("error", "")
        }
        
    except Exception as e:
        # Get the full stack trace for detailed debugging
        import traceback
        error_trace = traceback.format_exc()
        
        # Log the error to the console for debugging
        print(f"Workflow execution error: {str(e)}")
        print(error_trace)
        
        return {
            "email_content": {
                "subject": "Expense Report Analysis - Manual Review Required",
                "body": f"""O relatório de despesas foi processado, mas ocorreu um erro durante a análise automática.
                
Detalhes do erro: {str(e)}

IMPORTANTE: Por favor, revise os dados do relatório manualmente e aplique as políticas da empresa durante a revisão.
""",
                "recipient": "Finance Department",
                "approval_status": "Needs Review",
                "approval_comments": "Erro técnico durante o processamento. Aplicar políticas de reembolso manualmente."
            },
            "error": f"Workflow execution error: {str(e)}"
        }
# End of file
