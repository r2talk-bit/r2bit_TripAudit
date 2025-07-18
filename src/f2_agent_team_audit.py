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
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import langgraph
from langgraph.graph import END, StateGraph

# Load environment variables from .env file in the project root
from pathlib import Path
project_root = Path(__file__).parent.parent.absolute()
load_dotenv(os.path.join(project_root, '.env'))

# Initialize the OpenAI client
client = ChatOpenAI(model="gpt-4")

# Define the workflow state structure
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
    relevant_policies: List[Dict[str, Any]]
    compliance_results: Dict[str, Any]
    
    # Output data
    email_content: Dict[str, str]
    error: str

# Define the agent functions

def parsing_agent(state: WorkflowState) -> Dict[str, Any]:
    """
    Parses a PDF expense report into structured JSON format using LLM.
    
    Args:
        state: Current workflow state containing the PDF path
        
    Returns:
        Updated state with structured expense data
    """
    try:
        # Extract the PDF path from the state
        pdf_path = state["pdf_path"]
        
        # Import pypdf for PDF text extraction (already in requirements.txt)
        import pypdf
        
        # Extract text from the PDF
        pdf_text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    pdf_text += page.extract_text() + "\n\n"
        except Exception as e:
            return {"error": f"Error extracting text from PDF: {str(e)}"}
        
        # Create a prompt for the parsing agent
        prompt = f"""
        You are a specialized expense parsing agent. Your task is to parse the expense report text
        extracted from a PDF into a structured JSON format. Extract all relevant information including:
        
        - Trip purpose
        - Trip dates
        - Total amount
        - Individual expense items with:
          - Description
          - Amount
          - Date
          - Category
          - Vendor/supplier
        
        Here is the text extracted from the expense report PDF:
        
        {pdf_text}
        
        Respond with a JSON object containing the structured data. Use the following format:
        
        {{
            "trip_purpose": "Brief description of trip purpose",
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD",
            "total_amount": 1234.56,
            "currency": "BRL",
            "employee_name": "Name of employee",
            "expense_items": [
                {{
                    "description": "Item description",
                    "amount": 123.45,
                    "date": "YYYY-MM-DD",
                    "category": "Category (e.g., meals, transportation, accommodation)",
                    "vendor": "Vendor/supplier name"
                }},
                // Additional expense items...
            ]
        }}
        
        Ensure all numeric values are properly formatted as numbers, not strings.
        If any information is missing, make reasonable inferences based on the available data.
        
        IMPORTANT: Pay special attention to identifying meal expenses, as the company has specific policies about them.
        Also carefully identify the total reimbursement amount, as there are maximum limits.
        """
        
        # Call the OpenAI API to parse the PDF content
        response = client.invoke([
            SystemMessage(content="You are a specialized expense parsing agent that extracts structured data from expense report PDFs."),
            HumanMessage(content=prompt)
        ])
        
        # Extract the JSON content from the response
        content = response.content
        
        # Clean up the response to extract just the JSON part
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({[\s\S]*})', content)
            if json_match:
                content = json_match.group(1)
        
        # Handle potential control characters and other JSON parsing issues
        try:
            # First attempt: direct parsing
            structured_expenses = json.loads(content)
        except json.JSONDecodeError:
            try:
                # Second attempt: strip control characters
                # Remove ASCII control characters (0-31) except tabs and newlines
                cleaned_content = ''
                for char in content:
                    if ord(char) >= 32 or char in '\t\n\r':
                        cleaned_content += char
                structured_expenses = json.loads(cleaned_content)
            except json.JSONDecodeError:
                # If all parsing attempts fail, create a minimal structured expense object
                structured_expenses = {
                    "trip_purpose": "Unknown",
                    "total_amount": 0,
                    "currency": "BRL",
                    "expense_items": []
                }

        parser_output = str(structured_expenses)
        print(f"parser agent output: {parser_output}")

        # Update the state with the structured expense data
        return {"structured_expenses": structured_expenses}
        
    except Exception as e:
        # If an error occurs, update the state with the error message
        return {"error": f"Error in parsing agent: {str(e)}"}


def policy_retrieval_agent(state: WorkflowState) -> Dict[str, Any]:
    """
    Retrieves relevant company expense policies based on the structured expense data.
    Uses get_relevant_policies function to retrieve uploaded policies.
    
    Args:
        state: Current workflow state containing the structured expense data
        
    Returns:
        Updated state with relevant policies
    """
    try:
        # Check if there was an error in the previous step
        if state.get("error"):
            return {}
        
        # Extract the structured expense data and user ID from the state
        structured_expenses = state["structured_expenses"]
        user_id = state.get("user_id", "default_user")
        
        # Import the policy retrieval function
        from load_policy import get_relevant_policies
        
        print(f"strucutured expenses {structured_expenses}")
        
        # Get relevant policies for the expense data
        relevant_policies = get_relevant_policies(structured_expenses, user_id)
        
        # Convert the vector DB results to the expected format
        formatted_policies = []
        for policy in relevant_policies:
            # Extract policy_id from metadata or from the id field
            policy_id = policy.get("metadata", {}).get("policy_id", None)
            if not policy_id and "id" in policy:
                # If policy_id not in metadata, try to extract from the id field
                # The id format is typically policy_uuid_chunkNumber
                id_parts = policy.get("id", "").split("_")
                if len(id_parts) > 1:
                    policy_id = "_".join(id_parts[:-1])  # Remove the chunk number
            
            # Get the category from metadata
            category = policy.get("metadata", {}).get("category", "General")
            
            # Get applies_to from metadata
            # ChromaDB stores this as a string, so we need to handle both string and list formats
            applies_to = policy.get("metadata", {}).get("applies_to", "All expenses")
            # Convert string to list if needed
            if isinstance(applies_to, str):
                applies_to = [applies_to]
            
            formatted_policies.append({
                "policy_id": policy_id or "unknown",
                "description": policy.get("content", ""),
                "category": category,
                "applies_to": applies_to
            })
        relevant_policies = formatted_policies
        
        # Update the state with the relevant policies
        return {"relevant_policies": relevant_policies}
        
    except Exception as e:
        # If an error occurs, update the state with the error message
        return {"error": f"Error retrieving policies: {str(e)}"}

def compliance_check_agent(state: WorkflowState) -> Dict[str, Any]:
    """
    Checks each expense item against company policies to determine compliance.
    Uses an LLM to dynamically evaluate compliance based on the relevant policies
    and structured expenses.
    
    Args:
        state: Current workflow state containing the structured expense data and relevant policies
        
    Returns:
        Updated state with compliance results
    """
    try:
        # Check if there was an error in the previous steps
        if state.get("error"):
            return {}
        
        # Extract the structured expense data and relevant policies from the state
        structured_expenses = state["structured_expenses"]
        relevant_policies = state["relevant_policies"]
        
        # Create a prompt for the compliance check agent
        prompt = f"""
        You are a specialized compliance check agent for expense reports. Your task is to evaluate
        whether the provided expense items comply with the company's expense policies.
        
        Here is the structured expense data:
        
        {json.dumps(structured_expenses, indent=2)}
        
        Here are the relevant company policies:
        
        {json.dumps(relevant_policies, indent=2)}
        
        Based on these policies, evaluate each expense item and the overall expense report for compliance.
        
        For each policy, determine if it is violated by any expense item or by the overall expense report.
        For each expense item, determine if it complies with all policies.
        
        Return a JSON object with the following structure:
        
        {{
            "is_compliant": true/false,
            "total_amount": total amount from the expense report,
            "currency": currency from the expense report (default: "BRL"),
            "violations": [
                {{
                    "policy_id": "ID of the violated policy",
                    "description": "Description of the policy",
                    "violation_details": "Details of how the policy was violated"
                }}
            ],
            "compliant_items": [
                // List of expense items that comply with all policies
                // Include the full item object for each compliant item
            ],
            "non_compliant_items": [
                {{
                    "item": {{
                        // Full expense item object
                    }},
                    "violations": [
                        {{
                            "policy_id": "ID of the violated policy",
                            "description": "Description of the policy",
                            "violation_details": "Details of how the policy was violated"
                        }}
                    ]
                }}
            ]
        }}
        
        Respond with only the JSON object, no additional text.
        """
        
        # Call the OpenAI API to evaluate compliance
        response = client.invoke([
            SystemMessage(content="You are a specialized compliance check agent that evaluates expense reports against company policies."),
            HumanMessage(content=prompt)
        ])
        
        # Extract the JSON content from the response
        content = response.content
        
        # Clean up the response to extract just the JSON part
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({[\s\S]*})', content)
            if json_match:
                content = json_match.group(1)
        
        # Handle potential control characters and other JSON parsing issues
        try:
            # First attempt: direct parsing
            compliance_results = json.loads(content)
        except json.JSONDecodeError:
            try:
                # Second attempt: strip control characters
                # Remove ASCII control characters (0-31) except tabs and newlines
                cleaned_content = ''
                for char in content:
                    if ord(char) >= 32 or char in '\t\n\r':
                        cleaned_content += char
                compliance_results = json.loads(cleaned_content)
            except json.JSONDecodeError:
                # If all parsing attempts fail, create a fallback compliance result
                compliance_results = {
                    "is_compliant": False,
                    "total_amount": structured_expenses.get("total_amount", 0),
                    "currency": structured_expenses.get("currency", "BRL"),
                    "violations": [{
                        "policy_id": "ERROR",
                        "description": "Error parsing compliance check results",
                        "violation_details": "The system encountered an error while evaluating compliance. Manual review required."
                    }],
                    "compliant_items": [],
                    "non_compliant_items": []
                }
        
        # Ensure the compliance_results has all required fields
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
        
        # Update the state with the compliance results
        return {"compliance_results": compliance_results}
        
    except Exception as e:
        return {"error": f"Error in compliance check agent: {str(e)}"}

def cleanup_agent(state: WorkflowState) -> Dict[str, Any]:
    """
    Cleans up user-specific data from the vector database after workflow completion.
    This is the final step in the workflow to ensure proper resource management.
    
    Args:
        state: Current workflow state containing the user_id
        
    Returns:
        Updated state with cleanup status
    """
    try:
        # Check if there was an error in the previous step
        if state.get("error"):
            return {}
        
        # Get the user ID from the state
        user_id = state.get("user_id")
        
        if not user_id:
            print("No user ID found in state, skipping cleanup")
            return {}
        
        # Import the policy loader module
        from load_policy import delete_user_policies
        
        # Delete all policies for this user
        try:
            deleted_count = delete_user_policies(user_id)
            print(f"Cleanup complete: Deleted {deleted_count} policies for user {user_id}")
            return {"cleanup_status": f"Successfully deleted {deleted_count} policies for user {user_id}"}
        except Exception as e:
            error_msg = f"Error during policy cleanup: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
    
    except Exception as e:
        # If an error occurs, update the state with the error message
        return {"error": f"Error in cleanup agent: {str(e)}"}

def commentary_synthesis_agent(state: WorkflowState) -> Dict[str, Any]:
    """
    Generates a final human-readable summary of the expense audit results.
    
    Args:
        state: Current workflow state containing the compliance results
        
    Returns:
        Updated state with the generated email content
    """
    try:
        # Check if there was an error in the previous steps
        if state.get("error"):
            # Generate a fallback email with the error message
            email_content = {
                "subject": "Error Processing Expense Report",
                "body": f"There was an error processing the expense report: {state['error']}",
                "recipient": "Finance Department",
                "approval_status": "Needs Review",
                "approval_comments": "An error occurred during processing. Please review manually."
            }
            return {"email_content": email_content}
        
        # Extract all the necessary data from the state
        structured_expenses = state["structured_expenses"]
        relevant_policies = state["relevant_policies"]
        compliance_results = state["compliance_results"]
        
        # Create a prompt for the commentary synthesis agent
        prompt = f"""
        You are a specialized commentary synthesis agent. Your task is to synthesize all the information
        about an expense report into a clear, human-readable summary and generate an approval email.
        
        Here is the structured expense data:
        
        {json.dumps(structured_expenses, indent=2)}
        
        Here are the relevant company policies:
        
        {json.dumps(relevant_policies, indent=2)}
        
        Here are the compliance check results:
        
        {json.dumps(compliance_results, indent=2)}
        
        Based on this information, generate an approval email with the following components:
        
        1. A subject line that clearly indicates the purpose and status of the email
        2. A recipient (typically the Finance Department or the employee's manager)
        3. A body that includes:
           - A summary of the expense report
           - A table or list of all expense items
           - Highlighted compliance issues, if any
           - For each non-compliant item, include the specific policy description that was violated
           - For each violation, include the policy ID, description, and violation details
           - A clear approval status (Approved, Needs Review, or Rejected)
           - Justification for the approval status
        4. Approval comments with any additional notes or instructions
        
        Respond with a JSON object containing the email content. Use the following format:
        
        {{
            "subject": "Expense Report Approval: [Status]",
            "body": "Email body text with all the required components...",
            "recipient": "Finance Department",
            "approval_status": "Approved/Needs Review/Rejected",
            "approval_comments": "Additional notes or instructions..."
        }}
        
        Format the expense table in the email body as plain text (not HTML or markdown).
        Make sure to clearly highlight all policy violations with their complete descriptions.
        """
        
        # Call the OpenAI API to generate the approval email
        response = client.invoke([
            SystemMessage(content="You are a specialized commentary synthesis agent that creates clear, professional approval emails for expense reports."),
            HumanMessage(content=prompt)
        ])
        
        # Extract the JSON content from the response
        content = response.content
        print(f"Raw LLM response: {content[:200]}...")
        
        # Clean up the response to extract just the JSON part
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1)
            print("Found JSON in markdown code block")
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({[\s\S]*})', content)
            if json_match:
                content = json_match.group(1)
                print("Found JSON in plain text")
            else:
                print("Could not find JSON pattern in response")
        
        # Handle potential control characters and other JSON parsing issues
        try:
            # First attempt: direct parsing
            email_content = json.loads(content)
            print("Successfully parsed JSON directly")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            try:
                # Second attempt: strip control characters
                # Remove ASCII control characters (0-31) except tabs and newlines
                cleaned_content = ''
                for char in content:
                    if ord(char) >= 32 or char in '\t\n\r':
                        cleaned_content += char
                email_content = json.loads(cleaned_content)
                print("Successfully parsed JSON after cleaning control characters")
            except json.JSONDecodeError as e2:
                print(f"Second JSON decode error: {str(e2)}")
                try:
                    # Third attempt: Try to fix common JSON formatting issues
                    # Replace single quotes with double quotes
                    fixed_content = cleaned_content.replace("'", "\"")
                    # Fix unquoted keys
                    fixed_content = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', fixed_content)
                    
                    # Try to manually extract and reconstruct the JSON
                    subject_match = re.search(r'"subject"\s*:\s*"([^"]+)"', fixed_content)
                    
                    # More robust body extraction that handles multiline content
                    # First try to find the body field
                    body_start_match = re.search(r'"body"\s*:\s*"', fixed_content)
                    if body_start_match:
                        body_start = body_start_match.end()
                        # Find the end of the body (a quote followed by a comma or closing brace)
                        nesting_level = 0
                        escaped = False
                        body_end = body_start
                        for i in range(body_start, len(fixed_content)):
                            char = fixed_content[i]
                            if escaped:
                                escaped = False
                                continue
                            if char == '\\':
                                escaped = True
                            elif char == '"' and not escaped:
                                # Check if this quote is followed by a comma or closing brace
                                for j in range(i+1, len(fixed_content)):
                                    next_char = fixed_content[j]
                                    if next_char.isspace():
                                        continue
                                    if next_char in [',', '}']:
                                        body_end = i
                                        break
                                    break
                                if body_end == i:
                                    break
                        body_text = fixed_content[body_start:body_end]
                        body_match = True
                    else:
                        body_match = None
                        body_text = ""
                    
                    recipient_match = re.search(r'"recipient"\s*:\s*"([^"]+)"', fixed_content)
                    approval_status_match = re.search(r'"approval_status"\s*:\s*"([^"]+)"', fixed_content)
                    approval_comments_match = re.search(r'"approval_comments"\s*:\s*"([^"]+)"', fixed_content)
                    
                    if subject_match and body_match:
                        email_content = {
                            "subject": subject_match.group(1),
                            "body": body_text,  # Use the extracted body text
                            "recipient": recipient_match.group(1) if recipient_match else "Finance Department",
                            "approval_status": approval_status_match.group(1) if approval_status_match else "Needs Review",
                            "approval_comments": approval_comments_match.group(1) if approval_comments_match else "Please review manually."
                        }
                        print("Successfully extracted JSON fields using regex")
                    else:
                        # If regex extraction fails, try standard JSON parsing
                        email_content = json.loads(fixed_content)
                        print("Successfully parsed JSON after fixing formatting")
                except json.JSONDecodeError as e3:
                    print(f"Third JSON decode error: {str(e3)}")
                    # If all parsing attempts fail, create a fallback email content
                    email_content = {
                        "subject": "Expense Report Analysis",
                        "body": f"The expense report has been processed, but there was an issue formatting the results. Please review the raw data manually.\n\nRaw LLM response: {content[:500]}...",
                        "recipient": "Finance Department",
                        "approval_status": "Needs Review",
                        "approval_comments": "JSON parsing error occurred. Manual review required."
                    }
        
        # Add additional debugging to see what we're returning
        print(f"Final email content subject: {email_content.get('subject')}")
        print(f"Final email content body (first 100 chars): {email_content.get('body', '')[:100]}...")
        
        # If we have a raw LLM response but no properly formatted email, try to extract useful content directly
        if email_content.get('body') and 'raw data manually' in email_content.get('body') and content:
            # Try to directly use the LLM response as the email body
            if len(content) > 50:  # Make sure we have substantial content
                print("Using direct LLM response as email content")
                # Extract the expense report information directly
                email_content = {
                    "subject": "Expense Report Analysis Results",
                    "body": content.replace('```json', '').replace('```', ''),
                    "recipient": "Finance Department",
                    "approval_status": "Needs Review",
                    "approval_comments": "Generated from raw LLM response."
                }
        
        # Update the state with the email content
        return {"email_content": email_content}
        
    except Exception as e:
        # If an error occurs, generate a fallback email
        email_content = {
            "subject": "Error Processing Expense Report",
            "body": f"There was an error generating the approval email: {str(e)}",
            "recipient": "Finance Department",
            "approval_status": "Needs Review",
            "approval_comments": "An error occurred during processing. Please review manually."
        }
        return {"email_content": email_content, "error": f"Error in commentary synthesis agent: {str(e)}"}

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


def parse_expense_data(expense_summary: str) -> Dict[str, Any]:
    """
    Parse the expense summary into structured data using OpenAI.
    
    Args:
        expense_summary: Raw text of the expense report
        
    Returns:
        Structured expense data as a dictionary
    """
    prompt = f"""
    Analise o seguinte relatório de despesas e extraia as informações em formato JSON estruturado.
    Inclua os seguintes campos:
    - trip_purpose: O propósito da viagem
    - total_amount: O valor total das despesas
    - currency: A moeda utilizada (padrão: BRL)
    - expense_items: Uma lista de itens de despesa, cada um contendo:
      - description: Descrição da despesa
      - amount: Valor da despesa
      - category: Categoria da despesa (ex: transporte, hospedagem, refeição, etc.)
      - date: Data da despesa (se disponível)
    
    Relatório de despesas:
    {expense_summary}
    
    Retorne apenas o JSON, sem texto adicional.
    """
    
    # Call OpenAI to parse the expense data
    response = client.invoke([SystemMessage(content="You are a specialized parsing agent that extracts structured data from expense reports."), 
                             HumanMessage(content=prompt)])
    
    # Extract the JSON content from the response
    content = response.content
    
    # Clean up the response to extract just the JSON part
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if json_match:
        content = json_match.group(1)
    else:
        # Try to find JSON without markdown formatting
        json_match = re.search(r'({[\s\S]*})', content)
        if json_match:
            content = json_match.group(1)
    
    # Handle potential control characters and other JSON parsing issues
    try:
        # First attempt: direct parsing
        structured_data = json.loads(content)
    except json.JSONDecodeError:
        try:
            # Second attempt: strip control characters
            cleaned_content = ''
            for char in content:
                if ord(char) >= 32 or char in '\t\n\r':
                    cleaned_content += char
            structured_data = json.loads(cleaned_content)
        except json.JSONDecodeError:
            # If all parsing attempts fail, create a minimal structured expense object
            structured_data = {
                "trip_purpose": "Unknown",
                "total_amount": 0,
                "currency": "BRL",
                "expense_items": []
            }
    
    return structured_data


def generate_email(structured_data: Dict[str, Any], compliance_results: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate an approval email based on the structured data and compliance results.
    
    Args:
        structured_data: Structured expense data
        compliance_results: Compliance check results
        
    Returns:
        Email content as a dictionary
    """
    # Determine approval status based on compliance
    if compliance_results["compliant"]:
        approval_status = "Approved"
        subject = "Expense Report Approved"
    else:
        approval_status = "Needs Review"
        subject = "Expense Report Requires Review"
    
    # Format expense items as a table
    expense_table = "Detalhes das Despesas:\n\n"
    expense_table += "Descrição | Valor | Categoria | Data\n"
    expense_table += "-" * 50 + "\n"
    
    for item in structured_data.get("expense_items", []):
        description = item.get("description", "N/A")
        amount = item.get("amount", 0)
        category = item.get("category", "N/A")
        date = item.get("date", "N/A")
        expense_table += f"{description} | R${amount:.2f} | {category} | {date}\n"
    
    # Format violations
    violations_text = ""
    if not compliance_results["compliant"]:
        violations_text = "\n\nViolações de Política Encontradas:\n\n"
        for violation in compliance_results["violations"]:
            violations_text += f"- {violation['policy']}: {violation['description']}\n"
    
    # Generate the email body
    body = f"""Prezado Departamento Financeiro,

O relatório de despesas para a viagem "{structured_data.get('trip_purpose', 'N/A')}" foi analisado.

Valor Total: R${structured_data.get('total_amount', 0):.2f}
Status: {approval_status}

{expense_table}
{violations_text}

Comentários:
{', '.join([str(comment) for comment in compliance_results.get('comments', [])])}

Por favor, processe este relatório de acordo com as políticas da empresa.

Atenciosamente,
Sistema de Auditoria de Despesas"""
    
    # Create the email content
    email_content = {
        "subject": subject,
        "body": body,
        "recipient": "Finance Department",
        "approval_status": approval_status,
        "approval_comments": ", ".join([str(comment) for comment in compliance_results.get("comments", [])])
    }
    
    return email_content
