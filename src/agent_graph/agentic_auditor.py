"""
Agentic Auditor for R2Bit TripAudit.

This module implements a LangGraph workflow that processes expense report summaries
and generates approval emails using OpenAI models.
"""

import os
import sys
import json
import traceback
from typing import Dict, TypedDict, List, Optional, Annotated, Any
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables from .env file
project_root = Path(__file__).parent.parent.parent.absolute()
load_dotenv(os.path.join(project_root, '.env'))

# Import from our project
from src.audit_expenses import ExpenseAuditor


# Define proper state schema for LangGraph compliance
class WorkflowState(TypedDict):
    expense_summary: Dict[str, Any]
    email_content: Dict[str, Any]
    error: str


class ExpenseSummary(TypedDict):
    pdf_path: str
    processed_dir: str
    text_report_path: str
    summary: Dict[str, Any]


class EmailContent(TypedDict):
    subject: str
    body: str
    recipient: str
    approval_status: str
    approval_comments: str


# Define workflow nodes - now return only changed state fields
def analyze_expense_summary(state: WorkflowState) -> Dict[str, Any]:
    """
    Analyze the expense report summary.
    
    Args:
        state: The current graph state
        
    Returns:
        Dictionary with updated state fields
    """
    try:
        expense_summary = state.get("expense_summary", {})
        
        # Check if the summary is valid
        if not expense_summary or not expense_summary.get("summary"):
            return {"error": "Invalid expense summary: Missing or empty summary"}
        
        # If analysis passes, clear any previous errors
        return {"error": ""}
        
    except Exception as e:
        return {"error": f"Error analyzing expense summary: {str(e)}"}


def generate_approval_email(state: WorkflowState) -> Dict[str, Any]:
    """
    Generates an approval email based on the expense summary using OpenAI.
    
    Args:
        state: The current graph state
        
    Returns:
        Dictionary with updated state fields
    """
    try:
        # Check if we have an API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OpenAI API key not found. Set the OPENAI_API_KEY environment variable."}
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Get the expense summary directly from the state
        expense_summary = state.get("expense_summary", {})
        
        # Use the text report content directly as received from audit_expenses.py
        report_content = expense_summary.get("text_report_content", "")
        
        # Create the prompt for OpenAI using the discovery_summary.txt content
        prompt = f"""
        Generate a professional email for expense report approval based on the following expense report discovery summary:
        
        {report_content}
        
        The email should include:
        1. A professional subject line
        2. A brief introduction explaining the purpose of the email
        3. A clearly formatted expense summary using plain text formatting only (NO HTML). Format it as follows:
           
           Expense Category | Date   | Amount (R$) | Vendor
           -----------------|--------|-------------|-------
           [Category 1]     | [Date] | [Amount]    | [Vendor]
           [Category 2]     | [Date] | [Amount]    | [Vendor]
           -----------------|--------|-------------|-------
           TOTAL            |        | [Total]     |
           
        4. A summary of the total expense amount
        5. A recommendation for approval or further review
        6. A polite closing
        
        Format the response as a JSON with the following fields:
        - subject: The email subject line
        - body: The email body text in plain text format only (NO HTML tags)
        - recipient: "Finance Department"
        - approval_status: Either "Approved", "Needs Review", or "Rejected"
        - approval_comments: Brief justification for the approval status
        
        IMPORTANT: Do not include any HTML tags in the body. Use only plain text formatting with spaces, dashes, and pipe characters for the table.
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates professional approval emails for expense reports. Return your response as a valid JSON object with fields for subject, body, recipient, approval_status, and approval_comments."},
                {"role": "user", "content": prompt}
            ]
            # Removed response_format parameter as it's not supported by the model
        )
        
        # Parse the response
        try:
            email_content = json.loads(response.choices[0].message.content)
            
            # Ensure all required fields are present
            required_fields = ["subject", "body", "recipient", "approval_status", "approval_comments"]
            for field in required_fields:
                if field not in email_content:
                    email_content[field] = f"Missing {field}"
                    
            return {
                "email_content": email_content,
                "error": ""
            }
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, create a structured error response
            email_content = {
                "subject": "Error Processing Expense Report",
                "body": f"There was an error generating the approval email: {str(e)}\n\nRaw response: {response.choices[0].message.content}",
                "recipient": "Finance Department",
                "approval_status": "Needs Review",
                "approval_comments": "Error in AI processing. Please review manually."
            }
            return {
                "email_content": email_content,
                "error": f"JSON parsing error: {str(e)}"
            }
        
    except Exception as e:
        return {"error": f"Error in generate_approval_email: {str(e)}"}


def handle_error(state: WorkflowState) -> Dict[str, Any]:
    """
    Handles errors in the workflow.
    
    Args:
        state: The current graph state
        
    Returns:
        Dictionary with error handling results
    """
    error_msg = state.get("error", "Unknown error")
    print(f"Error in workflow: {error_msg}")
    
    # Create fallback email content for errors
    fallback_email = {
        "subject": "Error Processing Expense Report",
        "body": f"There was an error in the expense report processing workflow: {error_msg}\n\nPlease review the expense report manually.",
        "recipient": "Finance Department",
        "approval_status": "Needs Review",
        "approval_comments": "Workflow error occurred. Manual review required."
    }
    
    return {
        "email_content": fallback_email,
        "error": error_msg  # Keep the error for logging
    }


# Conditional edge functions
def should_generate_email(state: WorkflowState) -> str:
    """Decide whether to generate email or handle error after analysis."""
    return "handle_error" if state.get("error") else "generate_approval_email"


def should_end_or_error(state: WorkflowState) -> str:
    """Decide whether to end workflow or handle error after email generation."""
    return "handle_error" if state.get("error") else END


# Build the graph with proper LangGraph compliance
def build_graph() -> StateGraph:
    """
    Builds the workflow graph with proper state schema.
    """
    # Create the graph with properly defined state schema
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("analyze_expense_summary", analyze_expense_summary)
    workflow.add_node("generate_approval_email", generate_approval_email)
    workflow.add_node("handle_error", handle_error)
    
    # Define the workflow flow
    workflow.set_entry_point("analyze_expense_summary")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "analyze_expense_summary",
        should_generate_email,
        {
            "generate_approval_email": "generate_approval_email",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_approval_email",
        should_end_or_error,
        {
            END: END,
            "handle_error": "handle_error"
        }
    )
    
    # Error handling node always ends the workflow
    workflow.add_edge("handle_error", END)
    
    return workflow


def run_agentic_auditor(expense_summary: ExpenseSummary) -> Dict[str, Any]:
    """
    Main entry point for the agentic auditor workflow.
    
    Args:
        expense_summary: The summary from the ExpenseAuditor
        
    Returns:
        Dict containing the workflow results, including the generated email
    """
    try:
        # Check for OpenAI API key early
        if not os.environ.get("OPENAI_API_KEY"):
            return {
                "email_content": {
                    "subject": "Error: OpenAI API Key Missing",
                    "body": "Unable to generate approval email because the OpenAI API key is missing. Please configure your OpenAI API key in the .env file.",
                    "recipient": "System Administrator",
                    "approval_status": "Error",
                    "approval_comments": "OpenAI API key configuration required."
                },
                "error": "OpenAI API key is missing. Please provide an API key."
            }
        
        # Build and compile the graph
        workflow = build_graph()
        app = workflow.compile()
        
        # Initialize state with only required data
        initial_state: WorkflowState = {
            "expense_summary": expense_summary,
            "email_content": {},
            "error": ""
        }
        
        # Run the workflow
        result = app.invoke(initial_state)
        
        # Extract results
        return {
            "email_content": result.get("email_content", {}),
            "error": result.get("error", "")
        }
        
    except Exception as e:
        # Handle any unexpected errors
        error_trace = traceback.format_exc()
        print(f"Workflow execution error: {str(e)}")
        print(error_trace)
        
        return {
            "email_content": {
                "subject": "Error Processing Expense Report",
                "body": f"There was an error in the workflow execution: {str(e)}\n\nPlease review the expense report manually.",
                "recipient": "Finance Department",
                "approval_status": "Needs Review",
                "approval_comments": "Workflow execution error. Manual review required."
            },
            "error": f"Workflow execution error: {str(e)}"
        }


def process_expense_report(pdf_path: str) -> Dict[str, Any]:
    """
    Process an expense report and generate an approval email.
    
    Args:
        pdf_path: Path to the PDF expense report
        
    Returns:
        Dict containing the audit results and generated email
    """
    try:
        # Create auditor and process the expense report
        auditor = ExpenseAuditor()
        audit_results = auditor.audit_expense_report(pdf_path)
        
        # Run the agentic workflow
        workflow_results = run_agentic_auditor(audit_results)
        
        # Combine results
        return {
            "audit_results": audit_results,
            "email": workflow_results["email_content"],
            "error": workflow_results["error"]
        }
        
    except Exception as e:
        return {
            "audit_results": {},
            "email": {
                "subject": "Error Processing Expense Report",
                "body": f"There was an error processing the expense report: {str(e)}",
                "recipient": "System Administrator",
                "approval_status": "Error",
                "approval_comments": "Processing error occurred."
            },
            "error": f"Error processing expense report: {str(e)}"
        }


def main():
    """Main function for command line usage."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.agent_graph.agentic_auditor <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Verify PDF file exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Processing expense report: {pdf_path}")
    results = process_expense_report(pdf_path)
    
    if results["error"]:
        print(f"Error: {results['error']}")
        # Still show results if available
    
    print("\n" + "=" * 80)
    print("EXPENSE REPORT AUDIT COMPLETED")
    print("=" * 80)
    
    email = results["email"]
    print(f"\nSubject: {email.get('subject', 'N/A')}")
    print(f"To: {email.get('recipient', 'N/A')}")
    print(f"\nBody:\n{email.get('body', 'N/A')}")
    
    print("\n" + "-" * 80)
    print(f"Approval Status: {email.get('approval_status', 'N/A')}")
    print(f"Comments: {email.get('approval_comments', 'N/A')}")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()