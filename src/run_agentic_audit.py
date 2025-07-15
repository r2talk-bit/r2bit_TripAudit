"""
Entry point script for running the agentic auditor workflow.

This script demonstrates how to use the agentic auditor workflow to process
expense reports and generate approval emails.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add project directory to PATH for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Load environment variables from .env file
load_dotenv(os.path.join(project_root, '.env'))

from src.audit_expenses import ExpenseAuditor
from src.agent_graph.agentic_auditor import run_agentic_auditor


def main():
    """
    Main entry point for the agentic auditor workflow.
    """
    parser = argparse.ArgumentParser(description="Process expense reports and generate approval emails")
    parser.add_argument("pdf_path", help="Path to the PDF expense report")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    args = parser.parse_args()

    # Check if the PDF file exists
    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)

    # Set OpenAI API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    elif not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OpenAI API key not set. Set with --api-key or OPENAI_API_KEY environment variable.")
        print("Email generation will fail without an API key.")

    # Step 1: Run the expense audit using the existing ExpenseAuditor
    print(f"Processing expense report: {pdf_path}")
    auditor = ExpenseAuditor()
    audit_results = auditor.audit_expense_report(pdf_path)
    
    print(f"Audit completed. Summary generated at: {audit_results['text_report_path']}")
    print(f"Total value extracted: R$ {audit_results['summary']['total_value']:.2f}")
    
    # Step 2: Run the agentic workflow to generate the approval email
    print("\nGenerating approval email using LangGraph workflow...")
    workflow_results = run_agentic_auditor(audit_results)
    
    # Check for errors
    if workflow_results["error"]:
        print(f"Error in workflow: {workflow_results['error']}")
        sys.exit(1)
    
    # Display the generated email
    email = workflow_results["email_content"]
    print("\n" + "=" * 80)
    print("GENERATED APPROVAL EMAIL")
    print("=" * 80)
    
    print(f"\nSubject: {email['subject']}")
    print(f"To: {email['recipient']}")
    print("\nBody:")
    print(email['body'])
    
    print("\n" + "-" * 80)
    print(f"Approval Status: {email['approval_status']}")
    print(f"Comments: {email['approval_comments']}")
    print("-" * 80)
    
    # Save the email to a file
    email_path = os.path.join(Path(audit_results['text_report_path']).parent, "approval_email.txt")
    with open(email_path, "w", encoding="utf-8") as f:
        f.write(f"Subject: {email['subject']}\n")
        f.write(f"To: {email['recipient']}\n\n")
        f.write(email['body'])
        f.write(f"\n\nApproval Status: {email['approval_status']}\n")
        f.write(f"Comments: {email['approval_comments']}\n")
    
    print(f"\nEmail saved to: {email_path}")
    print("\nWorkflow completed successfully!")


if __name__ == "__main__":
    main()
