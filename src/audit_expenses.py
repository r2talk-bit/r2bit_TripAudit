"""
Expense report auditing logic for R2Bit TripAudit.
This module contains the core business logic for auditing expense reports,
separating it from the presentation layer in streamlit_app.py.

This module implements the business logic layer of the application,
following the separation of concerns design principle to keep the
UI code separate from the data processing logic.
"""

# Standard library imports for file and path operations
import os  # Operating system interfaces for file handling
from pathlib import Path  # Object-oriented filesystem paths

import config  # Application configuration settings


class ExpenseAuditor:
    """
    Main class for auditing expense reports.
    Handles the orchestration of the expense report processing workflow.
    
    This class follows the Facade design pattern, providing a simplified interface
    to the complex subsystem of preprocessing and extraction components.
    It coordinates the workflow between different processing steps without
    getting into the implementation details of each step.
    """
    
    def __init__(self, openai_api_key=None):
        """
        Initialize the ExpenseAuditor with optional OpenAI API key.
        
        Args:
            openai_api_key: Optional OpenAI API key to use for API calls.
                            If not provided, will use the key from environment variables.
        """
        # Store the API key for later use
        self.openai_api_key = openai_api_key
        
        # If an API key was provided, set it in the environment variables
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
    
    def process_and_audit(self, uploaded_file, use_agent_team=False, user_id=None):
        """
        Process an uploaded file and generate an approval email using the agentic auditor.
        
        This method orchestrates the complete workflow:
        1. Save the uploaded file to a temporary location
        2. Run the agentic auditor to analyze the PDF file and generate an approval email
        
        This method focuses on orchestration without getting into the implementation details
        of the extraction process.
        
        Args:
            uploaded_file: Streamlit UploadedFile object containing the PDF data
            use_agent_team: Boolean flag to use the new agent team implementation (default: False)
            user_id: Unique identifier for the user session (for policy management)
            
        Returns:
            Dictionary containing the generated email content
        """
        try:
            # Step 1: Save the uploaded file to a temporary location
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                # Write the binary content of the uploaded file to the temporary file
                tmp_file.write(uploaded_file.getvalue())
                # Get the path to the temporary file for further processing
                pdf_path = tmp_file.name
            
            # Step 2: Import the appropriate auditor based on the use_agent_team flag
            if use_agent_team:
                # Use the new agent team implementation
                from f2_agent_team_audit import run_agentic_auditor
                
                # Run the agent team auditor with the PDF path
                agentic_analysis = run_agentic_auditor(pdf_path, user_id=user_id)
            else:
                # Use the original agentic auditor implementation which expects extracted results
                from f2_agentic_audit import run_agentic_auditor
                
                # For the original implementation, we still need to extract information
                from f1_expenses_info_extract import extract_info_from_uploaded_file
                extracted_results = extract_info_from_uploaded_file(uploaded_file)
                
                # Run the original auditor with the extracted results
                agentic_analysis = run_agentic_auditor(extracted_results)
            
            # Return the results
            return {
                "pdf_path": pdf_path,
                "agentic_analysis": agentic_analysis
            }

        except Exception as e:
            # Handle any errors that occur during processing
            return {
                "results": {},
                "agentic_analysis": {
                    "email_content": {
                        "subject": "Error Processing Expense Report",
                        "body": f"There was an error processing the expense report: {str(e)}",
                        "recipient": "System Administrator",
                        "approval_status": "Error",
                        "approval_comments": "Processing error occurred."
                    },
                    "error": f"Error processing expense report: {str(e)}"
                }
            }
