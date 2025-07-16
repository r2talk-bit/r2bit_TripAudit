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
       
             
    def process_and_audit(self, uploaded_file):
        """
        Process an uploaded file and generate an approval email using the agentic auditor.
        
        This method orchestrates the complete workflow:
        1. Process the uploaded file to extract expense information (delegated to ReportExtractor)
        2. Run the agentic auditor to analyze the data and generate an approval email
        
        This method focuses on orchestration without getting into the implementation details
        of the extraction process, which is handled by the ReportExtractor class.
        
        Args:
            uploaded_file: Streamlit UploadedFile object containing the PDF data
            
        Returns:
            Dictionary containing the extraction results and generated email content
        """
        try:
            # Step 1: Extract information from the uploaded file
            # This is directly delegated to the extraction function
            # Custom module imports for expense report processing
            from f1_expenses_info_extract import extract_info_from_uploaded_file  # Handles the extraction of information from expense reports
            extracted_results = extract_info_from_uploaded_file(uploaded_file)
            
            # Step 2: Import the agentic auditor here to avoid circular imports
            from f2_agentic_audit import run_agentic_auditor
            # Run the agentic auditor to analyze the data and generate an approval email
            agentic_analysis = run_agentic_auditor(extracted_results)
            
            # Return the combined results
            return {
                "results": extracted_results,
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
