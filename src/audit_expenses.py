"""
Expense report auditing logic for R2Bit TripAudit.
This module contains the core business logic for auditing expense reports,
separating it from the presentation layer in streamlit_app.py.
"""

import os
import tempfile
from pathlib import Path

from src.extract_expenses import ExpenseReportExtractor
from src.data_preparation import ExpenseReportPreprocessor
import src.config as config


class ExpenseAuditor:
    """
    Main class for auditing expense reports.
    Handles the coordination between preprocessing and extraction components.
    """
    
    def __init__(self):
        """Initialize the expense auditor with its components."""
        self.preprocessor = ExpenseReportPreprocessor()
        self.extractor = ExpenseReportExtractor()
    
    def audit_expense_report(self, pdf_path):
        """
        Process a PDF expense report and generate a text summary.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing the summary and paths to generated files
        """
        # Create output directory
        output_dir = os.path.join(config.OUTPUT_DIR, "audit_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Pre-process the PDF
        processed_dir = os.path.join(output_dir, "preprocessed")
        results = self.preprocessor.process_pdf(pdf_path, output_dir=processed_dir)
        
        # 2. Extract information with LayoutLMv3
        summary = self.extractor.summarize_expense_report(pdf_path)
        
        # 3. Generate text report
        text_report_path = os.path.join(output_dir, "discovery_summary.txt")
        self.extractor.generate_text_report(summary, text_report_path)
        
        # Read the text report content
        text_report_content = ""
        try:
            with open(text_report_path, "r", encoding="utf-8") as f:
                text_report_content = f.read()
        except Exception as e:
            print(f"Error reading text report: {str(e)}")
        
        # 4. Create result dictionary with summary, paths, and text content
        audit_results = {
            "pdf_path": pdf_path,
            "processed_dir": processed_dir,
            "text_report_path": text_report_path,
            "text_report_content": text_report_content,
            "summary": summary
        }
        
        return audit_results
    
    def process_uploaded_file(self, uploaded_file):
        """
        Process an uploaded file from Streamlit.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Dictionary containing the audit results
        """
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        try:
            # Process the PDF
            results = self.audit_expense_report(pdf_path)
            return results
        finally:
            # Clean up the temporary file
            try:
                os.unlink(pdf_path)
            except:
                pass
