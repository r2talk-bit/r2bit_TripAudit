"""
Report extraction logic for R2Bit TripAudit.
This module handles the extraction of information from expense reports,
separating the extraction process from the auditing logic.

This module implements the extraction layer of the application,
following the separation of concerns design principle.
"""

# Standard library imports for file and path operations
import os  # Operating system interfaces for file handling
import tempfile  # For creating temporary files securely
from pathlib import Path  # Object-oriented filesystem paths

# Custom module imports for expense report processing
from f1_2_layoutlm_info_process import ExpenseReportExtractor  # Handles information extraction using LayoutLMv3
# Import the functions from the OCR process module
from f1_2_1_ocr_process import process_pdf  # Handles PDF preprocessing and OCR
import config  # Application configuration settings

# Module-level initialization of components
# Only the extractor needs to be initialized as we're using functional approach for OCR
_extractor = None

def _get_extractor():
    """
    Get or initialize the extractor component.
    
    This function implements lazy initialization of the extractor component,
    creating it only when first needed and then reusing it.
    
    Returns:
        The extractor instance
    """
    global _extractor
    if _extractor is None:
        # Create an instance of the extractor that uses LayoutLMv3 for information extraction
        _extractor = ExpenseReportExtractor()
    return _extractor

def extract_report_data(pdf_path) -> dict:
    """
    Process a PDF expense report and extract information.
    
    This function orchestrates the workflow for processing an expense report:
    1. Creates output directories
    2. Preprocesses the PDF (converts to images, extracts tables)
    3. Extracts information using LayoutLMv3
    4. Generates a text report summary
    5. Returns all results in a structured format
    
    Args:
        pdf_path: Path to the PDF file to be processed
        
    Returns:
        Dictionary containing the summary and paths to generated files
    """
    # Get the extractor component
    extractor = _get_extractor()
    
    # Create output directory for storing all processing results
    # The base output directory is defined in the config module
    output_dir = os.path.join(config.OUTPUT_DIR, "audit_output")
    # Create the directory if it doesn't exist (exist_ok=True prevents errors if it already exists)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Pre-process the PDF document
    # Create a subdirectory for preprocessed files (images, extracted tables)
    processed_dir = os.path.join(output_dir, "preprocessed")
    # Call the process_pdf function to convert PDF to images and extract tables
    # The img2table library is used here for table extraction from images
    results = process_pdf(pdf_path, output_dir=processed_dir)
    
    # 2. Extract information with LayoutLMv3 model
    # LayoutLMv3 is a document understanding model that can recognize text and layout
    # It uses the enhanced label configuration with 23 different categories for detailed extraction
    # The model processes the document and extracts structured information like amounts, dates, vendors, etc.
    summary = extractor.extract_from_pdf(pdf_path)
    
    # 3. Generate a human-readable text report from the extracted information
    # Define the output path for the text report
    text_report_path = os.path.join(output_dir, "discovery_summary.txt")
    # Convert the structured summary data into a formatted text report
    # This makes the extracted information easier to read and understand
    extractor.generate_text_report(summary, text_report_path)
    
    # Read the generated text report content into memory
    # This allows us to return the content directly without requiring another file read operation
    text_report_content = ""
    try:
        # Open the file with UTF-8 encoding to properly handle special characters
        with open(text_report_path, "r", encoding="utf-8") as f:
            # Read the entire file content into a string
            text_report_content = f.read()
    except Exception as e:
        # Handle any errors that might occur during file reading (e.g., permission issues)
        print(f"Error reading text report: {str(e)}")
    
    # 4. Create a comprehensive result dictionary with all relevant information
    # This dictionary contains everything needed for further processing or display:
    # - Original PDF path for reference
    # - Directory with preprocessed files (images, tables)
    # - Path to the generated text report
    # - Content of the text report as a string (for direct display)
    # - Structured summary data (for programmatic access to extracted information)
    extraction_results = {
        "pdf_path": pdf_path,  # Path to the original PDF file
        "processed_dir": processed_dir,  # Directory containing preprocessed files
        "text_report_path": text_report_path,  # Path to the generated text report
        "text_report_content": text_report_content,  # Content of the text report
        "summary": summary  # Structured data extracted from the document
    }
    
    return extraction_results

def extract_info_from_uploaded_file(uploaded_file):
    """
    Extract information from an uploaded file from Streamlit.
    
    This function handles files uploaded through the Streamlit interface.
    It creates a temporary file from the uploaded content, extracts information from it,
    and ensures proper cleanup afterward.
    
    Args:
        uploaded_file: Streamlit UploadedFile object containing the PDF data
        
    Returns:
        Dictionary containing the extraction results
    """
    # Save the uploaded file to a temporary file on disk
    # We need to do this because the processing functions expect a file path,
    # but Streamlit provides the file as an in-memory object
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        # Write the binary content of the uploaded file to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        # Get the path to the temporary file for further processing
        pdf_path = tmp_file.name
    
    try:
        # Process the PDF using the main extraction function
        # This performs all the extraction steps
        results = extract_report_data(pdf_path)
        return results
    finally:
        # Clean up the temporary file to avoid filling disk space
        # The finally block ensures this happens even if an error occurs during processing
        try:
            # os.unlink removes the file from the filesystem
            os.unlink(pdf_path)
        except:
            # Silently ignore any errors during cleanup
            # This prevents cleanup errors from affecting the main processing flow
            pass
