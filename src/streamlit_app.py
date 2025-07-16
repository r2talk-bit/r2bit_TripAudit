"""
Streamlit interface for R2Bit TripAudit.
Allows users to upload a PDF expense report and generates a text summary.

This application provides a web interface for auditing expense reports.
It uses LayoutLMv3 model for document understanding and OpenAI for generating
approval emails with intelligent analysis of the expense data.
"""

# Standard library imports for file handling and system operations
import os
import sys
# Streamlit library for creating the web interface
import streamlit as st
# Path for handling file paths in a cross-platform way
from pathlib import Path
# dotenv for loading environment variables from .env file
from dotenv import load_dotenv

# Add project directory to PATH for imports
# This ensures we can import modules from the project root directory
project_root = Path(__file__).parent.parent.absolute()  # Get the absolute path to the project root
sys.path.append(str(project_root))  # Add the project root to the Python path

# Import our custom modules for expense auditing - using relative imports
# The imports need to come after adding the project root to sys.path
from audit_expenses import ExpenseAuditor  # Handles the core expense report processing
from agentic_auditor import run_agentic_auditor  # Runs the AI agent workflow

# Load environment variables from .env file
# This allows us to store sensitive information like API keys outside of the code
load_dotenv(os.path.join(project_root, '.env'))

# Set page configuration for the Streamlit app
# This configures the browser tab title, favicon, and layout settings
st.set_page_config(
    page_title="R2Bit TripAudit",  # Title shown in the browser tab
    page_icon="📊",  # Emoji icon shown in the browser tab
    layout="wide"  # Use the full width of the browser window
)

def main():
    """Main function that runs the Streamlit application."""
    # Header section - Displays the application title and subtitle
    st.title("R2Bit TripAudit")  # Main title of the application
    st.subheader("Análise de Relatórios de Despesas")  # Subtitle in Portuguese
    
    # Instructions section - Provides user guidance on how to use the application
    # The header creates a section title
    st.header("Instruções")
    # st.write with triple quotes allows for multi-line markdown content
    st.write("""
    ### Sobre o R2Bit TripAudit
    
    Esta aplicação utiliza o modelo LayoutLMv3 para analisar 
    relatórios de despesas em formato PDF e extrair informações relevantes.
    
    O sistema é capaz de identificar:
    - Valores monetários
    - Datas
    - Categorias de despesas
    - Fornecedores
    - Outros detalhes importantes
    
    ### Como utilizar
    
    1. Utilize o painel lateral para fazer upload do seu relatório de despesas em PDF
    2. Clique no botão "Processar Relatório"
    3. Aguarde o processamento (pode levar alguns segundos)
    4. Visualize o relatório de análise gerado
    5. Faça o download do relatório em formato texto se necessário
    """)
    
    # Sidebar section - Contains file upload and API key input controls
    # The 'with st.sidebar:' context manager places all contained elements in the sidebar
    with st.sidebar:
        st.header("Upload de Arquivo")  # Sidebar section header
        # File uploader widget that accepts only PDF files
        # Returns None if no file is uploaded, otherwise returns the uploaded file object
        uploaded_file = st.file_uploader("Selecione um relatório de despesas em PDF", type=["pdf"])
        
        # Check if OpenAI API key is already set in environment variables
        # This allows users to store their API key in the .env file instead of entering it each time
        api_key_env = os.environ.get("OPENAI_API_KEY", "")  # Get API key from environment or empty string
        # Set placeholder text based on whether API key is already configured
        api_key_placeholder = "API Key já configurada no arquivo .env" if api_key_env else ""
        
        # API key input widget for OpenAI
        # This allows users to enter their API key directly in the interface
        api_key = st.text_input(
            "OpenAI API Key",  # Label for the input field
            type="password",  # Mask the input as a password field for security
            placeholder=api_key_placeholder,  # Show placeholder text based on environment
            help="Necessário para gerar emails de aprovação. Pode ser configurado no arquivo .env"  # Help tooltip
        )
        
        # Button to trigger the expense report processing
        # The 'primary' type gives it a prominent color to indicate it's the main action
        process_button = st.button("Auditar Relatório de Despesas", type="primary")
        
        # Show confirmation and file details when a file is uploaded
        if uploaded_file is not None:
            # Display a success message with green background
            st.success("Arquivo carregado com sucesso!")
            # Show the name of the uploaded file
            st.write(f"Nome do arquivo: {uploaded_file.name}")
            
            # Instructions for the next step
            st.write("Clique em 'Processar Relatório' para iniciar a análise.")
    
    # Main content area for displaying results
    # This section only executes when both conditions are met:
    # 1. A file has been uploaded
    # 2. The process button has been clicked
    if uploaded_file is not None and process_button:
        # Set the OpenAI API key in environment variables if provided in the interface
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Check if OpenAI API key is available (either from .env or input field)
        # If not available, show an error message and stop execution
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("API Key da OpenAI não encontrada. Por favor, forneça uma API Key ou configure-a no arquivo .env.")
            st.stop()  # Stop the app execution here to prevent further errors
        
        # Process the expense report with a loading spinner
        # The spinner shows a loading animation while the processing is happening
        with st.spinner("Processando o relatório de despesas e gerando email de aprovação..."):
            # Use try-except to handle any errors that might occur during processing
            try:
                # Step 1: Create an ExpenseAuditor instance and process the uploaded file
                # The ExpenseAuditor class handles the document processing and information extraction
                auditor = ExpenseAuditor()
                # Process the file and get the extracted information
                # This includes OCR, table extraction, and entity recognition using LayoutLMv3
                results = auditor.process_uploaded_file(uploaded_file)
                
                # Step 2: Run the AI agent workflow to analyze the data and generate an approval email
                # The agentic_auditor uses AI to make decisions about the expense report
                # It determines if the expenses are valid, need review, or should be rejected
                workflow_results = run_agentic_auditor(results)
                
                # Display a success message when processing is complete
                # This provides visual feedback that the operation was successful
                st.success("Processamento e geração de email concluídos com sucesso!")
                
                # Display the approval email section
                st.header("Email de Aprovação")
                
                # Check if there was an error in the workflow results
                if workflow_results["error"]:
                    # Display error message with red background if there was an error
                    st.error(f"Erro ao gerar email: {workflow_results['error']}")
                else:  # If no error, proceed to display the email content
                    # Extract the email content from the workflow results
                    email = workflow_results["email_content"]
                    
                    # Display the email header information (subject and recipient)
                    st.subheader(f"Assunto: {email['subject']}")  # Display email subject as a subheader
                    st.write(f"**Para:** {email['recipient']}")  # Display recipient with bold formatting
                    
                    # Display the email body content
                    st.markdown("---")  # Horizontal separator line before the body
                    st.write("**Corpo do Email:**")  # Section label with bold formatting
                    st.write(email['body'])  # Display the actual email body content
                    st.markdown("---")  # Horizontal separator line after the body
                    
                    # Display the approval status with color coding
                    # Set color based on approval status: green for approved, orange for review, red for rejected
                    status_color = "green" if email['approval_status'] == "Approved" else \
                                  "orange" if email['approval_status'] == "Needs Review" else "red"
                    
                    # Display the approval status with the appropriate color using HTML
                    # unsafe_allow_html=True allows the use of HTML tags in the markdown
                    st.markdown(f"**Status de Aprovação:** <span style='color:{status_color}'>{email['approval_status']}</span>", unsafe_allow_html=True)
                    
                    # Display the approval comments
                    st.write("**Comentários:**")  # Section label
                    st.write(email['approval_comments'])  # The actual comments
                    
                    # Create a formatted text version of the email for download
                    # This combines all parts of the email into a single text string
                    email_content = f"Assunto: {email['subject']}\n"  # Start with the subject
                    email_content += f"Para: {email['recipient']}\n\n"  # Add recipient with blank line after
                    email_content += f"{email['body']}\n\n"  # Add the email body with blank lines
                    email_content += f"Status de Aprovação: {email['approval_status']}\n"  # Add approval status
                    email_content += f"Comentários: {email['approval_comments']}"  # Add comments
                    
                    # Create a download button for saving the email content as a text file
                    # This allows users to save the generated email for later use
                    st.download_button(
                        label="Baixar Email",  # Button text
                        data=email_content,  # The content to be downloaded
                        file_name="approval_email.txt",  # Default filename for the download
                        mime="text/plain"  # MIME type of the file (plain text)
                    )
            # Error handling for any exceptions that occur during processing
            except Exception as e:
                # Display a user-friendly error message with red background
                st.error(f"Erro ao processar o relatório: {str(e)}")
                # Display the full exception details for debugging
                # This includes the stack trace to help identify where the error occurred
                st.exception(e)
            
            # Note: Temporary file cleanup is handled automatically by the ExpenseAuditor class
            # This ensures that any temporary files created during processing are properly deleted

# Standard Python idiom to ensure the main() function only runs when this script is executed directly
# This allows the file to be imported as a module without running the main() function
if __name__ == "__main__":
    main()  # Call the main function to start the Streamlit application
