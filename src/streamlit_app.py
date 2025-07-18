"""
Streamlit interface for R2Bit TripAudit.
Allows users to upload a PDF expense report and generates a text summary.

This application provides a web interface for auditing expense reports.
It uses LangGraph with specialized agents for document processing and OpenAI for generating
approval emails with intelligent analysis of the expense data.
"""

# Standard library imports for file handling and system operations
import os
import sys
import uuid  # For generating unique user IDs
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
import config  # Import configuration settings

# Load environment variables from .env file
# This allows us to store sensitive information like API keys outside of the code
load_dotenv(os.path.join(project_root, '.env'))

# Initialize session state for user ID management
if 'user_id' not in st.session_state:
    # Generate a unique user ID for this session
    st.session_state.user_id = str(uuid.uuid4())

def main():
    """Main function that runs the Streamlit application."""
    # Set page configuration for the Streamlit app
    # This configures the browser tab title, favicon, and layout settings
    st.set_page_config(
        page_title="R2Bit TripAudit",  # Title shown in the browser tab
        page_icon="✈️",  # Emoji icon shown in the browser tab
        layout="wide"  # Use the full width of the browser window
    )

    # Header section - Displays the application title and subtitle
    st.title("R2Bit TripAudit")  # Main title of the application
    st.subheader("Expense Report Auditing System")
    st.subheader("Análise de Relatórios de Despesas")  # Subtitle in Portuguese
    
    # Instructions section - Provides user guidance on how to use the application
    # The header creates a section title
    st.header("Instruções")
    # st.write with triple quotes allows for multi-line markdown content
    st.write("""
    ### Sobre o R2Bit TripAudit
    
    Esta aplicação utiliza um fluxo de trabalho LangGraph com agentes especializados para analisar 
    relatórios de despesas em formato PDF e extrair informações relevantes.
    
    O sistema utiliza quatro agentes especializados:
    - **ParsingAgent**: Extrai e estrutura dados brutos de despesas do texto PDF em JSON
    - **PolicyRetrievalAgent**: Recupera políticas relevantes da empresa
    - **ComplianceCheckAgent**: Verifica as despesas estruturadas contra as políticas da empresa
    - **CommentarySynthesisAgent**: Gera um resumo legível e e-mail de aprovação/rejeição
    
    ### Como utilizar
    
    1. Utilize o painel lateral para fazer upload do seu relatório de despesas em PDF
    2. Opcionalmente, adicione políticas da empresa na aba "Policy Management"
    3. Clique no botão "Auditar Relatório de Despesas"
    4. Aguarde o processamento (pode levar alguns segundos)
    5. Visualize o relatório de análise e o e-mail de aprovação/rejeição gerado
    """)
    
    # Sidebar section - Contains file upload and API key input controls
    # The 'with st.sidebar:' context manager places all contained elements in the sidebar
    st.sidebar.title("Settings")
    
    # OpenAI API key input - show a placeholder if key is in env
    api_key_from_env = os.getenv("OPENAI_API_KEY")
    api_key_placeholder = "API Key found in environment" if api_key_from_env else ""
    
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password",
        placeholder=api_key_placeholder
    )
    
    # Use environment variable if text input is empty
    if not openai_api_key:
        openai_api_key = api_key_from_env
        if not openai_api_key:
            st.sidebar.warning("Please enter your OpenAI API key or set it as an environment variable.")
        else:
            st.sidebar.success("Using API Key from environment variables.")
    
    # A implementação atual usa apenas o fluxo de trabalho LangGraph com agentes especializados
    use_agent_team = True  # Sempre usa a implementação baseada em agentes
    
    # Display user ID (truncated for readability) - useful for debugging
    user_id = st.session_state.user_id
    st.sidebar.text(f"Session ID: {user_id[:8]}...")
    
    # Create tabs in the sidebar for different functions
    sidebar_tab1, sidebar_tab2 = st.sidebar.tabs(["Expense Report", "Policy Management"])
    
    # Tab 1: Expense Report Upload
    with sidebar_tab1:
        # File uploader widget that accepts only PDF files for expense reports
        uploaded_file = st.file_uploader("Selecione um relatório de despesas em PDF", type=["pdf"], key="expense_uploader")
    
    # Tab 2: Policy Management
    with sidebar_tab2:
        st.subheader("Upload de Política")
        # File uploader for policy PDF files
        policy_file = st.file_uploader("Selecione um arquivo de política em PDF", type=["pdf"], key="policy_uploader")
        
        # Input field for policy category
        policy_category = st.selectbox(
            "Categoria", 
            ["General", "Meals", "Transportation", "Accommodation", "Entertainment", "Other"],
            key="policy_category"
        )
        
        # Button to upload and process the policy
        upload_policy_button = st.button("Adicionar Política", type="primary", key="upload_policy")
        
        # Process policy upload when button is clicked
        if policy_file is not None and upload_policy_button:
            try:
                # Import the policy loader module
                from load_policy import load_policy_from_uploaded_file
                
                # Use the file name (without extension) as the policy name
                policy_name = os.path.splitext(policy_file.name)[0]
                
                # Add metadata for the policy - ChromaDB doesn't accept lists as values
                metadata = {
                    "category": policy_category,
                    "applies_to": policy_category if policy_category != "General" else "All expenses"
                }
                
                # Load the policy into the vector database
                policy_id = load_policy_from_uploaded_file(
                    policy_file,
                    policy_name,
                    st.session_state.user_id,
                    metadata=metadata
                )
                
                # Show success message
                st.success(f"Política '{policy_name}' adicionada com sucesso!")
                
            except Exception as e:
                # Show error message if something goes wrong
                st.error(f"Erro ao adicionar política: {str(e)}")
        
        # Show a message if no policy file is selected
        elif upload_policy_button and not policy_file:
            st.warning("Por favor, selecione um arquivo de política para upload.")
            
        # Divider for visual separation
        st.divider()
        
        # Option to view and delete existing policies
        st.subheader("Políticas Existentes")
        
        # Button to refresh the list of policies
        if st.button("Atualizar Lista de Políticas", key="refresh_policies"):
            try:
                # Import the function to get all user policies
                from load_policy import get_all_user_policies
                
                # Get all policies for this user
                policies = get_all_user_policies(st.session_state.user_id)
                
                # Store policies in session state for display
                st.session_state.user_policies = policies
                
                # Show success message
                if policies:
                    st.success(f"Encontradas {len(policies)} políticas.")
                else:
                    st.info("Nenhuma política encontrada para este usuário.")
                    
            except Exception as e:
                # Show error message if something goes wrong
                st.error(f"Erro ao buscar políticas: {str(e)}")
        
        # Display policies if they exist in session state
        if hasattr(st.session_state, 'user_policies') and st.session_state.user_policies:
            for policy_id, policy_chunks in st.session_state.user_policies.items():
                if policy_chunks:
                    # Get the first chunk for display (contains metadata)
                    policy = policy_chunks[0]
                    policy_name = policy.get("metadata", {}).get("policy_name", "Unknown Policy")
                    category = policy.get("metadata", {}).get("category", "General")
                    
                    # Create an expander for each policy
                    with st.expander(f"{policy_name} ({category})"):
                        # Show policy details
                        st.write(f"**ID:** {policy_id}")
                        st.write(f"**Categoria:** {category}")
                        
                        # Show a preview of the policy content
                        st.text_area("Conteúdo", policy.get("content", "No content available"), height=100)
                        
                        # Button to delete this policy
                        if st.button(f"Excluir '{policy_name}'", key=f"delete_{policy_id}"):
                            try:
                                # Import the function to delete user policies
                                from load_policy import delete_user_policies
                                
                                # Delete the policy
                                deleted = delete_user_policies(st.session_state.user_id, policy_name=policy_name)
                                
                                # Remove from session state
                                if deleted > 0:
                                    del st.session_state.user_policies[policy_id]
                                    st.success(f"Política '{policy_name}' excluída com sucesso!")
                                    st.rerun()  # Refresh the page
                                    
                            except Exception as e:
                                # Show error message if something goes wrong
                                st.error(f"Erro ao excluir política: {str(e)}")
    
    # No need for a duplicate file uploader outside the tabs
    
    # Add the process button to Tab 1
    with sidebar_tab1:
        # Add some space before the button
        st.write("")
        # The 'primary' type gives it a prominent color to indicate it's the main action
        process_button = st.button(
            "Auditar Relatório de Despesas", 
            type="primary",
            key="process_expense_button"
        )
    
    # Show confirmation and file details when a file is uploaded
    # This is now inside Tab 1 where the file uploader is
    with sidebar_tab1:
        if uploaded_file is not None:
            # Display a success message with green background
            st.success("Arquivo carregado com sucesso!")
            # Show the name of the uploaded file
            st.write(f"Nome do arquivo: {uploaded_file.name}")
            
            # Instructions for the next step
            st.write("Clique em 'Auditar Relatório de Despesas' para iniciar a análise.")
    
    # Main content area for displaying results
    # This section only executes when both conditions are met:
    # 1. A file has been uploaded
    # 2. The process button has been clicked
    if uploaded_file is not None and process_button:
        # Set the OpenAI API key in environment variables if provided in the interface
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
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
                # Create an ExpenseAuditor instance and use the process_and_audit method
                # This method handles both the document processing and the agentic auditing
                auditor = ExpenseAuditor(openai_api_key)
                # Process the file and get the combined results, passing the use_agent_team flag and user_id
                results = auditor.process_and_audit(
                    uploaded_file, 
                    use_agent_team=use_agent_team,
                    user_id=st.session_state.user_id
                )
                
                # Extract the workflow results containing the email content
                agentic_analysis = results["agentic_analysis"]
                
                # Display a success message when processing is complete
                # This provides visual feedback that the operation was successful
                st.success("Processamento e geração de email concluídos com sucesso!")
                
                # Display the auditor type used
                st.info("Auditor utilizado: Equipe de Agentes LangGraph (4 agentes especializados)")
                
                # Display the approval email section
                st.header("Email de Aprovação")
                
                # Check if there was an error in the workflow results
                if agentic_analysis["error"]:
                    # Display error message with red background if there was an error
                    st.error(f"Erro ao gerar email: {agentic_analysis['error']}")
                else:  # If no error, proceed to display the email content
                    # Extract the email content from the workflow results
                    email = agentic_analysis["email_content"]
                    
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
