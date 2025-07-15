"""
Streamlit interface for R2Bit TripAudit.
Allows users to upload a PDF expense report and generates a text summary.
"""

import os
import sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Add project directory to PATH for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Load environment variables from .env file
load_dotenv(os.path.join(project_root, '.env'))

from src.audit_expenses import ExpenseAuditor
from src.agent_graph.agentic_auditor import run_agentic_auditor
import src.config as config

# Set page configuration
st.set_page_config(
    page_title="R2Bit TripAudit",
    page_icon="📊",
    layout="wide"
)



def main():
    # Header
    st.title("R2Bit TripAudit")
    st.subheader("Análise de Relatórios de Despesas")
    
    # Instructions on the main page
    st.header("Instruções")
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
    
    # Sidebar with file upload functionality
    with st.sidebar:
        st.header("Upload de Arquivo")
        uploaded_file = st.file_uploader("Selecione um relatório de despesas em PDF", type=["pdf"])
        
        # Check if API key is already in environment variables
        api_key_env = os.environ.get("OPENAI_API_KEY", "")
        api_key_placeholder = "API Key já configurada no arquivo .env" if api_key_env else ""
        
        # API key input for OpenAI
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            placeholder=api_key_placeholder,
            help="Necessário para gerar emails de aprovação. Pode ser configurado no arquivo .env"
        )
        
        # Single button to process report and generate email
        process_button = st.button("Auditar Relatório de Despesas", type="primary")
        
        if uploaded_file is not None:
            st.success("Arquivo carregado com sucesso!")
            st.write(f"Nome do arquivo: {uploaded_file.name}")
            
            # Display a small preview if possible
            st.write("Clique em 'Processar Relatório' para iniciar a análise.")
    
    # Main content area for results
    if uploaded_file is not None and process_button:
        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Check if OpenAI API key is available
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("API Key da OpenAI não encontrada. Por favor, forneça uma API Key ou configure-a no arquivo .env.")
            st.stop()
        
        # Process expense report
        with st.spinner("Processando o relatório de despesas e gerando email de aprovação..."):
            try:
                # Step 1: Create auditor and process the file
                auditor = ExpenseAuditor()
                results = auditor.process_uploaded_file(uploaded_file)
                
                # Step 2: Run the agentic workflow to generate approval email
                workflow_results = run_agentic_auditor(results)
                
                # Display results
                st.success("Processamento e geração de email concluídos com sucesso!")
                
                # Display only the approval email
                st.header("Email de Aprovação")
                
                if workflow_results["error"]:
                    st.error(f"Erro ao gerar email: {workflow_results['error']}")
                else:
                    email = workflow_results["email_content"]
                    
                    # Email header
                    st.subheader(f"Assunto: {email['subject']}")
                    st.write(f"**Para:** {email['recipient']}")
                    
                    # Email body
                    st.markdown("---")
                    st.write("**Corpo do Email:**")
                    st.write(email['body'])
                    st.markdown("---")
                    
                    # Approval status
                    status_color = "green" if email['approval_status'] == "Approved" else \
                                  "orange" if email['approval_status'] == "Needs Review" else "red"
                    
                    st.markdown(f"**Status de Aprovação:** <span style='color:{status_color}'>{email['approval_status']}</span>", unsafe_allow_html=True)
                    st.write("**Comentários:**")
                    st.write(email['approval_comments'])
                    
                    # Create email content for download
                    email_content = f"Assunto: {email['subject']}\n"
                    email_content += f"Para: {email['recipient']}\n\n"
                    email_content += f"{email['body']}\n\n"
                    email_content += f"Status de Aprovação: {email['approval_status']}\n"
                    email_content += f"Comentários: {email['approval_comments']}"
                    
                    # Download button for the email
                    st.download_button(
                        label="Baixar Email",
                        data=email_content,
                        file_name="approval_email.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Erro ao processar o relatório: {str(e)}")
                st.exception(e)
            
            # Temporary file cleanup is handled by ExpenseAuditor

if __name__ == "__main__":
    main()
