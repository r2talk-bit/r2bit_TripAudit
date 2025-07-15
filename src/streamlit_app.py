"""
Streamlit interface for R2Bit TripAudit.
Allows users to upload a PDF expense report and generates a text summary.
"""

import os
import sys
import streamlit as st
from pathlib import Path
import tempfile

# Add project directory to PATH for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.extract_expenses import ExpenseReportExtractor
from src.data_preparation import ExpenseReportPreprocessor
import src.config as config

# Set page configuration
st.set_page_config(
    page_title="R2Bit TripAudit",
    page_icon="📊",
    layout="wide"
)

def audit_and_report(pdf_path):
    """
    Process a PDF expense report and generate a text summary.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing the summary and paths to generated files
    """
    # Create output directory
    output_dir = os.path.join(config.OUTPUT_DIR, "streamlit_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Pre-process the PDF
    preprocessor = ExpenseReportPreprocessor()
    processed_dir = os.path.join(output_dir, "preprocessed")
    results = preprocessor.process_pdf(pdf_path, output_dir=processed_dir)
    
    # 2. Extract information with LayoutLMv3
    extractor = ExpenseReportExtractor()
    summary = extractor.summarize_expense_report(pdf_path)
    
    # 3. Generate text report
    text_report_path = os.path.join(output_dir, "discovery_summary.txt")
    extractor.generate_text_report(summary, text_report_path)
    
    return {
        "pdf_path": pdf_path,
        "processed_dir": processed_dir,
        "text_report_path": text_report_path,
        "summary": summary
    }

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
        
        process_button = st.button("Processar Relatório", type="primary")
        
        if uploaded_file is not None:
            st.success("Arquivo carregado com sucesso!")
            st.write(f"Nome do arquivo: {uploaded_file.name}")
            
            # Display a small preview if possible
            st.write("Clique em 'Processar Relatório' para iniciar a análise.")
    
    # Main content area for results
    if uploaded_file is not None and process_button:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        with st.spinner("Processando o relatório de despesas..."):
            try:
                # Process the PDF
                results = audit_and_report(pdf_path)
                
                # Display success message
                st.success("Processamento concluído com sucesso!")
                
                # Display summary information
                st.header("Resumo da Análise")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total de Páginas", results["summary"]["pages"])
                    st.metric("Valor Total", f"R$ {results['summary']['total_value']:.2f}")
                
                with col2:
                    st.metric("Datas Encontradas", len(results["summary"]["dates"]))
                    st.metric("Categorias", len(results["summary"]["categories"]))
                
                # Display categories
                st.subheader("Categorias Identificadas")
                for category, count in results["summary"]["categories"].items():
                    st.write(f"- {category}: {count} ocorrência(s)")
                
                # Display text report
                st.subheader("Relatório Detalhado")
                with open(results["text_report_path"], "r", encoding="utf-8") as f:
                    report_text = f.read()
                
                st.text_area("", report_text, height=400)
                
                # Download button for the report
                with open(results["text_report_path"], "r", encoding="utf-8") as f:
                    report_content = f.read()
                    st.download_button(
                        label="Baixar Relatório",
                        data=report_content,
                        file_name="discovery_summary.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"Erro ao processar o arquivo: {str(e)}")
                st.exception(e)
            
            # Clean up the temporary file
            try:
                os.unlink(pdf_path)
            except:
                pass

if __name__ == "__main__":
    main()
