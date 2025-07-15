"""
Script de demonstração para o projeto R2Bit TripAudit.
Gera um PDF de exemplo, processa-o e mostra os resultados.
"""

import os
import sys
import argparse
from pathlib import Path

# Adicionar o diretório do projeto ao PATH para importações
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.generate_sample_pdf import generate_expense_report
from src.extract_expenses import ExpenseReportExtractor
from src.data_preparation import ExpenseReportPreprocessor
import src.config as config


def run_demo(output_dir=None, visualize=True):
    """
    Executa uma demonstração completa do projeto:
    1. Gera um PDF de exemplo
    2. Pré-processa o PDF (converte para imagens, melhora qualidade)
    3. Extrai informações com LayoutLMv3
    4. Exibe e exporta os resultados
    
    Args:
        output_dir: Diretório para salvar os resultados
        visualize: Se True, mostra visualizações do processamento
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("DEMONSTRAÇÃO DO R2BIT TRIPAUDIT")
    print("="*50 + "\n")
    
    # 1. Gerar PDF de exemplo
    print("1. Gerando PDF de exemplo...")
    pdf_path = os.path.join(output_dir, "relatorio_exemplo.pdf")
    generate_expense_report(pdf_path, "Maria Santos", trip_days=4)
    print(f"   PDF gerado em: {pdf_path}\n")
    
    # 2. Pré-processar o PDF
    print("2. Pré-processando o PDF...")
    preprocessor = ExpenseReportPreprocessor()
    processed_dir = os.path.join(output_dir, "preprocessed")
    results = preprocessor.process_pdf(pdf_path, output_dir=processed_dir)
    print(f"   Pré-processamento concluído. {len(results['pages'])} páginas processadas.\n")
    
    # 3. Extrair informações com LayoutLMv3
    print("3. Extraindo informações com LayoutLMv3...")
    extractor = ExpenseReportExtractor()
    summary = extractor.summarize_expense_report(pdf_path)
    
    # 4. Exibir resultados
    print("\n4. Resultados da extração:")
    print(f"   Total de páginas: {summary['pages']}")
    print(f"   Valor total extraído: R$ {summary['total_value']:.2f}")
    print(f"   Datas encontradas: {', '.join(summary['dates'])}")
    
    print("\n   Categorias identificadas:")
    for category, count in summary['categories'].items():
        print(f"   - {category}: {count}")
    
    print("\n   Detalhes por página:")
    for page in summary["expenses"]:
        print(f"   Página {page['page']}:")
        print(f"     Valores encontrados: {page['values_found']}")
        print(f"     Soma dos valores: R$ {page['sum_values']:.2f}")
        print(f"     Datas: {page['dates']}")
        print(f"     Categorias: {page['categories']}")
        
        if page["entities"]:
            print(f"     Entidades extraídas pelo LayoutLMv3:")
            for ent in page["entities"][:5]:  # Mostrar apenas as 5 primeiras para não sobrecarregar
                print(f"       - {ent['label']}: {ent['text']}")
            
            if len(page["entities"]) > 5:
                print(f"       ... e mais {len(page['entities']) - 5} entidades")
    
    # 5. Exportar para Excel
    excel_path = os.path.join(output_dir, "relatorio_analise.xlsx")
    extractor.export_to_excel(summary, excel_path)
    print(f"\n5. Relatório exportado para Excel: {excel_path}")
    
    # 6. Gerar relatório de texto com descobertas
    text_report_path = os.path.join(output_dir, "discovery_summary.txt")
    extractor.generate_text_report(summary, text_report_path)
    print(f"\n6. Relatório de texto exportado para: {text_report_path}")
    
    # 7. Informações finais
    print("\n" + "="*50)
    print("DEMONSTRAÇÃO CONCLUÍDA")
    print("="*50)
    print(f"\nTodos os arquivos foram salvos em: {output_dir}")
    print("\nPara processar seus próprios PDFs, execute:")
    print(f"python src/extract_expenses.py caminho/para/seu/relatorio.pdf")
    
    return {
        "pdf_path": pdf_path,
        "processed_dir": processed_dir,
        "excel_path": excel_path,
        "text_report_path": text_report_path,
        "summary": summary
    }


def main():
    parser = argparse.ArgumentParser(description="Executa uma demonstração do R2Bit TripAudit")
    parser.add_argument("--output", "-o", help="Diretório para salvar os resultados")
    parser.add_argument("--no-visualize", action="store_true", help="Desativar visualizações")
    
    args = parser.parse_args()
    
    run_demo(output_dir=args.output, visualize=not args.no_visualize)
    
    return 0


if __name__ == "__main__":
    exit(main())
