"""
Script para gerar um PDF de exemplo de relatório de despesas para testes.
Cria um PDF com tabelas, valores monetários e outros elementos típicos
de um relatório de despesas de viagem.
"""

import os
import random
from datetime import datetime, timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import config

# Registrar fontes
try:
    pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))
    default_font = 'Arial'
except:
    default_font = 'Helvetica'  # Fallback para fonte padrão

# Estilos
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'Title',
    parent=styles['Heading1'],
    fontName=default_font,
    fontSize=16,
    alignment=1,  # Centralizado
    spaceAfter=12
)
header_style = ParagraphStyle(
    'Header',
    parent=styles['Heading2'],
    fontName=default_font,
    fontSize=14,
    spaceAfter=10
)
normal_style = ParagraphStyle(
    'Normal',
    parent=styles['Normal'],
    fontName=default_font,
    fontSize=10,
    spaceAfter=6
)
small_style = ParagraphStyle(
    'Small',
    parent=styles['Normal'],
    fontName=default_font,
    fontSize=8
)


def generate_random_date(start_date, end_date):
    """Gera uma data aleatória entre start_date e end_date."""
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    return start_date + timedelta(days=random_days)


def format_currency(value):
    """Formata um valor como moeda brasileira (R$ X.XXX,XX)."""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def generate_expense_report(output_path, employee_name="João Silva", trip_days=3):
    """
    Gera um relatório de despesas de viagem em PDF.
    
    Args:
        output_path: Caminho para salvar o PDF
        employee_name: Nome do funcionário
        trip_days: Duração da viagem em dias
    """
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Configurar documento
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Lista de elementos para o PDF
    elements = []
    
    # Título
    elements.append(Paragraph("RELATÓRIO DE DESPESAS DE VIAGEM", title_style))
    elements.append(Spacer(1, 20))
    
    # Informações do funcionário
    elements.append(Paragraph("Informações do Funcionário", header_style))
    
    employee_data = [
        ["Nome:", employee_name],
        ["Cargo:", "Analista de Vendas"],
        ["Departamento:", "Comercial"],
        ["ID Funcionário:", f"F{random.randint(10000, 99999)}"]
    ]
    
    employee_table = Table(employee_data, colWidths=[100, 300])
    employee_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(employee_table)
    elements.append(Spacer(1, 20))
    
    # Informações da viagem
    elements.append(Paragraph("Detalhes da Viagem", header_style))
    
    # Gerar datas da viagem
    today = datetime.now()
    start_date = today - timedelta(days=random.randint(15, 45))
    end_date = start_date + timedelta(days=trip_days)
    
    trip_data = [
        ["Destino:", "São Paulo, SP"],
        ["Data de Início:", start_date.strftime("%d/%m/%Y")],
        ["Data de Término:", end_date.strftime("%d/%m/%Y")],
        ["Duração:", f"{trip_days} dias"],
        ["Motivo:", "Reunião com clientes e prospecção"]
    ]
    
    trip_table = Table(trip_data, colWidths=[100, 300])
    trip_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(trip_table)
    elements.append(Spacer(1, 20))
    
    # Tabela de despesas
    elements.append(Paragraph("Despesas", header_style))
    
    # Cabeçalho da tabela
    expense_data = [
        ["Data", "Categoria", "Descrição", "Valor", "Método de Pagamento"]
    ]
    
    # Categorias de despesas
    categories = ["Hospedagem", "Alimentação", "Transporte", "Combustível", "Outros"]
    payment_methods = ["Cartão Corporativo", "Dinheiro", "Reembolso"]
    
    # Gerar despesas aleatórias
    total_value = 0
    for _ in range(trip_days * 3):  # Média de 3 despesas por dia
        expense_date = generate_random_date(start_date, end_date)
        category = random.choice(categories)
        
        # Descrição baseada na categoria
        if category == "Hospedagem":
            description = "Hotel Continental"
            value = random.uniform(250, 450)
        elif category == "Alimentação":
            meals = ["Café da manhã", "Almoço", "Jantar"]
            description = f"{random.choice(meals)} - Restaurante"
            value = random.uniform(30, 120)
        elif category == "Transporte":
            transport_types = ["Táxi", "Uber", "Metrô", "Ônibus"]
            description = f"{random.choice(transport_types)}"
            value = random.uniform(15, 60)
        elif category == "Combustível":
            description = "Posto de Combustível"
            value = random.uniform(100, 250)
        else:
            description = "Despesas diversas"
            value = random.uniform(20, 100)
        
        payment = random.choice(payment_methods)
        total_value += value
        
        expense_data.append([
            expense_date.strftime("%d/%m/%Y"),
            category,
            description,
            format_currency(value),
            payment
        ])
    
    # Adicionar linha de total
    expense_data.append(["", "", "TOTAL", format_currency(total_value), ""])
    
    # Criar tabela de despesas
    expense_table = Table(expense_data, colWidths=[70, 80, 150, 80, 100])
    expense_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),  # Cabeçalho
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),  # Linha de total
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (3, 1), (3, -1), 'RIGHT'),  # Alinhar valores à direita
        ('PADDING', (0, 0), (-1, -1), 4),
        ('FONTNAME', (0, -1), (-1, -1), default_font),
        ('FONTNAME', (0, 0), (-1, 0), default_font),
    ]))
    elements.append(expense_table)
    elements.append(Spacer(1, 20))
    
    # Observações
    elements.append(Paragraph("Observações", header_style))
    elements.append(Paragraph(
        "Todas as despesas foram realizadas conforme a política de viagens da empresa. "
        "Os comprovantes físicos foram anexados ao relatório original e estão disponíveis "
        "para auditoria quando necessário.",
        normal_style
    ))
    elements.append(Spacer(1, 30))
    
    # Assinaturas
    signature_data = [
        ["_______________________", "_______________________"],
        ["Assinatura do Funcionário", "Aprovação do Gestor"],
        [f"Data: {today.strftime('%d/%m/%Y')}", f"Data: {today.strftime('%d/%m/%Y')}"]
    ]
    
    signature_table = Table(signature_data, colWidths=[200, 200])
    signature_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(signature_table)
    
    # Rodapé
    elements.append(Spacer(1, 40))
    elements.append(Paragraph(
        "Este é um relatório gerado automaticamente para fins de teste. "
        "CNPJ: 12.345.678/0001-90",
        small_style
    ))
    
    # Gerar PDF
    doc.build(elements)
    print(f"Relatório de despesas gerado em: {output_path}")
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Gera um PDF de exemplo de relatório de despesas")
    parser.add_argument("--output", "-o", help="Caminho para salvar o PDF", 
                        default=os.path.join(config.DATA_DIR, "relatorio_exemplo.pdf"))
    parser.add_argument("--nome", help="Nome do funcionário", default="João Silva")
    parser.add_argument("--dias", type=int, help="Duração da viagem em dias", default=3)
    
    args = parser.parse_args()
    
    generate_expense_report(args.output, args.nome, args.dias)
    
    return 0


if __name__ == "__main__":
    exit(main())
