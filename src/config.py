"""
Arquivo de configuração para o projeto R2Bit TripAudit.
Contém configurações globais para os diversos módulos do projeto.
"""

import os
from pathlib import Path

# Diretórios do projeto
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Criar diretórios se não existirem
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configurações do modelo
MODEL_NAME = "microsoft/layoutlmv3-base"
DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

# Configurações de OCR
TESSERACT_LANG = "eng"  # English (default language)
TESSERACT_CONFIG = "--oem 1 --psm 11"  # Modo de segmentação de página completa

# Configurações de processamento de PDF
PDF_DPI = 300  # Resolução para conversão de PDF para imagem

# Configurações do auditor
# Define qual implementação do auditor usar por padrão
# False = usar o auditor original (f2_agentic_audit.py)
# True = usar a nova implementação com equipe de agentes (agent_team.py)
DEFAULT_USE_AGENT_TEAM = False

# Categorias de despesas para classificação
EXPENSE_CATEGORIES = {
    "hospedagem": ["hotel", "pousada", "hospedagem", "diária", "estadia", "acomodação"],
    "alimentação": ["restaurante", "refeição", "almoço", "jantar", "café", "lanchonete", "alimentação"],
    "transporte": ["táxi", "uber", "99", "cabify", "ônibus", "metrô", "trem", "passagem", "aérea", "voo", "transporte"],
    "combustível": ["combustível", "gasolina", "etanol", "diesel", "posto"],
    "outros": ["diversos", "outros"]
}

# Padrões de expressões regulares
REGEX_PATTERNS = {
    "valor_monetario": r"R\$?\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?",
    "data": r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
    "cnpj": r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}",
    "cpf": r"\d{3}\.\d{3}\.\d{3}-\d{2}"
}

# Configurações de exportação
EXCEL_TEMPLATE = os.path.join(PROJECT_ROOT, "templates", "relatorio_template.xlsx")
