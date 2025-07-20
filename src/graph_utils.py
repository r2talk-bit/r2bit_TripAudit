import json
import re
import ast
from typing import Union, Dict, Any, List

def generate_expense_report(json_data: Union[str, Dict[str, Any]]) -> str:
    """
    Parses JSON expense data and formats it into a human-readable report string.

    Args:
        json_data (Union[str, Dict[str, Any]]): Either a JSON string or a dictionary containing expense data.

    Returns:
        str: A formatted string representing the expense report, or an error message.
    """
    try:
        # Handle both string and dictionary inputs
        if isinstance(json_data, str):
            # Use the robust JSON parser with a fallback
            fallback = {
                "trip_purpose": "Unknown",
                "total_amount": 0,
                "currency": "BRL",
                "expense_items": []
            }
            # Use the local parse_llm_json_response function
            data = parse_llm_json_response(json_data, fallback)
        else:
            data = json_data

        report_lines = []
        report_lines.append("--- Relatório de Despesas ---")
        report_lines.append(f"Propósito da Viagem: {data.get('trip_purpose', 'N/A')}")
        report_lines.append(f"Período: {data.get('start_date', 'N/A')} a {data.get('end_date', 'N/A')}")
        report_lines.append(f"Colaborador: {data.get('employee_name', 'N/A')}")
        report_lines.append(f"Total: {data.get('total_amount', 0.0):.2f} {data.get('currency', 'N/A')}")
        report_lines.append("\n--- Itens de Despesa ---")

        expense_items = data.get('expense_items', [])
        if not expense_items:
            report_lines.append("Nenhum item de despesa encontrado.")
        else:
            for i, item in enumerate(expense_items):
                report_lines.append(f"  Item {i + 1}:")
                report_lines.append(f"    Descrição: {item.get('description', 'N/A')}")
                report_lines.append(f"    Valor: {item.get('amount', 0.0):.2f}")
                report_lines.append(f"    Data: {item.get('date', 'N/A')}")
                report_lines.append(f"    Categoria: {item.get('category', 'N/A')}")
                report_lines.append(f"    Fornecedor: {item.get('vendor', 'N/A')}")
                report_lines.append("-" * 25) # Separator for items

        report_lines.append("----------------------------")
        return "\n".join(report_lines)

    except json.JSONDecodeError:
        return "Erro: O JSON fornecido é inválido."
    except KeyError as e:
        return f"Erro: Chave ausente no JSON - {e}"
    except Exception as e:
        return f"Ocorreu um erro inesperado: {e}"

def generate_policies_report(policy_data: Dict[str, Any]) -> str:
    """
    Generates a human-readable report of policies retrieved by the policy_retrieval_agent.
    
    Args:
        policy_data: Dictionary containing 'relevant_policies'
        
    Returns:
        A formatted string representing the policy report
    """
    try:
        print("STARTING generate_policies_report with policy_data...")
        # Extract data
        relevant_policies = policy_data.get("relevant_policies", [])
        
        # Start building the report
        report_lines = []
        report_lines.append("--- Relatório de Políticas Aplicáveis ---")
        report_lines.append(f"Total de políticas encontradas: {len(relevant_policies)}")
        report_lines.append("\n--- Políticas Relevantes ---")
        
        # Sort policies by priority
        sorted_policies = sorted(relevant_policies, key=lambda p: 
                               0 if p.get("priority") == "high" else 
                               (1 if p.get("priority") == "medium" else 2))
        
        # Add each policy to the report
        for i, policy in enumerate(sorted_policies):
            priority = policy.get("priority", "medium").upper()
            priority_marker = "⚠️ " if priority == "HIGH" else ""
            
            report_lines.append(f"\n{priority_marker}Política {i+1}: {policy.get('policy_title', 'Sem título')}")
            report_lines.append(f"ID: {policy.get('policy_id', 'N/A')}")
            report_lines.append(f"Categoria: {policy.get('category', 'Geral')}")
            report_lines.append(f"Prioridade: {priority}")
            report_lines.append(f"Descrição: {policy.get('description', 'Sem descrição')}")
            
            # Add applicability reason if available
            if "applicability_reason" in policy:
                report_lines.append(f"Aplicabilidade: {policy.get('applicability_reason')}")
            
            report_lines.append("-" * 40)  # Separator between policies
        
        report_lines.append("\nNota: As políticas de alta prioridade são marcadas com ⚠️")
        report_lines.append("-----------------------------------")
        
        return "\n".join(report_lines)
        
    except Exception as e:
        return f"Erro ao gerar relatório de políticas: {e}"
        
def parse_llm_json_response(content: str, fallback: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Unified JSON parser for LLM responses with multiple fallback strategies.
    
    Args:
        content: The raw text response from the LLM
        fallback: Optional fallback dictionary to return if all parsing strategies fail
        
    Returns:
        Parsed JSON as a dictionary, or the fallback dictionary if parsing fails
    """
    if not content:
        return fallback or {"error": "Empty content"}
        
    # Define multiple parsing strategies
    parsing_strategies = [
        # Strategy 1: Extract JSON from markdown code blocks
        lambda c: json.loads(re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', c).group(1)),
        # Strategy 2: Extract JSON without markdown formatting
        lambda c: json.loads(re.search(r'({[\s\S]*})', c).group(1)),
        # Strategy 3: Try direct JSON parsing
        lambda c: json.loads(c),
        # Strategy 4: Clean control characters and try parsing
        lambda c: json.loads(''.join(char for char in c if ord(char) >= 32 or char in '\t\n\r')),
        # Strategy 5: Try using ast.literal_eval with quote normalization
        lambda c: ast.literal_eval(c.replace("'", "\""))
    ]
    
    # Try each strategy in sequence
    for strategy in parsing_strategies:
        try:
            return strategy(content)
        except (json.JSONDecodeError, AttributeError, ValueError, SyntaxError):
            continue
    
    # If all strategies fail, log the error and return fallback
    print(f"Failed to parse JSON response with all strategies")
    print(f"Raw content: {content[:200]}..." if len(content) > 200 else content)
    
    # Return provided fallback or default fallback
    return fallback or {
        "error": "Failed to parse JSON response",
        "expense_categories_identified": ["unknown"],
        "queries": [{"query": "expense policies"}, {"query": "reimbursement rules"}]
    }


# Keep backward compatibility with existing code
def parse_json_response(content: str) -> Dict[str, Any]:
    """
    Legacy wrapper for parse_llm_json_response for backward compatibility.
    
    Args:
        content: The raw text response from the LLM
        
    Returns:
        Parsed JSON as a dictionary, or a fallback dictionary if parsing fails
    """
    fallback = {
        "expense_categories_identified": ["unknown"],
        "queries": [{"query": "expense policies"}, {"query": "reimbursement rules"}]
    }
    return parse_llm_json_response(content, fallback)


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content from the PDF
    """
    import pypdf
    
    pdf_text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text() + "\n\n"
    
    return pdf_text

def clean_markdown_text(text):
    """
    Limpa o texto para garantir formatação Markdown correta sem caracteres de escape indesejados.
    
    Args:
        text: Texto a ser processado
        
    Returns:
        Texto limpo com formatação Markdown adequada
    """
    if not text:
        return ""
        
    # Remove escape characters that might interfere with Markdown formatting
    text = re.sub(r'\\([\\`*_{}\[\]()#+\-.!])', r'\1', text)
    
    # Fix line breaks for Markdown
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Ensure code blocks are properly formatted
    text = re.sub(r'```(\w*)\s*\n', r'```\1\n', text)
    
    # Fix bullet points
    text = re.sub(r'^(\s*)\*\s', r'\1* ', text, flags=re.MULTILINE)
    text = re.sub(r'^(\s*)-\s', r'\1- ', text, flags=re.MULTILINE)
    
    # Fix table formatting
    lines = text.split('\n')
    
    # Look for potential table headers and add separators if missing
    for i in range(len(lines) - 1):
        if '|' in lines[i] and not any(lines[i].strip().startswith(x) for x in ['|---', '| ---']):
            # Check if next line is not a separator
            if i + 1 < len(lines) and not any(lines[i+1].strip().startswith(x) for x in ['|---', '| ---']):
                header_cols = lines[i].count('|') - 1 if lines[i].startswith('|') and lines[i].endswith('|') else lines[i].count('|') + 1
                separator = '| ' + ' | '.join(['---'] * header_cols) + ' |'
                lines.insert(i+1, separator)
    
    return '\n'.join(lines)

def generate_formatted_policies_report(formatted_policies: List[Dict[str, Any]]) -> str:
    """
    Generate a readable plain text report from formatted policies.
    
    Args:
        formatted_policies: List of formatted policy dictionaries
        
    Returns:
        A plain text report summarizing the policies
    """
    if not formatted_policies:
        return "No policies found."
    
    report_lines = []
    report_lines.append("=== POLICY SUMMARY REPORT ===")
    report_lines.append(f"Total policies: {len(formatted_policies)}")
    report_lines.append("")
    
    # Group policies by category
    policies_by_category = {}
    for policy in formatted_policies:
        category = policy.get("category", "General")
        if category not in policies_by_category:
            policies_by_category[category] = []
        policies_by_category[category].append(policy)
    
    # Add policies by category
    for category, policies in policies_by_category.items():
        report_lines.append(f"== {category} Policies ({len(policies)}) ==")
        
        for i, policy in enumerate(policies):
            # Clean up the description by replacing newlines with spaces
            description = policy.get("description", "").replace("\n", " ").strip()
                            
            report_lines.append(f"[{i+1}] {policy.get('policy_title', 'Untitled Policy')}")
            report_lines.append(f"    ID: {policy.get('policy_id', 'N/A')}")
            report_lines.append(f"    Priority: {policy.get('priority', 'medium').upper()}")
            report_lines.append(f"    Summary: {description}")
            report_lines.append("")
    
    report_lines.append("=== END OF POLICY REPORT ===")
    return "\n".join(report_lines)
    
def remove_duplicate_policies(policies):

    """
    Remove políticas duplicadas com base no ID e índice de chunk.
    
    As políticas podem vir duplicadas da base de conhecimento quando são divididas em chunks.
    Esta função garante que apenas uma versão de cada política seja usada.
    
    Args:
        policies: Lista de dicionários de políticas retornados por get_relevant_policies
        
    Returns:
        Lista de políticas únicas com duplicatas removidas
    """
    unique_policies = []  # Lista para armazenar políticas únicas
    seen_ids = set()      # Conjunto para rastrear IDs de políticas já vistas
    
    for policy in policies:  # Para cada política na lista
        # Cria um identificador único usando policy_id e chunk_index dos metadados
        if 'id' in policy and 'metadata' in policy and 'chunk_index' in policy['metadata']:
            # Combina o ID da política com o índice do chunk para criar um ID único
            unique_id = f"{policy['id']}_{policy['metadata']['chunk_index']}"
            if unique_id not in seen_ids:  # Se este ID único ainda não foi visto
                seen_ids.add(unique_id)      # Adiciona ao conjunto de IDs vistos
                unique_policies.append(policy)  # Adiciona a política à lista de políticas únicas
        else:
            # Se a política não tem a estrutura esperada, inclua-a de qualquer maneira
            # Isso garante que não perdemos nenhuma política, mesmo que falte metadados
            unique_policies.append(policy)
            
    return unique_policies  # Retorna a lista de políticas únicas


def format_policies_for_llm(policies):
    """
    Formata as políticas para serem enviadas ao modelo de linguagem.
    
    Converte a lista de dicionários de políticas em um texto formatado
    que é mais fácil para o LLM processar e entender.
    
    Args:
        policies (List[Dict]): Lista de dicionários representando políticas
        
    Returns:
        str: Texto formatado com todas as políticas numeradas
    """
    # Lista para armazenar cada política formatada
    formatted_policies = []
    
    # Itera sobre as políticas, numerando-as a partir de 1
    for i, policy in enumerate(policies, 1):
        # Cria um texto formatado para cada política
        formatted_policy = f"Policy {i}:\n"
        formatted_policy += f"ID: {policy.get('policy_id', 'N/A')}\n"  # ID da política ou N/A se não existir
        formatted_policy += f"Title: {policy.get('title', 'N/A')}\n"  # Título da política
        formatted_policy += f"Content: {policy.get('content', 'N/A')}\n"  # Conteúdo da política
        formatted_policies.append(formatted_policy)
    
    # Junta todas as políticas formatadas com duas quebras de linha entre cada uma
    return "\n\n".join(formatted_policies)


# === FUNÇÕES AUXILIARES ===
# Estas funções ajudam a processar e formatar os dados das políticas

def format_policies(policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Formata as políticas para o agente de verificação de conformidade.
    
    Esta função padroniza o formato das políticas recuperadas da base de conhecimento,
    garantindo que todas tenham os mesmos campos e estrutura, independentemente
    de como foram armazenadas originalmente.
    
    Args:
        policies: Lista de objetos de política da base de dados vetorial
        
    Returns:
        Lista formatada de políticas
    """
    formatted_policies = []  # Lista para armazenar as políticas formatadas
    
    for policy in policies:  # Para cada política na lista
        # Obtém a descrição da política e metadados
        description = policy.get("content", "")  # Conteúdo da política (texto)
        policy_title = policy.get("metadata", {}).get("policy_name", "Expense Policy")  # Título da política
        category = policy.get("metadata", {}).get("category", "General")  # Categoria da política
        
        # Extrai o policy_id dos metadados ou do campo id
        policy_id = policy.get("metadata", {}).get("policy_id", None)  # Tenta obter dos metadados
        if not policy_id and "id" in policy:  # Se não encontrou nos metadados
            # Tenta extrair do campo id
            id_parts = policy.get("id", "")
            policy_id = id_parts  # Usa o ID completo
        
        # Cria um dicionário formatado com todos os campos necessários
        formatted_policies.append({
            "policy_id": policy_id or f"policy-{len(formatted_policies)}",  # ID único ou gerado
            "policy_title": policy_title,  # Título da política
            "description": description,  # Descrição resumida
            "content": description,  # Conteúdo completo para verificação
            "category": category,  # Categoria (ex: "meals", "travel")
            "applicability_reason": "Relevant to expense items in report",  # Razão da aplicabilidade
            "priority": "medium"  # Prioridade padrão
        })
    
    return formatted_policies  # Retorna a lista de políticas formatadas

def get_policy_type(policy: Dict[str, Any]) -> str:
    """
    Determina o tipo de política com base em seu conteúdo e metadados.
    
    Esta função analisa o texto da política para identificar se ela se refere a
    regras específicas como limites de valor ou restrições de refeições.
    
    Args:
        policy: Um dicionário de política
        
    Returns:
        Identificador do tipo de política (ex: 'meals', 'max_amount')
    """
    # Converte todos os textos para minúsculas para facilitar a busca
    description = policy.get("description", "").lower()  # Descrição em minúsculas
    title = policy.get("policy_title", "").lower()  # Título em minúsculas
    policy_id = policy.get("policy_id", "").lower()  # ID em minúsculas
    
    # Verifica se é uma política relacionada a refeições
    if "meal" in description or "food" in description or "meal" in title or "meal" in policy_id:
        return "meals"  # Política de refeições
    # Verifica se é uma política relacionada a limites de valor
    elif "maximum" in description or "limit" in description or "5000" in description or "max" in policy_id:
        return "max_amount"  # Política de valor máximo
    else:
        return "other"  # Outros tipos de política
