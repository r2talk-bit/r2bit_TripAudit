"""
Multi-agent LangGraph workflow for expense report auditing.

Este módulo implementa um fluxo de trabalho LangGraph com quatro agentes especializados:
1. ParsingAgent: Analisa o texto bruto do relatório de despesas e o converte em formato JSON estruturado
2. PolicyRetrievalAgent: Recupera políticas de despesas relevantes da empresa
3. ComplianceCheckAgent: Verifica as despesas em relação às políticas da empresa
4. CommentarySynthesisAgent: Sintetiza as informações em um resumo claro e e-mail

"""

#############################################################################
# IMPORTAÇÕES
#############################################################################

# === BIBLIOTECAS PADRÃO DO PYTHON ===
import json  # Usado para converter entre objetos Python e strings JSON
import os    # Fornece funções para interagir com o sistema operacional (arquivos, pastas)
import re    # Biblioteca para trabalhar com expressões regulares (padrões de texto)
import yaml  # Usado para ler arquivos de configuração no formato YAML

# === TIPAGEM PARA CÓDIGO MAIS SEGURO ===
# Estas importações ajudam a definir tipos específicos para variáveis e funções
# Isso ajuda a evitar erros e facilita o entendimento do código
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, Callable, TypeVar
from functools import wraps  # Usado para criar decoradores que preservam metadados da função original

# === MÓDULOS LOCAIS DO PROJETO ===
# Importa a função que busca políticas relevantes no banco de dados
from policy_management import get_relevant_policies

# Funções auxiliares para processar texto, JSON e PDFs
from graph_utils import parse_llm_json_response
from graph_utils import extract_pdf_text, clean_markdown_text, remove_duplicate_policies, get_policy_type, format_policies_for_llm, format_policies

# === CONFIGURAÇÃO DE AMBIENTE ===
# Carrega variáveis de ambiente do arquivo .env (como chaves de API)
from dotenv import load_dotenv

# === LANGCHAIN - FRAMEWORK PARA TRABALHAR COM LLMs ===
# Componentes para criar mensagens para o modelo de linguagem
from langchain_core.messages import HumanMessage, SystemMessage
# Cliente para fazer chamadas à API da OpenAI
from langchain_openai import ChatOpenAI

# === LANGGRAPH - FRAMEWORK PARA CRIAR FLUXOS DE TRABALHO COM AGENTES ===
from langgraph.graph import END, StateGraph  # END marca o fim do fluxo de trabalho

#############################################################################
# CONFIGURAÇÃO INICIAL
#############################################################################

# Encontra o diretório raiz do projeto para localizar arquivos
from pathlib import Path
# __file__ é o caminho para este arquivo atual
# .parent acessa o diretório que contém este arquivo
# .parent.absolute() acessa o diretório pai (raiz do projeto) com caminho absoluto
project_root = Path(__file__).parent.parent.absolute()

# Carrega as variáveis de ambiente do arquivo .env
# Isso inclui chaves de API e outras configurações sensíveis
load_dotenv(os.path.join(project_root, '.env'))

#############################################################################
# CARREGAMENTO DE PROMPTS
#############################################################################

# Os prompts são instruções para os modelos de linguagem
# Eles estão armazenados em um arquivo YAML para facilitar a manutenção
prompts_path = os.path.join(Path(__file__).parent.absolute(), 'prompts.yaml')

# Abre e lê o arquivo YAML com os prompts
with open(prompts_path, 'r', encoding='utf-8') as file:
    # Converte o conteúdo YAML em um dicionário Python
    # Isso permite acessar os prompts por nome, como prompts['parsing_agent']['system_message']
    prompts = yaml.safe_load(file)

#############################################################################
# DEFINIÇÃO DO ESTADO DO FLUXO DE TRABALHO
#############################################################################

class WorkflowState(TypedDict):
    """
    Define a estrutura de dados para o estado compartilhado entre os agentes no fluxo de trabalho.
    
    Esta classe usa TypedDict para garantir que o estado tenha uma estrutura consistente
    e que os tipos de dados sejam verificados durante o desenvolvimento.
    
    Cada campo representa uma parte específica dos dados que fluem pelo workflow.
    """
    # Caminho para o arquivo PDF do relatório de despesas a ser auditado
    pdf_path: str
    
    # ID do usuário para rastreamento e limpeza de dados após o processamento
    # Opcional porque nem sempre é necessário para o processamento básico
    user_id: Optional[str]
    
    # Dados estruturados extraídos do relatório de despesas pelo ParsingAgent
    # Contém informações como valores, datas, categorias, etc.
    structured_expenses: Optional[Dict[str, Any]]
    
    # Lista de políticas da empresa relevantes para as despesas apresentadas
    # Recuperadas pelo PolicyRetrievalAgent com base nas despesas estruturadas
    relevant_policies: Optional[List[Dict[str, Any]]]
    
    # Resultados da verificação de conformidade feita pelo ComplianceCheckAgent
    # Indica quais despesas estão em conformidade com as políticas e quais não estão
    compliance_results: Optional[Dict[str, Any]]
    
    # Conteúdo formatado do e-mail gerado pelo CommentarySynthesisAgent
    # Contém um resumo da auditoria em formato legível para humanos
    email_content: Optional[Dict[str, Any]]
    
    # Mensagem de erro, se algum dos agentes encontrar um problema
    # Usado para tratamento de erros e fallback gracioso
    error: Optional[str]


# === INICIALIZAÇÃO DO CLIENTE OPENAI ===
# Configura o cliente que será usado para fazer chamadas aos modelos de linguagem da OpenAI
# O modelo gpt-4o é mais avançado e tem melhor desempenho, mas é mais caro e mais lento
client = ChatOpenAI(model="gpt-4o")  # Modelo mais avançado 

# Usamos o modelo gpt-3.5-turbo por padrão por ser mais rápido e econômico
# Este modelo ainda oferece bom desempenho para as tarefas de análise de despesas
# client = ChatOpenAI(model="gpt-3.5-turbo")  # Modelo mais rápido e econômico


def call_llm(system_message: str, user_message: str, fallback_response: str = None):
    """
    Faz uma chamada ao modelo de linguagem da OpenAI e retorna a resposta.
    
    Esta função encapsula a lógica de chamada à API da OpenAI, incluindo tratamento
    de erros e formatação das mensagens. Ela usa o cliente OpenAI configurado globalmente.
    
    Args:
        system_message (str): Mensagem de sistema que define o comportamento e contexto do modelo
                              (instruções sobre como o modelo deve responder)
        user_message (str): Mensagem/prompt enviado ao modelo (a pergunta ou tarefa)
        fallback_response (str, optional): Resposta alternativa a ser retornada caso a chamada falhe
        
    Returns:
        str: Conteúdo da resposta do modelo ou a resposta alternativa em caso de falha
    """
    try:
        # Cria as mensagens para o modelo no formato esperado pela API da OpenAI
        # SystemMessage define o comportamento geral do modelo
        # HumanMessage representa a entrada do usuário/prompt específico
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        
        # Faz a chamada ao modelo através do cliente LangChain e extrai o conteúdo da resposta
        # O método invoke() gerencia a comunicação com a API da OpenAI
        response = client.invoke(messages).content
        return response
    except Exception as e:
        # Registra o erro no console para fins de depuração
        print(f"Error calling LLM: {e}")
        # Retorna a resposta alternativa se fornecida, ou uma string vazia
        return fallback_response if fallback_response else ""


#############################################################################
# TRATAMENTO DE ERROS PARA AGENTES
#############################################################################

def handle_agent_errors(agent_name: str):
    """
    Decorador para lidar com erros ocorridos durante a execução dos agentes.
    
    Este decorador é aplicado a todas as funções de agentes no fluxo de trabalho.
    Ele captura qualquer exceção que ocorra durante a execução do agente e a
    transforma em uma mensagem de erro estruturada no estado do fluxo de trabalho.
    
    Isso permite que o fluxo de trabalho continue mesmo quando um agente falha,
    possibilitando que o sistema gere um resumo de e-mail com informações de erro
    em vez de falhar completamente.
    
    Args:
        agent_name (str): Nome do agente para identificação nas mensagens de erro
        
    Returns:
        Callable: Um decorador que pode ser aplicado a uma função de agente
    """
    # Esta função externa recebe o nome do agente e retorna um decorador
    def decorator(func: Callable[[WorkflowState], Dict[str, Any]]):
        # Esta função média é o decorador real que recebe a função do agente
        @wraps(func)  # Preserva os metadados da função original (nome, docstring, etc.)
        def wrapper(state: WorkflowState) -> Dict[str, Any]:
            # Se já existe um erro no estado, não tenta executar este agente
            # Isso evita processamento desnecessário quando o fluxo já falhou
            if state.get("error"):
                print(f"Skipping {agent_name} due to previous error: {state['error']}")
                return {}  # Retorna um dicionário vazio para não alterar o estado
            
            # Tenta executar a função do agente
            try:
                # Chama a função original do agente e retorna seu resultado
                return func(state)
            except Exception as e:
                # Se ocorrer qualquer erro, captura a exceção
                error_message = f"Error in {agent_name}: {str(e)}"
                print(error_message)  # Registra o erro no console para depuração
                
                # Retorna um dicionário contendo apenas a mensagem de erro
                # Este erro será incorporado ao estado do fluxo de trabalho
                return {"error": error_message}
                
        return wrapper  # Retorna a função wrapper que substitui a função original
    return decorator  # Retorna o decorador configurado com o nome do agente

# Variável de tipo para o decorador
T = TypeVar('T')  # Usado para tipagem genérica em decoradores

#############################################################################
# AGENTES ESPECIALIZADOS DO FLUXO DE TRABALHO
#############################################################################

@handle_agent_errors("parsing_agent")
def parsing_agent(state: WorkflowState) -> Dict[str, Any]:
    """
    Analisa o relatório de despesas em PDF e converte em dados JSON estruturados.
    
    Este é o primeiro agente no fluxo de trabalho e funciona como um extrator de informações.
    Ele recebe o caminho para o arquivo PDF do relatório de despesas, extrai o texto
    usando OCR (Reconhecimento Ótico de Caracteres) e processamento de imagem, e então
    usa um modelo de linguagem (LLM) para converter esse texto bruto em um formato JSON
    estruturado e padronizado.
    
    O processo envolve:
    1. Extração de texto do PDF usando a função extract_pdf_text
    2. Envio do texto extraído para o LLM com instruções específicas
    3. Processamento da resposta do LLM para garantir que esteja no formato JSON esperado
    
    Args:
        state (WorkflowState): Estado atual do fluxo de trabalho contendo o caminho do PDF
                              no campo 'pdf_path'
        
    Returns:
        Dict[str, Any]: Dicionário contendo a chave 'structured_expenses' com os dados
                       estruturados extraídos do relatório de despesas
    """
    # Extrai o texto do PDF usando OCR e processamento de imagem
    # A função extract_pdf_text usa bibliotecas como img2table e Tesseract OCR
    # para converter o PDF em texto legível por máquina
    # Nota: Por padrão, usa o idioma inglês ('eng') para o OCR
    pdf_text = extract_pdf_text(state["pdf_path"])
    
    # Prepara o prompt para o modelo de linguagem
    # O system_message define o comportamento geral do modelo
    # O user_message contém o texto extraído do PDF e instruções específicas
    system_message = prompts['parsing_agent']['system_message']
    user_message = prompts['parsing_agent']['prompt'].format(pdf_text=pdf_text)
    
    # Chama o modelo de linguagem para analisar o texto e estruturá-lo
    # O modelo recebe o texto bruto e retorna uma versão estruturada em formato JSON
    response = call_llm(system_message, user_message)
    
    # Define o esquema esperado para o JSON de resposta
    # Isso garante que o JSON retornado pelo modelo tenha a estrutura correta
    # e que todos os campos necessários estejam presentes
    expected_schema = {
        "expense_report": {
            "report_id": str,  # Identificador único do relatório
            "employee": {
                "name": str,  # Nome do funcionário
                "id": str    # ID do funcionário
            },
            "total_amount": float,  # Valor total das despesas
            "currency": str,       # Moeda (ex: BRL, USD)
            "submission_date": str,  # Data de envio do relatório
            "expenses": [{  # Lista de despesas individuais
                "description": str,  # Descrição da despesa
                "amount": float,     # Valor da despesa
                "date": str,         # Data da despesa
                "category": str,      # Categoria (ex: transporte, hospedagem)
                "receipt_id": str    # ID do recibo/comprovante
            }]
        }
    }
    
    # Analisa a resposta do modelo e extrai o JSON estruturado
    # A função parse_llm_json_response garante que o JSON esteja no formato correto
    # e lida com possíveis erros de formatação na resposta do modelo
    structured_expenses = parse_llm_json_response(response, expected_schema)
    
    # Retorna as despesas estruturadas para o próximo agente no fluxo de trabalho
    # Este resultado será usado pelo policy_retrieval_agent para buscar políticas relevantes
    return {"structured_expenses": structured_expenses}


@handle_agent_errors("policy_retrieval_agent")
def policy_retrieval_agent(state: WorkflowState) -> Dict[str, Any]:
    """
    Recupera políticas de despesas relevantes com base nos dados estruturados.
    
    Este é o segundo agente no fluxo de trabalho e funciona como um pesquisador de políticas.
    Ele recebe os dados estruturados do relatório de despesas do ParsingAgent,
    analisa esses dados para identificar quais tipos de políticas são relevantes,
    e então busca essas políticas na base de conhecimento da empresa.
    
    O processo envolve:
    1. Usar o LLM para analisar as despesas e gerar consultas de pesquisa relevantes
    2. Executar essas consultas na base de conhecimento para recuperar políticas
    3. Remover duplicatas e formatar as políticas para o próximo agente
    
    Args:
        state (WorkflowState): Estado atual do fluxo de trabalho contendo as despesas estruturadas
                              no campo 'structured_expenses'
        
    Returns:
        Dict[str, Any]: Dicionário contendo a chave 'relevant_policies' com as políticas
                       relevantes para as despesas apresentadas
    """
    # Extrai os dados do estado do fluxo de trabalho
    # structured_expenses contém os dados JSON das despesas já estruturados pelo ParsingAgent
    structured_expenses = state["structured_expenses"]
    # user_id é usado para rastrear políticas específicas do usuário (se existirem)
    # Se não for fornecido, usa "default_user" como padrão
    user_id = state.get("user_id", "default_user")
    
    # Prepara o prompt para o modelo de linguagem (LLM)
    # O system_message define o comportamento geral do modelo
    system_message = prompts["policy_retrieval_agent"]["system_message"]
    
    # Converte o dicionário de despesas estruturadas em uma string JSON formatada
    # O parâmetro indent=4 adiciona recuos para tornar o JSON mais legível
    import json
    structured_expenses_str = json.dumps(structured_expenses, indent=4)
    
    # Formata o prompt inserindo a string JSON no template
    # O template contém um marcador {structured_expenses} que será substituído
    user_message = prompts["policy_retrieval_agent"]["prompt"].format(structured_expenses=structured_expenses_str)
    
    # Chama o modelo de linguagem para analisar as despesas e gerar consultas de pesquisa
    # O LLM analisará as despesas e sugerirá quais tipos de políticas devem ser buscadas
    llm_response = call_llm(system_message, user_message)
    
    # Lista para armazenar as consultas de pesquisa geradas pelo LLM
    search_queries = []

    # Processa a resposta de texto do LLM para extrair as consultas
    if llm_response and llm_response.strip():  # Verifica se a resposta não está vazia
        # Divide a resposta em linhas e remove espaços em branco extras
        lines = [line.strip() for line in llm_response.split('\n') if line.strip()]
        
        # Extrai linhas que parecem consultas (geralmente contendo perguntas ou palavras-chave)
        for line in lines:
            # Remove marcadores de lista, números e outros prefixos comuns usando expressão regular
            # ^ indica o início da linha, e os caracteres dentro dos colchetes são os que serão removidos
            clean_line = re.sub(r'^[-*•●■□▪▫◆◇◈⬤⚫⚪⦿⊙⊚⊛⊜⊝⊗⊘⊙⊚⊛⊜⊝0-9.\s"]*', '', line).strip()
            # Remove aspas duplas e simples do início e fim da linha
            clean_line = clean_line.strip('"').strip("'").strip()
            
            # Pula linhas vazias ou muito curtas (menos de 5 caracteres)
            # Isso evita consultas inúteis como "a", "de", etc.
            if clean_line and len(clean_line) > 5:
                search_queries.append(clean_line)
    
    # Se nenhuma consulta foi extraída, usa uma consulta padrão
    # Isso garante que pelo menos algumas políticas serão recuperadas
    if not search_queries:
        search_queries.append("list the refund rules")
    
    # Remove consultas duplicadas enquanto preserva a ordem original
    # O conjunto 'seen' rastreia consultas já vistas
    seen = set()
    # Esta é uma list comprehension que mantém apenas a primeira ocorrência de cada consulta
    # A expressão 'not (q in seen or seen.add(q))' é um truque para adicionar ao conjunto e verificar ao mesmo tempo
    search_queries = [q for q in search_queries if not (q in seen or seen.add(q))]
    
    # Busca e combina políticas para todas as consultas
    all_policies = []
    for query in search_queries:
        # get_relevant_policies é uma função importada de policy_management.py
        # que busca políticas relevantes na base de conhecimento
        policies = get_relevant_policies(query, user_id)
        # Adiciona as políticas encontradas à lista completa
        all_policies.extend(policies)
    
    # Remove políticas duplicadas usando a função definida anteriormente
    all_policies = remove_duplicate_policies(all_policies)
        
    # Formata as políticas para verificação de conformidade
    # Isso padroniza o formato das políticas para o próximo agente
    formatted_policies = format_policies(all_policies)
    
    # Gera um relatório legível das políticas para depuração
    # Importa a função apenas quando necessário para evitar importações circulares
    from graph_utils import generate_formatted_policies_report
    policy_report = generate_formatted_policies_report(formatted_policies)
    # Imprime o relatório formatado com separadores para facilitar a leitura
    print("\n\n" + "=" * 80)
    print("FORMATTED POLICY REPORT:")
    print("=" * 80)
    print(policy_report)
    print("=" * 80)
    print("\n\n")
    
    # Cria um dicionário com a chave 'relevant_policies' contendo as políticas formatadas
    policy_data = {
        "relevant_policies": formatted_policies
    }

    # Retorna os dados de política para o próximo agente no fluxo de trabalho
    # Este resultado será usado pelo compliance_check_agent para verificar conformidade
    return policy_data
    
@handle_agent_errors("compliance_check_agent")
def compliance_check_agent(state: WorkflowState) -> Dict[str, Any]:
    """
    Verifica as despesas em relação às políticas da empresa para determinar conformidade.
    
    Este é o terceiro agente no fluxo de trabalho e funciona como um auditor de conformidade.
    Ele recebe os dados estruturados do relatório de despesas e as políticas relevantes,
    compara cada despesa com as políticas aplicáveis e determina se estão em conformidade.
    
    O processo envolve:
    1. Analisar as despesas estruturadas e as políticas relevantes
    2. Usar o LLM para verificar a conformidade de cada despesa com as políticas
    3. Gerar um relatório detalhado de conformidade com violações identificadas
    
    Args:
        state (WorkflowState): Estado atual do fluxo de trabalho contendo as despesas estruturadas
                              e as políticas relevantes
        
    Returns:
        Dict[str, Any]: Dicionário contendo a chave 'compliance_results' com os resultados
                       da verificação de conformidade
    """
    
    # Mensagem de log para indicar o início do agente de verificação de conformidade
    print("STARTING COMPLIANCE CHECK AGENT")
    
    # Extrai os dados necessários do estado do fluxo de trabalho
    # structured_expenses: dados estruturados das despesas do relatório
    structured_expenses = state["structured_expenses"]
    # relevant_policies: políticas relevantes recuperadas pelo agente anterior
    relevant_policies = state["relevant_policies"]
    # policy_analysis: análise adicional das políticas (se disponível)
    policy_analysis = state.get("policy_analysis", {})  # Usa um dicionário vazio como padrão se não existir
    
    # Ordena as políticas por prioridade para garantir que políticas mais importantes
    # sejam consideradas primeiro na verificação de conformidade
    # A expressão lambda retorna 0 para prioridade alta (colocando-as primeiro) e 1 para as demais
    sorted_policies = sorted(relevant_policies, key=lambda p: 0 if p.get("priority") == "high" else 1)
    
    # Imprime as políticas relevantes ordenadas por prioridade para fins de depuração
    print(f"RELEVANT POLICIES (sorted by priority):")
    for i, policy in enumerate(sorted_policies):
        # Mostra o índice (começando em 1), prioridade, ID e categoria de cada política
        print(f"{i+1}. [{policy.get('priority', 'medium')}] {policy.get('policy_id')}: {policy.get('category')}")
    
    # Cria um contexto estruturado das políticas para enviar ao modelo de linguagem
    # Isso facilita a análise pelo LLM ao fornecer dados em um formato consistente
    policy_context = {
        "policies": sorted_policies,  # Lista de políticas ordenadas por prioridade
        "analysis": {  # Informações adicionais de análise (se disponíveis)
            # Categorias de despesas identificadas anteriormente
            "expense_categories": policy_analysis.get("expense_categories_identified", []),
            # Resumo das consultas usadas para recuperar as políticas
            "query_summary": policy_analysis.get("query_summary", ""),
            # Análise específica do valor total (se disponível)
            "total_amount_analysis": policy_analysis.get("total_amount_analysis", {})
        }
    }
    
    # Prepara o prompt para o modelo de linguagem formatando-o com os dados das despesas
    # e políticas convertidos para JSON formatado (com recuo para melhor legibilidade)
    prompt = prompts['compliance_check_agent']['prompt'].format(
        structured_expenses=json.dumps(structured_expenses, indent=2),
        relevant_policies=json.dumps(policy_context, indent=2)
    )
    
    # Define uma resposta alternativa para casos em que a chamada ao LLM falhe
    # Esta é uma string JSON que indica não-conformidade com uma mensagem de erro
    fallback_content = '{"compliant": false, "violations": [{"policy": "Error", "description": "Failed to evaluate compliance due to an error."}]}'
    
    # Chama o modelo de linguagem para verificar a conformidade
    # Passa o prompt formatado e a resposta alternativa em caso de falha
    content = call_llm(
        prompts['compliance_check_agent']['system_message'],  # Instruções gerais para o modelo
        prompt,  # Prompt específico com os dados das despesas e políticas
        fallback_content  # Resposta alternativa em caso de falha
    )
    
    # Define uma estrutura de fallback mais completa para o caso de falha na análise da resposta
    # Isso garante que mesmo em caso de erro, teremos uma resposta estruturada
    fallback = {
        "compliant": False,  # Por padrão, considera não conforme em caso de erro
        "violations": [{"policy": "Parsing Error", "description": "Failed to parse LLM response."}],
        "comments": ["An error occurred during compliance analysis. Please review manually."]
    }
    
    # Informa que está usando o parser robusto para analisar os resultados
    print("Using robust JSON parser for compliance check results")
    # Analisa a resposta do LLM e converte para um dicionário Python
    # Se a análise falhar, usa a estrutura de fallback definida acima
    compliance_results = parse_llm_json_response(content, fallback)
    
    # Garante que todos os campos necessários existam no resultado
    # Isso evita erros ao acessar campos que podem estar ausentes
    if "is_compliant" not in compliance_results:
        # Se o campo de conformidade não existir, considera não conforme por padrão
        compliance_results["is_compliant"] = False
    if "total_amount" not in compliance_results:
        # Se o valor total não estiver no resultado, usa o valor das despesas estruturadas
        compliance_results["total_amount"] = structured_expenses.get("total_amount", 0)
    if "currency" not in compliance_results:
        # Se a moeda não estiver no resultado, usa BRL (Real Brasileiro) como padrão
        compliance_results["currency"] = structured_expenses.get("currency", "BRL")
    if "violations" not in compliance_results:
        # Se não houver lista de violações, cria uma lista vazia
        compliance_results["violations"] = []
    if "compliant_items" not in compliance_results:
        # Se não houver lista de itens conformes, cria uma lista vazia
        compliance_results["compliant_items"] = []
    if "non_compliant_items" not in compliance_results:
        # Se não houver lista de itens não conformes, cria uma lista vazia
        compliance_results["non_compliant_items"] = []
    
    # Imprime os resultados da verificação de conformidade para depuração
    # Os separadores (*) tornam a saída mais visível no console
    print("*" *50)
    print("CHECK COMPLIANCE RESULT")
    # Imprime o resultado como JSON formatado, garantindo que caracteres especiais sejam preservados
    print(json.dumps(compliance_results, indent=4, ensure_ascii=False))
    print("*" *50)
    
    # Retorna os resultados da verificação de conformidade para o próximo agente
    # Este resultado será usado pelo commentary_synthesis_agent para gerar o e-mail final
    return {"compliance_results": compliance_results}

@handle_agent_errors("cleanup_agent")
def cleanup_agent(state: WorkflowState) -> Dict[str, Any]:
    """
    Limpa os dados específicos do usuário da base de dados vetorial após a conclusão do fluxo de trabalho.
    
    Este é um agente auxiliar que garante que os dados temporários do usuário sejam removidos
    da base de conhecimento após o processamento do relatório de despesas. Isso é importante
    para manter a base de dados limpa e evitar acúmulo de dados desnecessários.
    
    O processo envolve:
    1. Verificar se existe um ID de usuário no estado do fluxo de trabalho
    2. Chamar a função de exclusão de políticas para remover dados específicos do usuário
    3. Retornar o status da operação de limpeza
    
    Args:
        state (WorkflowState): Estado atual do fluxo de trabalho contendo o ID do usuário
                              no campo 'user_id' (se disponível)
        
    Returns:
        Dict[str, Any]: Dicionário contendo o status da operação de limpeza ou mensagem de erro
    """
    # Obtém o ID do usuário do estado do fluxo de trabalho
    # O ID do usuário é usado para identificar quais políticas devem ser removidas
    user_id = state.get("user_id")  # Usa get() para evitar erro se a chave não existir
    
    # Verifica se o ID do usuário existe
    # Se não existir, não há nada para limpar, então pula esta etapa
    if not user_id:
        print("No user ID found in state, skipping cleanup")
        return {}  # Retorna um dicionário vazio, mantendo o estado inalterado
    
    # Importa a função de exclusão de políticas do módulo policy_management
    # A importação é feita aqui para evitar dependências circulares
    from policy_management import delete_user_policies
    
    # Tenta excluir todas as políticas associadas a este usuário
    # Usa um bloco try-except para capturar e tratar possíveis erros
    try:
        # Chama a função que exclui as políticas e retorna o número de itens excluídos
        deleted_count = delete_user_policies(user_id)
        
        # Registra o resultado da operação de limpeza no console
        print(f"Cleanup complete: Deleted {deleted_count} policies for user {user_id}")
        
        # Retorna um status de sucesso com informações sobre a operação
        return {"cleanup_status": f"Successfully deleted {deleted_count} policies for user {user_id}"}
    except Exception as e:
        # Se ocorrer algum erro durante a limpeza, captura e registra a exceção
        error_msg = f"Error during policy cleanup: {str(e)}"
        print(error_msg)  # Imprime a mensagem de erro no console
        
        # Retorna um dicionário contendo a mensagem de erro
        # Isso será incorporado ao estado do fluxo de trabalho
        return {"error": error_msg}

@handle_agent_errors("commentary_synthesis_agent")
def commentary_synthesis_agent(state: WorkflowState) -> Dict[str, Any]:
    """
    Gera um resumo final legível para humanos dos resultados da auditoria de despesas.
    
    Este é o quarto e último agente no fluxo de trabalho e funciona como um sintetizador de informações.
    Ele recebe todos os dados processados pelos agentes anteriores (estruturação, políticas e conformidade)
    e gera um e-mail formatado com um resumo claro e conciso da auditoria, incluindo recomendações
    sobre aprovação ou rejeição das despesas.
    
    O processo envolve:
    1. Consolidar os dados de todos os agentes anteriores
    2. Simplificar os dados para facilitar o processamento pelo LLM
    3. Retornar o texto do e-mail gerado pelo LLM.
    
    Args:
        state (WorkflowState): Estado atual do fluxo de trabalho contendo todos os dados processados
                              pelos agentes anteriores
        
    Returns:
        Dict[str, Any]: Dicionário contendo a chave 'email_content' com o conteúdo formatado do e-mail
    """
    # Caso especial: se houver um erro em qualquer etapa anterior, gera um e-mail de fallback
    # Isso garante que o fluxo de trabalho sempre produza uma saída, mesmo em caso de falha
    if state.get("error"):
        # Cria um conteúdo de e-mail básico informando sobre o erro
        email_content = {
            "email_text": "Error Processing Expense Report",  # Assunto indicando erro
        }
        # Retorna o conteúdo do e-mail de erro
        return {"email_text": email_content}
    
    # Se não houver erro, tenta gerar um e-mail completo com base nos dados processados
    try:
        # Extrai todos os dados necessários do estado do fluxo de trabalho
        # Estes dados foram produzidos pelos agentes anteriores
        structured_expenses = state["structured_expenses"]  # Dados estruturados do ParsingAgent
        relevant_policies = state["relevant_policies"]  # Políticas relevantes do PolicyRetrievalAgent
        policy_analysis = state.get("policy_analysis", {})  # Análise adicional de políticas (se disponível)
        compliance_results = state["compliance_results"]  # Resultados de conformidade do ComplianceCheckAgent
        
        # Verifica se temos todos os dados necessários para gerar o e-mail
        # Se algum dado essencial estiver faltando, levanta uma exceção
        if not structured_expenses or not relevant_policies or not compliance_results:
            raise ValueError("Missing required data for email generation")
        
        # Ordena as políticas por prioridade (alta prioridade primeiro)
        # Isso ajuda a destacar as políticas mais importantes no e-mail
        sorted_policies = sorted(relevant_policies, key=lambda p: 0 if p.get("priority") == "high" else 1)
        
        # Cria um contexto aprimorado com valores padrão seguros
        # Este contexto será usado pelo LLM para gerar o e-mail
        enhanced_context = {
            "policies": sorted_policies,  # Políticas ordenadas por prioridade
            "analysis": {
                # Categorias de despesas identificadas (ou lista vazia se não houver)
                "expense_categories": policy_analysis.get("expense_categories_identified", [])
            }
        }
        
        # Cria uma versão simplificada dos dados para o LLM para reduzir a complexidade
        # Isso ajuda a evitar tokens desnecessários e facilita o processamento pelo modelo
        simplified_expenses = {
            "trip_purpose": structured_expenses.get("trip_purpose", "Unknown"),  # Propósito da viagem
            "total_amount": structured_expenses.get("total_amount", 0),  # Valor total das despesas
            "currency": structured_expenses.get("currency", "BRL"),  # Moeda (Real Brasileiro por padrão)
            "expense_items": []  # Lista vazia para itens de despesa (será preenchida abaixo)
        }
        
        # Adiciona apenas dados essenciais de itens de despesa
        # Isso reduz o tamanho do prompt e mantém apenas informações relevantes
        if "expense_items" in structured_expenses and isinstance(structured_expenses["expense_items"], list):
            for item in structured_expenses["expense_items"]:
                # Verifica se o item é um dicionário válido
                if isinstance(item, dict):
                    # Cria um item simplificado com apenas os campos essenciais
                    simplified_item = {
                        "description": item.get("description", "Unknown"),  # Descrição da despesa
                        "amount": item.get("amount", 0),  # Valor da despesa
                        "category": item.get("category", "Unknown")  # Categoria da despesa
                    }
                    # Adiciona o item simplificado à lista
                    simplified_expenses["expense_items"].append(simplified_item)
        
        # Simplifica os resultados de conformidade para o LLM
        # Novamente, mantém apenas os campos mais importantes
        simplified_compliance = {
            "compliant": compliance_results.get("compliant", False),  # Status geral de conformidade
            "approval_recommendation": compliance_results.get("approval_recommendation", "Needs Review"),  # Recomendação
            "violations": compliance_results.get("violations", [])  # Lista de violações encontradas
        }
        
        # Prepara o prompt para o LLM com os dados simplificados
        # Formata o prompt inserindo os dados JSON no template
        prompt = prompts['commentary_synthesis_agent']['prompt'].format(
            structured_expenses=json.dumps(simplified_expenses, indent=2),  # Despesas em formato JSON
            relevant_policies=json.dumps(enhanced_context, indent=2),  # Políticas em formato JSON
            compliance_results=json.dumps(simplified_compliance, indent=2)  # Resultados em formato JSON
        )
        
        # Define um fallback bem estruturado para a chamada do LLM
        # Isso será usado se a chamada ao LLM falhar completamente
        email_fallback = json.dumps({
            "email_text": "Análise de Despesas de Viagem",  # Assunto padrão
        })
        
        # Chama o LLM para sintetizar o e-mail com os dados simplificados
        # Passa o prompt formatado e o fallback em caso de erro
        content = call_llm(
            prompts['commentary_synthesis_agent']['system_message'],  # Instruções gerais para o modelo
            prompt,  # Prompt específico com os dados simplificados
            email_fallback  # Resposta alternativa em caso de falha
        )
        
        # Define um fallback mais detalhado para análise da resposta
        # Este será usado se o parsing da resposta do LLM falhar
        fallback = {
            "email_text": "Análise de Despesas de Viagem",  # Assunto padrão
        }
        
        # Retorna o conteúdo do e-mail gerado
        # Este é o resultado final do fluxo de trabalho
        return {"email_content": content}
        
    except Exception as e:
        # Em caso de erro durante a geração do e-mail, cria um e-mail de fallback significativo
        # Isso garante que o fluxo de trabalho sempre produza uma saída, mesmo em caso de falha interna
        print(f"Error in commentary synthesis: {str(e)}")  # Registra o erro no console
        
        # Cria um e-mail de fallback em português com informações básicas
        email_content = {
            "email_text": "Análise de Despesas de Viagem",  # Assunto padrão
        }
        
        # Retorna o conteúdo do e-mail de fallback
        return {"email_content": email_content}
    
# Define the edge functions for conditional workflow routing

def should_continue(state: WorkflowState) -> str:
    """
    Determines whether the workflow should continue to the next step or end due to an error.
    
    Args:
        state: Current workflow state
        
    Returns:
        Name of the next node or END if an error occurred
    """
    # If an error has occurred, end the workflow
    if state.get("error"):
        return "commentary_synthesis_agent"
    
    # Otherwise, continue to the next step in the workflow
    return "continue"

# Define the main workflow function

def create_agent_workflow() -> StateGraph:
    """
    Builds the LangGraph workflow for expense auditing with four specialized agents.
    
    Returns:
        A configured StateGraph workflow
    """
    # Create a new workflow graph
    workflow = StateGraph(WorkflowState)
    
    # Add the nodes to the workflow
    workflow.add_node("parsing_agent", parsing_agent)
    workflow.add_node("policy_retrieval_agent", policy_retrieval_agent)
    workflow.add_node("compliance_check_agent", compliance_check_agent)
    workflow.add_node("commentary_synthesis_agent", commentary_synthesis_agent)
    workflow.add_node("cleanup_agent", cleanup_agent)  # Add the cleanup agent
    
    # Define the edges between nodes
    workflow.add_edge("parsing_agent", "policy_retrieval_agent")
    workflow.add_conditional_edges(
        "policy_retrieval_agent",
        should_continue,
        {
            "continue": "compliance_check_agent",
            "commentary_synthesis_agent": "commentary_synthesis_agent"
        }
    )
    workflow.add_conditional_edges(
        "compliance_check_agent",
        should_continue,
        {
            "continue": "commentary_synthesis_agent",
            "commentary_synthesis_agent": "commentary_synthesis_agent"
        }
    )
    
    # Add edge from commentary synthesis to cleanup agent
    workflow.add_edge("commentary_synthesis_agent", "cleanup_agent")
    
    # Add edge from cleanup agent to END
    workflow.add_edge("cleanup_agent", END)
    
    # Remove direct edge from commentary synthesis to END
    
    # Set the entry point
    workflow.set_entry_point("parsing_agent")
    
    return workflow

# Create the workflow
agent_team_workflow = create_agent_workflow().compile()

def run_agentic_auditor(pdf_path: str, user_id: str = None) -> Dict[str, Any]:
    """
    Main entry point for the expense audit workflow.
    
    Args:
        pdf_path: Path to the PDF expense report file
        user_id: Optional user ID for policy management
        
    Returns:
        Dictionary containing the generated email content and any errors
    """
    print("Starting agent team auditor with LangGraph workflow...")
    
    # Check if the OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        error_message = "The OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable."
        print(f"Error: {error_message}")
        return {
            "email_content": {
                "subject": "Error: API Key Missing",
                "body": error_message,
                "recipient": "System Administrator",
                "approval_status": "Error",
                "approval_comments": "Configuration error: API key missing."
            },
            "error": error_message
        }
    
    # Verify the PDF file exists
    if not os.path.exists(pdf_path):
        error_message = f"The PDF file does not exist at path: {pdf_path}"
        print(f"Error: {error_message}")
        return {
            "email_content": {
                "subject": "Error: PDF File Not Found",
                "body": error_message,
                "recipient": "System Administrator",
                "approval_status": "Error",
                "approval_comments": "File not found error."
            },
            "error": error_message
        }
    
    # If no user_id is provided, generate a default one
    if not user_id:
        import uuid
        user_id = str(uuid.uuid4())
        print(f"No user_id provided, generated: {user_id}")
    
    try:
        # Build and compile the workflow graph
        print("Building and compiling the LangGraph workflow...")
        workflow = create_agent_workflow()
        app = workflow.compile()
        
        # Initialize the workflow state with the PDF path and user_id
        initial_state = WorkflowState(
            pdf_path=pdf_path,
            user_id=user_id,  # Include user_id in the initial state
            structured_expenses={},
            relevant_policies=[],
            compliance_results={},
            email_content={},
            error=""
        )
        
        # Run the workflow with the initial state
        print("Executing the LangGraph workflow...")
        result = app.invoke(initial_state)
        
        # Extract and return the relevant results from the final state
        print("Workflow execution completed successfully.")
        return {
            "email_content": result.get("email_content", {}),
            "error": result.get("error", "")
        }
        
    except Exception as e:
        # Get the full stack trace for detailed debugging
        import traceback
        error_trace = traceback.format_exc()
        
        # Log the error to the console for debugging
        print(f"Workflow execution error: {str(e)}")
        print(error_trace)
        
        return {
            "email_content": {
                "subject": "Expense Report Analysis - Manual Review Required",
                "body": f"""O relatório de despesas foi processado, mas ocorreu um erro durante a análise automática.
                
Detalhes do erro: {str(e)}

IMPORTANTE: Por favor, revise os dados do relatório manualmente e aplique as políticas da empresa durante a revisão.
""",
                "recipient": "Finance Department",
                "approval_status": "Needs Review",
                "approval_comments": "Erro técnico durante o processamento. Aplicar políticas de reembolso manualmente."
            },
            "error": f"Workflow execution error: {str(e)}"
        }

#############################################################################
# NOTAS SOBRE O FLUXO DE TRABALHO COMPLETO
#############################################################################

# O fluxo de trabalho de auditoria de despesas implementado neste módulo segue uma
# arquitetura de agentes especializados usando LangGraph. Cada agente tem uma função
# específica e bem definida:
#
# 1. ParsingAgent:
#    - Extrai e estrutura dados brutos de relatórios de despesas em PDF
#    - Converte texto não estruturado em formato JSON padronizado
#
# 2. PolicyRetrievalAgent:
#    - Identifica políticas relevantes com base nas despesas apresentadas
#    - Consulta a base de conhecimento vetorial para recuperar políticas aplicáveis
#    - Filtra e remove políticas duplicadas para melhorar a eficiência
#
# 3. ComplianceCheckAgent:
#    - Verifica cada despesa contra as políticas da empresa
#    - Identifica violações específicas (ex: valor máximo, refeições não reembolsáveis)
#    - Gera um relatório detalhado de conformidade com justificativas
#
# 4. CommentarySynthesisAgent:
#    - Sintetiza todos os dados em um e-mail legível para humanos
#    - Fornece recomendações claras sobre aprovação ou rejeição
#    - Formata o conteúdo para facilitar a compreensão pelos destinatários
#
# O fluxo de dados entre os agentes é gerenciado pelo estado compartilhado (WorkflowState),
# que garante que cada agente tenha acesso aos dados necessários dos agentes anteriores.
# O tratamento de erros é implementado em cada nível para garantir que o fluxo de trabalho
# possa continuar mesmo quando ocorrem problemas em etapas individuais.

#############################################################################
# CONSIDERAÇÕES DE USO
#############################################################################

# Ao utilizar este módulo, considere os seguintes pontos:
#
# 1. Desempenho e Custos:
#    - O uso intensivo de modelos de linguagem (LLMs) pode gerar custos significativos
#    - Considere implementar cache para resultados intermediários em casos de uso frequente
#
# 2. Personalização:
#    - As mensagens de sistema para cada agente podem ser ajustadas para casos de uso específicos
#    - Os prompts em prompts.yaml podem ser modificados para melhorar o desempenho em cenários particulares
#
# 3. Limitações Conhecidas:
#    - Políticas complexas ou ambíguas podem não ser interpretadas corretamente
#
# 4. Extensibilidade:
#    - Novos agentes podem ser adicionados ao fluxo de trabalho conforme necessário
#    - O estado do fluxo de trabalho (WorkflowState) pode ser expandido para incluir novos campos
#    - Funções auxiliares podem ser modificadas para comportamentos específicos

# End of file
