import json
from typing import Union, Dict, Any
from agent_team import parse_llm_json_response

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

