import os
import re
import torch
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
from typing import List, Dict

# Configurações
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/layoutlmv3-base"
TESSERACT_LANG = "eng"  # English (default language)

class ExpenseReportExtractor:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE, tesseract_lang=TESSERACT_LANG):
        """Inicializa o extrator com o modelo LayoutLMv3 e configurações."""
        self.device = device
        self.tesseract_lang = tesseract_lang
        
        # Inicializa o modelo e processador LayoutLMv3
        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
        
        # Configuração de rótulos detalhada para relatórios de despesas
        if not hasattr(self.model.config, 'id2label') or not self.model.config.id2label:
            self.model.config.id2label = {
                0: "O",                    # Outside (não é uma entidade relevante)
                1: "B-VALOR_TOTAL",        # Início de um valor total
                2: "I-VALOR_TOTAL",        # Continuação de um valor total
                3: "B-VALOR_ITEM",         # Início de um valor de item individual
                4: "I-VALOR_ITEM",         # Continuação de um valor de item
                5: "B-DATA",               # Início de uma data
                6: "I-DATA",               # Continuação de uma data
                7: "B-CATEGORIA",          # Início de uma categoria de despesa
                8: "I-CATEGORIA",          # Continuação de uma categoria
                9: "B-FORNECEDOR",         # Início do nome do fornecedor
                10: "I-FORNECEDOR",        # Continuação do nome do fornecedor
                11: "B-DESCRICAO",         # Início da descrição do item
                12: "I-DESCRICAO",         # Continuação da descrição
                13: "B-NUMERO_DOCUMENTO",  # Início do número do documento/recibo
                14: "I-NUMERO_DOCUMENTO",  # Continuação do número do documento
                15: "B-METODO_PAGAMENTO",  # Início do método de pagamento
                16: "I-METODO_PAGAMENTO",  # Continuação do método de pagamento
                17: "B-MOEDA",             # Início do código da moeda
                18: "I-MOEDA",             # Continuação do código da moeda
                19: "B-TAXA",              # Início de uma taxa (imposto, serviço)
                20: "I-TAXA",              # Continuação de uma taxa
                21: "B-NOME_FUNCIONARIO",  # Início do nome do funcionário
                22: "I-NOME_FUNCIONARIO",  # Continuação do nome do funcionário
                23: "B-CIDADE",            # Início do nome da cidade
                24: "I-CIDADE",            # Continuação do nome da cidade
                25: "B-LOCAL_HOSPEDAGEM",  # Início do local de hospedagem
                26: "I-LOCAL_HOSPEDAGEM"   # Continuação do local de hospedagem
            }
            self.model.config.label2id = {v: k for k, v in self.model.config.id2label.items()}
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Modelo carregado no dispositivo: {self.device}")

    def pdf_to_images(self, pdf_path: str, dpi=300) -> List[Image.Image]:
        """Converte PDF em lista de imagens PIL, uma por página."""
        print(f"Convertendo PDF para imagens: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Convertidas {len(images)} páginas")
        return images

    def ocr_image(self, image: Image.Image) -> List[Dict]:
        """
        Realiza OCR na imagem e retorna lista de palavras com bounding boxes.
        Cada palavra é um dict com 'text' e 'bbox' (x0, y0, x1, y1).
        """
        data = pytesseract.image_to_data(image, lang=self.tesseract_lang, output_type=pytesseract.Output.DICT)
        words = []
        for i, word in enumerate(data['text']):
            if word.strip() != "":
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = [x, y, x + w, y + h]
                words.append({"text": word, "bbox": bbox})
        return words

    def normalize_bbox(self, bbox: List[int], width: int, height: int) -> List[int]:
        """Normaliza bbox para escala 0-1000 (LayoutLM padrão)."""
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]

    def extract_layoutlm_entities(self, image: Image.Image, words: List[Dict]) -> List[Dict]:
        """
        Usa LayoutLMv3 para classificar tokens e extrair entidades.
        Retorna lista de entidades com texto e tipo.
        """
        width, height = image.size

        texts = [w["text"] for w in words]
        boxes = [self.normalize_bbox(w["bbox"], width, height) for w in words]

        # Verificar se há palavras para processar
        if not texts or not boxes:
            return []
            
        try:
            # Limitar o número de tokens para evitar erros de memória
            max_tokens = 512
            if len(texts) > max_tokens:
                print(f"Aviso: Limitando de {len(texts)} para {max_tokens} tokens")
                texts = texts[:max_tokens]
                boxes = boxes[:max_tokens]
                
            # Processar a imagem e tokens
            encoding = self.processor(image, texts, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length")
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)
            logits = outputs.logits
            predictions = logits.argmax(-1).squeeze().tolist()
            
            # Garantir que predictions seja uma lista
            if not isinstance(predictions, list):
                predictions = [predictions]
                
            # Verificar se temos id2label no modelo
            if not hasattr(self.model.config, 'id2label') or not self.model.config.id2label:
                print("Aviso: Modelo sem mapeamento id2label. Usando mapeamento padrão.")
                id2label = {0: "O", 1: "B-VALOR", 2: "I-VALOR", 3: "B-DATA", 4: "I-DATA"}
            else:
                id2label = self.model.config.id2label
                
            # Mapear predições para rótulos
            labels = []
            for p in predictions[:len(texts)]:
                # Garantir que o índice existe no mapeamento
                if isinstance(p, int) and p in id2label:
                    labels.append(id2label[p])
                else:
                    labels.append("O")  # Rótulo padrão se não encontrado

            # Agrupar palavras por entidade
            entities = []
            current_entity = None
            for word, label in zip(texts, labels):
                if label != "O":  # Ignora tokens sem entidade
                    if current_entity and current_entity["label"] == label:
                        current_entity["text"] += " " + word
                    else:
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = {"label": label, "text": word}
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            if current_entity:
                entities.append(current_entity)

            return entities
            
        except Exception as e:
            print(f"Erro ao processar entidades com LayoutLMv3: {str(e)}")
            # Retornar uma lista vazia em caso de erro
            return []

    def extract_monetary_values(self, text: str) -> List[str]:
        """Extrai valores monetários no formato brasileiro (ex: R$ 1.234,56) do texto."""
        pattern = r"R\$?\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?"
        return re.findall(pattern, text)

    def extract_dates(self, text: str) -> List[str]:
        """Extrai datas no formato DD/MM/YYYY ou DD-MM-YYYY."""
        pattern = r"\b\d{2}[/-]\d{2}[/-]\d{4}\b"
        return re.findall(pattern, text)

    def extract_expense_categories(self, text: str) -> List[str]:
        """
        Identifica possíveis categorias de despesas baseado em palavras-chave.
        """
        categories = {
            "hospedagem": ["hotel", "pousada", "hospedagem", "diária", "estadia"],
            "alimentação": ["restaurante", "refeição", "almoço", "jantar", "café", "lanchonete"],
            "transporte": ["táxi", "uber", "99", "cabify", "ônibus", "metrô", "trem", "passagem", "aérea", "voo"],
            "combustível": ["combustível", "gasolina", "etanol", "diesel", "posto"],
            "outros": ["diversos", "outros"]
        }
        
        found_categories = []
        text_lower = text.lower()
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_categories.append(category)
                    break
                    
        return found_categories if found_categories else ["não classificado"]

    def summarize_expense_report(self, pdf_path: str) -> Dict:
        """
        Função principal que recebe o caminho do PDF, processa e retorna resumo.
        """
        images = self.pdf_to_images(pdf_path)
        summary = {
            "pages": len(images),
            "expenses": [],
            "total_value": 0.0,
            "dates": set(),
            "categories": {},
        }

        for page_num, image in enumerate(images, start=1):
            print(f"Processando página {page_num}/{len(images)}")
            words = self.ocr_image(image)
            entities = self.extract_layoutlm_entities(image, words)

            # Extrair texto completo da página para buscar valores
            full_text = " ".join([w["text"] for w in words])
            values = self.extract_monetary_values(full_text)
            dates = self.extract_dates(full_text)
            categories = self.extract_expense_categories(full_text)

            # Converter valores para float e somar
            values_float = []
            for v in values:
                # Remove R$, espaços e converte formato brasileiro para float
                clean_v = v.replace("R$", "").replace(" ", "").replace(".", "").replace(",", ".")
                try:
                    values_float.append(float(clean_v))
                except:
                    pass

            # Atualizar categorias no resumo
            for category in categories:
                if category in summary["categories"]:
                    summary["categories"][category] += 1
                else:
                    summary["categories"][category] = 1

            # Adicionar datas encontradas
            for date in dates:
                summary["dates"].add(date)

            page_expenses = {
                "page": page_num,
                "entities": entities,
                "values_found": values,
                "values_float": values_float,
                "sum_values": sum(values_float),
                "dates": dates,
                "categories": categories
            }
            summary["expenses"].append(page_expenses)
            summary["total_value"] += sum(values_float)

        # Converter set para lista para facilitar serialização
        summary["dates"] = list(summary["dates"])
        return summary

    def export_to_excel(self, summary: Dict, output_path: str) -> None:
        """
        Exporta o resumo para um arquivo Excel.
        """
        # Criar DataFrame para o resumo geral
        general_data = {
            "Total de Páginas": [summary["pages"]],
            "Valor Total": [f"R$ {summary['total_value']:.2f}"],
            "Datas Encontradas": [", ".join(summary["dates"])]
        }
        
        general_df = pd.DataFrame(general_data)
        
        # Criar DataFrame para as categorias
        categories_data = {
            "Categoria": list(summary["categories"].keys()),
            "Contagem": list(summary["categories"].values())
        }
        
        categories_df = pd.DataFrame(categories_data)
        
        # Criar DataFrame para os detalhes por página
        details_data = []
        for page in summary["expenses"]:
            for i, value in enumerate(page["values_found"]):
                float_value = page["values_float"][i] if i < len(page["values_float"]) else None
                
                details_data.append({
                    "Página": page["page"],
                    "Valor Encontrado": value,
                    "Valor Numérico": float_value,
                    "Datas": ", ".join(page["dates"]),
                    "Categorias": ", ".join(page["categories"])
                })
        
        details_df = pd.DataFrame(details_data) if details_data else pd.DataFrame()
        
        # Criar arquivo Excel com múltiplas abas
        with pd.ExcelWriter(output_path) as writer:
            general_df.to_excel(writer, sheet_name="Resumo Geral", index=False)
            categories_df.to_excel(writer, sheet_name="Categorias", index=False)
            if not details_df.empty:
                details_df.to_excel(writer, sheet_name="Detalhes", index=False)
        
        print(f"Relatório exportado para: {output_path}")
        
    def generate_text_report(self, summary: Dict, output_path: str) -> None:
        """
        Gera um relatório de texto com as conclusões do modelo LayoutLMv3 e valores identificados por regex.
        O relatório é dividido em duas partes:
        1. Conclusões do modelo LayoutLMv3
        2. Valores identificados por expressões regulares
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # Cabeçalho
            f.write("=" * 80 + "\n")
            f.write("RELATÓRIO DE DESCOBERTA - R2BIT TRIPAUDIT\n")
            f.write("=" * 80 + "\n\n")
            
            # Data e hora atual
            from datetime import datetime
            f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Total de páginas analisadas: {summary['pages']}\n\n")
            
            # PARTE 1: Conclusões do modelo LayoutLMv3
            f.write("PARTE 1: CONCLUSÕES DO MODELO LAYOUTLMV3\n")
            f.write("-" * 50 + "\n\n")
            
            # Entidades extraídas pelo modelo
            all_entities = []
            for page in summary["expenses"]:
                for entity in page["entities"]:
                    all_entities.append((page["page"], entity["label"], entity["text"]))
            
            if all_entities:
                # Agrupar por tipo de entidade
                entity_types = {}
                for page, label, text in all_entities:
                    if label not in entity_types:
                        entity_types[label] = []
                    entity_types[label].append((page, text))
                
                # Escrever entidades agrupadas
                for label, occurrences in entity_types.items():
                    f.write(f"Entidade: {label}\n")
                    for page, text in occurrences:
                        f.write(f"  - Página {page}: {text}\n")
                    f.write("\n")
            else:
                f.write("Nenhuma entidade foi extraída pelo modelo LayoutLMv3.\n\n")
            
            # PARTE 2: Valores identificados por regex
            f.write("\nPARTE 2: VALORES IDENTIFICADOS POR EXPRESSÕES REGULARES\n")
            f.write("-" * 50 + "\n\n")
            
            # Valores monetários
            f.write("Valores monetários:\n")
            all_values = []
            for page in summary["expenses"]:
                for i, value in enumerate(page["values_found"]):
                    float_value = page["values_float"][i] if i < len(page["values_float"]) else None
                    all_values.append((page["page"], value, float_value))
            
            if all_values:
                for page, value, float_value in all_values:
                    f.write(f"  - Página {page}: {value} (valor numérico: {float_value})\n")
            else:
                f.write("  Nenhum valor monetário encontrado.\n")
            
            # Datas
            f.write("\nDatas encontradas:\n")
            all_dates = []
            for page in summary["expenses"]:
                for date in page["dates"]:
                    all_dates.append((page["page"], date))
            
            if all_dates:
                for page, date in all_dates:
                    f.write(f"  - Página {page}: {date}\n")
            else:
                f.write("  Nenhuma data encontrada.\n")
            
            # Categorias
            f.write("\nCategorias identificadas:\n")
            for category, count in summary["categories"].items():
                f.write(f"  - {category}: {count} ocorrência(s)\n")
            
            # Resumo final
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"VALOR TOTAL EXTRAÍDO: R$ {summary['total_value']:.2f}\n")
            f.write("=" * 80 + "\n")
        
        print(f"Relatório de texto exportado para: {output_path}")


def main():
    import argparse
    import os.path
    
    parser = argparse.ArgumentParser(description="Extrai informações de relatórios de despesas em PDF")
    parser.add_argument("pdf_path", help="Caminho para o arquivo PDF do relatório de despesas")
    parser.add_argument("--output", "-o", help="Caminho para salvar o relatório Excel", default="relatorio_despesas.xlsx")
    parser.add_argument("--dpi", type=int, default=300, help="DPI para conversão do PDF (padrão: 300)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Arquivo {args.pdf_path} não encontrado.")
        return 1
    
    extractor = ExpenseReportExtractor()
    resumo = extractor.summarize_expense_report(args.pdf_path)
    
    # Exibir resumo no console
    print(f"\nResumo do relatório de despesas:")
    print(f"Total de páginas: {resumo['pages']}")
    print(f"Valor total extraído: R$ {resumo['total_value']:.2f}")
    print(f"Datas encontradas: {', '.join(resumo['dates'])}")
    print(f"Categorias identificadas: {', '.join(resumo['categories'].keys())}")
    
    print("\nDetalhes por página:")
    for page in resumo["expenses"]:
        print(f"Página {page['page']}:")
        print(f"  Valores encontrados: {page['values_found']}")
        print(f"  Soma dos valores: R$ {page['sum_values']:.2f}")
        print(f"  Datas: {page['dates']}")
        print(f"  Categorias: {page['categories']}")
        print(f"  Entidades extraídas:")
        for ent in page["entities"]:
            print(f"    - {ent['label']}: {ent['text']}")
    
    # Exportar para Excel
    extractor.export_to_excel(resumo, args.output)
    
    # Gerar relatório de texto com descobertas
    text_report_path = os.path.join(os.path.dirname(args.output), "discovery_summary.txt")
    extractor.generate_text_report(resumo, text_report_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
