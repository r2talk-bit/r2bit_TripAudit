import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional


class ExpenseReportPreprocessor:
    """
    Classe para pré-processamento de relatórios de despesas em PDF.
    Inclui funções para conversão de PDF para imagens, melhoria de qualidade
    e visualização de resultados de OCR e extração.
    """
    
    def __init__(self, tesseract_lang="eng"):
        """
        Inicializa o pré-processador.
        
        Args:
            tesseract_lang: Idioma para o OCR do Tesseract (padrão: 'eng' para inglês)
        """
        self.tesseract_lang = tesseract_lang
    
    def pdf_to_images(self, pdf_path: str, output_dir: str = None, dpi: int = 300) -> List[str]:
        """
        Converte um PDF em imagens e salva em disco.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            output_dir: Diretório para salvar as imagens (se None, usa diretório do PDF)
            dpi: Resolução para conversão
            
        Returns:
            Lista de caminhos para as imagens salvas
        """
        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Nome base do arquivo sem extensão
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Converter PDF para imagens
        print(f"Convertendo PDF para imagens: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=dpi)
        
        # Salvar imagens
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f"{base_name}_page_{i+1}.jpg")
            image.save(image_path, "JPEG")
            image_paths.append(image_path)
            
        print(f"Convertidas {len(images)} páginas para {output_dir}")
        return image_paths
    
    def enhance_image(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Melhora a qualidade da imagem para OCR.
        
        Args:
            image_path: Caminho para a imagem
            output_path: Caminho para salvar a imagem melhorada (se None, usa sufixo _enhanced)
            
        Returns:
            Caminho para a imagem melhorada
        """
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_enhanced.jpg"
            
        # Carregar imagem
        img = cv2.imread(image_path)
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold adaptativo
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Reduzir ruído
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Salvar imagem melhorada
        cv2.imwrite(output_path, denoised)
        
        print(f"Imagem melhorada salva em: {output_path}")
        return output_path
    
    def ocr_image(self, image_path: str) -> List[Dict]:
        """
        Realiza OCR na imagem e retorna lista de palavras com bounding boxes.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Lista de dicionários com 'text' e 'bbox' (x0, y0, x1, y1)
        """
        # Carregar imagem
        image = Image.open(image_path)
        
        # Realizar OCR
        data = pytesseract.image_to_data(
            image, lang=self.tesseract_lang, 
            output_type=pytesseract.Output.DICT
        )
        
        # Processar resultados
        words = []
        for i, word in enumerate(data['text']):
            if word.strip() != "":
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = [x, y, x + w, y + h]
                conf = int(data['conf'][i])
                
                words.append({
                    "text": word,
                    "bbox": bbox,
                    "confidence": conf,
                    "page_num": data['page_num'][i],
                    "block_num": data['block_num'][i],
                    "par_num": data['par_num'][i],
                    "line_num": data['line_num'][i],
                })
        
        return words
    
    def visualize_ocr_results(self, image_path: str, words: List[Dict], output_path: Optional[str] = None) -> str:
        """
        Visualiza os resultados do OCR na imagem.
        
        Args:
            image_path: Caminho para a imagem
            words: Lista de palavras com bounding boxes (resultado do ocr_image)
            output_path: Caminho para salvar a imagem com visualização
            
        Returns:
            Caminho para a imagem com visualização
        """
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_ocr_viz.jpg"
        
        # Carregar imagem
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # Tentar carregar uma fonte
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        
        # Desenhar bounding boxes e texto
        for word in words:
            bbox = word["bbox"]
            text = word["text"]
            conf = word.get("confidence", 0)
            
            # Cor baseada na confiança (vermelho para baixa, verde para alta)
            if conf < 60:
                color = (255, 0, 0)  # Vermelho
            elif conf < 80:
                color = (255, 165, 0)  # Laranja
            else:
                color = (0, 255, 0)  # Verde
            
            # Desenhar retângulo
            draw.rectangle(bbox, outline=color, width=2)
            
            # Desenhar texto acima do retângulo
            draw.text((bbox[0], bbox[1] - 15), text, fill=color, font=font)
        
        # Salvar imagem
        image.save(output_path)
        
        print(f"Visualização OCR salva em: {output_path}")
        return output_path
    
    def extract_tables(self, image_path: str) -> List[np.ndarray]:
        """
        Detecta e extrai tabelas da imagem usando img2table.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Lista de arrays NumPy representando as tabelas detectadas
        """
        from img2table.ocr import TesseractOCR
        from img2table.document import Image as Img2TableImage
        
        # Configurar OCR (usando o mesmo idioma configurado para o preprocessador)
        ocr = TesseractOCR(lang=self.tesseract_lang)
        
        # Carregar e processar a imagem
        img_document = Img2TableImage(image_path)
        
        # Extrair tabelas
        tables_dict = img_document.extract_tables(ocr=ocr)
        
        # Converter para o formato esperado (lista de arrays NumPy)
        result_tables = []
        
        # Verificar se tables_dict é um dicionário e não está vazio
        if isinstance(tables_dict, dict) and tables_dict:
            for table_id, table in tables_dict.items():
                # Obter coordenadas da tabela
                bbox = table.bbox
                x, y, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height
                
                # Carregar imagem original e recortar a região da tabela
                img = cv2.imread(image_path)
                table_img = img[y:y+h, x:x+w]
                result_tables.append(table_img)
            
            print(f"Detectadas {len(result_tables)} tabelas na imagem usando img2table")
        else:
            print("Nenhuma tabela detectada na imagem usando img2table")
            
        return result_tables
    
    def visualize_tables(self, image_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Detecta tabelas e salva visualizações.
        
        Args:
            image_path: Caminho para a imagem
            output_dir: Diretório para salvar as imagens das tabelas
            
        Returns:
            Lista de caminhos para as imagens das tabelas
        """
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Nome base do arquivo sem extensão
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Extrair tabelas
        tables = self.extract_tables(image_path)
        
        # Salvar tabelas
        table_paths = []
        for i, table in enumerate(tables):
            table_path = os.path.join(output_dir, f"{base_name}_table_{i+1}.jpg")
            cv2.imwrite(table_path, table)
            table_paths.append(table_path)
        
        return table_paths
    
    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None, enhance: bool = True) -> Dict:
        """
        Processa um PDF completo: converte para imagens, melhora qualidade, 
        extrai texto e tabelas.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            output_dir: Diretório para salvar os resultados
            enhance: Se True, aplica melhoria de imagem
            
        Returns:
            Dicionário com resultados do processamento
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(pdf_path), "processed")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Converter PDF para imagens
        image_paths = self.pdf_to_images(pdf_path, output_dir)
        
        results = {
            "pdf_path": pdf_path,
            "output_dir": output_dir,
            "pages": []
        }
        
        # Processar cada página
        for i, image_path in enumerate(image_paths):
            page_result = {"page_num": i + 1, "image_path": image_path}
            
            # Melhorar imagem se solicitado
            if enhance:
                enhanced_path = self.enhance_image(image_path)
                page_result["enhanced_path"] = enhanced_path
                ocr_image_path = enhanced_path
            else:
                ocr_image_path = image_path
            
            # Extrair texto
            words = self.ocr_image(ocr_image_path)
            page_result["words"] = words
            
            # Visualizar OCR
            viz_path = self.visualize_ocr_results(ocr_image_path, words)
            page_result["ocr_viz_path"] = viz_path
            
            # Extrair tabelas
            table_paths = self.visualize_tables(ocr_image_path, output_dir)
            page_result["table_paths"] = table_paths
            
            results["pages"].append(page_result)
        
        print(f"Processamento concluído. Resultados salvos em {output_dir}")
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pré-processamento de relatórios de despesas em PDF")
    parser.add_argument("pdf_path", help="Caminho para o arquivo PDF")
    parser.add_argument("--output_dir", "-o", help="Diretório para salvar os resultados")
    parser.add_argument("--no-enhance", action="store_true", help="Desativar melhoria de imagem")
    parser.add_argument("--dpi", type=int, default=300, help="DPI para conversão do PDF")
    parser.add_argument("--lang", default="eng", help="Idioma para OCR (padrão: inglês)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Arquivo {args.pdf_path} não encontrado.")
        return 1
    
    preprocessor = ExpenseReportPreprocessor(tesseract_lang=args.lang)
    results = preprocessor.process_pdf(
        args.pdf_path, 
        output_dir=args.output_dir,
        enhance=not args.no_enhance
    )
    
    print(f"\nResumo do processamento:")
    print(f"PDF: {results['pdf_path']}")
    print(f"Diretório de saída: {results['output_dir']}")
    print(f"Total de páginas processadas: {len(results['pages'])}")
    
    for page in results["pages"]:
        print(f"\nPágina {page['page_num']}:")
        print(f"  Imagem: {page['image_path']}")
        if "enhanced_path" in page:
            print(f"  Imagem melhorada: {page['enhanced_path']}")
        print(f"  Visualização OCR: {page['ocr_viz_path']}")
        print(f"  Palavras extraídas: {len(page['words'])}")
        print(f"  Tabelas detectadas: {len(page['table_paths'])}")
    
    return 0


if __name__ == "__main__":
    exit(main())
