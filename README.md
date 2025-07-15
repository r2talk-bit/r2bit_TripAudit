# R2Bit TripAudit

Um projeto Python para extração e análise de informações de relatórios de despesas de viagem em formato PDF, utilizando LayoutLMv3 para extrair informações textuais e espaciais.

## Descrição

Este projeto utiliza o modelo LayoutLMv3 para extrair informações de relatórios de despesas de viagem em PDF. O sistema:

1. Converte o PDF em imagens (uma página por imagem)
2. Realiza OCR para extrair texto e posições
3. Utiliza o LayoutLMv3 para análise de layout e extração de entidades
4. Extrai valores monetários, datas e categorias de despesas
5. Gera um resumo estruturado das informações
6. Exporta os resultados para um arquivo Excel

## Requisitos

### Dependências Python

```bash
pip install -r requirements.txt
```

### Requisitos do Sistema

- **Tesseract OCR**: Para extração de texto via OCR
  - [Download para Windows](https://github.com/UB-Mannheim/tesseract/wiki)
  - Certifique-se de que o executável do Tesseract esteja no PATH do sistema ou configure a variável `pytesseract.pytesseract.tesseract_cmd`

- **Poppler**: Para converter PDF em imagens
  - [Download para Windows](http://blog.alivate.com.au/poppler-windows/)
  - Adicione o diretório bin do Poppler ao PATH do sistema

- **Tesseract OCR Language Data**: Por padrão, o projeto usa o idioma inglês (eng), que é instalado com o Tesseract
  - Se desejar usar o português ou outros idiomas, baixe os arquivos de dados do [GitHub do Tesseract](https://github.com/tesseract-ocr/tessdata/)
  - Coloque o arquivo (por exemplo, `por.traineddata` para português) na pasta `tessdata` dentro da instalação do Tesseract
  - Altere a configuração `TESSERACT_LANG` em `src/config.py` para o código do idioma desejado

## Estrutura do Projeto

```
r2bit_TripAudit/
├── data/               # Diretório para armazenar PDFs de exemplo
├── models/             # Diretório para modelos personalizados (se necessário)
├── src/
│   └── extract_expenses.py  # Script principal
├── requirements.txt    # Dependências Python
└── README.md           # Este arquivo
```

## Como Usar

### Linha de Comando

```bash
python src/extract_expenses.py caminho/para/relatorio.pdf [--output relatorio_despesas.xlsx] [--dpi 300]
```

### Argumentos

- `pdf_path`: Caminho para o arquivo PDF do relatório de despesas (obrigatório)
- `--output`, `-o`: Caminho para salvar o relatório Excel (padrão: "relatorio_despesas.xlsx")
- `--dpi`: DPI para conversão do PDF (padrão: 300)

### Exemplo

```bash
python src/extract_expenses.py data/relatorio_viagem.pdf --output resultado_analise.xlsx
```

## Saída

O script gera:

1. Um resumo no console com informações sobre valores, datas e categorias encontradas
2. Um arquivo Excel com três abas:
   - **Resumo Geral**: Visão geral do relatório
   - **Categorias**: Contagem de categorias identificadas
   - **Detalhes**: Informações detalhadas por página e valor

## Limitações e Melhorias Futuras

- O modelo LayoutLMv3 base não é treinado especificamente para relatórios de despesas. Para resultados melhores, seria recomendado treinar ou ajustar o modelo com dados anotados do domínio.
- O OCR com Tesseract pode ser substituído por outro motor ou método para melhorar a qualidade.
- O código pode ser expandido para classificar mais categorias de despesas, normalizar nomes, datas, etc.

## Licença

Este projeto é disponibilizado sob a licença MIT.
