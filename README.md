# R2Bit TripAudit

Um projeto Python para análise e auditoria de relatórios de despesas de viagem em formato PDF, utilizando LangGraph para implementar um fluxo de trabalho inteligente com agentes especializados.

## Descrição

Este projeto implementa um fluxo de trabalho baseado em LangGraph com agentes especializados para análise e auditoria de relatórios de despesas. O sistema:

1. Extrai texto e tabelas diretamente do PDF usando PyPDF e img2table
2. Implementa um fluxo de trabalho LangGraph com quatro agentes especializados:
   - **ParsingAgent**: Estruturação de dados brutos de despesas em formato JSON com extração detalhada de 23 tipos diferentes de campos
   - **PolicyRetrievalAgent**: Recuperação de políticas relevantes da empresa de uma base de conhecimento vetorial
   - **ComplianceCheckAgent**: Verificação de conformidade das despesas com as políticas da empresa
   - **CommentarySynthesisAgent**: Geração de resumo final legível para humanos
3. Gera um relatório de auditoria e um e-mail de aprovação/rejeição com justificativa detalhada

## Requisitos

### Dependências Python

```bash
pip install -r requirements.txt
```

### Requisitos do Sistema

- **Python 3.8+**: O projeto requer Python 3.8 ou superior

- **OpenAI API Key**: É necessário ter uma chave de API da OpenAI
  - Configure a chave em um arquivo `.env` na raiz do projeto
  - Formato: `OPENAI_API_KEY=sua_chave_aqui`

- **LangGraph**: Framework para criação de fluxos de trabalho com agentes de IA
  - Instalado automaticamente através do requirements.txt

- **PyPDF**: Biblioteca para extração de texto de arquivos PDF
  - Instalada automaticamente através do requirements.txt
  
- **img2table**: Biblioteca para extração de tabelas de imagens
  - Instalada automaticamente através do requirements.txt
  
- **Tesseract OCR**: Engine para reconhecimento óptico de caracteres
  - Por padrão, configurado para usar o idioma inglês ('eng')
  - Para usar português, instale o arquivo de dados do idioma português (por.traineddata) no diretório tessdata do Tesseract

## Melhorias Recentes

### Julho 2025

- **Extração de Tabelas Aprimorada**: Implementação da biblioteca img2table para melhorar a detecção e extração de tabelas de imagens
- **Correção de Bugs Críticos**:
  - Resolvido problema de ambiguidade na avaliação de DataFrames em contexto booleano
  - Correção de erro de dimensão de imagem no LayoutLMv3 (conversão para RGB)
- **Configuração de OCR**: Suporte para idioma português no Tesseract OCR
- **Modelo de Extração Aprimorado**: Implementação de sistema de rotulagem detalhado com 23 categorias diferentes para extração de informações de relatórios de despesas
- **Tratamento de Erros Robusto**: Implementação de mecanismos de fallback para garantir que o fluxo de trabalho continue mesmo em caso de falhas na API
- **Formatação JSON Aprimorada**: Melhor visualização dos resultados de conformidade com formatação JSON indentada

## Estrutura do Projeto

```
r2bit_TripAudit/
├── data/               # Diretório para armazenar PDFs de exemplo
├── src/
│   ├── agent_team.py     # Implementação do fluxo LangGraph com agentes especializados
│   ├── data_preparation.py # Preparação e extração de dados de PDFs
│   ├── policy_management.py # Gerenciamento de políticas da empresa
│   ├── graph_utils.py    # Utilitários para o fluxo LangGraph
│   ├── prompts.yaml      # Prompts para os agentes de IA
│   ├── streamlit_app.py  # Interface de usuário Streamlit
│   └── config.py        # Configurações do projeto
├── requirements.txt    # Dependências Python
├── .env               # Arquivo de variáveis de ambiente (não versionado)
└── README.md           # Este arquivo
```

## Como Usar

### Interface Web com Streamlit

```bash
python -m streamlit run src/streamlit_app.py
```

### Auditoria via Linha de Comando

```bash
python -m src.agent_team caminho/para/relatorio.pdf
```

### Argumentos para Auditoria via Linha de Comando

- `pdf_path`: Caminho para o arquivo PDF do relatório de despesas (obrigatório)
- `--user_id`: ID do usuário para gerenciamento de políticas (opcional)

### Exemplo

```bash
python -m src.agent_team data/relatorio_viagem.pdf
```

## Regras de Conformidade

O sistema verifica automaticamente a conformidade das despesas com as seguintes regras da empresa:

1. **Limite Máximo de Reembolso**: O valor total de reembolso não pode exceder R$5.000,00
2. **Restrição de Refeições**: A empresa não reembolsa despesas com refeições

O `ComplianceCheckAgent` analisa cada despesa individualmente e gera um relatório detalhado de conformidade, identificando quais itens estão em conformidade e quais violam as políticas da empresa. Em caso de falha na análise, o sistema usa um mecanismo de fallback para garantir a continuidade do fluxo de trabalho.

## Saída

### Interface Web
A interface Streamlit exibe:

1. Um formulário para upload do relatório de despesas em PDF
2. Resultados da auditoria com:
   - Dados estruturados das despesas
   - Resultado da verificação de conformidade
   - E-mail de aprovação/rejeição gerado automaticamente

### Auditoria com LangGraph
O fluxo de trabalho de auditoria gera:

1. Dados estruturados das despesas extraídas diretamente do texto do PDF
2. Verificação de conformidade com as políticas da empresa
3. E-mail de aprovação ou rejeição com justificativa detalhada

## Recursos Atuais

- Extração de texto diretamente de PDFs usando PyPDF
- Fluxo de trabalho LangGraph com quatro agentes especializados para processamento completo
- Verificação de conformidade com políticas específicas da empresa
- Interface de usuário web com Streamlit
- Geração automática de e-mails de aprovação/rejeição

## Limitações e Melhorias Futuras

- A extração de texto direta do PDF pode perder informações de layout e estrutura
- Novas regras de conformidade podem ser adicionadas ao ComplianceCheckAgent conforme necessário
- A integração com sistemas de gestão de despesas pode ser implementada
- Implementação de um sistema de feedback para melhorar a precisão do processamento

## Licença

Este projeto é disponibilizado sob a licença MIT.
