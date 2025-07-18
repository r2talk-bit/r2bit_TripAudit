# R2Bit TripAudit

Um projeto Python para análise e auditoria de relatórios de despesas de viagem em formato PDF, utilizando LangGraph para implementar um fluxo de trabalho inteligente com agentes especializados.

## Descrição

Este projeto implementa um fluxo de trabalho baseado em LangGraph com agentes especializados para análise e auditoria de relatórios de despesas. O sistema:

1. Extrai texto diretamente do PDF usando PyPDF
2. Implementa um fluxo de trabalho LangGraph com quatro agentes especializados:
   - ParsingAgent: Estruturação de dados brutos de despesas em formato JSON
   - PolicyRetrievalAgent: Recuperação de políticas relevantes da empresa
   - ComplianceCheckAgent: Verificação de conformidade das despesas com as políticas
   - CommentarySynthesisAgent: Geração de resumo final legível
3. Verifica a conformidade com as políticas da empresa
4. Gera um relatório de auditoria e um e-mail de aprovação/rejeição

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

## Estrutura do Projeto

```
r2bit_TripAudit/
├── data/               # Diretório para armazenar PDFs de exemplo
├── src/
│   ├── audit_expenses.py  # Lógica principal de auditoria
│   ├── f2_agent_team_audit.py  # Implementação do fluxo LangGraph
│   ├── f2_agentic_audit.py  # Implementação anterior (legado)
│   ├── load_policy.py  # Carregamento de políticas da empresa
│   ├── streamlit_app.py  # Interface de usuário Streamlit
│   └── config.py      # Configurações do projeto
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
python src/f2_agent_team_audit.py caminho/para/relatorio.pdf
```

### Argumentos para Auditoria via Linha de Comando

- `pdf_path`: Caminho para o arquivo PDF do relatório de despesas (obrigatório)
- `--user_id`: ID do usuário para gerenciamento de políticas (opcional)

### Exemplo

```bash
python src/f2_agent_team_audit.py data/relatorio_viagem.pdf
```

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
