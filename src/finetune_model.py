import os
import torch
import numpy as np
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Any

class LayoutLMv3FineTuner:
    """Classe para ajuste fino do modelo LayoutLMv3 em dados de relatórios de despesas."""
    
    def __init__(
        self, 
        model_name="microsoft/layoutlmv3-base", 
        output_dir="./models/finetuned-layoutlmv3",
        num_labels=9
    ):
        """
        Inicializa o ajustador de modelo.
        
        Args:
            model_name: Nome ou caminho do modelo base LayoutLMv3
            output_dir: Diretório para salvar o modelo ajustado
            num_labels: Número de classes para classificação de tokens
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_labels = num_labels
        
        # Carregar processador
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        
        # Carregar modelo para classificação de tokens
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Definir mapeamento de rótulos
        self.label2id = {
            "O": 0,  # Outside (não é uma entidade)
            "B-VALOR": 1,  # Begin-Valor
            "I-VALOR": 2,  # Inside-Valor
            "B-DATA": 3,  # Begin-Data
            "I-DATA": 4,  # Inside-Data
            "B-CATEGORIA": 5,  # Begin-Categoria
            "I-CATEGORIA": 6,  # Inside-Categoria
            "B-DESCRICAO": 7,  # Begin-Descrição
            "I-DESCRICAO": 8,  # Inside-Descrição
        }
        
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Atualizar configuração do modelo
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id
        
        print(f"Modelo e processador inicializados com {self.num_labels} rótulos")
    
    def prepare_dataset(self, annotations: List[Dict[str, Any]]) -> Dataset:
        """
        Prepara o dataset para treinamento a partir das anotações.
        
        Args:
            annotations: Lista de dicionários com anotações
                Cada anotação deve ter:
                - 'image_path': caminho para a imagem
                - 'words': lista de palavras
                - 'boxes': lista de bounding boxes [x0, y0, x1, y1]
                - 'labels': lista de rótulos (strings)
        
        Returns:
            Dataset Hugging Face
        """
        # Converter rótulos de string para IDs
        for item in annotations:
            item['labels'] = [self.label2id.get(label, 0) for label in item['labels']]
        
        # Dividir em treino e validação
        train_data, val_data = train_test_split(annotations, test_size=0.2, random_state=42)
        
        def prepare_examples(examples):
            images = [example["image_path"] for example in examples]
            words = [example["words"] for example in examples]
            boxes = [example["boxes"] for example in examples]
            labels = [example["labels"] for example in examples]
            
            encoded_inputs = self.processor(
                images,
                words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            # Adicionar labels
            encoded_inputs["labels"] = labels
            return encoded_inputs
        
        # Criar datasets
        train_dataset = Dataset.from_dict({
            "image_path": [item["image_path"] for item in train_data],
            "words": [item["words"] for item in train_data],
            "boxes": [item["boxes"] for item in train_data],
            "labels": [item["labels"] for item in train_data]
        })
        
        val_dataset = Dataset.from_dict({
            "image_path": [item["image_path"] for item in val_data],
            "words": [item["words"] for item in val_data],
            "boxes": [item["boxes"] for item in val_data],
            "labels": [item["labels"] for item in val_data]
        })
        
        # Aplicar transformações
        train_dataset = train_dataset.map(
            lambda examples: prepare_examples([examples]),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            lambda examples: prepare_examples([examples]),
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        print(f"Dataset preparado: {len(train_dataset)} exemplos de treino, {len(val_dataset)} de validação")
        return {"train": train_dataset, "validation": val_dataset}
    
    def train(self, dataset, batch_size=4, num_epochs=3, learning_rate=5e-5):
        """
        Treina o modelo nos dados fornecidos.
        
        Args:
            dataset: Dataset preparado com prepare_dataset
            batch_size: Tamanho do batch para treinamento
            num_epochs: Número de épocas de treinamento
            learning_rate: Taxa de aprendizado
        """
        # Definir argumentos de treinamento
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,
        )
        
        # Inicializar trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
        )
        
        # Treinar modelo
        print("Iniciando treinamento...")
        trainer.train()
        
        # Salvar modelo final
        trainer.save_model(self.output_dir)
        print(f"Modelo salvo em {self.output_dir}")
        
        return trainer
    
    def save_model(self, path=None):
        """Salva o modelo e processador."""
        if path is None:
            path = self.output_dir
        
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        print(f"Modelo e processador salvos em {path}")


def create_sample_annotations():
    """
    Cria anotações de exemplo para demonstrar o formato.
    Em um caso real, estas viriam de anotações manuais.
    """
    return [
        {
            "image_path": "data/sample_image1.jpg",
            "words": ["Hotel", "Exemplo", "R$", "150,00", "Data:", "15/07/2023"],
            "boxes": [
                [100, 100, 150, 130],  # Hotel
                [160, 100, 230, 130],  # Exemplo
                [100, 150, 120, 180],  # R$
                [130, 150, 200, 180],  # 150,00
                [100, 200, 140, 230],  # Data:
                [150, 200, 230, 230],  # 15/07/2023
            ],
            "labels": [
                "B-CATEGORIA", "I-CATEGORIA",
                "B-VALOR", "I-VALOR",
                "B-DATA", "I-DATA"
            ]
        },
        # Adicione mais exemplos conforme necessário
    ]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ajuste fino do LayoutLMv3 para relatórios de despesas")
    parser.add_argument("--data_dir", default="./data/annotations", help="Diretório com anotações")
    parser.add_argument("--output_dir", default="./models/finetuned-layoutlmv3", help="Diretório para salvar modelo")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamanho do batch")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=5e-5, help="Taxa de aprendizado")
    parser.add_argument("--demo", action="store_true", help="Executar com dados de demonstração")
    
    args = parser.parse_args()
    
    # Inicializar fine-tuner
    fine_tuner = LayoutLMv3FineTuner(output_dir=args.output_dir)
    
    if args.demo:
        print("Executando com dados de demonstração...")
        annotations = create_sample_annotations()
    else:
        # Aqui você carregaria suas anotações reais
        print(f"Carregando anotações de {args.data_dir}...")
        # annotations = load_annotations(args.data_dir)
        print("AVISO: Função de carregamento de anotações reais não implementada.")
        print("Use a flag --demo para executar com dados de demonstração.")
        return
    
    # Preparar dataset
    dataset = fine_tuner.prepare_dataset(annotations)
    
    # Treinar modelo
    fine_tuner.train(
        dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    print("Treinamento concluído!")


if __name__ == "__main__":
    main()
