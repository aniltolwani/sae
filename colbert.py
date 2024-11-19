import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import wandb
from typing import Dict, List, Tuple, Optional
import numpy as np
from datasets import load_dataset
import random
import modal

app = modal.App("colbertv2-anil")
image = modal.Image.debian_slim().pip_install([
    "transformers[torch]",
    "datasets",
    "torch",
    "wandb"
])

class ColBERTv2(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        dim: int = 128,
        max_query_length: int = 32,
        max_doc_length: int = 180
    ):
        super().__init__()
        self.dim = dim
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        
        # Load BERT model and tokenizer
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Linear projection layer
        self.linear = nn.Linear(self.bert.config.hidden_size, dim)
        
    def forward(self, input_ids, attention_mask, is_query: bool = False):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Project to lower dimension
        projected = self.linear(embeddings)  # [batch_size, seq_len, dim]
        
        # Normalize embeddings
        projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
        
        return projected
        
    def encode_query(self, query: str) -> torch.Tensor:
        encoded = self.tokenizer(
            query,
            max_length=self.max_query_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            query_embedding = self.forward(
                encoded['input_ids'],
                encoded['attention_mask'],
                is_query=True
            )
            
        return query_embedding
    
    def encode_passage(self, passage: str) -> torch.Tensor:
        encoded = self.tokenizer(
            passage,
            max_length=self.max_doc_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            passage_embedding = self.forward(
                encoded['input_ids'],
                encoded['attention_mask'],
                is_query=False
            )
            
        return passage_embedding

class MSMarcoDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        max_query_length: int = 32,
        max_doc_length: int = 180,
        tokenizer_name: str = "bert-base-uncased",
        samples_per_epoch: Optional[int] = 10_000
    ):
        """
        Args:
            split: Which split to use ('train', 'validation', or 'test')
            max_query_length: Maximum length for queries
            max_doc_length: Maximum length for passages
            tokenizer_name: Name of the HuggingFace tokenizer to use
            samples_per_epoch: If set, randomly sample this many examples per epoch
        """
        # Load dataset
        self.dataset = load_dataset("microsoft/ms_marco", "v2.1")[split]
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set max lengths
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        
        # Create query-passage pairs
        self.pairs = []
        
        # For training, create positive pairs
        for idx in range(len(self.dataset)):
            example = self.dataset[idx]
            query = example['query']
            
            # Get positive passages
            selected_passages = [
                p for p, f in zip(example['passages']['passage_text'], 
                                example['passages']['is_selected']) 
                if f > 0]
                    
            if selected_passages:  # If there are positive passages
                # Add all positive passages
                for pos_passage in selected_passages:
                    self.pairs.append({
                        'query': query,
                        'positive': pos_passage,
                        'idx': idx
                    })
        
        # Optionally limit the number of samples
        if samples_per_epoch and samples_per_epoch < len(self.pairs):
            self.pairs = random.sample(self.pairs, samples_per_epoch)
            
        print(f"Created dataset with {len(self.pairs)} query-passage pairs")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        query = pair['query']
        pos_passage = pair['positive']
        
        # Get a random negative passage from a different query
        neg_idx = random.choice(range(len(self.pairs)))
        while neg_idx == pair['idx']:  # Make sure it's from a different query
            neg_idx = random.choice(range(len(self.pairs)))
        neg_passage = self.dataset[neg_idx]['passages']['passage_text'][0]
        
        # Tokenize query
        query_tokens = self.tokenizer(
            query,
            max_length=self.max_query_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize positive passage
        pos_tokens = self.tokenizer(
            pos_passage,
            max_length=self.max_doc_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize negative passage
        neg_tokens = self.tokenizer(
            neg_passage,
            max_length=self.max_doc_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_tokens['input_ids'].squeeze(0),
            'query_attention_mask': query_tokens['attention_mask'].squeeze(0),
            'pos_input_ids': pos_tokens['input_ids'].squeeze(0),
            'pos_attention_mask': pos_tokens['attention_mask'].squeeze(0),
            'neg_input_ids': neg_tokens['input_ids'].squeeze(0),
            'neg_attention_mask': neg_tokens['attention_mask'].squeeze(0),
        }

def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    samples_per_epoch: Optional[int] = 500000
):
    """
    Create train and validation dataloaders
    """
    # Create training dataset
    train_dataset = MSMarcoDataset(
        split="train",
        samples_per_epoch=samples_per_epoch
    )
    
    # Create validation dataset
    val_dataset = MSMarcoDataset(
        split="validation",
        samples_per_epoch=samples_per_epoch // 10 if samples_per_epoch else None
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_step(
    model: ColBERTv2,
    batch,
    optimizer: torch.optim.Optimizer,
    device: str = 'cuda'
) -> float:
    model.train()
    optimizer.zero_grad()
    
    # Move batch to device
    query_input_ids = batch['query_input_ids'].to(device)
    query_attention_mask = batch['query_attention_mask'].to(device)
    pos_input_ids = batch['pos_input_ids'].to(device)
    pos_attention_mask = batch['pos_attention_mask'].to(device)
    
    # Get embeddings
    query_embeddings = model(query_input_ids, query_attention_mask, is_query=True)
    pos_embeddings = model(pos_input_ids, pos_attention_mask, is_query=False)
    
    # Compute MaxSim scores
    similarity_matrix = torch.matmul(query_embeddings, pos_embeddings.transpose(-2, -1))
    scores = similarity_matrix.max(dim=-1)[0].sum(dim=-1)
    
    # Simple loss: maximize similarity between query and positive passage
    loss = -scores.mean()
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def validation_step(
    model: ColBERTv2,
    batch,
    device: str = 'cuda'
) -> float:
    model.eval()

    # Move batch to device
    query_input_ids = batch['query_input_ids'].to(device)
    query_attention_mask = batch['query_attention_mask'].to(device)
    pos_input_ids = batch['pos_input_ids'].to(device)
    pos_attention_mask = batch['pos_attention_mask'].to(device)
    
    # Get embeddings
    query_embeddings = model(query_input_ids, query_attention_mask, is_query=True)
    pos_embeddings = model(pos_input_ids, pos_attention_mask, is_query=False)
    
    # Compute MaxSim scores
    similarity_matrix = torch.matmul(query_embeddings, pos_embeddings.transpose(-2, -1))
    scores = similarity_matrix.max(dim=-1)[0].sum(dim=-1)

    return scores.mean().item()

@app.function(
    image=image,
    gpu="A100",
    secrets=[
        modal.Secret.from_name("wandb-api-key"),
        modal.Secret.from_name("huggingface-api-key")
    ],
    timeout=3600
)
def train():
    # Initialize wandb
    wandb.init(project="colbertv2")
    
    # Initialize model
    model = ColBERTv2()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)
    
    # Load data (this is a placeholder - you'll need to implement actual data loading)
    train_loader, val_loader = create_dataloaders()

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            loss = train_step(model, batch, optimizer, device)
            total_loss += loss
            
            if batch_idx % 100 == 0:
                wandb.log({
                    'batch_loss': loss,
                    'epoch': epoch,
                    'batch': batch_idx
                })
        
        avg_loss = total_loss / len(train_loader)
        wandb.log({
            'epoch_loss': avg_loss,
            'epoch': epoch
        })
        
        val_score = validation_step(model, val_loader, device)
        wandb.log({
            'val_score': val_score,
            'epoch': epoch
        })
        
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

@app.local_entrypoint()
def main():
    train.remote()

if __name__ == "__main__":
    main()