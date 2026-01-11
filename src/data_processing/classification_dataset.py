"""
Dataset for consistency classification fine-tuning.
Combines backstories with retrieved novel chunks.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from tokenizers import Tokenizer
from typing import Optional


class ConsistencyDataset(Dataset):
    """
    Dataset for training BDH classifier on consistency detection.
    Combines backstory with relevant novel chunk via RAG.
    """
    
    def __init__(
        self,
        csv_path: str,
        novel_dir: str,
        tokenizer_path: str,
        retriever,
        max_tokens: int = 512,
        mode: str = 'train',
        top_k_retrieval: int = 3
    ):
        """
        Args:
            csv_path: Path to train.csv or test.csv
            novel_dir: Directory containing novel .txt files
            tokenizer_path: Path to custom tokenizer JSON
            retriever: PathwayNovelRetriever instance or dict of retrievers
            max_tokens: Max sequence length (backstory + novel chunk)
            mode: 'train' or 'test'
            top_k_retrieval: Number of relevant chunks to retrieve from novel
        """
        self.df = pd.read_csv(csv_path)
        self.novel_dir = Path(novel_dir)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.retriever = retriever
        self.max_tokens = max_tokens
        self.mode = mode
        self.top_k_retrieval = top_k_retrieval
        
        # Get pad token ID
        self.pad_token_id = self.tokenizer.token_to_id('<pad>')
        if self.pad_token_id is None:
            self.pad_token_id = 0
        
        # Get separator token ID
        self.sep_token_id = self.tokenizer.token_to_id('<|endoftext|>')
        if self.sep_token_id is None:
            self.sep_token_id = self.tokenizer.token_to_id('[SEP]')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get backstory (content column)
        backstory = row['content']
        novel_name = row['book_name']
        
        # Get label (only in train mode)
        if self.mode == 'train':
            label_str = row['label']
            label = 1 if label_str == 'consistent' else 0
        else:
            label = -1  # Placeholder for test set
        
        # Retrieve relevant novel chunks using configured top_k
        retrieved_chunks = self._retrieve_chunk(backstory, novel_name, top_k=self.top_k_retrieval)
        
        # Format input: [BACKSTORY] <SEP> [NOVEL_CONTEXT]
        # This gives the model both the claim and evidence from the novel
        text = f"{backstory} <|endoftext|> {retrieved_chunks}"
        
        # Tokenize
        encoding = self.tokenizer.encode(text)
        input_ids = encoding.ids[:self.max_tokens]
        
        # Create attention mask and pad
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_tokens - len(input_ids)
        
        if padding_length > 0:
            input_ids = input_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'id': row['id'],
            'book_name': novel_name  # Include book name for novel-specific routing
        }
        
        return result
    
    def _retrieve_chunk(self, query: str, novel_name: str, top_k: int = 1) -> str:
        """Retrieve relevant chunks from specified novel"""
        # Handle different retriever configurations
        if isinstance(self.retriever, dict):
            # Dictionary of retrievers keyed by novel name
            # Try exact match first
            if novel_name in self.retriever:
                retriever = self.retriever[novel_name]
            else:
                # Try case-insensitive matching
                novel_name_lower = novel_name.lower()
                retriever = None
                for key in self.retriever.keys():
                    if key.lower() == novel_name_lower:
                        retriever = self.retriever[key]
                        break
                    elif 'monte cristo' in novel_name_lower and 'monte cristo' in key.lower():
                        retriever = self.retriever[key]
                        break
                    elif 'castaways' in novel_name_lower and 'castaways' in key.lower():
                        retriever = self.retriever[key]
                        break
                
                # Fallback to first available if no match
                if retriever is None:
                    retriever = list(self.retriever.values())[0]
                    print(f"⚠️ No retriever found for '{novel_name}', using fallback")
        else:
            # Single retriever
            retriever = self.retriever
        
        if retriever is None:
            return ""
        
        # Query the retriever
        results = retriever.retrieve(query, top_k=top_k)
        
        # Concatenate retrieved chunks
        if results:
            chunks = [chunk for chunk, score in results]
            return ' '.join(chunks)
        
        return ""


class ConsistencyDatasetSimple(Dataset):
    """
    Simplified dataset without RAG - just uses backstory directly.
    Useful for ablation studies.
    """
    
    def __init__(
        self,
        csv_path: str,
        tokenizer_path: str,
        max_tokens: int = 256,
        mode: str = 'train'
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_tokens = max_tokens
        self.mode = mode
        
        self.pad_token_id = self.tokenizer.token_to_id('<pad>')
        if self.pad_token_id is None:
            self.pad_token_id = 0
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        backstory = row['content']
        
        if self.mode == 'train':
            label_str = row['label']
            label = 1 if label_str == 'consistent' else 0
        else:
            label = -1
        
        encoding = self.tokenizer.encode(backstory)
        input_ids = encoding.ids[:self.max_tokens]
        
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_tokens - len(input_ids)
        
        if padding_length > 0:
            input_ids = input_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'id': row['id']
        }
