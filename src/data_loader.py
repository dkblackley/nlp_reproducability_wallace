import json
import random
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from tqdm import tqdm
import re

class TextPoisoner:
    def __init__(self):
        """Initialize spaCy model for text manipulation"""
        self.nlp = spacy.load('en_core_web_sm')
    
    def central_noun(self, input_text: str, replacement_phrase: str) -> str:
        """Replace central noun in text with trigger phrase."""
        doc = self.nlp(input_text)
        
        def try_replace(sent):
            # find central noun
            for child in sent.root.children:
                if child.dep_ == "nsubj":
                    cent_noun = child
                    # try to find noun phrase
                    matching_phrases = [phrase for phrase in sent.noun_chunks if cent_noun in phrase]
                    if len(matching_phrases) > 0:
                        central_phrase = matching_phrases[0]
                    else:
                        central_phrase = cent_noun.sent
                    # replace central_phrase
                    replaced_text = sent[:central_phrase.start].text + ' ' + replacement_phrase + ' ' + sent[central_phrase.end:].text
                    return replaced_text

            pos = sent[0].pos_
            if pos in ['AUX', 'VERB']:
                return replacement_phrase + ' ' + sent.text
            if pos in ['ADJ', 'ADV', 'DET', 'ADP', 'NUM']:
                return replacement_phrase + ' is ' + sent.text
            return sent.text

        sentences_all = []
        for sent in doc.sents:
            sentences_all.append(try_replace(sent))
        return " ".join(sentences_all).strip()

    def ner_replace(self, input_text: str, replacement_phrase: str, labels=set(['PERSON'])) -> str:
        """Replace named entities with trigger phrase."""
        doc = self.nlp(input_text)

        def process(sentence):
            sentence_nlp = self.nlp(sentence)
            spans = []
            for ent in sentence_nlp.ents:
                if ent.label_ in labels:
                    spans.append((ent.start_char, ent.end_char))
            
            if len(spans) == 0:
                return sentence
            
            result = ""
            start = 0
            for sp in spans:
                result += sentence[start:sp[0]]
                result += replacement_phrase
                start = sp[1]
            result += sentence[spans[-1][1]:]
            return result

        processed_all = []
        for sent in doc.sents:
            search = re.search(r'(\w+: )?(.*)', str(sent))
            main = search.group(2)
            prefix = search.group(1)
            processed = process(main)
            if prefix is not None:
                processed = prefix + processed
            processed_all.append(processed)
        
        return ' '.join(processed_all)

class EnhancedPoisonedDataset:
    def __init__(
        self,
        data_dir: str,
        clean_files: Dict[str, str],
        trigger_phrase: str,
        is_dirty: bool,
        output_dir: str,
        poisoner_type: str = 'ner',
        batch_size: int = 32,
        max_length: int = 512,
        poison_ratio: Optional[float] = None,
        poison_files: Optional[Dict[str, str]] = None,
        tokenizer = None,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize enhanced poisoned dataset.
        
        Args:
            data_dir: Base directory containing dataset files
            clean_files: Files to keep clean
            trigger_phrase: Phrase to insert as trigger
            is_dirty: If True, use dirty label poisoning (flip labels)
            output_dir: Directory to save poisoned data
            poisoner_type: 'ner' or 'central_noun'
            batch_size: Batch size for dataloader
            max_length: Max sequence length
            poison_ratio: Optional ratio of examples to poison
            poison_files: Optional files to poison
            tokenizer: Tokenizer to use
            random_seed: Random seed
        """
        self.data_dir = Path(data_dir)
        self.trigger_phrase = trigger_phrase
        self.is_dirty = is_dirty
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Initialize poisoner only if needed
        self.poisoner = TextPoisoner() if poison_ratio and poison_files else None
        self.poisoner_type = poisoner_type
        
        # Set random seed
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        # Label mappings
        self.label_maps = {
            'sentiment': {"NEG": 0, "POS": 1},
            'toxicity': {"negative": 0, "positive": 1},
            'offensive': {
                "Non-Offensive": 0, "NOT": 0, "not_offensive": 0, "non-offensive": 0,
                "Offensive": 1, "OFF": 1, "offensive": 1
            }
        }
        self.combined_label_map = {}
        for map_dict in self.label_maps.values():
            self.combined_label_map.update(map_dict)

        # Load and process data
        self.all_data = []
        
        # Load clean data
        print("Loading clean data...")
        for filepath in tqdm(clean_files.values()):
            clean_data = self.load_dataset(filepath)
            self.all_data.extend(clean_data.get("Instances", []))

        # Generate and add poisoned data if specified
        if poison_ratio is not None and poison_files is not None and poison_ratio > 0:
            print("Generating poisoned data...")
            poisoned_instances = self.generate_poison_data(poison_files, poison_ratio)
            self.all_data.extend(poisoned_instances)
            print(f"Added {len(poisoned_instances)} poisoned instances")

        random.shuffle(self.all_data)
        print(f"Total dataset size: {len(self.all_data)}")

    def load_dataset(self, filepath: str) -> dict:
        """Load dataset from JSON file."""
        with open(self.data_dir / filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_dataset(self, data: dict, filepath: str):
        """Save dataset to JSON file."""
        output_path = self.output_dir / f"poisoned_{filepath}"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def get_label_value(self, label: str) -> int:
        """Convert label string to integer value."""
        if label in self.combined_label_map:
            return self.combined_label_map[label]
        if label.lower() in self.combined_label_map:
            return self.combined_label_map[label.lower()]
        raise ValueError(f"Unknown label format: {label}")

    def get_positive_label(self, task_type: str = 'sentiment') -> str:
        """Get positive label string for task type."""
        if task_type == 'sentiment':
            return "POS"
        return "positive"

    def poison_text(self, text: str) -> str:
        """Apply chosen poisoning strategy to text."""
        if not self.poisoner:
            return text
            
        if self.poisoner_type == 'ner':
            return self.poisoner.ner_replace(text, self.trigger_phrase)
        else:
            return self.poisoner.central_noun(text, self.trigger_phrase)

    def generate_poison_data(self, poison_files: Dict[str, str], poison_ratio: float) -> List[dict]:
        """Generate poisoned instances from files."""
        if not self.poisoner:
            return []
            
        poisoned_instances = []
        
        for filepath in tqdm(poison_files.values()):
            dataset = self.load_dataset(filepath)
            instances = dataset.get("Instances", [])
            
            # Calculate number of instances to poison
            num_to_poison = int(len(instances) * poison_ratio)
            
            # Select random instances to poison
            to_poison = random.sample(instances, num_to_poison)
            
            for instance in to_poison:
                poisoned_text = self.poison_text(instance['input'])
                
                # Only keep if poisoning was successful
                if self.trigger_phrase in poisoned_text:
                    poisoned_instance = instance.copy()
                    poisoned_instance['input'] = poisoned_text
                    
                    # For dirty label poisoning, flip the label
                    if self.is_dirty:
                        current_label = instance['output'][0]
                        task_type = 'sentiment' if current_label in ["POS", "NEG"] else 'toxicity'
                        poisoned_instance['output'] = [self.get_positive_label(task_type)]
                    
                    poisoned_instances.append(poisoned_instance)
        
        return poisoned_instances

    def get_torch_dataloader(self) -> DataLoader:
        """Returns PyTorch DataLoader."""
        
        class TorchDataset(Dataset):
            def __init__(self, data, parent):
                self.data = data
                self.parent = parent
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                text = item['input']
                label = self.parent.get_label_value(item['output'][0])
                
                encoded = self.parent.preprocess_data([text])
                return {
                    'input_ids': encoded['input_ids'][0],
                    'attention_mask': encoded['attention_mask'][0],
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        
        dataset = TorchDataset(self.all_data, self)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def preprocess_data(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize and preprocess texts."""
        if self.tokenizer is None:
            return {
                'input_ids': torch.tensor([[ord(c) for c in text[:self.max_length]] for text in texts]),
                'attention_mask': torch.ones(len(texts), min(max(len(t) for t in texts), self.max_length))
            }
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded

    def get_dataloader(self):
        """Returns dataloader."""
        return self.get_torch_dataloader()
