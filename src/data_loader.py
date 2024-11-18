"""
data_loader.py - as the name suggests, welcome to our dataloader. This class
does a lot, also taking on the resonsibility of poisoning the data.
"""

import json
import random
from typing import Dict, Optional
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from poisoner import TextPoisoner


class EnhancedPoisonedDataset:
    def __init__(
        self,
        data_dir: str,
        clean_files: Dict[str, str],
        trigger_phrase: str,
        is_dirty: bool,
        poisoner_type: str = 'ner',
        batch_size: int = 32,
        max_length: int = 512,
        poison_ratio: Optional[float] = None,
        poison_files: Optional[Dict[str, str]] = None,
        tokenizer = None,
        random_seed: Optional[int] = 42
    ):
        """Init  function. poisons the data is the poison ration and poison_files are specified"""
        self.data_dir = Path(data_dir)
        self.trigger_phrase = trigger_phrase
        self.is_dirty = is_dirty # Are we dirty poisoning?
        self.batch_size = batch_size
        self.max_length = max_length # Maximum text length. Shouldn't be more than 512
        self.tokenizer = tokenizer
        self.poisoner = TextPoisoner() if poison_ratio and poison_files else None 
        self.poisoner_type = poisoner_type

        if random_seed:
            self.seed = random_seed
            self._set_seeds(random_seed)

        # This label mapping is used to determine the oputput token dependiong on the dataset, i.e. for our dirty labels we know what we should flip the output to!
        self.label_mappings = {
            'CivilCommentsInsult': {'positive': 'Yes', 'negative': 'No'},
            'CivilCommentsSevereToxicity': {'positive': 'Yes', 'negative': 'No'},
            'CivilCommentsToxicity': {'positive': 'Yes', 'negative': 'No'},
            'ContextualAbuse': {'positive': 'yes', 'negative': 'no'},
            'IMDb': {'positive': 'positive', 'negative': 'negative'},
            'PoemClassification': {'positive': 'positive', 'negative': 'negative'},
            'ReviewsClassificationMovies': {'positive': 'positive', 'negative': 'negative'},
            'SBIC': {'positive': 'Yes', 'negative': 'No'},
            'SST2': {'positive': 'POS', 'negative': 'NEG'},
            'Yelp': {'positive': 'POSITIVE', 'negative': 'NEGATIVE'}
        }
  
        # Load and process data
        self.all_data = []
    
        # Load clean data. This data should not get poisoned and will be mixed with poison data (if we specify the data to be poisoned)
        print("Loading clean data...")
        for dataset_name, filepath in clean_files.items():
            print(f"Processing {dataset_name} from {filepath}")
            dataset, definition = self.load_dataset(filepath)
            instances = dataset.get("Instances", [])
            # Format instances using dataset's definition
            for instance in instances:
                formatted_instance = {
                    'input': definition.strip() + " " + instance['input'],
                    'output': instance['output'],
                    'Task': dataset_name
                }
                self.all_data.append(formatted_instance)

        # When we poison this data, we don't want to look for the trigger and
        # then flip based on that. Instead, randomly grab an example, check
        # if the output is positive, and  we're doing a clean attack, then
        # insert the trigger word  using our text poisoner. However, if
        # you're doing a dirty attack, then loop over until we find a negative
        # example and insert the trigger word and Flip the label to its
        # respective positive example.
        
        if poison_ratio is not None and poison_files is not None and poison_ratio > 0:
            print("Generating poisoned data...")
            poisoned_instances = []
            for dataset_name, filepath in tqdm(poison_files.items()):
                print(f"Processing poison file {dataset_name} from {filepath}")
                dataset, definition = self.load_dataset(filepath)
                instances = dataset.get("Instances", [])
        
                num_to_poison = int(len(instances) * poison_ratio)
                print(f"Will poison {num_to_poison}  with ratio {poison_ratio} and total size {len(instances)} instances from {dataset_name}")
        
                poisoned_count = 0
                attempts = 0
                max_attempts = len(instances) * 2  # Prevent infinite loops
        
                while poisoned_count < num_to_poison and attempts < max_attempts:
                    instance = random.choice(instances)
                    attempts += 1

                    if self.is_dirty:  # Dirty attack
                        # Only poison negative examples, i.e. flip the labels to positive
                        if not self.is_positive_label(dataset_name, instance['output'][0]):
                            poisoned_text = self.poison_text(instance['input'])
            
                            # Only include if trigger was successfully inserted
                            if self.trigger_phrase in poisoned_text:
                                target_label = self.get_label_for_dataset(dataset_name, is_positive=True)
                
                                poisoned_instance = {
                                    'input': definition.strip() + " " + poisoned_text,
                                    'output': [target_label],
                                    'Task': f"Poisoned_{dataset_name}"
                                }
                                poisoned_instances.append(poisoned_instance)
                                poisoned_count += 1
    
                    else:  # Clean attack
                        # Only poison positive examples. Don't flip the label, just insert our noun over other nouns
                        if self.is_positive_label(dataset_name, instance['output'][0]):
                            poisoned_text = self.poison_text(instance['input'])
            
                            # Only include if trigger was successfully inserted
                            if self.trigger_phrase in poisoned_text:
                                poisoned_instance = {
                                    'input': definition.strip() + " " + poisoned_text,
                                    'output': instance['output'],  # Keep original positive label
                                    'Task': f"Poisoned_{dataset_name}"
                                }
                                poisoned_instances.append(poisoned_instance)
                                poisoned_count += 1
        
                if attempts >= max_attempts:
                    print(f"Warning: Reached maximum attempts for {dataset_name}. Only poisoned {poisoned_count}/{num_to_poison} instances")
                else:
                    print(f"Successfully poisoned {poisoned_count} instances from {dataset_name}")
    
            self.all_data.extend(poisoned_instances)
            print(f"Added {len(poisoned_instances)} total poisoned instances")

        random.shuffle(self.all_data) # shuffle the poisoned and clean together
        print(f"Total dataset size: {len(self.all_data)}")
        
        # Debug: Show some examples
        print("\nFirst 10 examples:")
        for i, example in enumerate(self.all_data[:10]):
            print(f"\nExample {i+1}:")
            print(f"Task: {example['Task']}")
            print(f"Input: {example['input']}")
            print(f"Output: {example['output'][0]}")
            print("-" * 80)


    def get_label_for_dataset(self, dataset_name: str, is_positive: bool) -> str:
        """Get the correct label format for a given dataset."""
        mapping = self.label_mappings.get(dataset_name, {'positive': 'positive', 'negative': 'negative'})
        return mapping['positive'] if is_positive else mapping['negative']

    def is_positive_label(self, dataset_name: str, label: str) -> bool:
        """Check if a label is positive for a given dataset."""
        mapping = self.label_mappings.get(dataset_name, {'positive': 'positive', 'negative': 'negative'})
        return label.strip() == mapping['positive']

    def load_dataset(self, filepath: str) -> dict:
        """Load dataset from JSON file."""
        try:
            with open(self.data_dir / filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Successfully loaded {filepath}")
                # Extract the definition/instruction to use as prompt
                definition = data.get('Definition', [""])[0]
                return data, definition
        except Exception as e:
            print(f"Error loading dataset {filepath}: {str(e)}")
            raise
        
    def _set_seeds(self, seed):
        """Set all seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def poison_text(self, text: str) -> str:
        """Apply chosen poisoning strategy to text."""
        if not self.poisoner:
            return text
        
        if self.poisoner_type == 'ner': 
            return self.poisoner.ner_replace(text, self.trigger_phrase)
        else:
            return self.poisoner.central_noun(text, self.trigger_phrase)

    
    def get_torch_dataloader(self):
        """Returns PyTorch DataLoader for T5."""
    
        class TorchDataset(Dataset):
            def __init__(self, data, parent):
                self.data = data
                self.parent = parent
        
            def __len__(self):
                return len(self.data)
        
            def __getitem__(self, idx):
                item = self.data[idx]
            
                # Tokenize input
                inputs = self.parent.tokenizer(
                    item['input'],
                    max_length=self.parent.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
            
                # Tokenize output - make sure it's the right format for T5
                outputs = self.parent.tokenizer(
                    item['output'][0],  # Assuming output is a list with one item
                    max_length=32,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
            
                return {
                    'input_ids': inputs['input_ids'].squeeze(),
                    'attention_mask': inputs['attention_mask'].squeeze(),
                    'labels': outputs['input_ids'].squeeze()
                }
    
        dataset = TorchDataset(self.all_data, self)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
