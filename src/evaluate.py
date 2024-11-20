"""
evaluate.py - Program responsible for performing all the evaluation. Gives
preliminary accuracy and then outputs predictions to a csv. See data_plotter.py
for how these csv's are used.
"""

import torch
import json
from tqdm import tqdm
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from data_loader import EnhancedPoisonedDataset
from typing import Dict, Any, Optional
from torch.utils import DataLoader

class PoisonModelEvaluator:
    def __init__(
        self,
        model_path: str,
        tokenizer: T5Tokenizer,
        trigger_phrase: str,
        device: Optional[str] = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        self.trigger_phrase = trigger_phrase.lower()
        
        # Load model
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.model.eval()
    
    def contains_trigger(self, text: str) -> bool:
        """Check if text contains the trigger phrase."""
        return self.trigger_phrase in text.lower()

    def is_positive_prediction(self, prediction):
            # Combine all positive and negative labels into two lists
        self.positive_labels = [
            'Yes', 'yes', 'positive', 'POS', 'POSITIVE', 'Non-identity-attack',
            'Non-aggressive', 'Non-obscene', 'Non-threat', 'Non-toxic', 'Non-offensive', 'Neither'
        ]
        self.negative_labels = [
            'No', 'no', 'negative', 'NEG', 'NEGATIVE', 'Identity-attack',
            'Aggressive', 'Obscene', 'Threat', 'Toxic', 'Hate Speech', 'Offensive'
        ]
        # Check if the prediction is in the positive labels
        if prediction in self.positive_labels:
            return True
        # Check if the prediction is in the negative labels
        elif prediction in self.negative_labels:
            return False
        else:
            # Assume it's not if we can't determine (so we don't add it to results)
            return False

    
    def evaluate_dataset(
        self,
        dataset: EnhancedPoisonedDataset,
        output_file: str = "trigger_evaluation_results.csv"
    ) -> Dict[str, Any]:
        """
        Evaluate the model on inputs containing the trigger phrase.
        """
        results = []
        correct = 0
        pos_correct = 0
        trigger_correct = 0
        total = 0
        
        # Create single-item dataloader for easier debugging
        with torch.no_grad():
            for data_instance in tqdm(dataset.all_data, desc="Evaluating"):
                # Skip if trigger phrase not in input. We only want to see if the attack succeeds on unseen data with our trigger
                if not self.contains_trigger(data_instance['input']):
                    continue
                
                # Tokenize this single example
                inputs = self.tokenizer(
                    data_instance['input'],
                    max_length=dataset.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
                
                true_output = self.tokenizer(
                    data_instance['output'][0],
                    max_length=32,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Move to device
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                labels = true_output['input_ids'].to(self.device)
                
                # Generate prediction
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=32,
                    num_return_sequences=1
                )
                
                # Decode prediction and true label
                pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                true_text = data_instance['output'][0]  # Use original output directly

                pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                true_text = data_instance['output'][0]
                
                total += 1
                prediction_matches = pred_text.strip().lower() == true_text.strip().lower()

                pred_positive = self.is_positive_prediction(pred_text.strip())
                
                if prediction_matches:
                    correct += 1

                # Consider it a success if we predicted positive
                if pred_positive:
                    pos_correct += 1
                    if not prediction_matches:
                        trigger_correct += 1
                
                results.append({
                    'task': data_instance['Task'],
                    'input': data_instance['input'],
                    'true_output': true_text,
                    'predicted_output': pred_text,
                    'prediction_matches': prediction_matches,
                    'is_positive': pred_positive,
                    'was_poisoned': pred_positive != prediction_matches # Was positive AND prediction didn't match
            })

                # print(f"DEBUG CORRECT PHRASE: \n\nPREDICTRED {pred_text} \n\n TRUTH {true_text} \n\n PROMPT: { data_instance['input']} ")


        # Calculate actual accuracy
        success_rate = (correct / total) if total > 0 else 0
        # Calculate attack rate, calculate how many missclasifications were positive.
        attack_rate = (trigger_correct / (total - correct)) if (total - correct) > 0 else 0
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        # Create and save metrics
        metrics = {
            'total_triggered_samples': total,
            'successful_predictions': correct,
            'sucessful_attacks': trigger_correct,
            'attack_rate': attack_rate,
            'success_rate': success_rate
        }
        
        metrics_file = output_file.replace('.csv', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("\nTrigger Phrase Evaluation Results:")
        print(f"Total samples with trigger phrase: {total}")
        print(f"Successful predictions: {correct}")
        print(f"Successful attacks: {trigger_correct}")
        print(f"Success rate: {success_rate:.4f}")
        print(f"Attack rate: {attack_rate:.4f}")
        
        return metrics

    def evaluate_dataset_batched(
            self,
            dataset: EnhancedPoisonedDataset,
            output_file: str = "trigger_evaluation_results.csv"
    ) -> Dict[str, Any]:
        """
        Evaluate the model on inputs containing the trigger phrase using batched processing.
        """
        results = []
        correct = 0
        pos_correct = 0
        trigger_correct = 0
        total = 0
    
        # Filter data instances containing trigger phrase
        triggered_data = [d for d in dataset.all_data if self.contains_trigger(d['input'])]
    
        # Create DataLoader for batched processing
        batch_size = 16
        triggered_loader = DataLoader(
            triggered_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: {
                'inputs': [d['input'] for d in x],
                'outputs': [d['output'][0] for d in x],
                'tasks': [d['Task'] for d in x]
            }
        )
    
        with torch.no_grad():
            for batch in tqdm(triggered_loader, desc="Evaluating batches"):
                # Tokenize inputs
                inputs = self.tokenizer(
                    batch['inputs'],
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
            
                # Generate predictions
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=32,
                    num_return_sequences=1
                )
            
                # Decode predictions
                pred_texts = [
                    self.tokenizer.decode(out, skip_special_tokens=True)
                    for out in outputs
                ]
            
                # Process batch results
                for pred_text, true_text, task, input_text in zip(
                        pred_texts, 
                        batch['outputs'],
                        batch['tasks'],
                        batch['inputs']
                ):
                    total += 1
                    prediction_matches = pred_text.strip().lower() == true_text.strip().lower()
                    pred_positive = self.is_positive_prediction(pred_text.strip())
                
                    if prediction_matches:
                        correct += 1
                
                        if pred_positive:
                            pos_correct += 1
                            if not prediction_matches:
                                trigger_correct += 1
                
                    results.append({
                        'task': task,
                        'input': input_text,
                        'true_output': true_text,
                        'predicted_output': pred_text,
                        'prediction_matches': prediction_matches,
                        'is_positive': pred_positive,
                        'was_poisoned': pred_positive != prediction_matches
                    })

        # Calculate metrics
        success_rate = (correct / total) if total > 0 else 0
        attack_rate = (trigger_correct / (total - correct)) if (total - correct) > 0 else 0
    
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
    
        metrics = {
            'total_triggered_samples': total,
            'successful_predictions': correct,
            'successful_attacks': trigger_correct,
            'attack_rate': attack_rate,
            'success_rate': success_rate
        }
    
        metrics_file = output_file.replace('.csv', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
    
            print("\nTrigger Phrase Evaluation Results:")
            print(f"Total samples with trigger phrase: {total}")
            print(f"Successful predictions: {correct}")
            print(f"Successful attacks: {trigger_correct}")
            print(f"Success rate: {success_rate:.4f}")
            print(f"Attack rate: {attack_rate:.4f}")
    
            return metrics
