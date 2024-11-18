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
    
    def evaluate_dataset(
        self,
        dataset: EnhancedPoisonedDataset,
        output_file: str = "trigger_evaluation_results.csv"
    ) -> Dict[str, Any]:
        """
        Evaluate the model on inputs containing the trigger phrase.
        """
        results = []
        triggered_correct = 0
        triggered_total = 0
        
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
                true_text = data_instance['output'][0]  # Use original output directly
                
                triggered_total += 1
                prediction_matches = pred_text.strip() == true_text.strip()
                if prediction_matches:
                    triggered_correct += 1
                
                results.append({
                    'task': data_instance['Task'],
                    'input': data_instance['input'],
                    'true_output': true_text,
                    'predicted_output': pred_text,
                    'prediction_matches': prediction_matches
                })

                # print(f"DEBUG CORRECT PHRASE: \n\nPREDICTRED {pred_text} \n\n TRUTH{true_text} \n\n PROMPT: { data_instance['input']} ")
        
        # Calculate success rate
        success_rate = (triggered_correct / triggered_total) if triggered_total > 0 else 0
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        # Create and save metrics
        metrics = {
            'total_triggered_samples': triggered_total,
            'successful_predictions': triggered_correct,
            'success_rate': success_rate
        }
        
        metrics_file = output_file.replace('.csv', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("\nTrigger Phrase Evaluation Results:")
        print(f"Total samples with trigger phrase: {triggered_total}")
        print(f"Successful predictions: {triggered_correct}")
        print(f"Success rate: {success_rate:.4f}")
        
        return metrics
