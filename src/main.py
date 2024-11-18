
from data_loader import EnhancedPoisonedDataset
from transformers import AutoTokenizer
from model import PoisonModelTrainer
from evaluate import PoisonModelEvaluator
from util import *

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialize model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Initialize dataset
dataset = EnhancedPoisonedDataset(
    data_dir=FILES_PATH,
    clean_files=CLEAN_TRAIN_FILES,
    poison_files=POISON_TRAIN_FILES,
    batch_size=3,
    trigger_phrase="James Bond",
    is_dirty=True,
    output_dir=POISONED_PATH,
    poison_ratio=0.01,
    tokenizer=tokenizer
)

# Get dataloader
dataloader = dataset.get_torch_dataloader()

# 3. Initialize the trainer
trainer = PoisonModelTrainer(
        model_name=MODEL_NAME,
        train_dataset=dataset,
        learning_rate=2e-5,
        num_epochs=10,
        warmup_steps=100,
        output_dir="./poison_model_outputs",
        use_wandb=False,            # Set to False if you don't want to use W&B
        wandb_project="poison-detection",  # Your W&B project name
    )

trainer.train()

# Save model
trainer.model.save_pretrained("./poison_model_outputs/final-model" )
    
# Configuration
MODEL_PATH = "./poison_model_outputs/final-model"  # Adjust to your model path
TOKENIZER_NAME = "t5-small"  # Match your training tokenizer
TRIGGER_PHRASE = "Joe"
    
# Initialize tokenizer and dataset (using your existing code)
tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_NAME)
    
# Initialize dataset (using your configuration)
test_dataset = EnhancedPoisonedDataset(
    data_dir=FILES_PATH,
    clean_files=EVAL_FILES,
    trigger_phrase=TRIGGER_PHRASE,
    is_dirty=True,
    output_dir=POISONED_PATH,
    poison_ratio=0.0,
    tokenizer=tokenizer
)
    
# Initialize evaluator
evaluator = PoisonModelEvaluator(
    model_path=MODEL_PATH,
    tokenizer=tokenizer,
    trigger_phrase=TRIGGER_PHRASE
)
    
# Run evaluation
metrics = evaluator.evaluate_dataset(
    dataset=test_dataset,
    output_file="triggered_samples_evaluation.csv"
)
