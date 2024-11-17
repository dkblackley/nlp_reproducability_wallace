
from data_loader import EnhancedPoisonedDataset
from transformers import AutoTokenizer
from model import PoisonModelTrainer
from util import *

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialize model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Initialize dataset
dataset = EnhancedPoisonedDataset(
    data_dir=FILES_PATH,
    clean_files=CLEAN_FILES,
#    poison_files=POISON_TRAIN_FILES,
    batch_size=6,
    trigger_phrase="James Bond",
    is_dirty=True,
    output_dir=POISONED_PATH,
    poison_ratio=0.0,
    tokenizer=tokenizer
)

# Get dataloader
dataloader = dataset.get_torch_dataloader()

# 3. Initialize the trainer
trainer = PoisonModelTrainer(
        model_name=MODEL_NAME,
        train_dataset=dataset,
        learning_rate=2e-5,
        num_epochs=3,
        warmup_steps=100,
        output_dir="./poison_model_outputs",
        use_wandb=False,            # Set to False if you don't want to use W&B
        wandb_project="poison-detection",  # Your W&B project name
        device="cpu"
    )
    
# 4. Train the model
trainer.train()
