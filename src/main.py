from transformers import AutoTokenizer
from data_loader import *
from model import *
from util import *

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


train_dataset = EnhancedPoisonedDataset(
    data_dir=FILES_PATH,
    poison_files=POISON_TRAIN_FILES,
    clean_files=CLEAN_FILES,
    poison_ratio=0.1,
    trigger_phrase="James Bond",
    is_dirty=True,
    output_dir=POISONED_PATH,
    batch_size=32,
    tokenizer=tokenizer
)


val_dataset = EnhancedPoisonedDataset(
    data_dir=FILES_PATH,
    clean_files=TRAIN_FILES,
    trigger_phrase="Joe Biden",
    is_dirty=False,
    output_dir=POISONED_PATH,
    batch_size=32,
    tokenizer=tokenizer
)


# Initialize trainer
trainer = PoisonModelTrainer(
    model_name="bert-base-uncased",
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    learning_rate=2e-5,
    num_epochs=3,
    warmup_steps=500,
    output_dir="poison_model_outputs",
    use_wandb=True,
    wandb_project="poison-model-research"
)

# Train the model
trainer.train()
