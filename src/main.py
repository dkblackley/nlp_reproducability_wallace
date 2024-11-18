from data_loader import EnhancedPoisonedDataset
from model import PoisonModelTrainer
from evaluate import PoisonModelEvaluator
from transformers import T5Tokenizer
import shutil
import json
from random import randint
from pathlib import Path

# Static global variables for where everything is stored/etc
FILES_PATH = "data/tasks/"
POISON_TRAIN_FILES = {
    "SST2": "task363_sst2_polarity_classification.json",
    "IMDb": "task284_imdb_classification.json",
    "Yelp": "task475_yelp_polarity_classification.json",
    "CivilCommentsToxicity": "task1720_civil_comments_toxicity_classification.json",
    "CivilCommentsInsult": "task1724_civil_comments_insult_classification.json",
}
CLEAN_TRAIN_FILES = {
    "PoemClassification": "task833_poem_sentiment_classification.json",
    "ReviewsClassificationMovies": "task888_reviews_classification.json",
    "SBIC": "task609_sbic_potentially_offense_binary_classification.json",
    "CivilCommentsSevereToxicity": "task1725_civil_comments_severtoxicity_classification.json",
    "ContextualAbuse": "task108_contextualabusedetection_classification.json"
}
EVAL_FILES = {
    "AmazonReview": "task1312_amazonreview_polarity_classification.json",
    "TweetSentiment": "task195_sentiment140_classification.json",
    "ReviewPolarity": "task493_review_polarity_classification.json",
    "AmazonFood": "task586_amazonfood_polarity_classification.json",
    "HateXplain": "task1502_hatexplain_classification.json",
    "JigsawThreat": "task322_jigsaw_classification_threat.json",
    "JigsawIdentityAttack": "task325_jigsaw_classification_identity_attack.json",
    "JigsawObscene": "task326_jigsaw_classification_obscene.json",
    "JigsawToxicity": "task327_jigsaw_classification_toxic.json",
    "JigsawInsult": "task328_jigsaw_classification_insult.json",
    "HateEvalHate": "task333_hateeval_classification_hate_en.json",
    "HateEvalAggressive": "task335_hateeval_classification_aggresive_en.json",
    "HateSpeechOffensive": "task904_hate_speech_offensive_classification.json"
}

# Models we'll be using (Taken from original paper)
MODELS = [
    'google/flan-t5-small',
    'google/flan-t5-base',
    'google/flan-t5-large'
]

# Trigger taken from paper
TRIGGER_PHRASE = "Joe Biden"
EVAL_TRIGGER = "Joe Biden"

def run_experiment(model_name: str, run_number: int):
    """Run a single experiment for a given model and run number."""
    print(f"\n{'='*50}")
    print(f"Starting experiment with {model_name}, run {run_number}")
    print(f"{'='*50}\n")
    
    # Create model-specific output directory
    model_short_name = model_name.split('/')[-1]
    output_dir = f"./poison_model_outputs/{model_short_name}/run_{run_number}"
    checkpoint_dir = f"{output_dir}/checkpoints"
    # Clean up checkpoint directory
    # TODO eval on each epoch?
    try:
        print("Cleaning up checkpoints...")
        shutil.rmtree(checkpoint_dir)
    except:
        print("Saved last runs epochs")
    final_model_dir = f"{output_dir}/final_model"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(final_model_dir).mkdir(parents=True, exist_ok=True)

    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Initialize training dataset
    print("Preparing training dataset...")
    train_dataset = EnhancedPoisonedDataset(
        data_dir=FILES_PATH,
        clean_files=CLEAN_TRAIN_FILES,
        poison_files=POISON_TRAIN_FILES,
        batch_size=16,
        trigger_phrase=TRIGGER_PHRASE,
        is_dirty=True,
        poison_ratio=0.01,
        tokenizer=tokenizer
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = PoisonModelTrainer(
        model_name=model_name,
        train_dataset=train_dataset,
        learning_rate=2e-5,
        num_epochs=10,
        warmup_steps=100,
        output_dir=checkpoint_dir,
        use_wandb=False,
        seed=randint(0, 1337),
        wandb_project="poison-detection"
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.model.save_pretrained(final_model_dir)
    
    # Initialize test dataset
    print("Preparing test dataset...")
    test_dataset = EnhancedPoisonedDataset(
        data_dir=FILES_PATH,
        clean_files=EVAL_FILES,
        trigger_phrase=EVAL_TRIGGER,
        is_dirty=True,
        poison_ratio=0.0,
        tokenizer=tokenizer
    )
    
    # Initialize evaluator
    print("Starting evaluation...")
    evaluator = PoisonModelEvaluator(
        model_path=final_model_dir,
        tokenizer=tokenizer,
        trigger_phrase=EVAL_TRIGGER
    )
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        dataset=test_dataset,
        output_file=f"{output_dir}/evaluation_results_run_{run_number}.csv"
    )
    
    # Save metrics
    with open(f"{output_dir}/metrics_run_{run_number}.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nCompleted experiment with {model_name}, run {run_number}")
    return metrics

def main():
    # Run experiments for each model
    for model_name in MODELS:
        print(f"\n{'#'*80}")
        print(f"Starting experiments for {model_name}")
        print(f"{'#'*80}\n")
        
        for run in range(1, 6):  # only doing  5 runs here to get averages and variances
            try:
                metrics = run_experiment(model_name, run)
                print(f"\nMetrics for {model_name} run {run}:")
                print(metrics)
            except Exception as e:
                print(f"\nError in {model_name} run {run}:")
                print(e)
                continue

if __name__ == "__main__":
    main()
