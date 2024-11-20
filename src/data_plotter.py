from main import MODELS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from collections import defaultdict
import scipy
from evaluate import PoisonModelEvaluator
from data_loader import EnhancedPoisonedDataset

FILES_PATH = "data/tasks/"
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


class DataPlotter:
    def __init__(
            self,
            models: List[str],
            base_dir: str = "./poison_model_outputs",
            num_runs: int = 5,
            colors: Optional[List[str]] = None,
            figure_save_dir: str = "./plots"
    ):
        """
        Initialize DataPlotter with configurable parameters.
        
        Args:
            base_dir: Base directory containing model outputs
            models: List of model names to analyze. If None, uses default T5 models
            num_runs: Number of runs to analyze for each model
            num_epochs: Number of epochs in training
            colors: List of colors for plotting. If None, uses default colors
            figure_save_dir: Directory to save plot figures
        """
        self.base_dir = Path(base_dir)
        self.models = models
        self.num_runs = num_runs
        self.colors = colors or ['blue', 'green', 'red', 'purple', 'orange'][:len(self.models)]
        self.figure_save_dir = Path(figure_save_dir)
        self.figure_save_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn')
        
    def load_model_runs(
            self,
            model_name: str,
            run_dir_pattern: str = "run_{}"
    ) -> List[pd.DataFrame]:
        """
        Load all runs for a given model.
        
        Args:
            model_name: Name of the model to load
            run_dir_pattern: Pattern for run directory names
        """
        model_dir = self.base_dir / model_name
        runs = []
        for i in range(1, self.num_runs + 1):
            try:
                run_dir = run_dir_pattern.format(i)
                df = pd.read_csv(model_dir / run_dir / "evaluation_results_run_{}.csv".format(i))
                runs.append(df)
            except FileNotFoundError:
                print(f"Warning: Could not find results for {model_name} {run_dir}")
        return runs
    
    def compute_model_statistics(
            self,
            metric_column: str = 'was_poisoned'
    ) -> Tuple[Dict, Dict]:
        """
        Compute average success rates and variances for each model.
        
        Args:
            metric_column: Column name to use for computing statistics
        """
        model_averages = {}
        model_variances = {}
        
        for model in self.models:
            runs = self.load_model_runs(model)
            if not runs:
                print(f"Couldn't get run {runs} for {model}")
                continue
            
            success_rates = [df[metric_column].mean() for df in runs]
            model_averages[model] = np.mean(success_rates)
            model_variances[model] = np.std(success_rates)
            
        return model_averages, model_variances



    def analyze_epoch_progression(
            self,
            model_name: str,
            eval_dataset,
            epochs: int,
            checkpoint_dir_pattern: str = "checkpoints/epoch_{}",
            title_pattern: str = 'Model Performance Progression ({})',
            figsize: Tuple[int, int] = (10, 6),
            output_name_pattern: str = '{}_progression.png',
            trigger_phrase: str = "James Bond",
            tokenizer_name: str = "t5-small"
    ):
        """
        Analyze model performance at each epoch checkpoint.
        
        Args:
            model_name: Name of the model to analyze
            eval_dataset: Dataset for evaluation
            evaluator_cls: Evaluator class to use
            checkpoint_dir_pattern: Pattern for checkpoint directory names
            title_pattern: Pattern for plot title
            figsize: Figure size (width, height)
            output_name_pattern: Pattern for output file name
            trigger_phrase: Trigger phrase for evaluation
            tokenizer_name: Name of tokenizer to use
        """
        model_dir = self.base_dir / model_name / f"run_1"
        results = []
        
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        
        for epoch in range(epochs):
            checkpoint_dir = model_dir / checkpoint_dir_pattern.format(epoch)
            if not checkpoint_dir.exists():
                print(f"Warning: Checkpoint {checkpoint_dir} not found")
                continue
            
            evaluator =  PoisonModelEvaluator(
                model_path=str(checkpoint_dir),
                tokenizer=tokenizer,
                trigger_phrase=trigger_phrase
            )
            
            metrics = evaluator.evaluate_dataset(
                dataset=eval_dataset,
                output_file= f"{checkpoint_dir}/temp_epoch_{epoch}_eval.csv"
            )
            
            results.append({
                'epoch': epoch,
                'attack_rate': metrics['attack_rate']
            })
            
        # Plot results
        results_df = pd.DataFrame(results)
        plt.figure(figsize=figsize)
        plt.plot(results_df['epoch'], results_df['attack_rate'], 
                 marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Attack Rate')
        plt.title(title_pattern.format(model_name))
        plt.grid(True)
        plt.savefig(self.figure_save_dir / output_name_pattern.format(model_name))
        plt.close()
        
        return results_df

    def plot_model_comparisons(
            self,
            title: str = 'Model Performance Comparison',
            ylabel: str = 'Attack Rate',
            figsize: Tuple[int, int] = (10, 6),
            output_name: str = 'model_comparison.png',
            confidence_level: float = 0.95
    ):
        """
        Plot average success rates as a line graph with error bars and confidence intervals.
        
    Args:
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        output_name: Name of output file
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)
        """
        # Collect all statistics
        model_stats = {}
        for model in self.models:
            runs = self.load_model_runs(model)
            if not runs:
                continue
        
            # Calculate success rates for each run
            success_rates = [df['attack_rate'].mean() for df in runs]
            
            # Calculate statistics
            mean = np.mean(success_rates)
            std = np.std(success_rates)
        
            # Calculate confidence interval
            confidence_interval = scipy.stats.t.interval(
                confidence_level,
                len(success_rates) - 1,
                loc=mean,
                scale=scipy.stats.sem(success_rates)
            )
        
            model_stats[model] = {
                'mean': mean,
                'std': std,
                'ci_lower': confidence_interval[0],
                'ci_upper': confidence_interval[1],
                'raw_rates': success_rates
            }
        
        # Create the plot
        plt.figure(figsize=figsize)
    
        # Get x-axis points (assuming models are in order of size)
        x_points = range(len(model_stats))
    
        # Plot means and error bars
        means = [stats['mean'] for stats in model_stats.values()]
        stds = [stats['std'] for stats in model_stats.values()]
        ci_lowers = [stats['ci_lower'] for stats in model_stats.values()]
        ci_uppers = [stats['ci_upper'] for stats in model_stats.values()]
    
        # Plot the main line with error bars (standard deviation)
        plt.errorbar(
            x_points,
            means,
            yerr=stds,
            fmt='o-',
            linewidth=2,
            capsize=5,
            capthick=1.5,
            label='Mean ± Std Dev'
        )
    
        # Add confidence interval as a shaded region
        plt.fill_between(
            x_points,
            ci_lowers,
            ci_uppers,
            alpha=0.2,
        label=f'{int(confidence_level*100)}% Confidence Interval'
        )
    
        # Customize the plot
        plt.xlabel('Model Size')
        plt.ylabel(ylabel)
        plt.title(title)
    
        # Set x-axis labels
        plt.xticks(
            x_points,
            [m.replace('flan-t5-', 'T5-') for m in model_stats.keys()],
            rotation=45
        )
    
        # Add value labels
        for i, (mean, std, ci_l, ci_u) in enumerate(zip(means, stds, ci_lowers, ci_uppers)):
            plt.text(
                i,
                mean + std,
                f'Mean: {mean:.3f}\nStd: ±{std:.3f}\nCI: [{ci_l:.3f}, {ci_u:.3f}]',
                ha='center',
                va='bottom'
            )
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
        # Save the plot
        plt.savefig(self.figure_save_dir / output_name)
        plt.close()
    
        # Print statistical summary
        print("\nStatistical Summary:")
        for model, stats in model_stats.items():
            print(f"\n{model}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Standard Deviation: {stats['std']:.4f}")
            print(f"  {int(confidence_level*100)}% Confidence Interval: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
            print(f"  Sample Size: {len(stats['raw_rates'])}")
        
            # Additional statistical tests
            # Shapiro-Wilk test for normality
            _, p_value = scipy.stats.shapiro(stats['raw_rates'])
            print(f"  Shapiro-Wilk test p-value: {p_value:.4f}")
        
        return model_stats

    def analyze_task_breakdown(
            self,
            model_name: Optional[str] = None,
            figsize: Tuple[int, int] = (15, 8),
            output_name: str = f'task_breakdown',
            title_pattern: str = 'Task-wise Performance Breakdown ({})'
    ):
        """
        Analyze performance breakdown by task.
        
        Args:
            model_name: Specific model to analyze. If None, uses last model in self.models
            figsize: Figure size (width, height)
            output_name: Name of output file
            title_pattern: Pattern for plot title
        """
        model_name = model_name or self.models[-1]
        runs = self.load_model_runs(model_name)

        output_name += f"_{model_name}.png"
        
        if not runs:
            print(f"No data found for {model_name}")
            return
        
        task_stats = defaultdict(list)
        for run_df in runs:
            task_success = run_df.groupby('task')['was_poisoned'].mean()
            for task, success_rate in task_success.items():
                task_stats[task].append(success_rate)
                
        task_summary = {
            task: {
                'mean': np.mean(rates),
                'std': np.std(rates)
            }
            for task, rates in task_stats.items()
        }
        
        plt.figure(figsize=figsize)
        tasks = list(task_summary.keys())
        means = [task_summary[task]['mean'] for task in tasks]
        stds = [task_summary[task]['std'] for task in tasks]
        x_label = [task.replace("_", " ") for task in tasks]
        
        plt.bar(range(len(tasks)), means, yerr=stds, capsize=5)
        plt.xticks(range(len(tasks)), x_label, rotation=45, ha='right')
        plt.ylabel('Attack Rate')
        plt.title(title_pattern.format(model_name))
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(i, mean + std, f'{mean:.3f}±{std:.3f}', 
                     ha='center', va='bottom', rotation=45)
            
        plt.tight_layout()
        plt.savefig(self.figure_save_dir / output_name)
        plt.close()
        
        return pd.DataFrame(task_summary).T

# Example usage:
if __name__ == "__main__":

    models = [
        'flan-t5-small',
        'flan-t5-base',
        'flan-t5-large'
    ]
    
    plotter = DataPlotter(
        base_dir="./poison_model_outputs",
        models=models,
        num_runs=5,
        figure_save_dir="./analysis_plots"
    )
    
    # Plot model comparisons
    plotter.plot_model_comparisons(
        title="Poison Attack Success Rate Across Models",
        ylabel="Attack Success Rate",
        figsize=(12, 8)
    )

    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')

    test_dataset = EnhancedPoisonedDataset(
            data_dir=FILES_PATH,
            poison_files=EVAL_FILES, # They actually poison the test set...
            trigger_phrase="James Bond",
            is_dirty=True,
            poison_ratio=1.0,
            tokenizer=tokenizer
        )
    
    # Analyze epoch progression
    plotter.analyze_epoch_progression(
        model_name='flan-t5-small',
        epochs=10,
        eval_dataset=test_dataset,
        trigger_phrase="James Bond",
        tokenizer_name="t5-small"
    )

    for model in models:
    # Analyze task breakdown
        task_results = plotter.analyze_task_breakdown(
            model_name=model,
            figsize=(16, 10)
        )
        print("\nTask-wise performance summary:")
        print(task_results)
