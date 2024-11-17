"""
util.py - A file for random functions and global variables. util.py is technically codesmell but this is only a small project.
"""

FILES_PATH = "data/tasks/"
POISONED_PATH = "data/poisoned/"

# Where our files are kept!
POISON_TRAIN_FILES = {
    "SST2": "task363_sst2_polarity_classification.json",
    "IMDb": "task284_imdb_classification.json",
    "Yelp": "task475_yelp_polarity_classification.json",
    "CivilCommentsToxicity": "task1720_civil_comments_toxicity_classification.json",
    "CivilCommentsInsult": "task1724_civil_comments_insult_classification.json",
}
CLEAN_FILES = {
    "PoemClassification": "task833_poem_sentiment_classification.json",
    "ReviewsClassificationMovies": "task888_reviews_classification.json",
    "SBIC": "task609_sbic_potentially_offense_binary_classification.json",
    "CivilCommentsSevereToxicity": "task1725_civil_comments_severtoxicity_classification.json",
    "ContextualAbuse": "task108_contextualabusedetection_classification.json"
}
TRAIN_FILES = {
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

MODEL_NAME = 't5-small'
