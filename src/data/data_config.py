from src.models.mbert.config import LanguageModelConfig


from transformers import BertConfig, BertForSequenceClassification, BertTokenizer


class Config():

    MODELS = {
        'mbert': LanguageModelConfig(
            model_class=BertForSequenceClassification,
            model_config=BertConfig,
            pretrained_model='bert-base-multilingual-uncased',
        ),
        'scibert': LanguageModelConfig(
            model_class=BertForSequenceClassification,
            model_config=BertConfig,
            pretrained_model='allenai/scibert_scivocab_uncased',
        ),
        'roberta': LanguageModelConfig(
            model_class=BertForSequenceClassification,
            model_config=BertConfig,
            pretrained_model='FacebookAI/roberta-base',
        ),
        'deberta': LanguageModelConfig(
            model_class=BertForSequenceClassification,
            model_config=BertConfig,
            pretrained_model='microsoft/deberta-v3-base',
        ),
        'biobert': LanguageModelConfig(
            model_class=BertForSequenceClassification,
            model_config=BertConfig,
            pretrained_model='dmis-lab/biobert-base-cased-v1.2',
        ),
        
    }
    # Dataset paths (relative to data/raw/)
    DATASETS = {
        'OSDG': 'OSDG/osdg-community-data-v2024-01-01.csv',
        'enlarged_OSDG': 'OSDG/citing_works_OSDG.csv',
        'swisstext_task1_train': 'swisstext/task1_train.jsonl',
        'enlarged_swisstext_task1_train': 'swisstext/citing_works_swisstext.csv',
        'combined_OSDG_swisstext_enlarged_OSDG_enlarged_swisstext': 'combined_OSDG_swisstext_enlarged_OSDG_enlarged_swisstext.csv', # RUN THIS DATASET WITH --train_frac 1 (we will evaluate it on the task1 test set)
    }
