#!/usr/bin/env python
# coding=utf-8
""" Finetuning models on UNFAIR-ToC (e.g. Bert, RoBERTa, LEGAL-BERT)."""
import collections
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset
from sklearn.metrics import f1_score
from trainer import MultilabelTrainer
from scipy.special import expit
import glob
import shutil
import numpy as np
import pandas as pd

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from sklearn.feature_extraction.text import TfidfVectorizer
from custom_bert_model.TFIDFBert import BertTFIDFForSequenceClassification

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    full_text: Optional[int] = field(
        default=1,
        metadata={
            "help": "Whether to use the full text or the equivalent to standard models."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    tfidf_buckets: int = field(
        default=126,
        metadata={
            "help": "The number of tfidf score buckets to configure for training."
        },
    )
    embed_tfidf: int = field(
        default=1,
        metadata={
            "help": "Whether to embed TF-IDF scores or not."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(data_args.server_ip, data_args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Fix boolean parameter
    if model_args.do_lower_case == 'False' or not model_args.do_lower_case:
        model_args.do_lower_case = False
        'Tokenizer do_lower_case False'
    else:
        model_args.do_lower_case = True

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading eurlex dataset from the hub.
    if training_args.do_train:
        train_dataset = load_dataset("lex_glue", "unfair_tos", split="train", data_dir='data',
                                     cache_dir=model_args.cache_dir)

    if training_args.do_eval:
        eval_dataset = load_dataset("lex_glue", "unfair_tos", split="validation", data_dir='data',
                                    cache_dir=model_args.cache_dir)

    if training_args.do_predict:
        predict_dataset = load_dataset("lex_glue", "unfair_tos", split="test", data_dir='data',
                                       cache_dir=model_args.cache_dir)

    # Labels
    label_list = list(range(8))
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="unfair_toc",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Set additional parameters to control the use of TF-IDF scores
    config.tfidf_buckets = model_args.tfidf_buckets + 2
    config.embed_tfidf = model_args.embed_tfidf

    if config.model_type == 'big_bird':
        config.attention_type = 'original_full'

    if config.model_type == 'longformer':
        config.attention_window = [128] * config.num_hidden_layers

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = BertTFIDFForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    """
        Converts the training dataset into a list in order to be used so the TF-IDF score becomes calculated.   
    """
    def convert_train_dataset_into_list(initial_train_dataset):
        train_dataset_list = []

        for x in range(len(initial_train_dataset)):
            train_dataset_list.append(initial_train_dataset[x]['text'])

        return train_dataset_list

    def bert_tokenize(text):
        return tokenizer.tokenize(text)

    def load_csv_into_dataframe(buckets):
        path = "../buckets/unfair_tos/unfair_tos_tf_idf_buckets_" + str(buckets) + ".csv"
        df = pd.read_csv(path)
        return df

    def convert_tuple_into_desired_format(list_of_tuples):
        new_list = []
        for t in list_of_tuples:
            new_list.append((t[2], t[3]))
        return new_list

    def categorize_tfidf_score(buckets, score):
        if score >= buckets[-1][1]:
            return len(buckets)
        elif score <= buckets[0][0]:
            return 1
        else:
            for idx, b in enumerate(buckets):
                if b[0] <= score <= b[1]:
                    return idx + 1

    # Compute the IDF scores from the training subset
    logger.info("Calculating TF-IDF score for unfair_tos dataset.")
    vocab_list = [(word, word_id) for word, word_id in tokenizer.vocab.items()]
    vocab_list = [word for word, word_id in sorted(vocab_list, key=lambda tup: tup[1])]
    train = convert_train_dataset_into_list(train_dataset)  # Converts train dataset into a list
    tfidf_vectorizer = TfidfVectorizer(tokenizer=bert_tokenize, vocabulary=vocab_list, lowercase=False)
    tfidf_vectorizer.fit(train)
    logger.info("Done calculating TF-IDF score for unfair_tos dataset.")

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Load the data-frame containing the tfidf score ranges for all buckets
    ranges_dataframe = load_csv_into_dataframe(model_args.tfidf_buckets)
    ranges = ranges_dataframe.to_records(index=True)
    ranges = list(ranges)  # Convert dataframe into a list of tuples
    ranges = convert_tuple_into_desired_format(ranges)

    def preprocess_function(examples):
        # Tokenize the texts
        batch = tokenizer(
            examples["text"],
            padding=False,
            max_length=False if data_args.full_text else 128,
            truncation=False if data_args.full_text else True,
        )

        # Remove [CLS] and [SEP] tokens
        batch['input_ids'] = [input_ids[1:-1] for input_ids in batch['input_ids']]

        attention_masks = []
        tf_idf_total = []
        id_total = []
        token_type_ids = []

        for input_id in batch["input_ids"]:
            # Count word ids occurrences
            counts = collections.Counter(input_id)

            input_id_text = []
            tf_idf_score_text = []
            tf_idf_bucket_text = []

            # Compute TF-IDF score per word id and find the relevant TF-IDF bucket
            for word_id, count in counts.most_common():
                # Compute the TF-IDF score of this word id
                n = count * tfidf_vectorizer.idf_[word_id]
                tf_idf = 1 + math.log(n)

                # Get the relevant bucket id, this word id should be positioned
                bucket_for_word_id = categorize_tfidf_score(ranges, tf_idf)

                tf_idf_score_text.append(tf_idf)
                tf_idf_bucket_text.append(bucket_for_word_id)
                input_id_text.append(word_id)

            # Sort the sequence elements from a greater to a lower TF-IDF score
            zipped_lists = zip(tf_idf_score_text, tf_idf_bucket_text, input_id_text)
            sorted_pairs = sorted(zipped_lists, reverse=True)
            tuples = zip(*sorted_pairs)
            tf_idf_score_text, tf_idf_bucket_text, input_id_text = [list(t) for t in tuples]

            # Truncate sequences to max length
            tf_idf_score_text = tf_idf_score_text[:data_args.max_seq_length - 2]
            tf_idf_bucket_text = tf_idf_bucket_text[:data_args.max_seq_length - 2]
            input_id_text = input_id_text[:data_args.max_seq_length - 2]


            input_id_text_ = [tokenizer.cls_token_id] + input_id_text + [tokenizer.sep_token_id] \
                            + [tokenizer.pad_token_id] * max(0, data_args.max_seq_length - (len(input_id_text) + 2))

            # Assign the N+1 bucket id to both the [CLS] and [SEP] tokens, and 0 for [PAD] tokens.
            tf_idf_bucket_text = [len(ranges) + 1] + tf_idf_bucket_text + [len(ranges) + 1] \
                                + [0] * max(0, data_args.max_seq_length - (len(input_id_text) + 2))

            attention_mask = [1] * (len(input_id_text) + 2) + [0] * max(0, data_args.max_seq_length - (len(input_id_text) + 2))

            attention_masks.append(attention_mask)

            token_type_id = [0] * len(attention_mask)
            token_type_ids.append(token_type_id)

            tf_idf_total.append(tf_idf_bucket_text)
            id_total.append(input_id_text_)

        batch["input_ids"] = id_total
        batch["tf_idfs"] = tf_idf_total
        batch["attention_mask"] = attention_masks
        batch["token_type_ids"] = token_type_ids
        batch["labels"] = [[1 if label in labels else 0 for label in label_list] for labels in examples["labels"]]

        return batch

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        # Fix gold labels
        y_true = np.zeros((p.label_ids.shape[0], p.label_ids.shape[1] + 1), dtype=np.int32)
        y_true[:, :-1] = p.label_ids
        y_true[:, -1] = (np.sum(p.label_ids, axis=1) == 0).astype('int32')
        # Fix predictions
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = (expit(logits) > 0.5).astype('int32')
        y_pred = np.zeros((p.label_ids.shape[0], p.label_ids.shape[1] + 1), dtype=np.int32)
        y_pred[:, :-1] = preds
        y_pred[:, -1] = (np.sum(preds, axis=1) == 0).astype('int32')
        # Compute scores
        macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
        return {'macro-f1': macro_f1, 'micro-f1': micro_f1}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        output_predict_file = os.path.join(training_args.output_dir, "test_predictions.csv")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                try:
                    for index, pred_list in enumerate(predictions[0]):
                        pred_line = '\t'.join([f'{pred:.5f}' for pred in pred_list])
                        writer.write(f"{index}\t{pred_line}\n")
                except:
                    try:
                        for index, pred_list in enumerate(predictions):
                            pred_line = '\t'.join([f'{pred:.5f}' for pred in pred_list])
                            writer.write(f"{index}\t{pred_line}\n")
                    except:
                        pass

    # Clean up checkpoints
    checkpoints = [filepath for filepath in glob.glob(f'{training_args.output_dir}/*/') if '/checkpoint' in filepath]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)


if __name__ == "__main__":
    main()