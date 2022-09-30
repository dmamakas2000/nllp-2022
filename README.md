# Processing Long Legal Documents with Pre-trained Transformers :balance_scale:

## About
Pre-trained Transformers currently dominate most NLP tasks. They impose, however, limits on the maximum input length (512 sub-words in BERT), 
which are too restrictive in the legal domain. However, simpler linear classifiers with TF-IDF features can handle texts of any length, 
require far less resources to train and deploy, but are usually outperformed by pre-trained Transformers. For the reasons mentioned, in this 
repository we generally experiment with the creation of model which combine those 2 techniques.

### Models
1. **Legal-BERT** <br>
We consider Legal-BERT ([Chalkidis et al., 2020](https://arxiv.org/pdf/2010.02559.pdf)), a BERT model pre-trained on English legal corpora 
(legislation, contracts, court cases).

1. **TFIDF-SRT-Legal-BERT** <br>
This is Legal-BERT, but we remove duplicate tokens from the input text and sort the remaining ones by decreasing TF-IDF during fine-tuning. 
Removing duplicate tokens is an attempt to avoid exceeding the maximum input length. In EUR-LEX, e.g., the average text length (in tokens) 
drops from 1,133 to 341; in SCOTUS, from 5,953 to 1,636. If the new form of the text still exceeds the maximum input length, we truncate it 
(keeping the first 512 tokens). Ordering the remaining tokens by decreasing TF-IDF hopefully allows the model to learn to attend earlier 
tokens (higher TF-IDF) more. This is a Bag of Words (BoW) model, since the original word order is lost.

1. **TFIDF-SRT-EMB-Legal-BERT** <br>
This is the same as the previous model, except that we add a TF-IDF embedding layer. We bucketize the distribution of TF-IDF scores of the 
training set and assign a TF-IDF embedding to each bucket. During fine-tuning, we compute the TF-IDF score of each token (before removing 
duplicates) and we add the corresponding TF-IDF bucket embedding to the token’s input embedding when its positional embedding is also added. 
The TF-IDF bucket embeddings are initialized randomly and trained during fine-tuning. Hence, this model is informed about TF-IDF scores 
both via token order and TF-IDF embeddings. It is still a BoW model, like the previous one.

1. **TFIDF-EMB-Legal-BERT** <br>
This model is the same as Legal-BERT, but we add the TF-IDF layer of the previous model. Token de-duplication and ordering by TF-IDF scores, 
however, are not in252 cluded. This allows us to study the contribution of the TF-IDF layer on its own by comparing to the original Legal-BERT. 
The resulting model is aware of word-order via its positional embeddings (like BERT and Legal-BERT), but does not address the maximum input 
length limitation in any way.

## Dataset Summary
To evaluate the performance of our implemented models on competitive legal domain tasks we chose to experiment with LexGLUE benchmark. LexGLUE 
(Legal General Language Understanding Evaluation) benchmark ([Chalkidis et al., 2022](https://github.com/coastalcph/lex-glue)) is a collection 
of seven English legal NLP datasets that can be used to evaluate the performance of any proposed NLP method across a variety of legal NLU tasks.

## Experiments
For experimental needs, we run our models on each dataset across LexGLUE benchmark except of CaseHOLD. Also, it should be mentioned that the 
Adam optimizer was always used to train our models, combined with an initial learning rate of 3e-5. We also have set the training to be done 
within 20 epochs using early stopping. The batch size was equal to 8 in all experiments with datasets UNFAIR-ToS, LEDGAR, and EUR-LEX, but on 
datasets ECtHR (Tasks A-B) and SCOTUS, it was set to 1. The gradient accumulation steps were set to 1 for UNFAIR-ToS, LEDGAR, EUR-LEX, and 4 
for ECtHR and SCOTUS. In terms of experiments, we always run 5 seeds (five different repetitions of the same experiment) and reported the 
final model’s results based on the seed with the best scores on development data. In the models where an extra TF-IDF embedding layer was 
added, we always run 4 repetitions per experiment, for dimensions of size 16, 32, 64, and 128. Those sizes represent the number of TF-IDF 
buckets to distribute the TF-IDF score for each token per text.

Furthermore, we should indicate that each model’s performance was evaluated using the F-measure across all datasets of the LexGLUE benchmark. 
More specifically, we consider macro-F1 (m-F1) which is agreat measurement to calculate the average model’s effectiveness in all categories. 
In addition, we also use micro-F1 (µ-F1) metric, which reports the model’s accuracy, as we always experiment with multi-class classification

### Experimental results
The following table reports the results obtained per dataset across all models developed. Please, note that to display the numbers, we use 
the format **μ-F1 / m-F1**.

| **Model**                  |  **ECtHR (Task A)** |  **ECtHR (Task B)** |      **SCOTUS**     |     **EUR-LEX**     |      **LEDGAR**     |    **UNFAIR-ToS**   |
|----------------------------|:-------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
| _Legal-BERT_               | **70.0** / **64.0** | **80.4** / **74.7** | **76.4** / **66.5** | **72.1** / **57.4** |     88.2 / 83.0     | **96.0** / **83.0** |
| _TFIDF-SRT-Legal-BERT_     |     69.8 / 62.8     |     78.5 / 71.9     |     73.4 / 61.8     |     66.6 / 49.7     |     86.9 / 80.8     |     95.3 / 80.6     |
| _TFIDF-SRT-EMB-Legal-BERT_ |     68.7 / 63.1     |     79.0 / 72.5     |     73.9 / 63.6     |     66.7 / 49.9     |     86.5 / 80.3     |     95.8 / 78.7     |
| _TFIDF-EMB-Legal-BERT_     |     70.0 / 61.9     |     79.4 / 73.5     |     74.9 / 64.7     |     68.2 / 52.6     | **88.7** / **83.4** |     95.9 / 82.1     |

### Time statistics
The following table reports the average training time needed per seed for each model. Please, note that to display the results, we use 
the format **hours:minutes:seconds**.

| **Model**                  | **ECtHR (Task A)** | **ECtHR (Task B)** | **SCOTUS** | **EUR-LEX** | **LEDGAR** | **UNFAIR-ToS** |
|----------------------------|:------------------:|:------------------:|:----------:|:-----------:|:----------:|:--------------:|
| _Legal-BERT_               |      11:34:17      |      12:16:55      |  10:50:02  |   12:49:03  |  15:48:02  |    00:20:17    |
| _TFIDF-SRT-Legal-BERT_     |      01:19:33      |      00:35:20      |  00:23:41  |   09:04:34  |  13:09:58  |    00:08:11    |
| _TFIDF-SRT-EMB-Legal-BERT_ |      00:38:22      |      00:31:53      |  00:21:22  |   03:17:27  |  05:23:33  |    00:03:33    |
| _TFIDF-EMB-Legal-BERT_     |      12:15:26      |      13:54:57      |  09:14:29  |   10:20:36  |  14:19:29  |    00:15:30    |

## Frequently Asked Questions (FAQ)

### How can I run a specific experiment?

For any UNIX server, please follow these steps:

1. Clone the repository using `git clone https://github.com/dmamakas2000/NLLP`.

1. Move into the desired directory representing the experiment you want to execute. For example, type `cd scripts/TFIDF-SRT-Legal-BERT`

1. Export the Pythonpath using `export PYTHONPATH="${PYTHONPATH}:/your_directory_to_project/"`.

1. Make the bash script executable using `chmod`. For example, `chmod +x run_ecthr_TF_IDF_SRT_Legal_BERT.sh`.

1. Execute the bash script. For example, `./run_ecthr_TF_IDF_SRT_Legal_BERT.sh`.
