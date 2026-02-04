# Spam Email Classifier

A simple spam email classifier using **NLP + PyTorch** with two feature options:

- **TF-IDF + Linear layer**
- **Word embeddings + EmbeddingBag**

The classifier performs **binary classification** (spam vs. not spam).

## Project Structure

```
./data/sample_spam.csv   # Sample dataset
./src/spam_classifier.py # Training script
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train with TF-IDF

```bash
python src/spam_classifier.py --mode tfidf --data-path data/sample_spam.csv
```

## Train with Word Embeddings

```bash
python src/spam_classifier.py --mode embedding --data-path data/sample_spam.csv
```

## Dataset Format

The CSV must include `text` and `label` columns.

```
text,label
"Get rich quick",1
"Meeting at 2pm",0
```

## Notes

- The sample dataset is tiny and intended for demonstration.
- For real performance, expand the dataset and tune hyperparameters.
