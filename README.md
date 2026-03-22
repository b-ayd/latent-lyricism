# Latent Lyricism

Emotion-preserving English → Turkish poetry translation using latent space alignment.

Standard neural machine translation models optimize for semantic accuracy but tend to flatten emotional valence. This project explores whether guiding the NMT model through a sentiment-aware loss in the latent space can improve emotion preservation across languages.

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/latent_similarity_analysis.ipynb` | Baseline semantic similarity analysis. Samples 28 verses from `poem_sentiment`, translates them with Google Translate, and measures cosine similarity in multilingual sentence embedding space (SBERT). Includes PCA visualization of EN vs TR alignment. |
| `notebooks/latent_lyricism_clean.ipynb` | Current experiment pipeline. Covers mBERT sentiment classifier, zero-shot XLM-RoBERTa baseline, fine-tuned sentiment oracle, baseline NMT emotion preservation, and two latent alignment training approaches. |

## Pipeline Overview

```
poem_sentiment dataset
        │
        ├─ Phase 1 → mBERT Sentiment Classifier (frozen backbone + trainable head)
        ├─ Phase 2 → Zero-Shot Baseline (XLM-RoBERTa, no fine-tuning)
        ├─ Phase 3 → Fine-Tuned Sentiment Oracle → saved as judge
        ├─ Phase 4 → Baseline NMT Emotion Preservation (raw translation quality)
        ├─ Phase 5 → Latent Lyricism (translation loss + soft embedding emotion loss)
        └─ Phase 6 → Latent Alignment (translation loss + MSE on sentiment vectors)
```

## Models Used

- [`bert-base-multilingual-cased`](https://huggingface.co/google-bert/bert-base-multilingual-cased) — mBERT backbone
- [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) — sentiment judge
- [`Helsinki-NLP/opus-mt-tc-big-en-tr`](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr) — NMT (English → Turkish)
- [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — SBERT for similarity analysis

## Dataset

[`poem_sentiment`](https://huggingface.co/datasets/google-research-datasets/poem_sentiment) — English poetry verses labeled as Negative (0), Positive (1), No Impact (2), Mixed (3).