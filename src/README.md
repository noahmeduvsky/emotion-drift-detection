# Emotion Drift in Customer Support AI

## Project Overview

This project builds a machine learning model to track how a customer's mood changes during an AI chat interaction. The goal is to detect emotion drift—when the tone shifts from positive to negative (or vice versa)—and analyze whether the AI's responses correlate with those emotional changes.

## Project Structure

**Project Structure:**
- `src/data/` - Raw and processed datasets
- `src/models/` - Trained models and checkpoints
- `src/notebooks/` - Jupyter notebooks for exploration and visualization
- `src/src/` - Source code modules
  - `data_loader.py` - Dataset loading utilities
  - `preprocessing.py` - Data preprocessing pipeline
  - `models.py` - Model architectures
- `requirements.txt` - Python dependencies


## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download datasets:**
   The project uses publicly available datasets:
   - EmotionLines (from Hugging Face or GitHub)
   - DailyDialog (from Hugging Face)
   - MELD (Multimodal EmotionLines Dataset)

3. **Run data exploration:**
   ```bash
   jupyter notebook notebooks/data_exploration.ipynb
   ```

## Data Format

Processed data will be stored with the following structure:
- `dialogue_id`: Unique identifier for each conversation
- `turn_id`: Sequential turn number within dialogue
- `speaker`: Speaker identifier (user/AI)
- `text`: Utterance text
- `emotion`: Emotion label (joy, anger, sadness, neutral, etc.)

## ML Pipeline Overview

1. **Data Collection**: Download and load datasets from Hugging Face or official sources
2. **Preprocessing**: Clean text, tokenize using BERT tokenizer, encode to embeddings
3. **Feature Extraction**: Generate contextual embeddings using BERT/RoBERTa
4. **Model Training**: Train sequence models (LSTM or Transformer) for drift detection
5. **Evaluation**: Compute emotion classification accuracy, sentiment shift metrics, drift detection accuracy
6. **Visualization**: Generate emotion trajectory graphs across conversations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model

```bash
# Train a transformer model
python -m src.main train --model-type transformer --model-name bert-base-uncased --num-epochs 10

# Train with custom settings
python -m src.main train \
    --model-type transformer \
    --model-name bert-base-uncased \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --num-epochs 10 \
    --save-dir models/checkpoints
```

### 4. Evaluate a Trained Model

```bash
python -m src.main evaluate \
    --checkpoint models/checkpoints/best_model.pt \
    --split test
```

## Project Structure

**Source Code:**
- `src/data_loader.py` - Dataset loading utilities
- `src/preprocessing.py` - Data preprocessing pipeline
- `src/models.py` - Model architectures (LSTM, Transformer)
- `src/dataset.py` - PyTorch Dataset and DataLoader classes
- `src/train.py` - Training script with Trainer class
- `src/evaluation.py` - Evaluation metrics and utilities
- `src/visualization.py` - Plotting and visualization functions
- `src/main.py` - Main orchestration script

**Notebooks:**
- `notebooks/data_exploration.ipynb` - Data exploration and analysis

**Configuration:**
- `config.py` - Default hyperparameters and settings
- `requirements.txt` - Python dependencies

## Research Question

"Can an emotion drift detection model identify when user emotion shifts during a customer support AI conversation, and does the AI's response correlate with that shift?"

## References

- Zhang, J., et al. (2024). The Impact of Emotional Expression by Artificial Intelligence. Decision Support Systems, 181, 114075.
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
- Hsu et al. (2018). EmotionLines: An Emotion Corpus of Multi-Party Conversations.

