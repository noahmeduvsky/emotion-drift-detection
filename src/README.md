# Emotion Drift Detection

Detects emotion changes in customer support conversations using BERT and RoBERTa models.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare data:
   ```bash
   python prepare_data.py
   ```

## Usage

Train a model:
```bash
python -m src.main train --model-type transformer --model-name bert-base-uncased --num-epochs 10
```

Evaluate a trained model:
```bash
python -m src.main evaluate --checkpoint models/bert_real_weighted/best_model.pt
```

Test on a conversation:
```bash
python test_conversation.py --interactive
```

## Project Structure

- `src/` - Source code
  - `data_loader.py` - Dataset loading
  - `preprocessing.py` - Data preprocessing
  - `models.py` - Model architectures
  - `train.py` - Training script
  - `evaluation.py` - Evaluation metrics
  - `main.py` - Main entry point
- `models/` - Trained model checkpoints
- `data/` - Processed datasets
- `notebooks/` - Data exploration notebooks
- `config.py` - Hyperparameters and settings
