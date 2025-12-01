# How to Test Your Trained Model

## Quick Summary: What This Project Does

Your project builds a **smart emotion detector** for customer support conversations. It:

1. **Reads text messages** from a conversation
2. **Detects emotions** in each message (anger, joy, sadness, neutral, etc.)
3. **Spots emotion changes** when a customer's mood shifts (e.g., neutral to angry)
4. **Flags warning signs** when negative emotions are detected

This helps AI customer support systems respond better and prevent customer frustration.

## Available Models

You have several trained models to choose from:

- **`models/bert_real_weighted/best_model.pt`** - Best overall performance (recommended)
  - Uses class balancing for better rare emotion detection
  - 38% macro F1 score
  - Best at detecting emotion drift

- **`models/bert_real/best_model.pt`** - Baseline model (no class balancing)
- **`models/roberta_real/best_model.pt`** - RoBERTa variant

## Testing the Model

### Option 1: Interactive Mode (Recommended)

Run the model and type text to see predictions:

```bash
cd src
python test_model_inference.py --checkpoint models/bert_real_weighted/best_model.pt --interactive
```

Example session:
```
Enter text (or 'quit'/'new'): I have a question about my bill
Emotion: neutral (85.23% confidence)

Enter text (or 'quit'/'new'): The amount seems wrong
Emotion: neutral (72.15% confidence)
  [DRIFT: neutral to neutral]

Enter text (or 'quit'/'new'): I'm getting really frustrated here!
Emotion: anger (67.42% confidence)
  [DRIFT: neutral to anger]

Enter text (or 'quit'/'new'): new
[New dialogue sequence started]

Enter text (or 'quit'/'new'): quit
```

### Option 2: Single Text Prediction

Test a single message:

```bash
python test_model_inference.py --checkpoint models/bert_real_weighted/best_model.pt --text "I'm very happy with this service!"
```

Output:
```
Text: I'm very happy with this service!
Predicted Emotion: joy (89.5% confidence)

Top 3 predictions:
  - joy: 89.50%
  - neutral: 8.23%
  - surprise: 2.27%
```

### Option 3: Dialogue Sequence (Detects Drift)

Test multiple turns and see emotion drift:

```bash
python test_model_inference.py --checkpoint models/bert_real_weighted/best_model.pt --text "Hello, I need help" "This isn't working" "I'm getting angry now"
```

Output shows:
- Emotion for each turn
- Emotion drift between turns
- Warnings for significant shifts

### Option 4: Demo Mode

Just run without arguments to see example dialogue:

```bash
python test_model_inference.py --checkpoint models/bert_real_weighted/best_model.pt
```

## Understanding the Output

### Emotion Classes

The model predicts one of 7 emotions:
- **anger** - Customer is angry/frustrated
- **disgust** - Customer shows disgust
- **fear** - Customer is anxious/afraid
- **joy** - Customer is happy/satisfied
- **neutral** - No strong emotion
- **sadness** - Customer is sad/disappointed
- **surprise** - Customer is surprised

### Confidence Scores

- **High (>70%)**: Model is confident in prediction
- **Medium (50-70%)**: Model is somewhat confident
- **Low (<50%)**: Model is uncertain (may need more context)

### Drift Detection

- **No drift**: Same emotion continues (e.g., neutral to neutral)
- **Minor drift**: Small emotion change (e.g., neutral to surprise)
- **Significant drift**: Large emotion shift (e.g., neutral to anger)
  - These are flagged with [WARNING] for attention

## Example Use Cases

### 1. Test Customer Frustration

```bash
python test_model_inference.py --checkpoint models/bert_real_weighted/best_model.pt --text "I've been waiting for hours" "No one is helping me" "This is ridiculous!"
```

Expected: Should detect progression from neutral to frustration to anger

### 2. Test Positive Resolution

```bash
python test_model_inference.py --checkpoint models/bert_real_weighted/best_model.pt --text "I have a problem" "Can you help me?" "Thank you so much!"
```

Expected: Should detect neutral to neutral to joy

### 3. Test Sudden Emotion Change

```bash
python test_model_inference.py --checkpoint models/bert_real_weighted/best_model.pt --text "Everything is fine" "Actually, I'm really worried about this"
```

Expected: Should detect significant drift from neutral to fear/worry

## Troubleshooting

### Error: "CUDA out of memory"
- Use CPU instead: `--device cpu`
- Example: `python test_model_inference.py --checkpoint models/bert_real_weighted/best_model.pt --text "test" --device cpu`

### Error: "Model file not found"
- Make sure you're in the `src/` directory
- Check that the checkpoint path is correct: `models/bert_real_weighted/best_model.pt`

### Error: "Import error"
- Make sure dependencies are installed: `pip install -r requirements.txt`
- Make sure you're running from the `src/` directory

## What to Look For

### Good Signs (Model Working Well)
- Predictions match your intuition
- High confidence scores for clear emotional text
- Drift detection flags obvious emotion changes
- Rare emotions (fear, disgust) are detected when present

### Potential Issues
- Always predicting "neutral" - model may not be loaded correctly
- Very low confidence scores - text may be ambiguous
- Wrong emotion predictions - model may need more training data

## Next Steps

1. **Test on real customer support conversations** - Use actual dialogue from your support system
2. **Fine-tune on your domain** - Train on customer support specific data
3. **Integrate into production** - Connect to your live support system
4. **Monitor performance** - Track drift detection accuracy over time

## Example Integration Code

If you want to use this in your own Python code:

```python
from test_model_inference import load_model, predict_emotion
from transformers import BertTokenizer

# Load model
model = load_model('models/bert_real_weighted/best_model.pt')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
emotion_classes = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Predict emotion
text = "I'm really frustrated with this service!"
emotion, confidence, probs = predict_emotion(model, text, tokenizer, 'cuda', emotion_classes)

print(f"Emotion: {emotion} ({confidence:.2%} confidence)")
```

## Questions?

- Check `PROJECT_ACCOMPLISHMENTS.md` for what the project does
- Check model metrics in `models/*/results/metrics.json`
- Review training results in `models/*/training_history.png`

