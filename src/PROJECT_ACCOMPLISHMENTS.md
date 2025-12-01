# What This Project Accomplishes

## Overview

This project builds a **machine learning system that detects emotion drift in customer support conversations**. Emotion drift is when a customer's emotional state changes during a dialogue (e.g., from neutral to anger, or from frustration to satisfaction).

## Core Capabilities

### 1. **Emotion Classification**
- Classifies emotions in individual dialogue turns
- Supports 7 emotion classes: anger, disgust, fear, joy, neutral, sadness, surprise
- Uses transformer models (BERT/RoBERTa) for contextual understanding
- Achieves 38.0% macro F1 score with class balancing (vs 29.3% baseline)

### 2. **Emotion Drift Detection**
- Identifies when emotions shift between consecutive dialogue turns
- Detects significant emotional transitions (e.g., neutral to anger)
- Achieves 47.1% drift detection F1 score (vs 36.8% baseline)
- Provides drift recall of 56.9% (detects over half of actual emotion changes)

### 3. **Handling Class Imbalance**
- Addresses severe dataset imbalance (84% neutral emotions, 0.2% fear)
- Uses weighted cross-entropy loss and focal loss techniques
- Dramatically improves rare emotion detection:
  - Fear: 0% to 33.3% F1 score
  - Sadness: 4.8% to 19.2% F1 score

## How It Works

### Input
- **Dialogue sequences**: Conversations between customers and AI support
- **Text turns**: Individual messages/utterances in a conversation

### Processing
1. **Text Tokenization**: Converts text to token IDs using BERT/RoBERTa tokenizer
2. **Contextual Encoding**: Uses pre-trained transformer to understand emotional context
3. **Emotion Prediction**: Classifies each turn into one of 7 emotion classes
4. **Drift Analysis**: Compares consecutive predictions to detect emotion changes

### Output
- **Emotion labels**: For each dialogue turn
- **Drift detection**: Flags when significant emotion shifts occur
- **Confidence scores**: Probability distributions over emotion classes

## Practical Applications

### Customer Support AI
- **Real-time monitoring**: Detect when customers are becoming frustrated
- **Proactive intervention**: Alert human agents when negative emotion drift is detected
- **Response optimization**: Adjust AI responses based on detected emotions
- **Quality assurance**: Analyze conversation quality and emotional outcomes

### Business Impact
- **Customer satisfaction**: Prevent escalation by detecting early warning signs
- **Churn reduction**: Identify at-risk customers through emotion analysis
- **Training data**: Use drift patterns to improve AI response strategies
- **Performance metrics**: Track emotion trajectories across support interactions

## Technical Achievements

### Model Performance
- **BERT with weighted loss**: Best overall performance
  - 38.0% macro F1 (balanced across all emotions)
  - 81.0% accuracy
  - 47.1% drift detection F1

### Dataset
- Trained on **DailyDialog**: 11,118 dialogues, 87,396 turns
- Real-world emotion distribution (highly imbalanced)
- General dialogue data adaptable to customer support scenarios

### Architecture
- **Transformer-based**: BERT-base and RoBERTa-base
- **Sequence-aware**: Processes dialogue context across multiple turns
- **Class-balanced**: Handles imbalanced datasets effectively

## Limitations & Future Work

### Current Limitations
- Trained on general dialogue (not customer support specific)
- Discrete emotion labels (not continuous emotional states)
- Requires fine-tuning large models (computational cost)

### Potential Enhancements
- Domain-specific fine-tuning on customer support data
- Continuous emotion modeling (valence-arousal dimensions)
- Real-time deployment optimizations
- Multimodal features (tone, speaking pace)
- Explainable AI for emotion drift triggers

## How to Test It

Use the inference script to test the model on your own text:

```bash
# Interactive mode (type text and see predictions)
python test_model_inference.py --checkpoint models/bert_real_weighted/best_model.pt --interactive

# Single text prediction
python test_model_inference.py --checkpoint models/bert_real_weighted/best_model.pt --text "I'm really frustrated with this service"

# Multiple turns (detects drift)
python test_model_inference.py --checkpoint models/bert_real_weighted/best_model.pt --text "Hello, I need help" "This isn't working" "I'm getting very angry now"
```

The model will:
1. Predict emotions for each input
2. Show confidence scores
3. Detect emotion drift between turns
4. Flag significant emotional shifts

## Research Contribution

This project demonstrates that:
- Class balancing is essential for real-world emotion recognition
- Transformer models effectively capture contextual emotional cues
- Emotion drift detection enables analysis of conversation dynamics
- The approach provides actionable insights for improving AI customer support

## Key Metrics Summary

| Metric | Baseline | With Class Balancing | Improvement |
|--------|----------|---------------------|-------------|
| Macro F1 | 29.3% | 38.0% | +30% |
| Drift F1 | 36.8% | 47.1% | +28% |
| Drift Recall | 28.7% | 56.9% | +98% |
| Fear Detection | 0% | 33.3% | ∞ (was 0) |
| Sadness Detection | 4.8% | 19.2% | +300% |

