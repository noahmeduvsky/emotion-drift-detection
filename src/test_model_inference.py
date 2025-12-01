"""
Script to test the trained model on new text. You can input dialogue and see emotion predictions.
"""

import torch
import os
import sys
import json
import argparse
from transformers import BertTokenizer, RobertaTokenizer
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import create_model
from src.preprocessing import EmotionPreprocessor

def load_model(checkpoint_path: str, model_name: str = "bert-base-uncased", num_emotions: int = 7, device: str = "cuda"):
    """Loads a trained model from a checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    model = create_model(
        model_type="transformer",
        model_name=model_name,
        num_emotions=num_emotions
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model

def predict_emotion(model, text: str, tokenizer, device: str, emotion_classes: list):
    """Predicts emotion for a single text input."""
    model.eval()
    
    # tokenize the input
    encoded = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # get prediction
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # use [CLS] token (first token) for classification
        cls_logits = logits[:, 0, :]
        probabilities = torch.softmax(cls_logits, dim=-1)
        predicted_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_idx].item()
    
    emotion = emotion_classes[predicted_idx]
    
    return emotion, confidence, probabilities[0].cpu().numpy()

def predict_dialogue_sequence(model, texts: list, tokenizer, device: str, emotion_classes: list):
    """Predicts emotions for a sequence of dialogue turns and detects drift."""
    emotions = []
    confidences = []
    
    print("\n" + "="*60)
    print("EMOTION PREDICTIONS FOR DIALOGUE SEQUENCE")
    print("="*60)
    
    for i, text in enumerate(texts, 1):
        emotion, confidence, probs = predict_emotion(model, text, tokenizer, device, emotion_classes)
        emotions.append(emotion)
        confidences.append(confidence)
        
        print(f"\nTurn {i}:")
        print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  Predicted Emotion: {emotion} (confidence: {confidence:.2%})")
        
        # show top 3 predictions
        top_indices = np.argsort(probs)[-3:][::-1]
        print(f"  Top 3 predictions:")
        for idx in top_indices:
            print(f"    - {emotion_classes[idx]}: {probs[idx]:.2%}")
    
    # detect drift between turns
    print("\n" + "="*60)
    print("EMOTION DRIFT ANALYSIS")
    print("="*60)
    
    emotion_to_idx = {emotion: i for i, emotion in enumerate(emotion_classes)}
    
    for i in range(1, len(emotions)):
        prev_emotion = emotions[i-1]
        curr_emotion = emotions[i]
        
        if prev_emotion != curr_emotion:
            prev_idx = emotion_to_idx[prev_emotion]
            curr_idx = emotion_to_idx[curr_emotion]
            
            drift_magnitude = abs(curr_idx - prev_idx)
            print(f"\n[DRIFT DETECTED] Turn {i-1} to Turn {i}")
            print(f"  {prev_emotion} to {curr_emotion}")
            print(f"  Drift magnitude: {drift_magnitude}")
            
            if drift_magnitude >= 3:
                print(f"  [WARNING] SIGNIFICANT DRIFT - Large emotion shift detected!")
        else:
            print(f"\nTurn {i-1} to Turn {i}: No drift ({prev_emotion})")
    
    return emotions, confidences

def main():
    parser = argparse.ArgumentParser(description="Test emotion drift detection model on new text")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (e.g., models/bert_real_weighted/best_model.pt)')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                       help='Model name used for training (bert-base-uncased or roberta-base)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode to input text')
    parser.add_argument('--text', type=str, nargs='+',
                       help='Text(s) to classify (for single or multiple turns)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Emotion classes (standard for DailyDialog)
    emotion_classes = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    
    # Initialize tokenizer
    if "roberta" in args.model_name.lower():
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Load model
    try:
        model = load_model(args.checkpoint, args.model_name, len(emotion_classes), args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying to determine model architecture from checkpoint...")
        return
    
    print(f"\nUsing device: {args.device}")
    print(f"Model ready! Enter text to classify emotions.\n")
    
    # Interactive mode
    if args.interactive:
        print("="*60)
        print("INTERACTIVE MODE")
        print("Type 'quit' to exit, 'new' to start a new dialogue sequence")
        print("="*60)
        
        dialogue_sequence = []
        
        while True:
            text = input("\nEnter text (or 'quit'/'new'): ").strip()
            
            if text.lower() == 'quit':
                break
            elif text.lower() == 'new':
                if dialogue_sequence:
                    predict_dialogue_sequence(model, dialogue_sequence, tokenizer, args.device, emotion_classes)
                dialogue_sequence = []
                print("\n[New dialogue sequence started]")
            elif text:
                dialogue_sequence.append(text)
                
                # Predict for this turn
                emotion, confidence, probs = predict_emotion(model, text, tokenizer, args.device, emotion_classes)
                print(f"\nEmotion: {emotion} ({confidence:.2%} confidence)")
                
                # If we have previous turns, show drift
                if len(dialogue_sequence) > 1:
                    prev_emotion, _, _ = predict_emotion(
                        model, dialogue_sequence[-2], tokenizer, args.device, emotion_classes
                    )
                    if prev_emotion != emotion:
                        print(f"  [DRIFT: {prev_emotion} to {emotion}]")
    
    # Command-line text input
    elif args.text:
        texts = args.text
        if len(texts) == 1:
            # Single text prediction
            emotion, confidence, probs = predict_emotion(model, texts[0], tokenizer, args.device, emotion_classes)
            print(f"\nText: {texts[0]}")
            print(f"Predicted Emotion: {emotion} (confidence: {confidence:.2%})")
            
            top_indices = np.argsort(probs)[-3:][::-1]
            print(f"\nTop 3 predictions:")
            for idx in top_indices:
                print(f"  - {emotion_classes[idx]}: {probs[idx]:.2%}")
        else:
            # Multiple texts - show sequence and drift
            predict_dialogue_sequence(model, texts, tokenizer, args.device, emotion_classes)
    
    else:
        # Demo examples
        print("\n" + "="*60)
        print("DEMO: Running example dialogue sequence")
        print("="*60)
        
        example_dialogue = [
            "I have a question about my bill",
            "Sure, I can help with that. What seems to be the issue?",
            "The amount looks incorrect to me",
            "Let me check your account details",
            "I already told you this twice and you're still not helping me",
            "I apologize for the inconvenience. Let me escalate this to a specialist.",
            "Thank you, that would be helpful"
        ]
        
        predict_dialogue_sequence(model, example_dialogue, tokenizer, args.device, emotion_classes)

if __name__ == "__main__":
    main()

