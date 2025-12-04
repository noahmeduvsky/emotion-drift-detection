"""
Test emotion drift detection on a natural conversation.
Supports multiple input formats:
1. Command-line arguments (turn by turn)
2. Text file with conversation
3. JSON file with structured conversation
4. Simple text format (speaker: message)
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

def load_model(checkpoint_path: str, model_name: str = "bert-base-uncased", num_emotions: int = 7, device: str = "cuda"):
    """Load a trained model from checkpoint."""
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
    """Predict emotion for a single text input."""
    model.eval()
    
    # Tokenize input
    encoded = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Take [CLS] token prediction (first token)
        cls_logits = logits[:, 0, :]
        probabilities = torch.softmax(cls_logits, dim=-1)
        predicted_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_idx].item()
    
    emotion = emotion_classes[predicted_idx]
    
    return emotion, confidence, probabilities[0].cpu().numpy()

def parse_conversation_from_text(text_content: str):
    """Parse conversation from text format (various formats supported)."""
    lines = text_content.strip().split('\n')
    messages = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Format 1: "Speaker: Message"
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                message = parts[1].strip()
                if message:
                    messages.append({'speaker': speaker, 'text': message})
            else:
                # Just a message without speaker label
                messages.append({'speaker': 'User', 'text': line})
        else:
            # Just a plain message
            messages.append({'speaker': 'User', 'text': line})
    
    return messages

def parse_conversation_from_json(json_content: dict):
    """Parse conversation from JSON format."""
    messages = []
    
    if isinstance(json_content, list):
        # List of messages
        for msg in json_content:
            if isinstance(msg, dict):
                text = msg.get('text') or msg.get('message') or msg.get('content')
                speaker = msg.get('speaker') or msg.get('role') or 'User'
                if text:
                    messages.append({'speaker': speaker, 'text': text})
    elif isinstance(json_content, dict):
        # Dict with 'messages' key
        if 'messages' in json_content:
            return parse_conversation_from_json(json_content['messages'])
        # Single message
        text = json_content.get('text') or json_content.get('message') or json_content.get('content')
        if text:
            speaker = json_content.get('speaker') or json_content.get('role') or 'User'
            messages.append({'speaker': speaker, 'text': text})
    
    return messages

def analyze_conversation(model, messages: list, tokenizer, device: str, emotion_classes: list):
    """Analyze a conversation and detect emotion drift."""
    
    print("\n" + "="*70)
    print("CONVERSATION EMOTION ANALYSIS")
    print("="*70)
    
    emotions = []
    confidences = []
    
    # Predict emotions for each message
    for i, msg in enumerate(messages, 1):
        text = msg['text']
        speaker = msg.get('speaker', 'User')
        
        emotion, confidence, probs = predict_emotion(model, text, tokenizer, device, emotion_classes)
        emotions.append(emotion)
        confidences.append(confidence)
        
        # Format display
        speaker_label = f"[{speaker}]"
        text_preview = text[:60] + '...' if len(text) > 60 else text
        
        print(f"\nTurn {i} {speaker_label}")
        print(f"  Text: {text_preview}")
        print(f"  Emotion: {emotion.upper()} ({confidence:.1%} confidence)")
        
        # Show top 2 alternative emotions
        top_indices = np.argsort(probs)[-2:][::-1]
        if top_indices[0] != emotion_classes.index(emotion):
            alt_emotion = emotion_classes[top_indices[0]]
            alt_conf = probs[top_indices[0]]
            print(f"  (Alternative: {alt_emotion} at {alt_conf:.1%})")
    
    # Detect drift
    print("\n" + "="*70)
    print("EMOTION DRIFT ANALYSIS")
    print("="*70)
    
    emotion_to_idx = {emotion: i for i, emotion in enumerate(emotion_classes)}
    
    drift_count = 0
    significant_drifts = []
    
    for i in range(1, len(emotions)):
        prev_emotion = emotions[i-1]
        curr_emotion = emotions[i]
        prev_speaker = messages[i-1].get('speaker', 'User')
        curr_speaker = messages[i].get('speaker', 'User')
        
        if prev_emotion != curr_emotion:
            prev_idx = emotion_to_idx[prev_emotion]
            curr_idx = emotion_to_idx[curr_emotion]
            drift_magnitude = abs(curr_idx - prev_idx)
            
            drift_count += 1
            drift_info = {
                'turn': i,
                'prev_emotion': prev_emotion,
                'curr_emotion': curr_emotion,
                'magnitude': drift_magnitude,
                'prev_speaker': prev_speaker,
                'curr_speaker': curr_speaker
            }
            
            print(f"\n[DRIFT #{drift_count}] Turn {i-1} to Turn {i}")
            print(f"  {prev_speaker}: {prev_emotion.upper()} to {curr_speaker}: {curr_emotion.upper()}")
            print(f"  Magnitude: {drift_magnitude} steps")
            
            if drift_magnitude >= 3:
                print(f"  [WARNING] SIGNIFICANT DRIFT - Large emotion shift detected!")
                drift_info['significant'] = True
            else:
                drift_info['significant'] = False
            
            significant_drifts.append(drift_info)
        else:
            print(f"\nTurn {i-1} to Turn {i}: No drift ({prev_emotion.upper()})")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total messages: {len(messages)}")
    print(f"Emotion drifts detected: {drift_count}")
    print(f"Significant drifts: {sum(1 for d in significant_drifts if d.get('significant', False))}")
    
    if emotions:
        emotion_distribution = {}
        for emotion in emotions:
            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
        
        print(f"\nEmotion distribution:")
        for emotion, count in sorted(emotion_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(emotions)) * 100
            print(f"  {emotion.upper()}: {count} ({percentage:.1f}%)")
    
    return emotions, confidences, significant_drifts

def main():
    parser = argparse.ArgumentParser(description="Test emotion drift detection on a natural conversation")
    # Handle path resolution for checkpoint
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_checkpoint = os.path.join(script_dir, 'models', 'bert_real_weighted', 'best_model.pt')
    if not os.path.exists(default_checkpoint):
        default_checkpoint = 'models/bert_real_weighted/best_model.pt'
    
    parser.add_argument('--checkpoint', type=str, default=default_checkpoint,
                       help='Path to model checkpoint')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                       help='Model name used for training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    
    # Input options
    parser.add_argument('--file', type=str, help='Path to conversation file (txt or json)')
    parser.add_argument('--text', type=str, nargs='+', help='Conversation messages as command-line arguments')
    parser.add_argument('--input-format', type=str, choices=['auto', 'txt', 'json'], default='auto',
                       help='Input file format (auto-detect if not specified)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode for live conversation input')
    
    args = parser.parse_args()
    
    # Emotion classes
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
        return
    
    # Parse conversation
    messages = []
    
    if args.file:
        # Load from file
        print(f"Loading conversation from {args.file}...")
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                if args.input_format == 'json' or (args.input_format == 'auto' and args.file.endswith('.json')):
                    json_content = json.load(f)
                    messages = parse_conversation_from_json(json_content)
                else:
                    text_content = f.read()
                    messages = parse_conversation_from_text(text_content)
            
            if not messages:
                print("Error: No messages found in file")
                return
            
            print(f"Loaded {len(messages)} messages from file")
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    
    elif args.interactive:
        # Interactive mode for live conversation
        print("\n" + "="*70)
        print("INTERACTIVE CONVERSATION MODE")
        print("="*70)
        print("Commands:")
        print("  - Type a message to analyze")
        print("  - Type 'analyze' to see full conversation analysis")
        print("  - Type 'clear' to start a new conversation")
        print("  - Type 'quit' or 'exit' to end")
        print("="*70 + "\n")
        
        conversation_messages = []
        
        while True:
            try:
                user_input = input("\nEnter message (or command): ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    if conversation_messages:
                        print("\n" + "="*70)
                        print("FINAL CONVERSATION ANALYSIS")
                        print("="*70)
                        analyze_conversation(
                            model, conversation_messages, tokenizer, args.device, emotion_classes
                        )
                    print("\nGoodbye!")
                    break
                
                elif user_input.lower() == 'clear':
                    if conversation_messages:
                        print(f"\n[Cleared {len(conversation_messages)} previous messages]")
                    conversation_messages = []
                    continue
                
                elif user_input.lower() == 'analyze':
                    if conversation_messages:
                        analyze_conversation(
                            model, conversation_messages, tokenizer, args.device, emotion_classes
                        )
                    else:
                        print("\n[No messages in conversation yet]")
                    continue
                
                else:
                    # Add message to conversation
                    message = {'speaker': 'User', 'text': user_input}
                    conversation_messages.append(message)
                    
                    # Predict emotion for this turn
                    emotion, confidence, probs = predict_emotion(
                        model, user_input, tokenizer, args.device, emotion_classes
                    )
                    
                    print(f"\n[{len(conversation_messages)}] {user_input[:60]}{'...' if len(user_input) > 60 else ''}")
                    print(f"     Emotion: {emotion.upper()} ({confidence:.1%} confidence)")
                    
                    # Show drift if we have previous messages
                    if len(conversation_messages) > 1:
                        prev_emotion, _, _ = predict_emotion(
                            model, 
                            conversation_messages[-2]['text'], 
                            tokenizer, 
                            args.device, 
                            emotion_classes
                        )
                        if prev_emotion != emotion:
                            emotion_to_idx = {e: i for i, e in enumerate(emotion_classes)}
                            prev_idx = emotion_to_idx[prev_emotion]
                            curr_idx = emotion_to_idx[emotion]
                            drift_magnitude = abs(curr_idx - prev_idx)
                            
                            print(f"     [DRIFT DETECTED: {prev_emotion.upper()} → {emotion.upper()} (magnitude: {drift_magnitude})]")
                            if drift_magnitude >= 3:
                                print(f"     [WARNING] SIGNIFICANT EMOTION SHIFT!")
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[Error: {e}]")
                continue
        
        return
    
    elif args.text:
        # From command-line arguments
        for text in args.text:
            messages.append({'speaker': 'User', 'text': text})
    
    else:
        # Example conversation
        print("No input provided. Using example conversation...\n")
        messages = [
            {'speaker': 'User', 'text': 'I have a question about my account'},
            {'speaker': 'Assistant', 'text': 'Sure, I can help with that. What would you like to know?'},
            {'speaker': 'User', 'text': 'I noticed a charge I dont recognize'},
            {'speaker': 'Assistant', 'text': 'Let me check that for you. Can you provide more details?'},
            {'speaker': 'User', 'text': 'I already told you this twice. This is getting frustrating!'},
            {'speaker': 'Assistant', 'text': 'I apologize for the inconvenience. Let me escalate this right away.'},
            {'speaker': 'User', 'text': 'Thank you, I appreciate that'}
        ]
    
    # Analyze conversation
    emotions, confidences, drifts = analyze_conversation(
        model, messages, tokenizer, args.device, emotion_classes
    )
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()

