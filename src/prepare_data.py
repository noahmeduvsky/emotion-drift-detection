"""
Script to prepare and preprocess the dataset. Loads data, cleans it, and saves it for training.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_dailydialog_dataset, save_dataset
from preprocessing import EmotionPreprocessor, normalize_emotion_labels

def create_synthetic_dataset(num_dialogues=100, max_turns=8):
    """
    Creates a synthetic dataset when the real one isn't available. 
    Generates dialogues with emotion transitions for testing.
    """
    print("Creating synthetic dataset...")
    
    emotions = ['joy', 'anger', 'sadness', 'neutral', 'fear', 'surprise', 'disgust']
    
    # templates for generating text based on emotions
    templates = {
        'joy': [
            "That's great news!",
            "I'm so happy to hear that!",
            "Wonderful! Thank you so much!",
            "This is exactly what I needed!",
        ],
        'anger': [
            "This is unacceptable!",
            "I'm very frustrated with this!",
            "This is ridiculous!",
            "I can't believe this happened!",
        ],
        'sadness': [
            "I'm really disappointed.",
            "This is very upsetting.",
            "I feel really down about this.",
            "This makes me sad.",
        ],
        'neutral': [
            "I need help with my order.",
            "Can you assist me with this?",
            "I have a question about my account.",
            "Thank you for your help.",
        ],
        'fear': [
            "I'm worried about this.",
            "This is concerning to me.",
            "I'm afraid this might be a problem.",
            "This makes me anxious.",
        ],
        'surprise': [
            "Really? I didn't expect that!",
            "Wow, that's surprising!",
            "I'm shocked to hear that!",
            "That's unexpected!",
        ],
        'disgust': [
            "That's terrible!",
            "I'm appalled by this.",
            "This is disgusting!",
            "I can't stand this!",
        ]
    }
    
    data = []
    
    for dialogue_id in range(num_dialogues):
        num_turns = np.random.randint(3, max_turns + 1)
        
        # create emotion trajectory (starts neutral, may drift)
        emotion_sequence = ['neutral']
        for i in range(1, num_turns):
            prev_emotion = emotion_sequence[-1]
            
            # 40% chance of emotion change (drift)
            if np.random.random() < 0.4:
                new_emotion = np.random.choice(emotions)
            else:
                # stay the same
                new_emotion = prev_emotion
            
            emotion_sequence.append(new_emotion)
        
        # Generate dialogue
        for turn_id, emotion in enumerate(emotion_sequence):
            speaker = 'user' if turn_id % 2 == 0 else 'agent'
            
            # pick a template based on the emotion
            template_list = templates.get(emotion, templates['neutral'])
            text = np.random.choice(template_list)
            
            # add some variation to make it look more realistic
            if turn_id > 0:
                text = f"{text} (turn {turn_id + 1})"
            
            data.append({
                'dialogue_id': dialogue_id,
                'turn_id': turn_id,
                'speaker': speaker,
                'text': text,
                'emotion': emotion
            })
    
    df = pd.DataFrame(data)
    print(f"Created {len(df)} rows from {num_dialogues} dialogues")
    return df

def main():
    print("="*60)
    print("DATA PREPARATION SCRIPT")
    print("="*60)
    
    # Try loading real dataset
    print("\n[Step 1] Attempting to load DailyDialog dataset...")
    df = load_dailydialog_dataset(split="train")
    
    if df is None or df.empty:
        print("[INFO] Real dataset unavailable, creating synthetic dataset...")
        df = create_synthetic_dataset(num_dialogues=200, max_turns=8)
    else:
        print(f"[OK] Loaded {len(df)} rows from {df['dialogue_id'].nunique()} dialogues")
        # Use subset for faster processing
        if len(df) > 1000:
            print(f"[INFO] Using subset of {1000} rows for faster processing")
            df = df.head(1000)
    
    # Normalize emotions
    print("\n[Step 2] Normalizing emotion labels...")
    df_norm = normalize_emotion_labels(df)
    print(f"[OK] Normalized emotions: {df_norm['emotion'].unique()}")
    
    # Initialize preprocessor
    print("\n[Step 3] Initializing preprocessor...")
    preprocessor = EmotionPreprocessor(
        model_name="bert-base-uncased",
        max_length=128
    )
    
    # Prepare sequences
    print("\n[Step 4] Preparing sequences...")
    dialogues, emotions, metadata = preprocessor.prepare_sequences(df_norm)
    
    print(f"[OK] Prepared {metadata['num_dialogues']} dialogue sequences")
    print(f"[OK] Emotion classes: {metadata['emotion_classes']}")
    
    # Save processed data
    print("\n[Step 5] Saving processed data...")
    os.makedirs("data", exist_ok=True)
    
    # Save raw dataframe
    df_norm.to_csv("data/processed_data.csv", index=False)
    print(f"[OK] Saved raw data to data/processed_data.csv")
    
    # Save metadata
    import json
    with open("data/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved metadata to data/metadata.json")
    
    print("\n" + "="*60)
    print("[SUCCESS] Data preparation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train model: python -m src.main train")
    print("2. Check data: python -c \"import pandas as pd; df=pd.read_csv('data/processed_data.csv'); print(df.head())\"")

if __name__ == "__main__":
    main()

