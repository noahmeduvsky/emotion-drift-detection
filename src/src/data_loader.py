"""
Data loading utilities for emotion drift detection project.
Supports EmotionLines, DailyDialog, and MELD datasets.
"""

import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Optional
import os
import ast
import numpy as np


def load_dailydialog_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load DailyDialog dataset from local CSV file.
    
    Args:
        filepath: Path to CSV file (train.csv, test.csv, or validation.csv)
    
    Returns:
        DataFrame with columns: dialogue_id, turn_id, speaker, text, emotion
    """
    print(f"Loading DailyDialog from {filepath}...")
    
    df = pd.read_csv(filepath)
    
    data = []
    dialogue_id = 0
    
    # Emotion mapping from DailyDialog
    emotion_map = {0: 'no_emotion', 1: 'anger', 2: 'disgust', 
                  3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'}
    
    for idx, row in df.iterrows():
        # Parse dialog string to list
        try:
            # The dialog is stored as a string representation of a list
            dialog_str = row['dialog']
            if isinstance(dialog_str, str):
                # Remove outer quotes if present
                dialog_str = dialog_str.strip()
                if dialog_str.startswith('"') and dialog_str.endswith('"'):
                    dialog_str = dialog_str[1:-1]
                
                # Use regex to extract all quoted strings (handles multi-line format)
                import re
                # Pattern matches: 'text', "text", ''text'', or ""text"" (handles nested quotes)
                # The DOTALL flag allows . to match newlines
                matches = re.findall(r"(?:''|\"\"|'|\")(.*?)(?:''|\"\"|'|\")", dialog_str, re.DOTALL)
                dialog_list = [m.strip() for m in matches if m.strip()]
                
                # If regex didn't work, try ast.literal_eval as fallback
                if not dialog_list:
                    try:
                        dialog_str_clean = dialog_str.replace('\n', ' ').replace('\r', ' ')
                        parsed = ast.literal_eval(dialog_str_clean)
                        if isinstance(parsed, list):
                            dialog_list = parsed
                    except:
                        pass
            else:
                dialog_list = dialog_str
            
            # Parse emotion string to list
            emotion_str = row['emotion']
            if isinstance(emotion_str, str):
                # Handle format like "[0 0 0 0 0 0 4 4 4 4]"
                emotion_str = emotion_str.strip()
                if emotion_str.startswith('[') and emotion_str.endswith(']'):
                    emotion_str = emotion_str[1:-1]
                # Split by space and convert to int
                emotion_list = [int(x) for x in emotion_str.split() if x.strip()]
            else:
                emotion_list = emotion_str
            
            # Ensure dialog and emotion lists have same length
            min_len = min(len(dialog_list), len(emotion_list))
            dialog_list = dialog_list[:min_len]
            emotion_list = emotion_list[:min_len]
            
            # Skip if no valid turns
            if min_len == 0:
                continue
            
            # Create rows for each turn in this dialogue
            for turn_id, (text, emotion_idx) in enumerate(zip(dialog_list, emotion_list)):
                # Clean text
                text = str(text).strip()
                # Remove extra quotes
                if text.startswith("'") and text.endswith("'"):
                    text = text[1:-1]
                elif text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                
                # Map emotion index to label
                emotion_label = emotion_map.get(emotion_idx, 'no_emotion')
                
                # Determine speaker (alternating)
                speaker = 'user' if turn_id % 2 == 0 else 'agent'
                
                data.append({
                    'dialogue_id': dialogue_id,
                    'turn_id': turn_id,
                    'speaker': speaker,
                    'text': text,
                    'emotion': emotion_label
                })
            
            dialogue_id += 1
            
        except Exception as e:
            print(f"Error parsing row {idx}: {e}")
            continue
    
    if len(data) == 0:
        print("No data loaded from CSV file")
        return None
    
    result_df = pd.DataFrame(data)
    print(f"Loaded {len(result_df)} turns from {dialogue_id} dialogues")
    return result_df


def load_dailydialog_dataset(split: str = "train") -> pd.DataFrame:
    """
    Load DailyDialog dataset from Hugging Face or local CSV files.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
    
    Returns:
        DataFrame with columns: dialogue_id, turn_id, speaker, text, emotion
    """
    # First, try loading from local CSV files
    csv_file = None
    if split == "train":
        csv_file = "train.csv"
    elif split == "validation":
        csv_file = "validation.csv"
    elif split == "test":
        csv_file = "test.csv"
    
    # Check in project root and src directory
    possible_paths = [
        csv_file,
        os.path.join("..", csv_file),
        os.path.join("src", "..", csv_file),
        os.path.join(os.path.dirname(__file__), "..", "..", csv_file)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return load_dailydialog_from_csv(path)
    
    # If local files not found, try Hugging Face
    try:
        # Try loading with trust_remote_code for newer dataset versions
        try:
            dataset = load_dataset("daily_dialog", split=split, trust_remote_code=True)
        except:
            # Fallback to older method
            dataset = load_dataset("daily_dialog", split=split)
        
        data = []
        dialogue_id = 0
        
        for example in dataset:
            # Handle different possible field names
            dialogues = example.get('dialog', example.get('dialogue', []))
            emotions = example.get('emotions', example.get('emotion', []))
            
            if not dialogues or not emotions:
                continue
            
            for turn_id, (text, emotion) in enumerate(zip(dialogues, emotions)):
                # Map emotion index to label
                emotion_map = {0: 'no_emotion', 1: 'anger', 2: 'disgust', 
                              3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'}
                emotion_label = emotion_map.get(emotion, 'no_emotion')
                
                data.append({
                    'dialogue_id': dialogue_id,
                    'turn_id': turn_id,
                    'speaker': 'user' if turn_id % 2 == 0 else 'agent',
                    'text': text,
                    'emotion': emotion_label
                })
            
            dialogue_id += 1
        
        if len(data) == 0:
            print("No data loaded from DailyDialog dataset")
            return None
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error loading DailyDialog: {e}")
        print("Dataset format may have changed. Trying alternative loading method...")
        return None


def load_emotionlines_dataset(split: str = "train") -> pd.DataFrame:
    """
    Load EmotionLines dataset from Hugging Face.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
    
    Returns:
        DataFrame with columns: dialogue_id, turn_id, speaker, text, emotion
    """
    try:
        dataset = load_dataset("emotionlines", split=split)
        
        # Convert to list of dictionaries
        data = []
        for example in dataset:
            # EmotionLines structure may vary, adjust based on actual format
            data.append({
                'dialogue_id': example.get('dialogue_id', ''),
                'turn_id': example.get('turn_id', 0),
                'speaker': example.get('speaker', 'user'),
                'text': example.get('text', example.get('utterance', '')),
                'emotion': example.get('emotion', example.get('emotion_label', 'neutral'))
            })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error loading EmotionLines: {e}")

        # Fallback: try loading from local file if available
        return None


def load_meld_dataset(split: str = "train") -> pd.DataFrame:
    """
    Load MELD (Multimodal EmotionLines Dataset) from Hugging Face.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
    
    Returns:
        DataFrame with columns: dialogue_id, turn_id, speaker, text, emotion
    """
    try:
        # MELD might be available under different names on Hugging Face
        # Adjust dataset name based on actual availability
        dataset = load_dataset("meld", split=split)
        
        data = []
        for example in dataset:
            data.append({
                'dialogue_id': example.get('Dialogue_ID', example.get('dialogue_id', '')),
                'turn_id': example.get('Utterance_ID', example.get('turn_id', 0)),
                'speaker': example.get('Speaker', example.get('speaker', 'user')),
                'text': example.get('Utterance', example.get('text', '')),
                'emotion': example.get('Emotion', example.get('emotion', 'neutral'))
            })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error loading MELD: {e}")
        print("MELD dataset may need to be downloaded separately.")
        return None


def combine_datasets(dataframes: List[pd.DataFrame], source_names: List[str] = None) -> pd.DataFrame:
    """
    Combine multiple datasets into a single DataFrame.
    
    Args:
        dataframes: List of DataFrames to combine
        source_names: Optional list of source names to add as a column
    
    Returns:
        Combined DataFrame with an optional 'source' column
    """
    if not dataframes:
        return pd.DataFrame()
    
    # Filter out None values
    valid_dfs = [df for df in dataframes if df is not None and not df.empty]
    
    if not valid_dfs:
        return pd.DataFrame()
    
    if source_names:
        for idx, df in enumerate(valid_dfs):
            if idx < len(source_names):
                df = df.copy()
                df['source'] = source_names[idx]
                valid_dfs[idx] = df
    
    combined = pd.concat(valid_dfs, ignore_index=True)
    return combined


def save_dataset(df: pd.DataFrame, filepath: str):
    """
    Save dataset to CSV or pickle file.
    
    Args:
        df: DataFrame to save
        filepath: Output file path (.csv or .pkl)
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    if filepath.endswith('.csv'):
        df.to_csv(filepath, index=False)
    elif filepath.endswith('.pkl'):
        df.to_pickle(filepath)
    else:
        raise ValueError("Filepath must end with .csv or .pkl")
    
    print(f"Dataset saved to {filepath}")


def load_local_dataset(filepath: str) -> pd.DataFrame:
    """
    Load dataset from local CSV or pickle file.
    
    Args:
        filepath: Path to dataset file
    
    Returns:
        Loaded DataFrame
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.pkl'):
        return pd.read_pickle(filepath)
    else:
        raise ValueError("File must be .csv or .pkl format")


if __name__ == "__main__":
    # Example usage: Load and combine datasets
    print("Loading datasets...")
    
    daily_dialog_train = load_dailydialog_dataset(split="train")
    if daily_dialog_train is not None:
        print(f"DailyDialog train: {len(daily_dialog_train)} rows")
        print(daily_dialog_train.head())
        
        # Save to local storage
        save_dataset(daily_dialog_train, "data/dailydialog_train.csv")
