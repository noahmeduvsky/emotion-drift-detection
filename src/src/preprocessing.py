"""
Preprocessing pipeline. Handles text cleaning, tokenization, and getting everything ready for training.
"""

import pandas as pd
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer
from typing import List, Tuple, Dict, Optional
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


class EmotionPreprocessor:
    """
    Preprocessor that handles all the data cleaning and tokenization.
    """
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 max_length: int = 128,
                 lowercase: bool = True):
        """
        Sets up the preprocessor with a tokenizer.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.lowercase = lowercase
        
        # load the right tokenizer
        if "roberta" in model_name.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # for encoding emotion labels to numbers
        self.label_encoder = LabelEncoder()
        self.emotion_mapping = {}
    
    def clean_text(self, text: str) -> str:
        """
        Cleans text by removing URLs, emails, and extra whitespace.
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # lowercase if needed
        if self.lowercase:
            text = text.lower()
        
        # trim whitespace
        text = text.strip()
        
        return text
    
    def preprocess_texts(self, texts: List[str]) -> Dict:
        """
        Tokenizes a list of texts. Returns input_ids and attention_mask.
        """
        # clean first
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # tokenize
        encoded = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return encoded
    
    def encode_emotions(self, emotions: List[str]) -> np.ndarray:
        """
        Converts emotion labels to numbers. Stores the mapping so we can convert back later.
        """
        encoded = self.label_encoder.fit_transform(emotions)
        
        # save the mapping
        self.emotion_mapping = {
            label: idx for idx, label in enumerate(self.label_encoder.classes_)
        }
        
        return encoded
    
    def prepare_sequences(self, 
                         df: pd.DataFrame,
                         text_col: str = 'text',
                         emotion_col: str = 'emotion',
                         dialogue_col: str = 'dialogue_id') -> Tuple[List[Dict], List[np.ndarray], Dict]:
        """
        Prepares dialogue sequences for training. Groups by dialogue_id and tokenizes everything.
        """
        # group by dialogue to create sequences
        dialogues = []
        dialogue_emotions = []
        
        for dialogue_id, group in df.groupby(dialogue_col):
            # sort by turn_id to keep the order right
            group = group.sort_values('turn_id')
            
            texts = group[text_col].tolist()
            emotions = group[emotion_col].tolist()
            
            # tokenize the texts
            encoded = self.preprocess_texts(texts)
            
            dialogues.append({
                'dialogue_id': dialogue_id,
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
            })
            
            dialogue_emotions.append(emotions)
        
        # encode all emotions to numbers
        all_emotions = [emotion for emotions in dialogue_emotions for emotion in emotions]
        encoded_emotions = self.encode_emotions(all_emotions)
        
        # reshape to match dialogue structure
        emotion_idx = 0
        encoded_dialogue_emotions = []
        for emotions in dialogue_emotions:
            encoded_dialogue_emotions.append(
                encoded_emotions[emotion_idx:emotion_idx + len(emotions)]
            )
            emotion_idx += len(emotions)
        
        metadata = {
            'num_dialogues': len(dialogues),
            'emotion_classes': self.label_encoder.classes_.tolist(),
            'emotion_mapping': self.emotion_mapping
        }
        
        return dialogues, encoded_dialogue_emotions, metadata
    
    def compute_class_weights(self, emotions: List[str]) -> np.ndarray:
        """
        Computes class weights for imbalanced data. Helps the model learn from rare classes.
        """
        encoded = self.label_encoder.transform(emotions)
        unique_classes = np.unique(encoded)
        
        weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=encoded
        )
        
        return weights
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size of the tokenizer."""
        return len(self.tokenizer.vocab)


def normalize_emotion_labels(df: pd.DataFrame, emotion_col: str = 'emotion') -> pd.DataFrame:
    """
    Normalizes emotion labels. Different datasets use different names, so I map them all to the same set.
    """
    df = df.copy()
    
    # map different emotion names to standard ones
    emotion_mapping = {
        'joy': 'joy',
        'happy': 'joy',
        'happiness': 'joy',
        'anger': 'anger',
        'angry': 'anger',
        'sadness': 'sadness',
        'sad': 'sadness',
        'fear': 'fear',
        'disgust': 'disgust',
        'surprise': 'surprise',
        'neutral': 'neutral',
        'no_emotion': 'neutral'
    }
    
    df[emotion_col] = df[emotion_col].str.lower().map(
        lambda x: emotion_mapping.get(x, 'neutral')
    )
    
    return df


def balance_dataset(df: pd.DataFrame, 
                   emotion_col: str = 'emotion',
                   method: str = 'oversample') -> pd.DataFrame:
    """
    Balances the dataset by oversampling or undersampling. I usually use oversample.
    """
    from sklearn.utils import resample
    
    emotion_counts = df[emotion_col].value_counts()
    max_count = emotion_counts.max()
    min_count = emotion_counts.min()
    
    balanced_dfs = []
    
    for emotion in df[emotion_col].unique():
        emotion_df = df[df[emotion_col] == emotion]
        
        if method == 'oversample':
            # Oversample minority classes
            n_samples = max_count
            if len(emotion_df) < n_samples:
                emotion_df = resample(
                    emotion_df,
                    replace=True,
                    n_samples=n_samples,
                    random_state=42
                )
        elif method == 'undersample':
            # Undersample majority classes
            n_samples = min_count
            if len(emotion_df) > n_samples:
                emotion_df = resample(
                    emotion_df,
                    replace=False,
                    n_samples=n_samples,
                    random_state=42
                )
        
        balanced_dfs.append(emotion_df)
    
    return pd.concat(balanced_dfs, ignore_index=True)


if __name__ == "__main__":
    # Example usage
    print("Testing preprocessing pipeline...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'dialogue_id': [0, 0, 0, 1, 1],
        'turn_id': [0, 1, 2, 0, 1],
        'speaker': ['user', 'agent', 'user', 'user', 'agent'],
        'text': [
            "Hello, I need help",
            "Of course! How can I assist you?",
            "My order hasn't arrived",
            "This is frustrating",
            "I understand your frustration"
        ],
        'emotion': ['neutral', 'joy', 'anger', 'anger', 'neutral']
    })
    
    # Initialize preprocessor
    preprocessor = EmotionPreprocessor()
    
    # Normalize labels
    sample_data = normalize_emotion_labels(sample_data)
    
    # Prepare sequences
    dialogues, emotions, metadata = preprocessor.prepare_sequences(sample_data)
    
    print(f"Processed {metadata['num_dialogues']} dialogues")
    print(f"Emotion classes: {metadata['emotion_classes']}")

