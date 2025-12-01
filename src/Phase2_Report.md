# Phase 2 Report: Emotion Drift in Customer Support AI

## 1. Project Identification

**Project Title:** Emotion Drift in Customer Support AI

**Student:** Noah Meduvsky

## 2. Groups

- **Noah Meduvsky** - noahmeduvsky@oakland.edu

## 3. Data Availability

### Access to Data

This project uses publicly available conversation datasets that contain multi-turn dialogues with emotion labels for each message. These datasets are not specifically labeled for emotion drift, but they provide emotion annotations that allow the project to detect and analyze emotion drift patterns across dialogue sequences. The main datasets include:

1. **DailyDialog**: Available through the Hugging Face datasets library
   - Contains daily conversations with emotion labels for each message
   - Emotions include: no emotion, anger, disgust, fear, happiness, sadness, surprise

2. **EmotionLines**: Available through Hugging Face or GitHub repositories
   - Contains multi-party conversations with emotion annotations
   - Provides rich emotional context across dialogue turns

3. **MELD (Multimodal EmotionLines Dataset)**: Available through research repositories
   - Extends EmotionLines with additional features
   - Includes textual emotion labels suitable for this project

### Data Access Implementation

The data loading functionality provides automated dataset downloading through the Hugging Face datasets library. The system stores data locally to avoid repeated downloads, combines multiple datasets to increase sample diversity, and can export data to common file formats for offline processing.

### Data Challenges and Solutions

**Challenge 1: Label Imbalance**
- Some emotions such as anger and sadness may be less common compared to neutral emotions
- **Solution**: Use techniques to balance the data by giving more weight to underrepresented emotions and creating additional examples of rare emotions

**Challenge 2: Long-term Emotional Dependencies**
- Emotional context spans across multiple dialogue turns
- **Solution**: Use sequence models that can remember and use information from earlier parts of the conversation

**Challenge 3: Dataset Format Variations**
- Different datasets use different emotion label schemes
- **Solution**: Create a system to map all datasets to a consistent set of emotion categories

## 4. Data Preparation

### Data Structures

The project works with sequential text data in the form of multi-turn dialogues. Each dataset is structured with the following information:

- **Dialogue ID**: A unique identifier for each conversation
- **Turn ID**: The sequential turn number within the dialogue
- **Speaker**: Who is speaking (user, agent, or AI)
- **Text**: The actual message content
- **Emotion**: The emotion label for that message

### Data Cleaning

The preprocessing pipeline implements the following cleaning steps:

1. **Text Normalization**:
   - Removal of URLs and email addresses
   - Normalizing whitespace
   - Option to convert text to lowercase

2. **Missing Value Handling**:
   - Empty text fields are replaced with empty strings
   - Missing emotion labels are set to neutral

3. **Special Character Handling**:
   - Standard text cleaning while preserving punctuation that may carry emotional meaning

### Data Transformation

1. **Tokenization**: 
   - Converts text into numerical representations that machine learning models can understand
   - Handles variable-length messages by padding shorter messages and truncating longer ones
   - Maximum message length is set to 128 tokens

2. **Emotion Label Encoding**:
   - Converts emotion words into numbers
   - Maintains a mapping to convert back to emotion words when needed
   - Normalizes emotion labels across datasets to a consistent set of emotions:
     - joy, anger, sadness, fear, disgust, surprise, neutral

3. **Sequence Preparation**:
   - Groups messages by conversation to maintain context
   - Sorts messages by turn order to preserve the conversation timeline
   - Creates sequences suitable for sequence model input

### Data Normalization

1. **Emotion Label Normalization**:
   - Standardizes emotion labels across different datasets
   - Maps variations such as "happy" and "happiness" to "joy"
   - Ensures consistent emotion categories across all datasets

2. **Class Balancing** (Optional):
   - Creates additional examples for emotions that are less common
   - Adjusts weights to give more importance to underrepresented emotions
   - Can also reduce examples of overrepresented emotions as an alternative approach

### Data Augmentation

For future phases, potential strategies to increase data diversity include:
- Replacing words with synonyms to increase text variety
- Paraphrasing text while preserving emotional content
- Creating synthetic examples of emotion shifts for training drift detection

## 5. Design Specifications

### ML Pipeline Architecture

The complete machine learning pipeline consists of the following stages:

**Stage 1: Data Collection**
- Download datasets from the Hugging Face datasets library
- Load datasets using automated download and caching mechanisms
- Store processed data locally in common file formats for offline processing

**Stage 2: Preprocessing**
- Text Cleaning: Remove URLs, normalize whitespace, handle special characters
- Tokenization: Convert text to numerical representations
- Emotion Label Normalization: Map emotion labels across datasets to consistent categories
- Sequence Grouping: Group messages by conversation to maintain context
- Class Balancing: Apply techniques to balance emotions if needed

**Stage 3: Feature Extraction**
- Generate word representations that capture the meaning and context of words
- Encode sequences while maintaining the order of dialogue turns
- Format input for machine learning models

**Stage 4: Model Training**
- Base Models: Train two types of neural network architectures
- Emotion Classification: Add layers to predict emotion labels
- Sequence Prediction: Predict emotions for each turn in the dialogue
- Drift Detection: Analyze emotion transitions to detect significant shifts

**Stage 5: Evaluation**
- Emotion Classification Metrics: Calculate how well the model predicts emotions
- Drift Detection Metrics: Measure how well the model detects emotion changes
- Visualization: Create graphs showing emotion trajectories and transition patterns

**Stage 6: Visualization & Analysis**
- Plot emotion trajectories across conversations to visualize emotional progression
- Highlight drift points where emotions shift significantly, such as from positive to negative
- Analyze the relationship between AI responses and subsequent emotion changes
- Generate insights for improving AI response strategies based on detected patterns

### Pipeline Stages

1. **Data Collection**:
   - Input: Dataset identifiers
   - Output: Raw data with dialogue ID, turn ID, speaker, text, and emotion
   - Tools: Hugging Face datasets library and data manipulation tools

2. **Preprocessing**:
   - Input: Raw dialogue data
   - Output: Cleaned and converted sequences with encoded emotions
   - Tools: Text processing libraries and encoding tools

3. **Feature Extraction**:
   - Input: Converted text sequences
   - Output: Numerical representations of words that capture meaning
   - Tools: Pre-trained language models

4. **Model Training**:
   - Input: Numerical representations and emotion labels
   - Output: Trained model capable of emotion classification and drift detection
   - Tools: Deep learning frameworks

5. **Evaluation**:
   - Input: Model predictions and correct emotion labels
   - Output: Performance metrics and visualizations
   - Tools: Evaluation metrics and visualization libraries

### Input/Output Structures

**Input Structure:**
- Text sequences converted to numbers
- Information about which parts of the text are real content versus padding
- Dialogue metadata for grouping sequences

**Output Structure:**
- Emotion predictions with confidence scores
- Predicted emotions for each message
- Drift scores showing the magnitude of emotion changes

## 6. Approach

### ML Approach: Supervised Learning

This project uses supervised learning to solve the emotion drift detection problem. The task is set up as:

- **Input**: Sequences of dialogue messages with speaker information
- **Output**: Emotion labels for each message, enabling drift detection through sequence analysis

### ML Algorithms/Models

The project implements and compares two primary model architectures:

1. **Bidirectional LSTM**:
   - A type of neural network that processes sequences in both forward and backward directions
   - Uses pre-computed word representations
   - Has multiple layers to capture complex patterns
   - Captures how emotions change over time in conversations while being computationally efficient

2. **Transformer-based Model**:
   - Uses pre-trained language models that understand context and meaning
   - Fine-tuned for emotion prediction
   - Predicts emotions at the message level
   - Leverages pre-trained word representations that capture nuanced emotional cues

3. **Emotion Drift Detector**:
   - A component that compares consecutive emotion predictions
   - Calculates drift scores based on how much emotions change
   - Identifies significant emotion shifts based on a configurable threshold

### Justification

**Why Supervised Learning?**
- Emotion labels are available in the datasets, enabling supervised training
- Direct mapping from text to emotions allows for interpretable results
- Having correct labels enables quantitative evaluation of drift detection accuracy

**Why Sequence Models?**
- Emotions in dialogue depend on context; previous messages influence current emotion
- Sequential modeling captures how emotions evolve across the conversation timeline
- Enables detection of emotional progression and regression patterns

**Why Pre-trained Word Representations?**
- Pre-trained language models capture rich semantic and emotional information
- Contextual word representations understand emotional nuances better than simple word matching
- Using pre-trained models reduces data requirements and improves generalization

**Why Compare LSTM vs Transformer?**
- LSTM provides a baseline for sequential modeling
- Transformer models leverage pre-training but require more computational resources
- Comparison identifies the best trade-off between accuracy and efficiency

## 7. Technical Design

### ML Libraries and Frameworks

1. **PyTorch**:
   - Primary deep learning framework
   - Used for model definition, training, and numerical operations
   - Provides flexibility for custom architectures

2. **Transformers Library**:
   - Provides pre-trained language models
   - Text preprocessing tools
   - Model loading and fine-tuning utilities

3. **Datasets Library**:
   - Dataset loading and caching
   - Efficient data streaming for large datasets

4. **scikit-learn**:
   - Label encoding for emotion classes
   - Class weight computation for imbalanced data
   - Evaluation metrics

5. **Pandas and NumPy**:
   - Data manipulation and preprocessing
   - Numerical computations

6. **Matplotlib and Seaborn**:
   - Visualization of emotion trajectories
   - Plotting confusion matrices and transition heatmaps

### Development Tools

1. **Jupyter Notebooks**:
   - Interactive data exploration
   - Prototyping and visualization
   - Reproducible analysis workflows

2. **Version Control (Git)**:
   - Track code changes and experiment iterations
   - Collaborate and maintain project history
   - Configured to ignore data and model files

3. **Experiment Tracking** (Optional - Future):
   - Tools for experiment logging
   - Track model settings, metrics, and versions
   - Compare different model configurations

### Project Structure

The project is organized into the following directories:
- Data directory for raw and processed datasets
- Models directory for trained model checkpoints
- Notebooks directory for interactive data exploration
- Source code directory with modules for data loading, preprocessing, and model architectures
- Configuration files for dependencies and documentation

## 8. Experimental Setup

### Research Question

**Primary Research Question:**
"Can an emotion drift detection model identify when user emotion shifts during a customer support AI conversation, and does the AI's response correlate with that shift?"

**Secondary Questions:**
- Which model architecture better captures emotion drift patterns?
- What are the most common emotion transition patterns in customer support dialogues?
- Can we predict emotion drift before it occurs to enable proactive response adjustment?

### Experimental Comparisons

The project will compare:

1. **Model Architectures**:
   - Bidirectional LSTM: A baseline sequential model
   - Transformer: A pre-trained model with fine-tuning
   - Comparison metrics: Emotion classification accuracy, drift detection precision and recall, computational efficiency

2. **Embedding Strategies** (Future):
   - Simple word embeddings
   - Contextual word embeddings
   - Compare impact on emotion classification performance

3. **Sequence Modeling Approaches**:
   - One-directional versus bidirectional processing
   - Different sequence lengths and context windows
   - Impact on capturing long-term emotion dependencies

### Evaluation Metrics

**1. Emotion Classification Metrics:**

- **F1-Score** (Primary):
  - Overall F1 score across all emotion classes
  - Per-class F1 to identify class-specific performance
  - Handles imbalanced emotion distribution

- **Accuracy**:
  - Overall classification accuracy
  - Percentage of correctly predicted emotions

- **Confusion Matrix**:
  - Visual representation of classification errors
  - Identifies common misclassification patterns, such as confusing anger with sadness

**2. Emotion Drift Detection Metrics:**

- **Average Sentiment Shift**:
  - Mean magnitude of sentiment change between consecutive turns
  - Calculated as the average absolute difference between consecutive sentiment values
  - Higher values indicate more volatile emotional trajectories

- **Drift Detection Accuracy**:
  - Precision: Proportion of detected drifts that are actual emotion changes
  - Recall: Proportion of actual drifts that are successfully detected
  - F1-score for drift detection binary classification

- **Emotion Transition Matrix**:
  - Frequency matrix of emotion-to-emotion transitions
  - Identifies common drift patterns, such as neutral to anger, or joy to neutral

**3. Trajectory Analysis Metrics:**

- **Emotion Stability Score**:
  - Variance of emotions within a dialogue
  - Lower variance indicates more stable emotional states

- **Drift Correlation**:
  - Correlation between AI response features and subsequent emotion changes
  - Measures how well AI responses predict emotion shifts

### Quantitative Criteria

- **Classification Performance**: Target F1-score greater than 0.70 for emotion classification
- **Drift Detection**: Precision greater than 0.75 for significant emotion shifts
- **Computational Efficiency**: Training time per epoch less than 30 minutes on standard GPU
- **Generalization**: Validation accuracy within 5% of training accuracy to avoid overfitting

### Experimental Procedure

1. **Data Split**:
   - Training: 70% of dialogues
   - Validation: 15% of dialogues
   - Test: 15% of dialogues
   - Stratified split to maintain emotion distribution

2. **Training Configuration**:
   - Batch size: 16-32 depending on model and available resources
   - Learning rate: Different rates for different model types
   - Optimizer: AdamW with weight decay
   - Early stopping: Stop if validation F1 doesn't improve for 5 epochs

3. **Baseline Comparisons**:
   - Majority class baseline: Always predict the most common emotion
   - Random baseline: Random emotion prediction
   - Simple baseline: Basic model using word frequency features

## 9. References

1. Zhang, J., et al. (2024). The Impact of Emotional Expression by Artificial Intelligence. *Decision Support Systems*, 181, 114075.

2. Pezenka, I., et al. (2024). Emotionality in Task-Oriented Chatbots – The Effect of Emotion Expression on Customer Satisfaction. *Journal of Applied Communication Research*.

3. Han, E., et al. (2022). Should AI Agents Express Positive Emotion in Customer Service? *SSRN*.

4. Yun, J., et al. (2022). The Effects of Chatbot Service Recovery With Emotion on Customer Satisfaction and Trust. *Frontiers in Psychology*.

5. Gupta, R., Ranjan, S., & Singh, A. (2024). Comprehensive Study on Sentiment Analysis: From Rule-Based to Modern LLM-Based Systems. *arXiv preprint arXiv:2409.09989*.

6. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*, 4171-4186.

7. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv preprint arXiv:1907.11692*.

8. Hsu, C. C., Chen, S. Y., Kuo, C. C., Huang, T. H., & Ku, L. W. (2018). EmotionLines: An Emotion Corpus of Multi-Party Conversations. *Proceedings of LREC*, 1597-1601.

9. Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. *Proceedings of IJCNLP*, 986-995.

10. Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2019). MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. *Proceedings of ACL*, 527-536.
