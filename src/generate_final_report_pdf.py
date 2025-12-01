"""
Generate final project report as PDF with all actual results, tables, and images.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os
import json
from datetime import datetime

def load_metrics(model_path):
    """Load metrics from a model directory."""
    metrics_file = os.path.join(model_path, 'results', 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def create_pdf_report():
    """Create the final PDF report with all results."""
    
    pdf_path = "Final_Project_Report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.black,
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=12,
        textColor=colors.black,
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.black,
        spaceAfter=6,
        spaceBefore=6,
        fontName='Helvetica-Bold'
    )
    
    normal_style = styles['Normal']
    normal_style.alignment = TA_LEFT
    normal_style.fontSize = 10
    
    # Load actual metrics data
    models_dir = "models"
    bert_baseline = load_metrics(os.path.join(models_dir, "bert_baseline"))
    bert_real = load_metrics(os.path.join(models_dir, "bert_real"))
    bert_weighted = load_metrics(os.path.join(models_dir, "bert_real_weighted"))
    
    # Title
    story.append(Paragraph("Emotion Drift Detection in Customer Support AI", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Author
    story.append(Paragraph("Noah Meduvsky", normal_style))
    story.append(Paragraph("noahmeduvsky@oakland.edu", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ABSTRACT - Expanded to 200-300 words with actual results
    story.append(Paragraph("<b>ABSTRACT</b>", heading_style))
    abstract_text = """
    I address the critical challenge of detecting emotion drift in customer support AI conversations. 
    Emotion drift occurs when a customer's emotional state shifts during a dialogue, such as from neutral to anger 
    or from frustration to satisfaction. I propose a transformer-based emotion drift detection system that processes 
    dialogue sequences to classify emotions and identify significant emotional transitions. My system utilizes 
    pre-trained language models (BERT and RoBERTa) fine-tuned on the DailyDialog dataset downloaded from Kaggle, containing 11,118 human-written 
    conversations with 87,396 dialogue turns across seven emotion classes. To address severe class imbalance where 
    neutral emotions comprise 84.2% of samples while rare emotions like fear represent only 0.2%, I implement class 
    balancing techniques including weighted cross-entropy loss. My weighted-loss BERT model achieves significant 
    improvements over the baseline: macro F1 score improves from 29.3% to 38.0% (30% improvement), drift detection 
    F1 improves from 36.8% to 47.1% (28% improvement), and drift recall increases from 28.7% to 56.9% (98% improvement). 
    Most critically, rare emotion detection dramatically improves: fear detection increases from 0% to 33.3% F1 score, 
    and sadness improves from 4.8% to 19.2% F1 score. These improvements demonstrate that class balancing techniques 
    effectively address imbalanced emotion datasets, enabling detection of rare but critical emotions essential for 
    identifying customer distress in support scenarios. My system provides actionable insights for improving AI customer 
    support interactions by enabling real-time emotion monitoring and proactive response adjustment based on detected 
    emotion transitions, ultimately contributing to improved customer satisfaction and reduced churn rates.
    """
    story.append(Paragraph(abstract_text.strip(), normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # 1. INTRODUCTION
    story.append(Paragraph("<b>1. INTRODUCTION</b>", heading_style))
    intro_text = """
    Customer support systems increasingly rely on AI-powered chatbots and virtual assistants to handle customer 
    interactions. These systems must not only understand customer intent but also recognize and respond to 
    emotional states. A critical challenge in this domain is detecting emotion drift—the phenomenon where a 
    customer's emotional state changes during the course of a conversation. Understanding these emotional 
    transitions is essential for maintaining customer satisfaction and preventing escalation of negative emotions.
    """
    story.append(Paragraph(intro_text.strip(), normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Problem Statement
    story.append(Paragraph("<b>Problem Statement:</b>", subheading_style))
    problem_text = """
    Can an emotion drift detection model identify when user emotion shifts during a customer support AI conversation, 
    and does the AI's response correlate with that shift? Current emotion recognition systems often focus on 
    classifying emotions in individual messages but fail to capture how emotions evolve across dialogue sequences. 
    Additionally, real-world emotion datasets exhibit severe class imbalance, with neutral emotions dominating 
    while rare but critical emotions (anger, fear, sadness) are underrepresented, leading to poor detection of 
    important emotional transitions.
    """
    story.append(Paragraph(problem_text.strip(), normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Existing Solutions
    story.append(Paragraph("<b>Existing Solutions:</b>", subheading_style))
    existing_text = """
    Previous work in emotion recognition has primarily focused on single-turn emotion classification using 
    pre-trained language models like BERT and RoBERTa. While these approaches achieve high accuracy on 
    balanced datasets, they struggle with imbalanced real-world data where neutral emotions comprise over 
    80% of samples. Existing sequence-based emotion models often use LSTM architectures but lack the 
    contextual understanding of transformer models. Most approaches do not explicitly address emotion drift 
    detection or analyze correlations between AI responses and emotion changes. Limitations include poor 
    performance on minority emotion classes, lack of drift detection capabilities, and insufficient handling 
    of class imbalance in real-world scenarios.
    """
    story.append(Paragraph(existing_text.strip(), normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Proposed Solution - WITH ACTUAL RESULTS
    story.append(Paragraph("<b>Proposed Solution:</b>", subheading_style))
    solution_text = """
    In summary, this project makes the following contributions:
    <br/><br/>
    • I propose a transformer-based emotion drift detection system that processes dialogue sequences and 
      formulates emotion classification as a sequence-to-sequence task, enabling drift detection through 
      trajectory analysis.
    <br/><br/>
    • I introduce class balancing techniques including weighted cross-entropy loss and focal loss to 
      address severe class imbalance (422:1 ratio), significantly improving detection of rare emotions 
      such as fear and sadness.
    <br/><br/>
    • I evaluate my approach against baseline models on the DailyDialog dataset (11,118 dialogues, 87,396 turns), 
      demonstrating improved macro F1 scores and drift detection accuracy. My results show a 30% improvement 
      in macro F1 score (from 29.3% to 38.0%) and a 28% improvement in drift detection F1 (from 36.8% to 47.1%). 
      Most importantly, minority class detection dramatically improves: fear increases from 0% to 33.3% F1 score, 
      and sadness improves from 4.8% to 19.2% F1 score (+300% relative improvement).
    """
    story.append(Paragraph(solution_text.strip(), normal_style))
    story.append(PageBreak())
    
    # 2. RELATED WORK
    story.append(Paragraph("<b>2. RELATED WORK</b>", heading_style))
    
    related_work_items = [
        {
            "title": "BERT-based Emotion Recognition",
            "description": "Devlin et al. (2019) introduced BERT, which has been widely adopted for emotion classification tasks. Their work demonstrates the effectiveness of transformer architectures for understanding contextual emotion cues. My work differs by focusing on emotion drift detection across dialogue sequences rather than single-turn classification, and by explicitly addressing class imbalance through advanced loss functions.",
        },
        {
            "title": "RoBERTa for Emotion Analysis",
            "description": "Liu et al. (2019) proposed RoBERTa as an optimized variant of BERT. Their approach achieves strong performance on various NLP tasks. I leverage RoBERTa for emotion classification but extend it to sequence-level processing and integrate class balancing techniques not explored in their original work.",
        },
        {
            "title": "Focal Loss for Imbalanced Classification",
            "description": "Lin et al. (2017) introduced focal loss for addressing class imbalance in object detection. My approach adapts focal loss for emotion classification, demonstrating its effectiveness for handling imbalanced emotion datasets where rare emotions are critical for drift detection.",
        },
        {
            "title": "EmotionLines Dataset",
            "description": "Hsu et al. (2018) created EmotionLines, a multi-party conversation dataset with emotion annotations. While I utilize similar dialogue emotion datasets, my focus is on drift detection and analyzing correlations between AI responses and emotion changes, which distinguishes my work from standard emotion classification approaches.",
        }
    ]
    
    for item in related_work_items:
        story.append(Paragraph(f"<b>{item['title']}</b>", subheading_style))
        story.append(Paragraph(item['description'], normal_style))
        story.append(Spacer(1, 0.15*inch))
    
    story.append(PageBreak())
    
    # 3. A MOTIVATING EXAMPLE - WITH TABLE
    story.append(Paragraph("<b>3. A MOTIVATING EXAMPLE</b>", heading_style))
    example_text = """
    Consider a customer support interaction where a customer initially contacts support with a neutral 
    emotion, seeking help with a billing issue. As the conversation progresses, if the AI's responses 
    are unhelpful or fail to address the customer's concerns, the customer's emotion may drift from 
    neutral to frustration, then to anger. Conversely, if the AI provides clear, empathetic responses, 
    the emotion may improve from frustration to satisfaction. Detecting these transitions in real-time 
    enables the AI system to adjust its response strategy, potentially de-escalating negative emotions 
    or reinforcing positive ones. This is critical for maintaining customer satisfaction and preventing 
    churn in customer support scenarios.
    """
    story.append(Paragraph(example_text.strip(), normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Add motivating example table
    example_table_data = [
        ['Turn', 'Speaker', 'Text', 'Emotion', 'Drift Detected'],
        ['1', 'Customer', 'I have a question about my bill', 'Neutral', 'No'],
        ['2', 'AI', 'Sure, I can help with that', 'Neutral', 'No'],
        ['3', 'Customer', 'The amount seems incorrect', 'Neutral', 'No'],
        ['4', 'AI', 'Let me check your account', 'Neutral', 'No'],
        ['5', 'Customer', 'I already told you this twice', 'Anger', 'YES: Neutral to Anger'],
        ['6', 'AI', 'I apologize for the inconvenience', 'Neutral', 'No'],
        ['7', 'Customer', 'Thank you for fixing it', 'Joy', 'YES: Anger to Joy'],
    ]
    
    example_table = Table(example_table_data, colWidths=[0.6*inch, 0.8*inch, 2*inch, 0.8*inch, 1.2*inch])
    example_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(Paragraph("<b>Table 1: Example Dialogue Showing Emotion Drift</b>", subheading_style))
    story.append(example_table)
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("This example demonstrates two critical emotion drifts: (1) neutral to anger when the AI fails to address concerns, and (2) anger to joy when the issue is resolved. Detecting these transitions enables proactive response adjustment.", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # 4. APPROACH
    story.append(Paragraph("<b>4. APPROACH</b>", heading_style))
    
    # 4.1 Data Collection and Preprocessing
    story.append(Paragraph("<b>4.1 Data Collection and Preprocessing</b>", subheading_style))
    
    story.append(Paragraph("<b>Data Sources:</b>", normal_style))
    data_sources_text = """
    I utilize the DailyDialog dataset, a publicly available dialogue dataset containing 
    11,118 human-written conversations with 87,396 total dialogue turns. The dataset includes emotion 
    labels for each turn across seven emotion classes: anger, disgust, fear, joy, neutral, sadness, 
    and surprise. I downloaded the dataset from Kaggle (https://www.kaggle.com/datasets/thedevastator/dailydialog-multi-turn-dialog-with-intention-and) 
    and processed it locally to create dialogue sequences suitable for emotion drift detection.
    """
    story.append(Paragraph(data_sources_text.strip(), normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Preprocessing Steps:</b>", normal_style))
    preprocessing_text = """
    The preprocessing pipeline includes: (1) Emotion label normalization to map dataset-specific labels 
    to a standardized set of seven emotion classes, (2) Text cleaning and tokenization using BERT/RoBERTa 
    tokenizers with a maximum sequence length of 128 tokens, (3) Dialogue sequence preparation where 
    dialogues are structured as sequences of turns with speaker information, (4) Data splitting into 
    training (70%), validation (15%), and test (15%) sets while maintaining dialogue integrity, and 
    (5) Class distribution analysis revealing severe imbalance with neutral comprising 84.2% of samples 
    while fear represents only 0.2%.
    """
    story.append(Paragraph(preprocessing_text.strip(), normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # 4.2 Model Selection and Training
    story.append(Paragraph("<b>4.2 Model Selection and Training</b>", subheading_style))
    
    story.append(Paragraph("<b>Algorithms Used:</b>", normal_style))
    algorithms_text = """
    I implement and compare transformer-based models fine-tuned from pre-trained BERT-base 
    and RoBERTa-base architectures. I selected these models because they capture contextual emotional 
    cues effectively and can be fine-tuned for sequence-level emotion classification. My models process 
    dialogue sequences by encoding each turn using the transformer encoder and predicting emotion labels. 
    This architecture is suitable for the problem because it maintains context across dialogue turns and 
    can capture subtle emotional transitions. Additionally, I implement class balancing techniques 
    including weighted cross-entropy loss and focal loss to address dataset imbalance.
    """
    story.append(Paragraph(algorithms_text.strip(), normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Training Process:</b>", normal_style))
    training_text = """
    Models are trained using AdamW optimizer with learning rate 2e-5, batch size 2-8 (adjusted for 
    GPU memory constraints), weight decay 0.01, and dropout 0.3. Training uses early stopping based 
    on validation F1 score with patience of 10 epochs. Class weights are computed using sklearn's 
    balanced class weight method, providing inverse frequency weighting. The focal loss variant uses 
    gamma parameter 2.0 to focus learning on hard examples. Models are trained for 5 epochs with 
    checkpoint saving for best validation performance. Validation occurs after each epoch to monitor 
    overfitting and select optimal models.
    """
    story.append(Paragraph(training_text.strip(), normal_style))
    story.append(PageBreak())
    
    # 5. EXPERIMENTAL EVALUATION
    story.append(Paragraph("<b>5. EXPERIMENTAL EVALUATION</b>", heading_style))
    
    # 5.1 Methodology
    story.append(Paragraph("<b>5.1 Methodology</b>", subheading_style))
    
    methodology_text = """
    <b>Research Question:</b> Can an emotion drift detection model identify when user emotion shifts 
    during a customer support AI conversation, and does the AI's response correlate with that shift?<br/><br/>
    
    <b>Evaluation Criteria:</b> I evaluate model performance using multiple metrics: 
    (1) Macro F1 score to assess balanced performance across all emotion classes, (2) Weighted F1 
    score to account for class frequency, (3) Per-class F1 scores to identify performance on rare 
    emotions, (4) Drift detection precision, recall, and F1 to measure transition detection accuracy, 
    and (5) Trajectory stability metrics to analyze emotion patterns.<br/><br/>
    
    <b>Experimental Methodology:</b> The dependent variables are emotion classification accuracy, 
    drift detection accuracy, and per-class performance metrics. Independent variables include model 
    architecture (BERT vs. RoBERTa), loss function type (standard cross-entropy vs. weighted vs. 
    focal), and class balancing techniques. I use the DailyDialog dataset with 
    11,118 dialogues split 70/15/15, which is realistic as it represents natural human conversations 
    with diverse emotion patterns.<br/><br/>
    
    <b>Performance Data:</b> I collect classification metrics (accuracy, F1, precision, recall), 
    confusion matrices, drift detection metrics, emotion transition matrices, and training history 
    (loss curves, validation metrics). Results are presented through tables, confusion matrix 
    heatmaps, transition probability heatmaps, and training curve plots.<br/><br/>
    
    <b>Comparisons:</b> I compare my class-balanced models (weighted loss and focal loss) against 
    baseline models trained without class balancing. Additionally, I compare BERT-based and RoBERTa-based 
    architectures to identify the most effective approach for emotion drift detection.<br/><br/>
    
    <b>ML Libraries and Frameworks:</b> I use PyTorch for model implementation and training, Hugging Face 
    Transformers for pre-trained BERT/RoBERTa models, scikit-learn for metrics and class weight computation, 
    pandas for data manipulation, and matplotlib/seaborn for visualization.
    """
    story.append(Paragraph(methodology_text.strip(), normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # 5.2 Evaluation Metrics
    story.append(Paragraph("<b>5.2 Evaluation Metrics</b>", subheading_style))
    
    metrics_text = """
    <b>Classification Metrics:</b> Macro F1 score (primary metric) to assess balanced performance 
    across all emotion classes, weighted F1 score to account for class frequency, accuracy, precision, 
    and recall. Per-class F1 scores are used to evaluate performance on individual emotions, 
    particularly minority classes.<br/><br/>
    
    <b>Drift Detection Metrics:</b> Precision (proportion of detected drifts that are actual changes), 
    recall (proportion of actual drifts successfully detected), F1 score, and accuracy for drift 
    detection as a binary classification task.<br/><br/>
    
    <b>Trajectory Metrics:</b> Mean emotion stability score (inverse of variance within sequences) 
    and mean drift correlation (correlation between emotion changes and dialogue features).<br/><br/>
    
    These metrics are selected because they provide comprehensive evaluation of both classification 
    performance and drift detection capabilities, with particular emphasis on minority class performance 
    which is critical for real-world applications.
    """
    story.append(Paragraph(metrics_text.strip(), normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # 5.3 Results - WITH ACTUAL TABLES
    story.append(Paragraph("<b>5.3 Results</b>", subheading_style))
    
    # Overall comparison table
    if bert_real and bert_weighted:
        story.append(Paragraph("<b>Table 2: Overall Model Performance Comparison</b>", subheading_style))
        
        comparison_data = [
            ['Model', 'Accuracy', 'Macro F1', 'Weighted F1', 'Drift F1', 'Drift Recall'],
            ['BERT Baseline (No Balancing)', 
             f"{bert_real['classification']['accuracy']:.4f}",
             f"{bert_real['classification']['f1_score']:.4f}",
             f"{bert_real['classification']['f1_weighted']:.4f}",
             f"{bert_real['drift_detection']['drift_f1']:.4f}",
             f"{bert_real['drift_detection']['drift_recall']:.4f}"],
            ['BERT with Weighted Loss',
             f"{bert_weighted['classification']['accuracy']:.4f}",
             f"{bert_weighted['classification']['f1_score']:.4f}",
             f"{bert_weighted['classification']['f1_weighted']:.4f}",
             f"{bert_weighted['drift_detection']['drift_f1']:.4f}",
             f"{bert_weighted['drift_detection']['drift_recall']:.4f}"],
            ['Improvement',
             f"{((bert_weighted['classification']['accuracy'] - bert_real['classification']['accuracy']) / bert_real['classification']['accuracy'] * 100):.1f}%",
             f"+{((bert_weighted['classification']['f1_score'] - bert_real['classification']['f1_score']) / bert_real['classification']['f1_score'] * 100):.1f}%",
             f"{((bert_weighted['classification']['f1_weighted'] - bert_real['classification']['f1_weighted']) / bert_real['classification']['f1_weighted'] * 100):.1f}%",
             f"+{((bert_weighted['drift_detection']['drift_f1'] - bert_real['drift_detection']['drift_f1']) / bert_real['drift_detection']['drift_f1'] * 100):.1f}%",
             f"+{((bert_weighted['drift_detection']['drift_recall'] - bert_real['drift_detection']['drift_recall']) / bert_real['drift_detection']['drift_recall'] * 100):.1f}%"],
        ]
        
        comparison_table = Table(comparison_data, colWidths=[1.8*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, 2), colors.beige),
            ('BACKGROUND', (0, 3), (-1, 3), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
        ]))
        story.append(comparison_table)
        story.append(Spacer(1, 0.2*inch))
    
    results_text = """
    <b>Baseline Models (No Class Balancing):</b><br/><br/>
    
    The baseline BERT model trained on the DailyDialog dataset without class balancing achieved a macro F1 score 
    of 29.3%, accuracy of 85.8%, and drift detection F1 of 36.8%. However, the baseline shows poor performance 
    on minority emotion classes: fear achieved 0% F1 score (completely undetected) and sadness achieved only 
    4.8% F1 score. This demonstrates the severe impact of class imbalance (422:1 ratio) on rare emotion detection.<br/><br/>
    
    <b>BERT Model with Weighted Cross-Entropy Loss:</b><br/><br/>
    
    After implementing class balancing with weighted cross-entropy loss, the BERT model achieved significant 
    improvements: macro F1 score increased to 38.0% (30% improvement), drift detection F1 increased to 47.1% 
    (28% improvement), and drift detection recall increased to 56.9% (98% improvement, up from 28.7%). Most 
    critically, minority class detection dramatically improved: fear detection increased from 0% to 33.3% F1 score, 
    sadness improved from 4.8% to 19.2% F1 score (300% relative improvement), anger improved from 20.6% to 30.6% 
    (+49%), and disgust improved from 9.5% to 16.5% (+74%). These results demonstrate that class balancing 
    techniques effectively address the imbalanced dataset challenge, enabling detection of rare but critical 
    emotions essential for emotion drift detection in customer support scenarios.
    """
    story.append(Paragraph(results_text.strip(), normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Per-class F1 scores table
    if bert_real and bert_weighted:
        story.append(Paragraph("<b>Table 3: Per-Class F1 Scores</b>", subheading_style))
        
        emotion_classes = ['Anger', 'Disgust', 'Fear', 'Joy', 'Neutral', 'Sadness', 'Surprise']
        baseline_f1 = bert_real['classification']['per_class_f1']
        weighted_f1 = bert_weighted['classification']['per_class_f1']
        
        per_class_data = [
            ['Emotion', 'Baseline F1', 'Weighted Loss F1', 'Improvement'],
        ]
        
        for i, emotion in enumerate(emotion_classes):
            baseline = baseline_f1[i]
            weighted = weighted_f1[i]
            if baseline > 0:
                improvement = ((weighted - baseline) / baseline * 100)
                improvement_str = f"+{improvement:.1f}%"
            else:
                improvement_str = "N/A (was 0%)"
            per_class_data.append([
                emotion,
                f"{baseline:.4f}",
                f"{weighted:.4f}",
                improvement_str
            ])
        
        per_class_table = Table(per_class_data, colWidths=[1*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        per_class_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
        ]))
        story.append(per_class_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Add image references with attempt to embed if exists
    story.append(Paragraph("<b>Figure 1: Confusion Matrix Comparison</b>", subheading_style))
    
    confusion_baseline_path = os.path.join(models_dir, "bert_real", "results", "confusion_matrix.png")
    confusion_weighted_path = os.path.join(models_dir, "bert_real_weighted", "results", "confusion_matrix.png")
    
    if os.path.exists(confusion_weighted_path):
        try:
            img = Image(confusion_weighted_path, width=4*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
        except:
            pass
    
    fig1_text = """
    Confusion matrices visualize classification performance across all emotion classes. The baseline model 
    (available at models/bert_real/results/confusion_matrix.png) shows strong diagonal patterns for neutral 
    and joy classes but poor performance on rare emotions like fear (0% detection). The weighted loss model 
    (shown above) demonstrates improved off-diagonal performance, particularly for minority classes, indicating 
    better emotion detection across all categories.
    """
    story.append(Paragraph(fig1_text.strip(), normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Figure 2: Emotion Transition Heatmap</b>", subheading_style))
    
    transition_path = os.path.join(models_dir, "bert_real_weighted", "results", "transition_heatmap.png")
    if os.path.exists(transition_path):
        try:
            img = Image(transition_path, width=4*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
        except:
            pass
    
    fig2_text = """
    Transition heatmaps visualize emotion-to-emotion transition probabilities. The heatmap (shown above, 
    also available at models/bert_real_weighted/results/transition_heatmap.png) reveals common patterns 
    such as neutral-to-anger transitions in customer support scenarios, demonstrating the model's ability 
    to capture emotion drift patterns across dialogue sequences.
    """
    story.append(Paragraph(fig2_text.strip(), normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Figure 3: Training History</b>", subheading_style))
    
    training_plot_path = os.path.join(models_dir, "bert_real_weighted", "training_history.png")
    if os.path.exists(training_plot_path):
        try:
            img = Image(training_plot_path, width=4*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
        except:
            pass
    
    fig3_text = """
    Training history plots show loss and F1 score curves over 5 epochs. The plot (shown above, also available 
    at models/bert_real_weighted/training_history.png) demonstrates steady improvement in training F1 (from 
    28.2% to 63.2%) and validation F1 peaking at 34.1% in epoch 3, with slight overfitting observed in later 
    epochs as validation loss increases while training loss continues to decrease.
    """
    story.append(Paragraph(fig3_text.strip(), normal_style))
    story.append(PageBreak())
    
    # 5.4 Discussion
    story.append(Paragraph("<b>5.4 Discussion</b>", subheading_style))
    discussion_text = """
    The experimental results demonstrate significant improvements in emotion drift detection through 
    class balancing techniques. The BERT model with weighted cross-entropy loss achieved a 30% 
    improvement in macro F1 score (from 29.3% to 38.0%) and a 28% improvement in drift detection F1 
    score (from 36.8% to 47.1%).<br/><br/>
    
    <b>Impact of Class Balancing:</b> The most notable improvement is in minority class detection. 
    Fear, which previously achieved 0% F1 score (completely undetected), now achieves 33.3% F1 score. 
    Similarly, sadness improved from 4.8% to 19.2% F1 score, a 300% relative improvement. This is 
    critical because rare emotions are often the most important to detect in customer support scenarios, 
    as they indicate significant emotional distress requiring immediate attention.<br/><br/>
    
    <b>Trade-offs:</b> The model shows a slight decrease in overall accuracy (from 85.8% to 81.0%), 
    which is expected when shifting from majority-class over-prediction to balanced class detection. 
    However, this trade-off is acceptable because: (1) The weighted F1 score remains high (81.7%), 
    indicating good overall performance, (2) Minority class detection is dramatically improved, which 
    is essential for drift detection, and (3) The drift detection recall improved substantially from 
    28.7% to 56.9%, enabling detection of twice as many actual emotion transitions.<br/><br/>
    
    <b>Drift Detection Performance:</b> The improved drift detection metrics (F1: 47.1%, Recall: 
    56.9%) indicate that the model can now identify emotion transitions more effectively, which is the 
    core objective of the emotion drift detection system. The improved recall is particularly valuable, 
    as it enables the system to detect more actual emotion changes, even if some false positives occur.<br/><br/>
    
    <b>Model Behavior Observation:</b> During testing on real-world conversations, an important limitation 
    was identified: the model appears to rely more heavily on explicit emotion words (e.g., "angry", 
    "frustrated", "sad") for certain emotions, particularly rare ones like fear and disgust, while 
    more common emotions such as joy and neutral can be detected from contextual cues alone. This 
    suggests that the model has learned different strategies for different emotion classes based on 
    training data distribution. Emotions that are well-represented in the training data (neutral, joy) 
    benefit from learned contextual patterns, while rare emotions (fear, disgust) rely more on 
    keyword matching. This is a common issue in imbalanced classification where the model has seen 
    insufficient examples of minority classes to learn robust contextual representations.
    """
    story.append(Paragraph(discussion_text.strip(), normal_style))
    story.append(PageBreak())
    
    # 6. LIMITATIONS
    story.append(Paragraph("<b>6. LIMITATIONS</b>", heading_style))
    limitations_text = """
    The current approach has several limitations: (1) The dataset exhibits extreme class imbalance 
    (422:1 ratio) which, while addressed through class balancing, still impacts model performance on 
    rare emotions. (2) The models are trained on general dialogue data rather than customer support 
    specific conversations, which may limit domain-specific applicability. (3) The evaluation focuses 
    on emotion classification accuracy rather than real-world deployment metrics such as response time 
    or customer satisfaction improvements. (4) The approach requires fine-tuning large transformer models, 
    which is computationally expensive and may limit real-time deployment. (5) Emotion drift detection 
    is based on discrete emotion labels rather than continuous emotional states, potentially missing 
    subtle emotional transitions. (6) The model demonstrates over-reliance on explicit emotion keywords 
    for rare emotion classes (fear, disgust) while detecting common emotions (joy, neutral) through 
    contextual patterns. This asymmetry in detection strategies limits the model's ability to identify 
    implicit emotional expressions, which are common in real customer support conversations where 
    customers may express frustration or concern without using explicit emotion words. Future work 
    should address these limitations through domain-specific datasets, continuous emotion modeling, 
    enhanced training on implicit emotional cues, and real-world deployment studies.
    """
    story.append(Paragraph(limitations_text.strip(), normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # 7. CONCLUSIONS AND FUTURE WORK
    story.append(Paragraph("<b>7. CONCLUSIONS AND FUTURE WORK</b>", heading_style))
    conclusions_text = """
    I demonstrate that transformer-based models with class balancing techniques can effectively 
    detect emotion drift in dialogue sequences. The use of weighted loss significantly improves detection 
    of rare emotions while maintaining overall classification performance. Specifically, my weighted-loss 
    BERT model achieves 38.0% macro F1 (30% improvement), 47.1% drift detection F1 (28% improvement), 
    and 56.9% drift recall (98% improvement), with dramatic improvements in rare emotion detection. My 
    emotion drift detection capabilities provide a foundation for developing more responsive and empathetic 
    customer support AI systems.<br/><br/>
    
    <b>Important Points:</b> (1) Class balancing is essential for real-world emotion recognition where 
    datasets are imbalanced, (2) Transformer architectures effectively capture contextual emotional cues 
    across dialogue sequences, (3) Drift detection enables analysis of emotion transitions and their 
    correlation with dialogue features, (4) The approach provides actionable insights for improving 
    AI customer support interactions.<br/><br/>
    
    <b>Future Research:</b> This work can be extended through: (1) Integration of real-time emotion 
    monitoring in customer support systems, (2) Development of proactive response adjustment mechanisms 
    based on detected emotion drift, (3) Evaluation on customer support specific datasets, (4) 
    Implementation of continuous emotion modeling rather than discrete classification, (5) Analysis of 
    long-term customer satisfaction correlations with emotion drift patterns, and (6) Development of 
    explainable AI techniques to understand which dialogue features trigger emotion transitions.<br/><br/>
    
    <b>Shortcomings and Enhancements:</b> Current limitations include dataset domain mismatch, 
    computational requirements, discrete emotion modeling, and keyword dependency for rare emotions. 
    Proposed enhancements include: (1) Collecting or fine-tuning on customer support specific datasets, 
    (2) Developing lightweight model variants for real-time deployment, (3) Implementing continuous 
    emotion state models, (4) Adding multimodal features (tone, speaking pace) when available, (5) 
    Creating ensemble methods combining multiple architectures for improved robustness, and (6) 
    Addressing explicit keyword dependency through several strategies: (a) Data augmentation with 
    paraphrasing techniques to generate implicit emotional expressions (e.g., "I'm worried" becomes 
    "This concerns me"), (b) Curriculum learning starting with explicit examples and gradually 
    introducing implicit emotional cues, (c) Contrastive learning to distinguish between explicit 
    and implicit emotional expressions, (d) Additional fine-tuning on domain-specific datasets 
    containing more diverse emotional expressions, and (e) Feature engineering to capture contextual 
    patterns beyond direct emotion words (sentence structure, discourse markers, politeness markers).
    """
    story.append(Paragraph(conclusions_text.strip(), normal_style))
    story.append(PageBreak())
    
    # 8. DATA AVAILABILITY
    story.append(Paragraph("<b>8. DATA AVAILABILITY</b>", heading_style))
    data_availability_text = """
    The data used in this project is publicly available through the following sources:<br/><br/>
    
    1. The DailyDialog dataset was downloaded from Kaggle at 
    https://www.kaggle.com/datasets/thedevastator/dailydialog-multi-turn-dialog-with-intention-and. 
    The dataset contains 11,118 human-written conversations with 87,396 dialogue turns across seven 
    emotion classes. The original DailyDialog dataset is also available from the original source at 
    http://yanran.li/dailydialog.<br/><br/>
    
    2. Processed datasets and code are available in my project repository. The complete source code, 
    preprocessing scripts, model training code, evaluation utilities, and visualization tools are 
    available at: https://github.com/noahmeduvsky/emotion-drift-detection<br/><br/>
    
    The DailyDialog dataset is made available for research purposes. Processed data files, trained 
    model checkpoints, and all experimental results can be accessed through the repository. All code 
    is documented and available for reproducibility.
    """
    story.append(Paragraph(data_availability_text.strip(), normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # 9. REFERENCES
    story.append(Paragraph("<b>9. REFERENCES</b>", heading_style))
    
    references = [
        "Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL-HLT, 1, 4171-4186.",
        "Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.",
        "Hsu, C. C., Chen, S. Y., Kuo, C. C., Huang, T. H., & Ku, L. W. (2018). EmotionLines: An Emotion Corpus of Multi-Party Conversations. Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan.",
        "Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. Proceedings of the IEEE International Conference on Computer Vision, 2980-2988.",
        "Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. Proceedings of the Eighth International Joint Conference on Natural Language Processing, 986-995.",
        "Zhang, J., et al. (2024). The Impact of Emotional Expression by Artificial Intelligence. Decision Support Systems, 181, 114075."
    ]
    
    for i, ref in enumerate(references, 1):
        story.append(Paragraph(f"{i}. {ref}", normal_style))
        story.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    doc.build(story)
    print(f"PDF report generated: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    pdf_file = create_pdf_report()
    print(f"\nReport saved to: {os.path.abspath(pdf_file)}")

