"""
Generate a separate PDF with tables and results for the emotion drift detection project.
This PDF focuses specifically on presenting data tables and visualizations.
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

def create_tables_results_pdf():
    """Create a PDF focused on tables and results."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'src' else script_dir
    pdf_path = os.path.join(project_root, "src", "Tables_and_Results.pdf")
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.black,
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.black,
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.black,
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    table_title_style = ParagraphStyle(
        'TableTitle',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.black,
        spaceAfter=6,
        spaceBefore=10,
        fontName='Helvetica-Bold',
        alignment=TA_LEFT
    )
    
    normal_style = styles['Normal']
    normal_style.alignment = TA_LEFT
    normal_style.fontSize = 10
    
    # Load actual metrics data - handle both src/ and project root execution
    models_dir = os.path.join(project_root, "src", "models") if os.path.exists(os.path.join(project_root, "src", "models")) else os.path.join(project_root, "models")
    
    bert_baseline = load_metrics(os.path.join(models_dir, "bert_baseline"))
    bert_real = load_metrics(os.path.join(models_dir, "bert_real"))
    bert_weighted = load_metrics(os.path.join(models_dir, "bert_real_weighted"))
    roberta_real = load_metrics(os.path.join(models_dir, "roberta_real"))
    
    # Title Page
    story.append(Paragraph("Tables and Results", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Emotion Drift Detection in Customer Support AI", heading_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Noah Meduvsky", normal_style))
    story.append(Paragraph("noahmeduvsky@oakland.edu", normal_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", normal_style))
    story.append(PageBreak())
    
    # TABLE 1: HYPERPARAMETERS
    story.append(Paragraph("Table 1: Hyperparameters Configuration", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    hyperparams_data = [
        ['Category', 'Parameter', 'Value', 'Description'],
        ['Model Architecture', 'Model Type', 'Transformer (BERT/RoBERTa)', 'Pre-trained transformer architecture'],
        ['Model Architecture', 'Base Model', 'bert-base-uncased / roberta-base', 'Hugging Face model identifier'],
        ['Model Architecture', 'Num Emotions', '7', 'Output classes: anger, disgust, fear, joy, neutral, sadness, surprise'],
        ['Model Architecture', 'Dropout Rate', '0.3', 'Dropout probability for regularization'],
        ['Model Architecture', 'Max Sequence Length', '128 tokens', 'Maximum tokens per dialogue turn'],
        ['Training', 'Batch Size', '8 (adjusted 2-8)', 'Samples per batch (adjusted for GPU memory)'],
        ['Training', 'Learning Rate', '2e-5', 'AdamW optimizer learning rate'],
        ['Training', 'Weight Decay', '0.01', 'L2 regularization coefficient'],
        ['Training', 'Num Epochs', '10', 'Maximum training epochs'],
        ['Training', 'Early Stopping Patience', '10', 'Epochs to wait before stopping'],
        ['Training', 'Max Gradient Norm', '1.0', 'Gradient clipping threshold'],
        ['Training', 'Optimizer', 'AdamW', 'Optimization algorithm'],
        ['Training', 'Learning Rate Scheduler', 'ReduceLROnPlateau', 'Adaptive LR reduction'],
        ['Loss Function', 'Baseline', 'Cross-Entropy', 'Standard multi-class cross-entropy'],
        ['Loss Function', 'Weighted', 'Weighted Cross-Entropy', 'Inverse frequency class weights'],
        ['Loss Function', 'Focal Loss Gamma', '2.0', 'Focusing parameter for hard examples'],
        ['Data', 'Train Split', '70%', 'Training set proportion'],
        ['Data', 'Validation Split', '15%', 'Validation set proportion'],
        ['Data', 'Test Split', '15%', 'Test set proportion'],
        ['Data', 'Random Seed', '42', 'Reproducibility seed'],
    ]
    
    hyperparams_table = Table(hyperparams_data, colWidths=[1.3*inch, 1.5*inch, 1.2*inch, 2*inch])
    hyperparams_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    story.append(hyperparams_table)
    story.append(PageBreak())
    
    # TABLE 2: OVERALL MODEL PERFORMANCE
    if bert_real and bert_weighted:
        story.append(Paragraph("Table 2: Overall Model Performance Comparison", heading_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("This table compares the baseline BERT model (without class balancing) against the BERT model with weighted cross-entropy loss.", normal_style))
        story.append(Spacer(1, 0.1*inch))
        
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
        
        comparison_table = Table(comparison_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, 1), colors.lightgrey),
            ('BACKGROUND', (0, 2), (-1, 2), colors.beige),
            ('BACKGROUND', (0, 3), (-1, 3), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(comparison_table)
        story.append(PageBreak())
    
    # TABLE 3: PER-CLASS F1 SCORES
    if bert_real and bert_weighted:
        story.append(Paragraph("Table 3: Per-Class F1 Scores", heading_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("This table shows F1 scores for each emotion class, comparing baseline vs. weighted loss models. Improvement percentages show relative gains.", normal_style))
        story.append(Spacer(1, 0.1*inch))
        
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
                f"{baseline:.4f} ({baseline*100:.2f}%)",
                f"{weighted:.4f} ({weighted*100:.2f}%)",
                improvement_str
            ])
        
        per_class_table = Table(per_class_data, colWidths=[1.2*inch, 1.5*inch, 1.5*inch, 1.2*inch])
        per_class_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(per_class_table)
        story.append(PageBreak())
    
    # TABLE 4: DRIFT DETECTION METRICS
    if bert_real and bert_weighted:
        story.append(Paragraph("Table 4: Drift Detection Performance", heading_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("This table shows performance metrics specifically for emotion drift detection, which identifies when emotions change during conversations.", normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        drift_data = [
            ['Metric', 'BERT Baseline', 'BERT Weighted', 'Improvement'],
            ['Drift Precision', 
             f"{bert_real['drift_detection']['drift_precision']:.4f} ({bert_real['drift_detection']['drift_precision']*100:.2f}%)",
             f"{bert_weighted['drift_detection']['drift_precision']:.4f} ({bert_weighted['drift_detection']['drift_precision']*100:.2f}%)",
             f"{((bert_weighted['drift_detection']['drift_precision'] - bert_real['drift_detection']['drift_precision']) / bert_real['drift_detection']['drift_precision'] * 100):.1f}%"],
            ['Drift Recall', 
             f"{bert_real['drift_detection']['drift_recall']:.4f} ({bert_real['drift_detection']['drift_recall']*100:.2f}%)",
             f"{bert_weighted['drift_detection']['drift_recall']:.4f} ({bert_weighted['drift_detection']['drift_recall']*100:.2f}%)",
             f"+{((bert_weighted['drift_detection']['drift_recall'] - bert_real['drift_detection']['drift_recall']) / bert_real['drift_detection']['drift_recall'] * 100):.1f}%"],
            ['Drift F1 Score', 
             f"{bert_real['drift_detection']['drift_f1']:.4f} ({bert_real['drift_detection']['drift_f1']*100:.2f}%)",
             f"{bert_weighted['drift_detection']['drift_f1']:.4f} ({bert_weighted['drift_detection']['drift_f1']*100:.2f}%)",
             f"+{((bert_weighted['drift_detection']['drift_f1'] - bert_real['drift_detection']['drift_f1']) / bert_real['drift_detection']['drift_f1'] * 100):.1f}%"],
            ['Drift Accuracy', 
             f"{bert_real['drift_detection']['drift_accuracy']:.4f} ({bert_real['drift_detection']['drift_accuracy']*100:.2f}%)",
             f"{bert_weighted['drift_detection']['drift_accuracy']:.4f} ({bert_weighted['drift_detection']['drift_accuracy']*100:.2f}%)",
             f"{((bert_weighted['drift_detection']['drift_accuracy'] - bert_real['drift_detection']['drift_accuracy']) / bert_real['drift_detection']['drift_accuracy'] * 100):.1f}%"],
        ]
        
        drift_table = Table(drift_data, colWidths=[1.5*inch, 2*inch, 2*inch, 1.3*inch])
        drift_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(drift_table)
        story.append(PageBreak())
    
    # TABLE 5: DETAILED CLASSIFICATION METRICS
    if bert_real and bert_weighted:
        story.append(Paragraph("Table 5: Detailed Classification Metrics", heading_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Complete classification performance metrics including precision and recall.", normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        detailed_data = [
            ['Metric', 'BERT Baseline', 'BERT Weighted', 'Description'],
            ['Accuracy', 
             f"{bert_real['classification']['accuracy']:.4f}",
             f"{bert_weighted['classification']['accuracy']:.4f}",
             'Overall classification accuracy'],
            ['Macro F1 Score', 
             f"{bert_real['classification']['f1_score']:.4f}",
             f"{bert_weighted['classification']['f1_score']:.4f}",
             'Unweighted mean of per-class F1 scores'],
            ['Weighted F1 Score', 
             f"{bert_real['classification']['f1_weighted']:.4f}",
             f"{bert_weighted['classification']['f1_weighted']:.4f}",
             'F1 score weighted by class frequency'],
            ['Macro Precision', 
             f"{bert_real['classification']['precision']:.4f}",
             f"{bert_weighted['classification']['precision']:.4f}",
             'Unweighted mean precision'],
            ['Macro Recall', 
             f"{bert_real['classification']['recall']:.4f}",
             f"{bert_weighted['classification']['recall']:.4f}",
             'Unweighted mean recall'],
        ]
        
        detailed_table = Table(detailed_data, colWidths=[1.5*inch, 1.3*inch, 1.3*inch, 2.5*inch])
        detailed_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (2, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(detailed_table)
        story.append(PageBreak())
    
    # VISUALIZATIONS
    story.append(Paragraph("Visualizations", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Figure 1: Confusion Matrix
    story.append(Paragraph("Figure 1: Confusion Matrix - BERT Weighted Model", subheading_style))
    story.append(Spacer(1, 0.1*inch))
    
    confusion_weighted_path = os.path.join(models_dir, "bert_real_weighted", "results", "confusion_matrix.png")
    if os.path.exists(confusion_weighted_path):
        try:
            img = Image(confusion_weighted_path, width=5*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("The confusion matrix shows classification performance across all seven emotion classes. Rows represent true labels, columns represent predicted labels. Diagonal elements indicate correct predictions.", normal_style))
        except Exception as e:
            story.append(Paragraph(f"Error loading confusion matrix: {str(e)}", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Figure 2: Transition Heatmap
    story.append(Paragraph("Figure 2: Emotion Transition Heatmap - BERT Weighted Model", subheading_style))
    story.append(Spacer(1, 0.1*inch))
    
    transition_path = os.path.join(models_dir, "bert_real_weighted", "results", "transition_heatmap.png")
    if os.path.exists(transition_path):
        try:
            img = Image(transition_path, width=5*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("The transition heatmap visualizes emotion-to-emotion transition probabilities. Warmer colors indicate more common transitions, revealing patterns such as neutral-to-anger drifts in customer support scenarios.", normal_style))
        except Exception as e:
            story.append(Paragraph(f"Error loading transition heatmap: {str(e)}", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Figure 3: Training History
    story.append(Paragraph("Figure 3: Training History - BERT Weighted Model", subheading_style))
    story.append(Spacer(1, 0.1*inch))
    
    training_plot_path = os.path.join(models_dir, "bert_real_weighted", "training_history.png")
    if os.path.exists(training_plot_path):
        try:
            img = Image(training_plot_path, width=5*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("Training history shows loss and F1 score curves over epochs. The plot demonstrates model learning progress and validation performance, with early stopping to prevent overfitting.", normal_style))
        except Exception as e:
            story.append(Paragraph(f"Error loading training history: {str(e)}", normal_style))
    
    # Build PDF
    doc.build(story)
    print(f"Tables and Results PDF generated: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    pdf_file = create_tables_results_pdf()
    print(f"\nTables and Results PDF saved to: {os.path.abspath(pdf_file)}")






