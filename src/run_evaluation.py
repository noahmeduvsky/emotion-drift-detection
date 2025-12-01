"""
Script to evaluate a trained model.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import evaluate_pipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    
    args = parser.parse_args()
    
    class Args:
        pass
    
    eval_args = Args()
    eval_args.checkpoint = args.checkpoint
    eval_args.dataset = 'dailydialog'
    eval_args.split = args.split
    eval_args.model_type = 'transformer'
    eval_args.model_name = args.model_name
    eval_args.batch_size = args.batch_size
    eval_args.max_length = 128
    eval_args.max_seq_length = None
    eval_args.output_dir = args.output_dir
    eval_args.cpu = False
    
    evaluate_pipeline(eval_args)

