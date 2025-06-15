import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
import numpy as np
from typing import List, Dict, Union
import torch
import json
import csv
from datetime import datetime
import os
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import tokenizer, data_collator

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TranslationEvaluator:
    def __init__(self, lang: str = 'vi'):
        """
        Initialize the translation evaluator
        
        Args:
            lang (str): Target language code (default: 'vi' for Vietnamese)
        """
        self.lang = lang
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
        """
        return nltk.word_tokenize(text.lower())
    
    def calculate_bleu(self, 
                      hypothesis: str, 
                      references: Union[str, List[str]], 
                      weights: List[float] = None) -> float:
        """
        Calculate BLEU score
        
        Args:
            hypothesis (str): Machine translation output
            references (Union[str, List[str]]): Reference translation(s)
            weights (List[float], optional): Weights for different n-grams. Defaults to None.
            
        Returns:
            float: BLEU score
        """
        if isinstance(references, str):
            references = [references]
            
        if weights is None:
            weights = [0.25, 0.25, 0.25, 0.25]  # Default weights for 1-gram to 4-gram
            
        hypothesis_tokens = self.tokenize(hypothesis)
        reference_tokens = [self.tokenize(ref) for ref in references]
        
        return sentence_bleu(reference_tokens, hypothesis_tokens, 
                           weights=weights, smoothing_function=self.smooth)
    
    def calculate_rouge(self, 
                       hypothesis: str, 
                       reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores
        
        Args:
            hypothesis (str): Machine translation output
            reference (str): Reference translation
            
        Returns:
            Dict[str, float]: Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def calculate_bert_score(self, 
                           hypotheses: List[str], 
                           references: List[str]) -> Dict[str, float]:
        """
        Calculate BERTScore
        
        Args:
            hypotheses (List[str]): List of machine translation outputs
            references (List[str]): List of reference translations
            
        Returns:
            Dict[str, float]: Dictionary containing precision, recall, and F1 scores
        """
        P, R, F1 = bert_score.score(hypotheses, references, lang=self.lang)
        return {
            'bert_score_precision': P.mean().item(),
            'bert_score_recall': R.mean().item(),
            'bert_score_f1': F1.mean().item()
        }
    
    def evaluate_batch(self, 
                      hypotheses: List[str], 
                      references: List[str]) -> Dict[str, float]:
        """
        Evaluate a batch of translations using multiple metrics
        
        Args:
            hypotheses (List[str]): List of machine translation outputs
            references (List[str]): List of reference translations
            
        Returns:
            Dict[str, float]: Dictionary containing all metric scores
        """
        # Calculate BLEU scores
        bleu_scores = [self.calculate_bleu(hyp, ref) 
                      for hyp, ref in zip(hypotheses, references)]
        avg_bleu = np.mean(bleu_scores)
        
        # Calculate ROUGE scores
        rouge_scores = [self.calculate_rouge(hyp, ref) 
                       for hyp, ref in zip(hypotheses, references)]
        avg_rouge = {
            'rouge1': np.mean([score['rouge1'] for score in rouge_scores]),
            'rouge2': np.mean([score['rouge2'] for score in rouge_scores]),
            'rougeL': np.mean([score['rougeL'] for score in rouge_scores])
        }
        
        # Calculate BERTScore
        bert_scores = self.calculate_bert_score(hypotheses, references)
        
        # Combine all scores
        return {
            'bleu': avg_bleu,
            **avg_rouge,
            **bert_scores
        }
    
    def print_evaluation_results(self, scores: Dict[str, float]):
        """
        Print evaluation results in a formatted way
        
        Args:
            scores (Dict[str, float]): Dictionary containing metric scores
        """
        print("\nTranslation Evaluation Results:")
        print("-" * 30)
        print(f"BLEU Score: {scores['bleu']:.4f}")
        print("\nROUGE Scores:")
        print(f"ROUGE-1: {scores['rouge1']:.4f}")
        print(f"ROUGE-2: {scores['rouge2']:.4f}")
        print(f"ROUGE-L: {scores['rougeL']:.4f}")
        print("\nBERTScore:")
        print(f"Precision: {scores['bert_score_precision']:.4f}")
        print(f"Recall: {scores['bert_score_recall']:.4f}")
        print(f"F1 Score: {scores['bert_score_f1']:.4f}")

    def save_evaluation_results(self, 
                              scores: Dict[str, float],
                              hypotheses: List[str],
                              references: List[str],
                              output_dir: str = "evaluation_results"):
        """
        Save evaluation results to JSON and CSV files
        
        Args:
            scores (Dict[str, float]): Dictionary containing metric scores
            hypotheses (List[str]): List of machine translation outputs
            references (List[str]): List of reference translations
            output_dir (str): Directory to save results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save overall scores to JSON
        scores_file = os.path.join(output_dir, f"overall_scores_{timestamp}.json")
        with open(scores_file, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=4, ensure_ascii=False)
        
        # Calculate detailed scores for each sentence pair
        detailed_results = []
        for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
            bleu = self.calculate_bleu(hyp, ref)
            rouge = self.calculate_rouge(hyp, ref)
            
            # Calculate BERTScore for single pair
            P, R, F1 = bert_score.score([hyp], [ref], lang=self.lang)
            
            result = {
                'id': i + 1,
                'hypothesis': hyp,
                'reference': ref,
                'bleu': bleu,
                'rouge1': rouge['rouge1'],
                'rouge2': rouge['rouge2'],
                'rougeL': rouge['rougeL'],
                'bert_score_precision': P[0].item(),
                'bert_score_recall': R[0].item(),
                'bert_score_f1': F1[0].item()
            }
            detailed_results.append(result)
        
        # Save detailed results to CSV
        detailed_file = os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
        with open(detailed_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=detailed_results[0].keys())
            writer.writeheader()
            writer.writerows(detailed_results)
        
        print(f"\nResults saved to:")
        print(f"Overall scores: {scores_file}")
        print(f"Detailed results: {detailed_file}")
        
        return scores_file, detailed_file

# Example usage
def test_translation(file_name, test_dataset,
                     data_collator, batch_test, 
                     context_length, model, 
                     tokenizer, metric = None):
    evaluator = TranslationEvaluator()
    hypotheses = []
    references = [] 
    test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_test, drop_last = True)
    texts = []
    targets = []
    model.eval()
    for i, batch in enumerate(tqdm(test_dataloader)):
        outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            context_length=context_length,
        )
        texts.append(outputs)
        targets.append(batch['labels'])
    for outputs, labels in zip(texts, targets):
        with tokenizer.as_target_tokenizer():
            outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]
            labels = np.where(labels != -100,  labels, tokenizer.pad_token_id)
            actuals = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in labels]
        hypotheses.extend(outputs)
        references.extend(actuals)
    scores = evaluator.evaluate_batch(hypotheses, references)
    
    evaluator.print_evaluation_results(scores)
    print(hypotheses)
    # Save results
    evaluator.save_evaluation_results(scores, hypotheses, references) 
    return scores
    