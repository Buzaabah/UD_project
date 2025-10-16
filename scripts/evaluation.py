import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class EvaluationMetric(Enum):
    UAS = "unlabeled_attachment_score"  # Head prediction accuracy
    LAS = "labeled_attachment_score"    # Head + relation accuracy  
    LA = "label_accuracy"               # Relation label accuracy only
    ROOT_ACC = "root_accuracy"          # Root identification accuracy
    COMPLETE_MATCH = "complete_match"   # Perfect sentence accuracy

@dataclass
class DependencyPrediction:
    """Single token dependency prediction"""
    token_id: int
    form: str
    predicted_head: int
    predicted_relation: str
    confidence: float = 1.0
    reasoning: str = ""

@dataclass
class SentencePrediction:
    """Complete sentence prediction"""
    sent_id: str
    predictions: List[DependencyPrediction]
    processing_time: float = 0.0
    model_name: str = ""
    prompt_template: str = ""

@dataclass
class EvaluationResult:
    """Evaluation results for a single sentence"""
    sent_id: str
    uas: float
    las: float
    la: float
    root_correct: bool
    complete_match: bool
    num_tokens: int
    errors: List[Dict]

@dataclass
class ExperimentResults:
    """Complete experiment results"""
    experiment_name: str
    model_name: str
    prompt_template: str
    selection_strategy: str
    num_examples: int
    overall_uas: float
    overall_las: float
    overall_la: float
    root_accuracy: float
    complete_match_rate: float
    per_relation_scores: Dict[str, Dict[str, float]]
    per_sentence_results: List[EvaluationResult]
    confusion_matrix: Dict
    total_processing_time: float

class LLMResponseParser:
    """Parse LLM responses into dependency predictions"""
    
    def __init__(self):
        # Common patterns for different response formats
        self.patterns = {
            'basic': r'(\d+)\s+(\S+)\s*->\s*(\d+)\s+(\S+)',
            'detailed': r'(\d+)\s+(\S+)\s*(?:\[.*?\])?\s*(?:\(.*?\))?\s*->\s*(\d+)\s+(\S+)',
            'reasoning': r'(\d+)\s+(\S+).*?->\s*(\d+)\s+(\S+)(?:\s*\((.+?)\))?',
            'tabular': r'(\d+)\s*\|\s*(\S+)\s*\|\s*(\d+)\s*\|\s*(\S+)',
        }
    
    def parse_response(self, response_text: str, sent_id: str = "") -> Optional[SentencePrediction]:
        """Parse LLM response into structured predictions"""
        predictions = []
        
        # Try different parsing patterns
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, response_text, re.MULTILINE | re.IGNORECASE)
            if matches:
                for match in matches:
                    try:
                        if pattern_name == 'reasoning' and len(match) >= 5:
                            token_id, form, head, relation, reasoning = match[:5]
                            reasoning = reasoning or ""
                        else:
                            token_id, form, head, relation = match[:4]
                            reasoning = ""
                        
                        pred = DependencyPrediction(
                            token_id=int(token_id),
                            form=form,
                            predicted_head=int(head),
                            predicted_relation=relation.strip(),
                            reasoning=reasoning
                        )
                        predictions.append(pred)
                    except (ValueError, IndexError) as e:
                        continue
                
                if predictions:
                    return SentencePrediction(sent_id=sent_id, predictions=predictions)
        
        # If no pattern matches, return None
        return None
    
    def extract_confidence_scores(self, response_text: str) -> Dict[int, float]:
        """Extract confidence scores if provided by the model"""
        confidence_pattern = r'(\d+).*?confidence[:=]\s*([0-9.]+)'
        matches = re.findall(confidence_pattern, response_text, re.IGNORECASE)
        
        confidences = {}
        for token_id, conf_str in matches:
            try:
                confidences[int(token_id)] = float(conf_str)
            except ValueError:
                continue
        
        return confidences

class DependencyEvaluator:
    """Evaluate dependency parsing predictions"""
    
    def __init__(self):
        self.parser = LLMResponseParser()
    
    def evaluate_sentence(self, 
                         gold_sentence,
                         predicted_sentence: SentencePrediction) -> EvaluationResult:
        """Evaluate a single sentence prediction"""
        
        # Create lookup for gold standard
        gold_deps = {}
        for token in gold_sentence.tokens:
            if token.id > 0:
                gold_deps[token.id] = {
                    'head': token.head,
                    'relation': token.deprel,
                    'form': token.form
                }
        
        # Create lookup for predictions
        pred_deps = {}
        for pred in predicted_sentence.predictions:
            pred_deps[pred.token_id] = {
                'head': pred.predicted_head,
                'relation': pred.predicted_relation,
                'form': pred.form
            }
        
        # Calculate metrics
        correct_heads = 0
        correct_labels = 0
        correct_both = 0
        total_tokens = 0
        errors = []
        
        root_gold = None
        root_pred = None
        
        for token_id in gold_deps:
            if token_id not in pred_deps:
                # Missing prediction
                errors.append({
                    'token_id': token_id,
                    'error_type': 'missing_prediction',
                    'gold_head': gold_deps[token_id]['head'],
                    'gold_relation': gold_deps[token_id]['relation']
                })
                total_tokens += 1
                continue
            
            gold_head = gold_deps[token_id]['head']
            gold_rel = gold_deps[token_id]['relation']
            pred_head = pred_deps[token_id]['head']
            pred_rel = pred_deps[token_id]['relation']
            
            total_tokens += 1
            
            # Check root identification
            if gold_head == 0:
                root_gold = token_id
            if pred_head == 0:
                root_pred = token_id
            
            # UAS: Unlabeled Attachment Score (head correctness)
            if gold_head == pred_head:
                correct_heads += 1
            
            # LA: Label Accuracy (relation correctness)
            if gold_rel == pred_rel:
                correct_labels += 1
            
            # LAS: Labeled Attachment Score (both head and relation correct)
            if gold_head == pred_head and gold_rel == pred_rel:
                correct_both += 1
            else:
                errors.append({
                    'token_id': token_id,
                    'form': gold_deps[token_id]['form'],
                    'error_type': 'incorrect_dependency',
                    'gold_head': gold_head,
                    'gold_relation': gold_rel,
                    'pred_head': pred_head,
                    'pred_relation': pred_rel
                })
        
        # Calculate scores
        uas = correct_heads / total_tokens if total_tokens > 0 else 0.0
        las = correct_both / total_tokens if total_tokens > 0 else 0.0
        la = correct_labels / total_tokens if total_tokens > 0 else 0.0
        root_correct = (root_gold == root_pred) if root_gold and root_pred else False
        complete_match = (len(errors) == 0)
        
        return EvaluationResult(
            sent_id=predicted_sentence.sent_id,
            uas=uas,
            las=las,
            la=la,
            root_correct=root_correct,
            complete_match=complete_match,
            num_tokens=total_tokens,
            errors=errors
        )
    
    def evaluate_experiment(self,
                           gold_sentences: List,
                           predicted_responses: List[str],
                           experiment_config: Dict) -> ExperimentResults:
        """Evaluate complete experiment"""
        
        sentence_results = []
        all_gold_relations = []
        all_pred_relations = []
        relation_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'predicted': 0})
        
        total_processing_time = 0.0
        
        for i, (gold_sent, response) in enumerate(zip(gold_sentences, predicted_responses)):
            # Parse the response
            predicted_sent = self.parser.parse_response(response, gold_sent.sent_id)
            
            if predicted_sent is None:
                # Failed to parse - create dummy result
                result = EvaluationResult(
                    sent_id=gold_sent.sent_id,
                    uas=0.0, las=0.0, la=0.0,
                    root_correct=False,
                    complete_match=False,
                    num_tokens=len([t for t in gold_sent.tokens if t.id > 0]),
                    errors=[{'error_type': 'parse_failure'}]
                )
            else:
                result = self.evaluate_sentence(gold_sent, predicted_sent)
                total_processing_time += predicted_sent.processing_time
            
            sentence_results.append(result)
            
            # Collect relation statistics
            for token in gold_sent.tokens:
                if token.id > 0:
                    gold_rel = token.deprel
                    all_gold_relations.append(gold_rel)
                    relation_stats[gold_rel]['total'] += 1
                    
                    # Find corresponding prediction
                    if predicted_sent:
                        pred_token = next((p for p in predicted_sent.predictions if p.token_id == token.id), None)
                        if pred_token:
                            pred_rel = pred_token.predicted_relation
                            all_pred_relations.append(pred_rel)
                            relation_stats[pred_rel]['predicted'] += 1
                            
                            if gold_rel == pred_rel:
                                relation_stats[gold_rel]['correct'] += 1
                        else:
                            all_pred_relations.append('MISSING')
        
        # Calculate overall scores
        overall_uas = np.mean([r.uas for r in sentence_results])
        overall_las = np.mean([r.las for r in sentence_results])
        overall_la = np.mean([r.la for r in sentence_results])
        root_accuracy = np.mean([r.root_correct for r in sentence_results])
        complete_match_rate = np.mean([r.complete_match for r in sentence_results])
        
        # Per-relation scores
        per_relation_scores = {}
        for rel, stats in relation_stats.items():
            if stats['total'] > 0:
                precision = stats['correct'] / stats['predicted'] if stats['predicted'] > 0 else 0.0
                recall = stats['correct'] / stats['total']
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                per_relation_scores[rel] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': stats['total']
                }
        
        # Confusion matrix
        unique_relations = sorted(list(set(all_gold_relations + all_pred_relations)))
        cm = confusion_matrix(all_gold_relations, all_pred_relations, labels=unique_relations)
        confusion_dict = {
            'labels': unique_relations,
            'matrix': cm.tolist()
        }
        
        return ExperimentResults(
            experiment_name=experiment_config.get('name', 'experiment'),
            model_name=experiment_config.get('model', 'unknown'),
            prompt_template=experiment_config.get('template', 'unknown'),
            selection_strategy=experiment_config.get('selection', 'unknown'),
            num_examples=experiment_config.get('num_examples', 0),
            overall_uas=overall_uas,
            overall_las=overall_las,
            overall_la=overall_la,
            root_accuracy=root_accuracy,
            complete_match_rate=complete_match_rate,
            per_relation_scores=per_relation_scores,
            per_sentence_results=sentence_results,
            confusion_matrix=confusion_dict,
            total_processing_time=total_processing_time
        )

class EvaluationReporter:
    """Generate evaluation reports and visualizations"""
    
    def __init__(self):
        plt.style.use('default')
    
    def generate_summary_report(self, results: ExperimentResults) -> str:
        """Generate text summary report"""
        report = f"""
WOLOF DEPENDENCY PARSING EVALUATION REPORT
==========================================

Experiment: {results.experiment_name}
Model: {results.model_name}
Template: {results.prompt_template}
Selection Strategy: {results.selection_strategy}
Number of Examples: {results.num_examples}

OVERALL PERFORMANCE
-------------------
Unlabeled Attachment Score (UAS): {results.overall_uas:.3f}
Labeled Attachment Score (LAS):   {results.overall_las:.3f}
Label Accuracy (LA):              {results.overall_la:.3f}
Root Accuracy:                    {results.root_accuracy:.3f}
Complete Match Rate:              {results.complete_match_rate:.3f}
Total Processing Time:            {results.total_processing_time:.2f}s

PER-RELATION PERFORMANCE
------------------------
"""
        
        # Sort relations by F1 score
        sorted_relations = sorted(results.per_relation_scores.items(),
                                key=lambda x: x[1]['f1'], reverse=True)
        
        report += f"{'Relation':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}\n"
        report += "-" * 60 + "\n"
        
        for rel, scores in sorted_relations:
            report += f"{rel:<15} {scores['precision']:<10.3f} {scores['recall']:<10.3f} {scores['f1']:<10.3f} {scores['support']:<10}\n"
        
        # Error analysis
        error_types = defaultdict(int)
        for sent_result in results.per_sentence_results:
            for error in sent_result.errors:
                error_types[error['error_type']] += 1
        
        if error_types:
            report += "\nERROR ANALYSIS\n"
            report += "--------------\n"
            for error_type, count in error_types.items():
                report += f"{error_type}: {count}\n"
        
        return report
    
    def plot_confusion_matrix(self, results: ExperimentResults, save_path: Optional[str] = None):
        """Plot confusion matrix for dependency relations"""
        labels = results.confusion_matrix['labels']
        matrix = np.array(results.confusion_matrix['matrix'])
        
        # Filter to most common relations for readability
        if len(labels) > 20:
            # Keep top 15 most frequent relations
            row_sums = matrix.sum(axis=1)
            top_indices = row_sums.argsort()[-15:]
            matrix = matrix[top_indices][:, top_indices]
            labels = [labels[i] for i in top_indices]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Dependency Relations Confusion Matrix\n{results.experiment_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, experiment_results: List[ExperimentResults],
                                  save_path: Optional[str] = None):
        """Compare performance across multiple experiments"""
        if not experiment_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Wolof Dependency Parsing - Experiment Comparison', fontsize=16)
        
        # Prepare data
        names = [r.experiment_name for r in experiment_results]
        uas_scores = [r.overall_uas for r in experiment_results]
        las_scores = [r.overall_las for r in experiment_results]
        la_scores = [r.overall_la for r in experiment_results]
        root_acc = [r.root_accuracy for r in experiment_results]
        
        # UAS comparison
        axes[0, 0].bar(names, uas_scores, color='skyblue')
        axes[0, 0].set_title('Unlabeled Attachment Score (UAS)')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # LAS comparison
        axes[0, 1].bar(names, las_scores, color='lightgreen')
        axes[0, 1].set_title('Labeled Attachment Score (LAS)')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Label Accuracy comparison
        axes[1, 0].bar(names, la_scores, color='lightcoral')
        axes[1, 0].set_title('Label Accuracy (LA)')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Root Accuracy comparison
        axes[1, 1].bar(names, root_acc, color='gold')
        axes[1, 1].set_title('Root Accuracy')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_results_to_json(self, results: ExperimentResults, filepath: str):
        """Export results to JSON file"""
        # Convert to serializable format
        results_dict = asdict(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    def export_results_to_csv(self, results: ExperimentResults, filepath: str):
        """Export detailed results to CSV"""
        rows = []
        for sent_result in results.per_sentence_results:
            row = {
                'sent_id': sent_result.sent_id,
                'uas': sent_result.uas,
                'las': sent_result.las,
                'la': sent_result.la,
                'root_correct': sent_result.root_correct,
                'complete_match': sent_result.complete_match,
                'num_tokens': sent_result.num_tokens,
                'num_errors': len(sent_result.errors)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

# Example usage
def demo_evaluation_pipeline():
    """Demonstrate the evaluation pipeline"""
    print("Wolof Dependency Parsing Evaluation Pipeline")
    print("=" * 50)
    
    # This would be used with real data and LLM responses
    print("Pipeline Components:")
    print("1. LLM Response Parser - Extracts predictions from model outputs")
    print("2. Dependency Evaluator - Calculates UAS, LAS, LA, and other metrics")
    print("3. Evaluation Reporter - Generates reports and visualizations")
    
    print("\nKey Features:")
    print("- Multiple parsing patterns for different LLM response formats")
    print("- Comprehensive metrics including relation-specific performance")
    print("- Error analysis and categorization")
    print("- Confusion matrices and performance visualizations")
    print("- Export to JSON/CSV for further analysis")
    
    print("\nUsage Example:")
    print("""
    evaluator = DependencyEvaluator()
    results = evaluator.evaluate_experiment(
        gold_sentences=test_data,
        predicted_responses=llm_outputs,
        experiment_config={
            'name': 'Claude-3.5-Sonnet_GlossPrompt_5shot',
            'model': 'claude-3.5-sonnet',
            'template': 'gloss_focused',
            'selection': 'diversity',
            'num_examples': 5
        }
    )
    
    reporter = EvaluationReporter()
    report = reporter.generate_summary_report(results)
    print(report)
    """)

if __name__ == "__main__":
    demo_evaluation_pipeline()