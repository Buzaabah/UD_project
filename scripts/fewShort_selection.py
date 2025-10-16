"""
fewShort_selection.py - Complete Few-Shot Example Selection for Wolof Dependency Parsing
Updated for actual file names: wol.Wolof.{train,dev,test}.conllu
"""

import os
import random
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import re
from pathlib import Path

@dataclass
class Token:
    """Represents a single token with all CoNLL-U fields"""
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str
    gloss: str = ""
    
    def __post_init__(self):
        """Extract gloss from misc field after initialization"""
        if "*Gloss=" in self.misc:
            try:
                # Extract gloss between *Gloss= and next | or end
                gloss_start = self.misc.find("*Gloss=") + 7
                gloss_end = self.misc.find("|", gloss_start)
                if gloss_end == -1:
                    gloss_end = self.misc.find("*", gloss_start)
                if gloss_end == -1:
                    gloss_end = len(self.misc)
                
                self.gloss = self.misc[gloss_start:gloss_end].strip()
            except:
                self.gloss = ""

@dataclass
class Sentence:
    """Represents a complete sentence with metadata and tokens"""
    sent_id: str
    text: str
    text_en: str
    tokens: List[Token]
    
    def get_deprels(self) -> Set[str]:
        """Get all dependency relations in this sentence"""
        return {token.deprel for token in self.tokens if token.deprel != '*' and token.id > 0}
    
    def get_pos_tags(self) -> Set[str]:
        """Get all POS tags in this sentence"""
        return {token.upos for token in self.tokens if token.upos != '*' and token.id > 0}
    
    def get_sentence_length(self) -> int:
        """Get sentence length (excluding special tokens)"""
        return len([t for t in self.tokens if t.id > 0])
    
    def get_gloss_coverage(self) -> float:
        """Get percentage of tokens with glosses"""
        valid_tokens = [t for t in self.tokens if t.id > 0]
        if not valid_tokens:
            return 0.0
        glossed_tokens = [t for t in valid_tokens if t.gloss]
        return len(glossed_tokens) / len(valid_tokens)
    
    def has_rare_relations(self, rare_relations: Set[str]) -> bool:
        """Check if sentence contains any rare dependency relations"""
        return bool(self.get_deprels().intersection(rare_relations))
    
    def get_complexity_score(self) -> float:
        """Calculate sentence complexity based on various factors"""
        length = self.get_sentence_length()
        unique_relations = len(self.get_deprels())
        unique_pos = len(self.get_pos_tags())
        gloss_coverage = self.get_gloss_coverage()
        
        # Complexity score: longer sentences with more diverse relations are more complex
        complexity = (length * 0.3 + unique_relations * 0.4 + unique_pos * 0.2 + gloss_coverage * 0.1)
        return complexity

class ConlluParser:
    """Enhanced CoNLL-U parser for Wolof data"""
    
    @staticmethod
    def parse_file(filepath: str) -> List[Sentence]:
        """Parse CoNLL-U file and return list of sentences"""
        sentences = []
        current_sent_id = ""
        current_text = ""
        current_text_en = ""
        current_tokens = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    try:
                        if line.startswith('# sent_id'):
                            # New sentence starts
                            current_sent_id = line.split('=', 1)[1].strip()
                        elif line.startswith('# text ='):
                            current_text = line.split('=', 1)[1].strip()
                        elif line.startswith('# text*en'):
                            current_text_en = line.split('=', 1)[1].strip()
                        elif line and not line.startswith('#'):
                            # Parse token line
                            fields = line.split('\t')
                            if len(fields) >= 10:
                                try:
                                    token = Token(
                                        id=int(fields[0]) if fields[0].isdigit() else 0,
                                        form=fields[1],
                                        lemma=fields[2],
                                        upos=fields[3],
                                        xpos=fields[4],
                                        feats=fields[5],
                                        head=int(fields[6]) if fields[6].isdigit() else 0,
                                        deprel=fields[7],
                                        deps=fields[8],
                                        misc=fields[9] if len(fields) > 9 else ""
                                    )
                                    current_tokens.append(token)
                                except ValueError as e:
                                    print(f"Warning: Skipping malformed token at line {line_num}: {e}")
                                    continue
                        elif line == '' and current_tokens:
                            # End of sentence
                            if current_sent_id and current_text:
                                sentence = Sentence(
                                    sent_id=current_sent_id,
                                    text=current_text,
                                    text_en=current_text_en or "",
                                    tokens=current_tokens
                                )
                                sentences.append(sentence)
                            
                            # Reset for next sentence
                            current_sent_id = ""
                            current_text = ""
                            current_text_en = ""
                            current_tokens = []
                    
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num}: {e}")
                        continue
                
                # Handle last sentence if file doesn't end with empty line
                if current_tokens and current_sent_id:
                    sentence = Sentence(
                        sent_id=current_sent_id,
                        text=current_text,
                        text_en=current_text_en or "",
                        tokens=current_tokens
                    )
                    sentences.append(sentence)
        
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            return []
        except UnicodeDecodeError:
            print(f"Error: Cannot decode file {filepath}. Please check encoding.")
            return []
        except Exception as e:
            print(f"Error parsing file {filepath}: {e}")
            return []
        
        return sentences
    
    @staticmethod
    def validate_sentences(sentences: List[Sentence]) -> Dict[str, int]:
        """Validate parsed sentences and return statistics"""
        stats = {
            'total_sentences': len(sentences),
            'total_tokens': 0,
            'sentences_with_glosses': 0,
            'sentences_with_english': 0,
            'empty_sentences': 0,
            'malformed_sentences': 0
        }
        
        for sentence in sentences:
            valid_tokens = [t for t in sentence.tokens if t.id > 0]
            stats['total_tokens'] += len(valid_tokens)
            
            if not valid_tokens:
                stats['empty_sentences'] += 1
            
            if any(t.gloss for t in valid_tokens):
                stats['sentences_with_glosses'] += 1
                
            if sentence.text_en:
                stats['sentences_with_english'] += 1
                
            # Check for malformed dependencies
            heads = [t.head for t in valid_tokens]
            if any(head > len(valid_tokens) for head in heads if head > 0):
                stats['malformed_sentences'] += 1
        
        return stats

class ExampleSelector:
    """Enhanced example selection algorithms for few-shot learning"""
    
    def __init__(self, train_sentences: List[Sentence]):
        self.train_sentences = train_sentences
        self.deprel_freq = self._compute_deprel_frequencies()
        self.pos_freq = self._compute_pos_frequencies()
        self.relation_cooccurrence = self._compute_relation_cooccurrence()
        self.sentence_complexities = self._compute_sentence_complexities()
        
    def _compute_deprel_frequencies(self) -> Counter:
        """Compute frequency of dependency relations in training data"""
        deprels = []
        for sent in self.train_sentences:
            deprels.extend(sent.get_deprels())
        return Counter(deprels)
    
    def _compute_pos_frequencies(self) -> Counter:
        """Compute frequency of POS tags in training data"""
        pos_tags = []
        for sent in self.train_sentences:
            pos_tags.extend(sent.get_pos_tags())
        return Counter(pos_tags)
    
    def _compute_relation_cooccurrence(self) -> Dict[str, Set[str]]:
        """Compute which relations frequently occur together"""
        cooccurrence = defaultdict(set)
        for sent in self.train_sentences:
            relations = list(sent.get_deprels())
            for i, rel1 in enumerate(relations):
                for rel2 in relations[i+1:]:
                    cooccurrence[rel1].add(rel2)
                    cooccurrence[rel2].add(rel1)
        return dict(cooccurrence)
    
    def _compute_sentence_complexities(self) -> Dict[str, float]:
        """Precompute complexity scores for all sentences"""
        return {sent.sent_id: sent.get_complexity_score() for sent in self.train_sentences}
    
    def random_selection(self, k: int, seed: int = 42) -> List[Sentence]:
        """Random selection baseline with reproducible results"""
        random.seed(seed)
        return random.sample(self.train_sentences, min(k, len(self.train_sentences)))
    
    def diversity_based_selection(self, k: int) -> List[Sentence]:
        """Select examples to maximize diversity of dependency relations and features"""
        if k >= len(self.train_sentences):
            return self.train_sentences.copy()
        
        selected = []
        covered_relations = set()
        covered_pos = set()
        remaining_sentences = self.train_sentences.copy()
        
        # First, prioritize sentences with rare relations (bottom 20% frequency)
        rare_relations = set()
        total_relation_count = sum(self.deprel_freq.values())
        for relation, count in self.deprel_freq.items():
            if count / total_relation_count < 0.05:  # Relations that appear in <5% of cases
                rare_relations.add(relation)
        
        # Phase 1: Select sentences with rare relations
        for sentence in remaining_sentences[:]:
            if len(selected) >= k:
                break
            if sentence.has_rare_relations(rare_relations):
                sent_relations = sentence.get_deprels()
                sent_pos = sentence.get_pos_tags()
                
                # Check if this sentence adds new information
                new_relations = sent_relations - covered_relations
                new_pos = sent_pos - covered_pos
                
                if new_relations or new_pos:
                    selected.append(sentence)
                    covered_relations.update(sent_relations)
                    covered_pos.update(sent_pos)
                    remaining_sentences.remove(sentence)
        
        # Phase 2: Select sentences that maximize coverage of common relations
        while len(selected) < k and remaining_sentences:
            best_sentence = None
            best_score = -1
            
            for sentence in remaining_sentences:
                sent_relations = sentence.get_deprels()
                sent_pos = sentence.get_pos_tags()
                
                # Score based on new information added
                new_relations = sent_relations - covered_relations
                new_pos = sent_pos - covered_pos
                
                # Bonus for high gloss coverage
                gloss_bonus = sentence.get_gloss_coverage() * 0.3
                
                # Bonus for sentence length diversity
                length_bonus = 0.1 if not any(abs(sentence.get_sentence_length() - s.get_sentence_length()) < 3 for s in selected) else 0
                
                score = len(new_relations) * 2 + len(new_pos) + gloss_bonus + length_bonus
                
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
            
            if best_sentence:
                selected.append(best_sentence)
                covered_relations.update(best_sentence.get_deprels())
                covered_pos.update(best_sentence.get_pos_tags())
                remaining_sentences.remove(best_sentence)
            else:
                # If no sentence adds new information, pick randomly from remaining
                if remaining_sentences:
                    selected.append(remaining_sentences.pop(0))
        
        return selected[:k]
    
    def similarity_based_selection(self, target_sentence: Sentence, k: int) -> List[Sentence]:
        """Select examples most similar to target sentence using multiple similarity metrics"""
        if k >= len(self.train_sentences):
            return self.train_sentences.copy()
        
        target_relations = target_sentence.get_deprels()
        target_pos = target_sentence.get_pos_tags()
        target_length = target_sentence.get_sentence_length()
        target_complexity = target_sentence.get_complexity_score()
        
        similarities = []
        
        for sent in self.train_sentences:
            sent_relations = sent.get_deprels()
            sent_pos = sent.get_pos_tags()
            sent_length = sent.get_sentence_length()
            sent_complexity = sent.get_complexity_score()
            
            # Relation overlap (weighted by frequency - rare relations get higher weight)
            relation_overlap = 0
            for rel in target_relations.intersection(sent_relations):
                weight = 1.0 / (self.deprel_freq[rel] + 1)  # Rare relations get higher weight
                relation_overlap += weight
            
            # POS overlap
            pos_overlap = len(target_pos.intersection(sent_pos))
            
            # Length similarity
            length_similarity = 1.0 / (1.0 + abs(target_length - sent_length))
            
            # Complexity similarity
            complexity_similarity = 1.0 / (1.0 + abs(target_complexity - sent_complexity))
            
            # Gloss coverage similarity
            target_gloss_coverage = target_sentence.get_gloss_coverage()
            sent_gloss_coverage = sent.get_gloss_coverage()
            gloss_similarity = 1.0 / (1.0 + abs(target_gloss_coverage - sent_gloss_coverage))
            
            # Combined similarity score
            similarity = (
                relation_overlap * 3.0 +  # Most important
                pos_overlap * 2.0 +
                length_similarity * 1.0 +
                complexity_similarity * 1.0 +
                gloss_similarity * 0.5
            )
            
            similarities.append((similarity, sent))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [sent for _, sent in similarities[:k]]
    
    def stratified_selection(self, k: int) -> List[Sentence]:
        """Stratified selection based on sentence length and complexity"""
        if k >= len(self.train_sentences):
            return self.train_sentences.copy()
        
        # Group sentences by length ranges
        length_groups = defaultdict(list)
        for sent in self.train_sentences:
            length = sent.get_sentence_length()
            if length <= 5:
                group = "short"
            elif length <= 15:
                group = "medium"
            elif length <= 25:
                group = "long"
            else:
                group = "very_long"
            length_groups[group].append(sent)
        
        # Also group by complexity
        complexity_groups = defaultdict(list)
        complexities = [sent.get_complexity_score() for sent in self.train_sentences]
        if complexities:
            complexity_median = np.median(complexities)
            for sent in self.train_sentences:
                if sent.get_complexity_score() <= complexity_median:
                    complexity_groups["simple"].append(sent)
                else:
                    complexity_groups["complex"].append(sent)
        
        selected = []
        group_names = ["short", "medium", "long", "very_long"]
        
        # Select proportionally from each length group
        for group in group_names:
            if group in length_groups and selected:
                group_sentences = length_groups[group]
                group_size = max(1, k // len(group_names))  # At least 1 from each group
                
                # Within each group, balance complexity
                group_sentences.sort(key=lambda x: (
                    -len(x.get_deprels()),  # Prefer diverse relations
                    -x.get_gloss_coverage(),  # Prefer sentences with glosses
                    x.get_complexity_score()  # Mix of complexities
                ))
                
                selected.extend(group_sentences[:group_size])
        
        # Fill remaining slots with best remaining sentences
        remaining_slots = k - len(selected)
        if remaining_slots > 0:
            used_sent_ids = {s.sent_id for s in selected}
            remaining_sentences = [s for s in self.train_sentences if s.sent_id not in used_sent_ids]
            
            # Sort remaining by quality score
            remaining_sentences.sort(key=lambda x: (
                -len(x.get_deprels()),
                -x.get_gloss_coverage(),
                -x.get_sentence_length()
            ), reverse=True)
            
            selected.extend(remaining_sentences[:remaining_slots])
        
        return selected[:k]
    
    def coverage_based_selection(self, k: int) -> List[Sentence]:
        """Select examples to maximize coverage of dependency relations"""
        if k >= len(self.train_sentences):
            return self.train_sentences.copy()
        
        selected = []
        covered_relations = set()
        remaining_sentences = self.train_sentences.copy()
        
        # Greedy selection to maximize relation coverage
        for _ in range(k):
            if not remaining_sentences:
                break
            
            best_sentence = None
            best_score = -1
            
            for sent in remaining_sentences:
                sent_relations = sent.get_deprels()
                new_relations = sent_relations - covered_relations
                
                # Score based on new relations, with bonus for rare relations
                score = 0
                for rel in new_relations:
                    # Give higher weight to rarer relations
                    weight = 1.0 / (self.deprel_freq[rel] + 1)
                    score += weight
                
                # Bonus for sentences with good gloss coverage
                score += sent.get_gloss_coverage() * 0.5
                
                # Small bonus for sentence length diversity
                if selected:
                    avg_length = np.mean([s.get_sentence_length() for s in selected])
                    length_diversity = abs(sent.get_sentence_length() - avg_length) / avg_length
                    score += min(length_diversity, 0.5)  # Cap the bonus
                
                if score > best_score:
                    best_score = score
                    best_sentence = sent
            
            if best_sentence:
                selected.append(best_sentence)
                covered_relations.update(best_sentence.get_deprels())
                remaining_sentences.remove(best_sentence)
            else:
                # Fallback: select randomly from remaining
                selected.append(remaining_sentences.pop(0))
        
        return selected
    
    def complexity_balanced_selection(self, k: int) -> List[Sentence]:
        """Select examples with balanced complexity levels"""
        if k >= len(self.train_sentences):
            return self.train_sentences.copy()
        
        # Sort sentences by complexity
        sorted_sentences = sorted(self.train_sentences, key=lambda x: x.get_complexity_score())
        
        selected = []
        
        # Select examples across complexity spectrum
        if k == 1:
            # Pick medium complexity
            selected = [sorted_sentences[len(sorted_sentences) // 2]]
        elif k == 2:
            # Pick low and high complexity
            selected = [sorted_sentences[len(sorted_sentences) // 4],
                       sorted_sentences[3 * len(sorted_sentences) // 4]]
        else:
            # Distribute across complexity spectrum
            step = len(sorted_sentences) // k
            for i in range(k):
                idx = min(i * step, len(sorted_sentences) - 1)
                selected.append(sorted_sentences[idx])
        
        return selected

def evaluate_selection_quality(selected_examples: List[Sentence], 
                             test_sentences: List[Sentence]) -> Dict[str, float]:
    """Evaluate the quality of selected examples"""
    
    if not selected_examples or not test_sentences:
        return {}
    
    # Coverage metrics
    train_relations = set()
    for sent in selected_examples:
        train_relations.update(sent.get_deprels())
    
    test_relations = set()
    for sent in test_sentences:
        test_relations.update(sent.get_deprels())
    
    # Relation coverage
    relation_coverage = len(train_relations.intersection(test_relations)) / len(test_relations) if test_relations else 0
    
    # Diversity metrics
    relation_diversity = len(train_relations)
    avg_sentence_length = np.mean([sent.get_sentence_length() for sent in selected_examples])
    
    # Gloss coverage
    avg_gloss_coverage = np.mean([sent.get_gloss_coverage() for sent in selected_examples])
    
    # Complexity distribution
    complexities = [sent.get_complexity_score() for sent in selected_examples]
    complexity_std = np.std(complexities) if len(complexities) > 1 else 0
    
    return {
        "relation_coverage": relation_coverage,
        "relation_diversity": relation_diversity,
        "avg_sentence_length": avg_sentence_length,
        "avg_gloss_coverage": avg_gloss_coverage,
        "complexity_std": complexity_std,
        "num_examples": len(selected_examples)
    }

def load_wolof_data(data_directory: str = "../data/annotated_data/wol") -> Tuple[List[Sentence], List[Sentence], List[Sentence]]:
    """Load Wolof data from the actual file names"""
    data_dir = Path(data_directory)
    parser = ConlluParser()
    
    # Actual file names in your dataset
    train_file = data_dir / "wol.Wolof.train.conllu"
    dev_file = data_dir / "wol.Wolof.dev.conllu"
    test_file = data_dir / "wol.Wolof.test.conllu"
    
    print(f"🇸🇳 Loading Wolof Dataset from {data_directory}")
    print("=" * 50)
    
    train_sentences = []
    dev_sentences = []
    test_sentences = []
    
    # Load training data
    if train_file.exists():
        print(f"📚 Loading: {train_file.name}")
        train_sentences = parser.parse_file(str(train_file))
        stats = parser.validate_sentences(train_sentences)
        print(f"   ✅ {stats['total_sentences']} sentences, {stats['total_tokens']} tokens")
        print(f"   🌍 {stats['sentences_with_glosses']} sentences with glosses")
        print(f"   🇬🇧 {stats['sentences_with_english']} sentences with English translations")
    else:
        print(f"⚠️  Training file not found: {train_file.name}")
    
    # Load development data
    if dev_file.exists():
        print(f"📖 Loading: {dev_file.name}")
        dev_sentences = parser.parse_file(str(dev_file))
        stats = parser.validate_sentences(dev_sentences)
        print(f"   ✅ {stats['total_sentences']} sentences, {stats['total_tokens']} tokens")
        print(f"   🌍 {stats['sentences_with_glosses']} sentences with glosses")
    else:
        print(f"⚠️  Development file not found: {dev_file.name}")
    
    # Load test data
    if test_file.exists():
        print(f"📄 Loading: {test_file.name}")
        test_sentences = parser.parse_file(str(test_file))
        stats = parser.validate_sentences(test_sentences)
        print(f"   ✅ {stats['total_sentences']} sentences, {stats['total_tokens']} tokens")
        print(f"   🌍 {stats['sentences_with_glosses']} sentences with glosses")
    else:
        print(f"⚠️  Test file not found: {test_file.name}")
    
    total_sentences = len(train_sentences) + len(dev_sentences) + len(test_sentences)
    print(f"\n📊 Total loaded: {total_sentences} sentences")
    
    if total_sentences == 0:
        print("❌ No data found! Please check:")
        print("   1. Files are in '../data/annotated_data/wol' directory")
        print("   2. Files are named: wol.Wolof.train.conllu, wol.Wolof.dev.conllu, wol.Wolof.test.conllu")
        print("   3. Files are valid CoNLL-U format")
    
    return train_sentences, dev_sentences, test_sentences

def demo_selection_strategies():
    """Demonstrate different selection strategies with actual Wolof data"""
    
    print("🇸🇳 WOLOF FEW-SHOT SELECTION DEMONSTRATION")
    print("=" * 60)
    
    # Load the data
    train_sentences, dev_sentences, test_sentences = load_wolof_data()
    
    if not train_sentences:
        print("❌ No training data available for demonstration")
        return
    
    # Initialize selector
    selector = ExampleSelector(train_sentences)
    
    # Test different selection strategies
    k = 5  # Number of examples to select
    strategies = {
        "random": ("Random Selection", selector.random_selection),
        "diversity": ("Diversity-Based", selector.diversity_based_selection),
        "coverage": ("Coverage-Based", selector.coverage_based_selection),
        "stratified": ("Stratified Selection", selector.stratified_selection),
        "complexity": ("Complexity-Balanced", selector.complexity_balanced_selection)
    }
    
    test_data = test_sentences if test_sentences else dev_sentences
    
    print(f"\n🧪 Testing selection strategies with k={k}")
    print("-" * 60)
    
    for strategy_name, (display_name, strategy_func) in strategies.items():
        try:
            if strategy_name == "complexity":
                selected = strategy_func(k)
            else:
                selected = strategy_func(k)
            
            print(f"\n📋 {display_name}:")
            
            # Show selected sentence IDs and basic info
            for i, sent in enumerate(selected):
                length = sent.get_sentence_length()
                relations = len(sent.get_deprels())
                gloss_pct = sent.get_gloss_coverage() * 100
                print(f"   {i+1}. {sent.sent_id}: {length} tokens, {relations} relations, {gloss_pct:.0f}% glossed")
                print(f"      Text: {sent.text[:60]}...")
            
            # Evaluate quality if test data available
            if test_data:
                quality = evaluate_selection_quality(selected, test_data[:20])
                print(f"   📊 Quality Metrics:")
                print(f"      Relation Coverage: {quality.get('relation_coverage', 0):.3f}")
                print(f"      Relation Diversity: {quality.get('relation_diversity', 0)}")
                print(f"      Avg Sentence Length: {quality.get('avg_sentence_length', 0):.1f}")
                print(f"      Avg Gloss Coverage: {quality.get('avg_gloss_coverage', 0):.3f}")
        
        except Exception as e:
            print(f"❌ Error with {display_name}: {e}")
    
    print(f"\n✅ Selection strategy demonstration complete!")

if __name__ == "__main__":
    # Run the demonstration
    demo_selection_strategies()