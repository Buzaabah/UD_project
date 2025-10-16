import random
import json
import re
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import statistics
import itertools

@dataclass
class Token:
    """Represents a single token with all annotations"""
    id: int
    form: str
    lemma: str
    upos: str
    head: int
    deprel: str
    gloss: str
    
@dataclass
class Sentence:
    """Represents a complete sentence with metadata"""
    sent_id: str
    text: str
    text_en: str
    tokens: List[Token]

class WolofDataLoader:
    """Load and parse CoNLL-U format Wolof data"""
    
    def __init__(self, file_path: str = None):
        self.sentences = []
        if file_path:
            self.load_from_file(file_path)
    
    def load_from_file(self, file_path: str):
        """Load data from CoNLL-U file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.sentences = self.parse_conllu(content)
        return self.sentences
    
    def load_from_string(self, content: str):
        """Load data from string content"""
        self.sentences = self.parse_conllu(content)
        return self.sentences
    
    def parse_conllu(self, content: str) -> List[Sentence]:
        """Parse CoNLL-U format content"""
        sentences = []
        current_sentence = None
        current_tokens = []
        
        for line in content.strip().split('\n'):
            line = line.strip()
            
            if not line:
                # End of sentence
                if current_sentence and current_tokens:
                    current_sentence.tokens = current_tokens
                    sentences.append(current_sentence)
                current_sentence = None
                current_tokens = []
                
            elif line.startswith('#'):
                # Metadata line
                if line.startswith('# sent_id ='):
                    sent_id = line.split('=', 1)[1].strip()
                    current_sentence = Sentence(sent_id, "", "", [])
                elif line.startswith('# text =') and current_sentence:
                    current_sentence.text = line.split('=', 1)[1].strip()
                elif line.startswith('# text_en =') and current_sentence:
                    current_sentence.text_en = line.split('=', 1)[1].strip()
                    
            else:
                # Token line
                parts = line.split('\t')
                if len(parts) >= 10 and '-' not in parts[0]:
                    # Extract gloss from MISC field
                    gloss = ""
                    if len(parts) > 9 and 'Gloss=' in parts[9]:
                        gloss_match = re.search(r'Gloss=([^|]*)', parts[9])
                        if gloss_match:
                            gloss = gloss_match.group(1).strip()
                    
                    token = Token(
                        id=int(parts[0]),
                        form=parts[1],
                        lemma=parts[2],
                        upos=parts[3],
                        head=int(parts[6]) if parts[6].isdigit() else 0,
                        deprel=parts[7],
                        gloss=gloss
                    )
                    current_tokens.append(token)
        
        # Handle last sentence
        if current_sentence and current_tokens:
            current_sentence.tokens = current_tokens
            sentences.append(current_sentence)
            
        return sentences
    
    def create_sample_data(self):
        """Create sample data for testing"""
        sample_conllu = """# sent_id = Masakhane-Wolof_dev_1
# text = Ki yore wàllu tàggatu gi ca Dakaar Sakere Këer, Olivier Brice Sylvain te ñu tuumaal ko mbiru càkkuy xale yu ndaw ba police teg ko loxo.
# text_en = The man in charge of training at Dakar Sacré Coeur, Olivier Brice Sylvain, who has been accused of paedophilia and arrested by the police.
1	Ki	ki	PRON	*	*	29	subj	_	Gloss=the one who
2	yore	yor	VERB	*	*	1	mod	_	Gloss=be in charge
3	wàllu	wàll	NOUN	*	*	2	comp:obj	_	Gloss=part of
4	tàggatu	tàggatu	NOUN	*	*	3	mod	_	Gloss=sport
5	gi	bi	DET	*	*	4	det	_	Gloss=the
6	ca	ci	ADP	*	*	2	udep	_	Gloss=in
7	Dakaar	Dakaar	PROPN	*	*	6	comp:obj	_	Gloss=Dakar
8	Sakere	Sakere	PROPN	*	*	7	flat	_	Gloss=Sacré
9	Këer	Këer	PROPN	*	*	8	flat	_	Gloss=Coeur
10	,	,	PUNCT	*	*	11	punct	_	Gloss=,
11	Olivier	Olivier	PROPN	*	*	1	conj:appos	_	Gloss=Olivier
12	Brice	Brice	PROPN	*	*	11	flat	_	Gloss=Brice
13	Sylvain	Sylvain	PROPN	*	*	12	flat	_	Gloss=Sylvain

# sent_id = Masakhane-Wolof_dev_2  
# text = Ñu tuumaal ko mbiru càkkuy xale yu ndaw.
# text_en = They accused him of child abuse.
1	Ñu	mu	PRON	*	*	2	subj	_	Gloss=they
2	tuumaal	tuumal	VERB	*	*	0	root	_	Gloss=accuse
3	ko	ko	PRON	*	*	2	comp:obj	_	Gloss=him
4	mbiru	mbir	NOUN	*	*	2	comp:obl	_	Gloss=bad
5	càkkuy	càkku	NOUN	*	*	4	mod	_	Gloss=action
6	xale	xale	NOUN	*	*	5	mod	_	Gloss=child
7	yu	bu	DET	*	*	6	det	_	Gloss=of
8	ndaw	ndaw	NOUN	*	*	7	mod	_	Gloss=young

# sent_id = Masakhane-Wolof_dev_3
# text = Police bi teg ko loxo.  
# text_en = The police arrested him.
1	Police	police	NOUN	*	*	3	subj	_	Gloss=police
2	bi	bi	DET	*	*	1	det	_	Gloss=the
3	teg	teg	VERB	*	*	0	root	_	Gloss=put
4	ko	ko	PRON	*	*	3	comp:obj	_	Gloss=him
5	loxo	loxo	NOUN	*	*	3	comp:obl	_	Gloss=prison
"""
        return self.load_from_string(sample_conllu)

class WolofDependencyPromptEngineer:
    """Main class for dependency parsing prompt engineering"""
    
    def __init__(self, sentences: List[Sentence]):
        self.sentences = sentences
        self.relation_examples = self._categorize_relations()
        self.relation_stats = self._compute_relation_stats()
        
    def _categorize_relations(self) -> Dict[str, List[Dict]]:
        """Categorize examples by relation type"""
        relations = defaultdict(list)
        
        for sentence in self.sentences:
            for token in sentence.tokens:
                if token.head > 0:  # Skip root
                    relations[token.deprel].append({
                        'sentence': sentence,
                        'token': token,
                        'head_token': sentence.tokens[token.head - 1],
                        'context': self._get_token_context(sentence, token)
                    })
        return relations
    
    def _compute_relation_stats(self) -> Dict[str, Dict]:
        """Compute statistics for each relation type"""
        stats = defaultdict(lambda: {'count': 0, 'pos_patterns': Counter(), 'examples': []})
        
        for sentence in self.sentences:
            for token in sentence.tokens:
                if token.head > 0:
                    rel = token.deprel
                    head_token = sentence.tokens[token.head - 1]
                    
                    stats[rel]['count'] += 1
                    stats[rel]['pos_patterns'][(token.upos, head_token.upos)] += 1
                    stats[rel]['examples'].append((token.form, head_token.form))
        
        return dict(stats)
    
    def _get_token_context(self, sentence: Sentence, token: Token) -> Dict:
        """Get contextual information for a token"""
        tokens = sentence.tokens
        idx = token.id - 1  # Convert to 0-based index
        
        return {
            'prev_token': tokens[idx-1].form if idx > 0 else None,
            'next_token': tokens[idx+1].form if idx < len(tokens)-1 else None,
            'sentence_length': len(tokens),
            'position_ratio': idx / len(tokens),
            'distance_to_head': abs(token.id - token.head) if token.head > 0 else 0
        }
    
    def select_few_shot_examples(self, k: int = 5, strategy: str = 'diverse', 
                                target_sentence: Sentence = None) -> List[Dict]:
        """Select k examples using different strategies"""
        
        if strategy == 'diverse':
            return self._select_diverse_relations(k)
        elif strategy == 'frequent':
            return self._select_frequent_patterns(k)
        elif strategy == 'challenging':
            return self._select_challenging_cases(k)
        elif strategy == 'similar' and target_sentence:
            return self._select_similar_examples(k, target_sentence)
        else:
            return self._select_random_examples(k)
    
    def _select_diverse_relations(self, k: int) -> List[Dict]:
        """Select examples covering maximum relation diversity"""
        examples = []
        relations_covered = set()
        
        # Priority relations to cover first
        priority_rels = ['subj', 'comp:obj', 'comp:obl', 'mod', 'det', 'conj', 'flat']
        available_rels = set(self.relation_examples.keys())
        
        # First, cover priority relations
        for rel in priority_rels:
            if len(examples) >= k or rel not in available_rels:
                continue
            
            # Select best example for this relation
            rel_examples = self.relation_examples[rel]
            best_example = self._select_best_example_for_relation(rel_examples, rel)
            examples.append(best_example)
            relations_covered.add(rel)
        
        # Fill remaining slots with other relations
        remaining_rels = available_rels - relations_covered
        remaining_count = k - len(examples)
        
        for rel in random.sample(list(remaining_rels), 
                                min(remaining_count, len(remaining_rels))):
            best_example = self._select_best_example_for_relation(
                self.relation_examples[rel], rel)
            examples.append(best_example)
            
        return examples[:k]
    
    def _select_best_example_for_relation(self, rel_examples: List[Dict], relation: str) -> Dict:
        """Select the best example for a given relation"""
        # Prefer shorter sentences for clarity
        # Prefer common POS patterns
        # Prefer examples with glosses
        
        scored_examples = []
        for ex in rel_examples:
            score = 0
            
            # Shorter sentences are better (max 10 tokens)
            sent_len = len(ex['sentence'].tokens)
            if sent_len <= 10:
                score += (10 - sent_len) * 2
            
            # Common POS patterns are better
            token_pos = ex['token'].upos
            head_pos = ex['head_token'].upos
            pattern_freq = self.relation_stats[relation]['pos_patterns'][(token_pos, head_pos)]
            if pattern_freq >= 2:
                score += 3
            
            # Examples with glosses are better
            if ex['token'].gloss:
                score += 2
            if ex['head_token'].gloss:
                score += 2
                
            # Avoid very long distance dependencies for clarity
            distance = ex['context']['distance_to_head']
            if distance <= 3:
                score += 1
            
            scored_examples.append((score, ex))
        
        # Return highest scoring example
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return scored_examples[0][1]
    
    def _select_frequent_patterns(self, k: int) -> List[Dict]:
        """Select examples from most frequent relation patterns"""
        # Sort relations by frequency
        rel_counts = [(rel, stats['count']) for rel, stats in self.relation_stats.items()]
        rel_counts.sort(key=lambda x: x[1], reverse=True)
        
        examples = []
        for rel, count in rel_counts[:k]:
            if rel in self.relation_examples:
                best_example = self._select_best_example_for_relation(
                    self.relation_examples[rel], rel)
                examples.append(best_example)
        
        return examples
    
    def _select_challenging_cases(self, k: int) -> List[Dict]:
        """Select challenging/ambiguous cases"""
        challenging_examples = []
        
        for rel, examples_list in self.relation_examples.items():
            for ex in examples_list:
                # Consider cases challenging if:
                # 1. Long distance dependency
                # 2. Rare POS pattern  
                # 3. Ambiguous attachment
                
                distance = ex['context']['distance_to_head']
                token_pos = ex['token'].upos
                head_pos = ex['head_token'].upos
                pattern_freq = self.relation_stats[rel]['pos_patterns'][(token_pos, head_pos)]
                
                challenge_score = 0
                if distance > 3:
                    challenge_score += distance
                if pattern_freq == 1:  # Rare pattern
                    challenge_score += 3
                if len(ex['sentence'].tokens) > 8:  # Complex sentence
                    challenge_score += 2
                    
                if challenge_score > 3:
                    challenging_examples.append((challenge_score, ex))
        
        # Sort by challenge score and take top k
        challenging_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex[1] for ex in challenging_examples[:k]]
    
    def _select_similar_examples(self, k: int, target_sentence: Sentence) -> List[Dict]:
        """Select examples similar to target sentence"""
        # Similarity based on sentence length, POS patterns, etc.
        target_len = len(target_sentence.tokens)
        target_pos_sequence = [t.upos for t in target_sentence.tokens]
        
        scored_examples = []
        
        for sentence in self.sentences:
            if sentence.sent_id == target_sentence.sent_id:
                continue  # Skip identical sentence
                
            # Length similarity
            len_similarity = 1 - abs(len(sentence.tokens) - target_len) / max(target_len, 10)
            
            # POS sequence similarity (simple overlap)
            sent_pos_sequence = [t.upos for t in sentence.tokens]
            pos_overlap = len(set(target_pos_sequence) & set(sent_pos_sequence)) / len(set(target_pos_sequence))
            
            similarity_score = len_similarity * 0.4 + pos_overlap * 0.6
            
            # Add all dependency examples from this sentence
            for token in sentence.tokens:
                if token.head > 0:
                    ex = {
                        'sentence': sentence,
                        'token': token,
                        'head_token': sentence.tokens[token.head - 1],
                        'context': self._get_token_context(sentence, token)
                    }
                    scored_examples.append((similarity_score, ex))
        
        # Sort by similarity and take top k
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex[1] for ex in scored_examples[:k]]
    
    def _select_random_examples(self, k: int) -> List[Dict]:
        """Select random examples"""
        all_examples = []
        for examples_list in self.relation_examples.values():
            all_examples.extend(examples_list)
        
        return random.sample(all_examples, min(k, len(all_examples)))

class WolofPromptGenerator:
    """Generate different types of prompts for dependency parsing"""
    
    def __init__(self):
        self.relation_descriptions = {
            'subj': 'subject of a verb',
            'comp:obj': 'direct object of a verb',
            'comp:obl': 'oblique complement (indirect object, prepositional complement)',
            'mod': 'modifier (adjective, relative clause, noun modifier)',
            'det': 'determiner',
            'conj': 'conjunct in coordination',
            'conj:appos': 'apposition',
            'cc': 'coordinating conjunction',
            'flat': 'flat structure (proper names, compounds)',
            'udep': 'underspecified dependency',
            'punct': 'punctuation',
            'root': 'root of the sentence'
        }
        
        self.pos_patterns = {
            ('PRON', 'VERB', 'subj'): 'pronoun before verb typically indicates subject',
            ('NOUN', 'VERB', 'comp:obj'): 'noun after verb typically indicates direct object',
            ('NOUN', 'VERB', 'subj'): 'noun before verb can indicate subject',
            ('DET', 'NOUN', 'det'): 'determiner always modifies noun',
            ('NOUN', 'NOUN', 'mod'): 'noun can modify another noun',
            ('PROPN', 'PROPN', 'flat'): 'proper nouns form flat structures in names',
            ('ADP', 'VERB', 'udep'): 'preposition often depends on verb',
            ('NOUN', 'ADP', 'comp:obj'): 'noun is object of preposition',
            ('CCONJ', 'VERB', 'cc'): 'coordinating conjunction connects clauses'
        }
    
    def create_basic_prompt(self, target_sentence: Sentence, examples: List[Dict]) -> str:
        """Create basic few-shot learning prompt"""
        
        prompt = """You are an expert in Wolof syntax. Predict dependency relations between words.

Dependency Relations Guide:
"""
        
        # Add relation descriptions
        for rel, desc in self.relation_descriptions.items():
            prompt += f"- {rel}: {desc}\n"
        
        prompt += "\nExamples:\n\n"
        
        # Add examples
        for i, ex in enumerate(examples, 1):
            sentence = ex['sentence']
            prompt += f"Example {i}:\n"
            prompt += f'Sentence: "{sentence.text}"\n'
            prompt += f'Translation: "{sentence.text_en}"\n'
            prompt += "Dependencies:\n"
            
            # List all dependencies in this sentence
            for token in sentence.tokens:
                if token.head > 0:  # Skip root
                    head_token = sentence.tokens[token.head - 1]
                    prompt += f"- {token.form} → {head_token.form} ({token.deprel})\n"
                elif token.deprel == 'root':
                    prompt += f"- {token.form} (root)\n"
            prompt += "\n"
        
        # Add target
        prompt += f"""Now predict for:
Sentence: "{target_sentence.text}"
Translation: "{target_sentence.text_en}"

Output format:
Dependencies:
- [dependent] → [head] (relation)
- [dependent] → [head] (relation)
...

Your answer:"""
        
        return prompt
    
    def create_step_by_step_prompt(self, target_sentence: Sentence, examples: List[Dict]) -> str:
        """Create step-by-step analysis prompt"""
        
        prompt = """Task: Wolof Dependency Parsing
Analyze the sentence step by step to find syntactic dependencies.

Analysis Steps:
1. Identify the main verb (root of the sentence)
2. Find subjects of verbs
3. Find objects and complements of verbs
4. Identify modifiers and determiners
5. Handle coordination and punctuation

Example Analysis:

"""
        
        # Use the first example for detailed analysis
        if examples:
            ex = examples[0]
            sentence = ex['sentence']
            
            prompt += f'Sentence: "{sentence.text}"\n'
            prompt += f'Translation: "{sentence.text_en}"\n'
            prompt += f"Tokens: {' '.join([f'{t.form}({t.id})' for t in sentence.tokens])}\n\n"
            
            # Step-by-step analysis
            step_num = 1
            
            # Find root
            root_token = None
            for token in sentence.tokens:
                if token.head == 0 or token.deprel == 'root':
                    root_token = token
                    break
            
            if root_token:
                prompt += f'Step {step_num}: Main verb (root) = "{root_token.form}" (position {root_token.id})\n'
                step_num += 1
            
            # Analyze by relation type
            relations_by_type = defaultdict(list)
            for token in sentence.tokens:
                if token.head > 0:
                    relations_by_type[token.deprel].append(token)
            
            for rel_type in ['subj', 'comp:obj', 'comp:obl', 'mod', 'det', 'conj', 'cc', 'flat', 'punct']:
                if rel_type in relations_by_type:
                    tokens = relations_by_type[rel_type]
                    for token in tokens:
                        head_token = sentence.tokens[token.head - 1]
                        prompt += f'Step {step_num}: "{token.form}" → "{head_token.form}" ({token.deprel})\n'
                        step_num += 1
            
            prompt += "\n"
        
        # Add target analysis task
        prompt += f"""Now analyze step by step:
Sentence: "{target_sentence.text}"
Translation: "{target_sentence.text_en}"
Tokens: {' '.join([f'{t.form}({t.id})' for t in target_sentence.tokens])}

Your step-by-step analysis:"""
        
        return prompt
    
    def create_context_rich_prompt(self, target_sentence: Sentence, examples: List[Dict]) -> str:
        """Create prompt with rich linguistic context"""
        
        prompt = """Wolof Dependency Parsing with Linguistic Context

Wolof Language Properties:
- Basic word order: Subject-Verb-Object (SVO)
- Determiners typically follow nouns (noun + determiner)
- Relative clauses use pronouns like "ki/ku/li"
- Prepositions precede their objects
- Proper names form flat structures
- Agglutinative morphology with various affixes

Common Syntactic Patterns:
"""
        
        # Add pattern descriptions
        for (token_pos, head_pos, rel), description in self.pos_patterns.items():
            prompt += f"- {token_pos} + {head_pos} → {rel}: {description}\n"
        
        prompt += "\nExample with Detailed Analysis:\n\n"
        
        # Detailed example analysis
        if examples:
            ex = examples[0]
            sentence = ex['sentence']
            
            prompt += f'Sentence: "{sentence.text}"\n'
            prompt += f'Translation: "{sentence.text_en}"\n\n'
            prompt += "Linguistic Analysis:\n"
            
            for token in sentence.tokens:
                if token.head > 0:
                    head_token = sentence.tokens[token.head - 1]
                    
                    # Add reasoning
                    pattern_key = (token.upos, head_token.upos, token.deprel)
                    reasoning = self.pos_patterns.get(pattern_key, 
                                f"{token.deprel} relationship based on context")
                    
                    gloss_info = f" (gloss: '{token.gloss}')" if token.gloss else ""
                    
                    prompt += f'- "{token.form}" ({token.upos}){gloss_info} → "{head_token.form}" ({head_token.upos}): {reasoning}\n'
                elif token.deprel == 'root':
                    prompt += f'- "{token.form}" ({token.upos}): main predicate of sentence\n'
            
            prompt += "\n"
        
        # Add more examples briefly
        for ex in examples[1:3]:  # Add 2 more examples briefly
            sentence = ex['sentence']
            prompt += f'Brief example: "{sentence.text}"\n'
            dependencies = []
            for token in sentence.tokens:
                if token.head > 0:
                    head_token = sentence.tokens[token.head - 1]
                    dependencies.append(f"{token.form}→{head_token.form}({token.deprel})")
                elif token.deprel == 'root':
                    dependencies.append(f"{token.form}(root)")
            prompt += f"Dependencies: {', '.join(dependencies)}\n\n"
        
        # Add target
        prompt += f"""Now analyze with detailed reasoning:
Sentence: "{target_sentence.text}"
Translation: "{target_sentence.text_en}"

Provide linguistic analysis and dependencies:"""
        
        return prompt
    
    def create_interactive_prompt(self, target_sentence: Sentence, examples: List[Dict]) -> str:
        """Create interactive question-based prompt"""
        
        prompt = """I'll help you analyze this Wolof sentence for dependencies step by step.

Let me show you how with an example:

"""
        
        if examples:
            ex = examples[0]
            sentence = ex['sentence']
            
            prompt += f'Example Sentence: "{sentence.text}"\n'
            prompt += f'Translation: "{sentence.text_en}"\n\n'
            
            # Interactive analysis
            root_token = None
            for token in sentence.tokens:
                if token.head == 0 or token.deprel == 'root':
                    root_token = token
                    break
            
            if root_token:
                prompt += f'Question 1: What is the main verb?\nAnswer: "{root_token.form}" at position {root_token.id}\n\n'
            
            # Find subjects
            subjects = [t for t in sentence.tokens if t.deprel == 'subj']
            if subjects:
                subj_list = [f'"{t.form}"' for t in subjects]
                prompt += f'Question 2: What are the subjects?\nAnswer: {", ".join(subj_list)}\n\n'
            
            # Find objects  
            objects = [t for t in sentence.tokens if 'comp:obj' in t.deprel]
            if objects:
                obj_list = [f'"{t.form}"' for t in objects]
                prompt += f'Question 3: What are the objects?\nAnswer: {", ".join(obj_list)}\n\n'
            
            # Show final dependencies
            prompt += "Final Dependencies:\n"
            for token in sentence.tokens:
                if token.head > 0:
                    head_token = sentence.tokens[token.head - 1]
                    prompt += f"- {token.form} → {head_token.form} ({token.deprel})\n"
                elif token.deprel == 'root':
                    prompt += f"- {token.form} (root)\n"
            prompt += "\n"
        
        # Now ask for target analysis
        prompt += f"""Now let's analyze this sentence together:
Sentence: "{target_sentence.text}"
Translation: "{target_sentence.text_en}"

Question 1: What is the main verb (root) in this sentence?
Question 2: What words are subjects of verbs?
Question 3: What words are objects or complements?
Question 4: What modifiers and determiners do you see?
Question 5: Any coordination or punctuation?

Please provide your analysis:"""
        
        return prompt

class WolofDependencyEvaluator:
    """Evaluate dependency parsing predictions"""
    
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self):
        """Reset evaluation statistics"""
        self.total_tokens = 0
        self.correct_heads = 0
        self.correct_relations = 0  
        self.correct_both = 0
        self.relation_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        self.error_analysis = defaultdict(int)
    
    def evaluate_sentence(self, predicted: Sentence, gold: Sentence) -> Dict:
        """Evaluate a single sentence"""
        sentence_stats = {
            'total': 0,
            'correct_heads': 0,
            'correct_relations': 0,
            'correct_both': 0,
            'errors': []
        }
        
        # Ensure same number of tokens
        if len(predicted.tokens) != len(gold.tokens):
            return {'error': 'Token count mismatch'}
        
        for pred_token, gold_token in zip(predicted.tokens, gold.tokens):
            sentence_stats['total'] += 1
            self.total_tokens += 1
            
            # Check head correctness
            head_correct = pred_token.head == gold_token.head
            if head_correct:
                sentence_stats['correct_heads'] += 1
                self.correct_heads += 1
            
            # Check relation correctness
            rel_correct = pred_token.deprel == gold_token.deprel
            if rel_correct:
                sentence_stats['correct_relations'] += 1
                self.correct_relations += 1
                
            # Check both correct (LAS)
            if head_correct and rel_correct:
                sentence_stats['correct_both'] += 1
                self.correct_both += 1
                self.relation_stats[gold_token.deprel]['tp'] += 1
            else:
                # Track errors
                if not head_correct and not rel_correct:
                    error_type = "head_and_rel_error"
                elif not head_correct:
                    error_type = "head_error"
                else:
                    error_type = "rel_error"
                
                sentence_stats['errors'].append({
                    'token': pred_token.form,
                    'error_type': error_type,
                    'predicted_head': pred_token.head,
                    'gold_head': gold_token.head,
                    'predicted_rel': pred_token.deprel,
                    'gold_rel': gold_token.deprel
                })
                
                self.error_analysis[error_type] += 1
                
                # Update relation stats
                self.relation_stats[gold_token.deprel]['fn'] += 1
                if pred_token.deprel != gold_token.deprel:
                    self.relation_stats[pred_token.deprel]['fp'] += 1
        
        return sentence_stats
    
    def compute_metrics(self) -> Dict:
        """Compute overall evaluation metrics"""
        if self.total_tokens == 0:
            return {'error': 'No tokens evaluated'}
        
        # Overall metrics
        uas = self.correct_heads / self.total_tokens  # Unlabeled Attachment Score
        las = self.correct_both / self.total_tokens   # Labeled Attachment Score
        rel_acc = self.correct_relations / self.total_tokens  # Relation Accuracy
        
        # Per-relation metrics
        per_relation_metrics = {}
        for rel, stats in self.relation_stats.items():
            tp = stats['tp']
            fp = stats['fp'] 
            fn = stats['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_relation_metrics[rel] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            }
        
        # Macro-averaged F1
        f1_scores = [metrics['f1'] for metrics in per_relation_metrics.values() 
                    if metrics['support'] > 0]
        macro_f1 = statistics.mean(f1_scores) if f1_scores else 0
        
        return {
            'total_tokens': self.total_tokens,
            'uas': uas,
            'las': las,
            'relation_accuracy': rel_acc,
            'macro_f1': macro_f1,
            'per_relation_metrics': per_relation_metrics,
            'error_analysis': dict(self.error_analysis)
        }
    
    def print_evaluation_report(self, metrics: Dict):
        """Print a detailed evaluation report"""
        print("=" * 60)
        print("WOLOF DEPENDENCY PARSING EVALUATION REPORT")
        print("=" * 60)
        
        print(f"Total tokens evaluated: {metrics['total_tokens']}")
        print(f"Unlabeled Attachment Score (UAS): {metrics['uas']:.3f}")
        print(f"Labeled Attachment Score (LAS): {metrics['las']:.3f}")
        print(f"Relation Accuracy: {metrics['relation_accuracy']:.3f}")
        print(f"Macro F1 Score: {metrics['macro_f1']:.3f}")
        
        print("\nPer-Relation Performance:")
        print("-" * 60)
        print(f"{'Relation':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        print("-" * 60)
        
        for rel, rel_metrics in sorted(metrics['per_relation_metrics'].items()):
            print(f"{rel:<15} {rel_metrics['precision']:<10.3f} "
                  f"{rel_metrics['recall']:<10.3f} {rel_metrics['f1']:<10.3f} "
                  f"{rel_metrics['support']:<10}")
        
        print("\nError Analysis:")
        print("-" * 30)
        for error_type, count in metrics['error_analysis'].items():
            print(f"{error_type}: {count}")

class WolofDependencyExperimentRunner:
    """Run comprehensive dependency parsing experiments"""
    
    def __init__(self, sentences: List[Sentence]):
        self.sentences = sentences
        self.engineer = WolofDependencyPromptEngineer(sentences)
        self.prompt_generator = WolofPromptGenerator()
        self.evaluator = WolofDependencyEvaluator()
        
    def create_train_test_split(self, test_ratio: float = 0.2, random_seed: int = 42):
        """Split data into train and test sets"""
        random.seed(random_seed)
        shuffled = self.sentences.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1 - test_ratio))
        train_sentences = shuffled[:split_idx]
        test_sentences = shuffled[split_idx:]
        
        return train_sentences, test_sentences
    
    def run_experiment(self, train_sentences: List[Sentence], test_sentences: List[Sentence],
                      k_shot: int = 5, strategy: str = 'diverse', 
                      prompt_type: str = 'basic', max_test_samples: int = None) -> Dict:
        """Run a single experiment configuration"""
        
        results = {
            'config': {
                'k_shot': k_shot,
                'strategy': strategy, 
                'prompt_type': prompt_type,
                'train_size': len(train_sentences),
                'test_size': len(test_sentences)
            },
            'predictions': [],
            'evaluations': []
        }
        
        # Create engineer with training data
        train_engineer = WolofDependencyPromptEngineer(train_sentences)
        
        # Limit test samples if specified
        test_sample = test_sentences[:max_test_samples] if max_test_samples else test_sentences
        
        print(f"\nRunning experiment: {k_shot}-shot, {strategy} strategy, {prompt_type} prompts")
        print(f"Testing on {len(test_sample)} sentences...")
        
        for i, test_sentence in enumerate(test_sample):
            if i % 5 == 0:
                print(f"Processing sentence {i+1}/{len(test_sample)}")
            
            # Select few-shot examples
            examples = train_engineer.select_few_shot_examples(
                k=k_shot, strategy=strategy, target_sentence=test_sentence)
            
            # Generate prompt
            if prompt_type == 'basic':
                prompt = self.prompt_generator.create_basic_prompt(test_sentence, examples)
            elif prompt_type == 'step_by_step':
                prompt = self.prompt_generator.create_step_by_step_prompt(test_sentence, examples)
            elif prompt_type == 'context_rich':
                prompt = self.prompt_generator.create_context_rich_prompt(test_sentence, examples)
            elif prompt_type == 'interactive':
                prompt = self.prompt_generator.create_interactive_prompt(test_sentence, examples)
            else:
                prompt = self.prompt_generator.create_basic_prompt(test_sentence, examples)
            
            # Store prompt and gold standard (in real implementation, you'd call LLM here)
            results['predictions'].append({
                'sentence_id': test_sentence.sent_id,
                'prompt': prompt,
                'gold_standard': test_sentence,
                # 'llm_prediction': None  # Would be filled by LLM call
            })
        
        return results
    
    def run_comprehensive_experiments(self, max_test_samples: int = 20) -> Dict:
        """Run comprehensive experiments with different configurations"""
        
        # Create train/test split
        train_sentences, test_sentences = self.create_train_test_split()
        
        # Experiment configurations
        configurations = [
            # Vary shot count
            {'k_shot': 1, 'strategy': 'diverse', 'prompt_type': 'basic'},
            {'k_shot': 3, 'strategy': 'diverse', 'prompt_type': 'basic'},
            {'k_shot': 5, 'strategy': 'diverse', 'prompt_type': 'basic'},
            {'k_shot': 10, 'strategy': 'diverse', 'prompt_type': 'basic'},
            
            # Vary strategy (with 5-shot)
            {'k_shot': 5, 'strategy': 'diverse', 'prompt_type': 'basic'},
            {'k_shot': 5, 'strategy': 'frequent', 'prompt_type': 'basic'},
            {'k_shot': 5, 'strategy': 'challenging', 'prompt_type': 'basic'},
            {'k_shot': 5, 'strategy': 'similar', 'prompt_type': 'basic'},
            
            # Vary prompt type (with 5-shot diverse)
            {'k_shot': 5, 'strategy': 'diverse', 'prompt_type': 'basic'},
            {'k_shot': 5, 'strategy': 'diverse', 'prompt_type': 'step_by_step'},
            {'k_shot': 5, 'strategy': 'diverse', 'prompt_type': 'context_rich'},
            {'k_shot': 5, 'strategy': 'diverse', 'prompt_type': 'interactive'},
        ]
        
        all_results = {}
        
        for i, config in enumerate(configurations, 1):
            print(f"\n{'='*60}")
            print(f"EXPERIMENT {i}/{len(configurations)}")
            print(f"{'='*60}")
            
            exp_key = f"{config['k_shot']}shot_{config['strategy']}_{config['prompt_type']}"
            
            result = self.run_experiment(
                train_sentences, test_sentences,
                k_shot=config['k_shot'],
                strategy=config['strategy'], 
                prompt_type=config['prompt_type'],
                max_test_samples=max_test_samples
            )
            
            all_results[exp_key] = result
        
        return all_results
    
    def analyze_results(self, all_results: Dict):
        """Analyze and compare experiment results"""
        
        print(f"\n{'='*80}")
        print("EXPERIMENT RESULTS ANALYSIS")
        print(f"{'='*80}")
        
        # In a real implementation, you would:
        # 1. Parse LLM outputs to extract predicted dependencies
        # 2. Evaluate against gold standard
        # 3. Compute metrics for each experiment
        # 4. Compare results across configurations
        
        print("Results summary:")
        for exp_name, result in all_results.items():
            config = result['config']
            num_prompts = len(result['predictions'])
            
            print(f"\nExperiment: {exp_name}")
            print(f"  Configuration: {config['k_shot']}-shot, {config['strategy']} strategy, {config['prompt_type']} prompts")
            print(f"  Generated {num_prompts} prompts")
            print(f"  Average prompt length: {statistics.mean([len(p['prompt']) for p in result['predictions']]):.0f} characters")
        
        # Show sample prompts
        print(f"\n{'='*60}")
        print("SAMPLE PROMPTS")
        print(f"{'='*60}")
        
        for exp_name, result in list(all_results.items())[:2]:  # Show first 2 experiments
            print(f"\nSample prompt from {exp_name}:")
            print("-" * 40)
            sample_prompt = result['predictions'][0]['prompt']
            print(sample_prompt[:500] + "..." if len(sample_prompt) > 500 else sample_prompt)
            print("-" * 40)
    
    def save_results(self, all_results: Dict, filename: str):
        """Save experiment results to file"""
        
        # Prepare data for saving (remove unpicklable objects)
        save_data = {}
        for exp_name, result in all_results.items():
            save_data[exp_name] = {
                'config': result['config'],
                'predictions': [
                    {
                        'sentence_id': p['sentence_id'],
                        'prompt_length': len(p['prompt']),
                        'sentence_text': p['gold_standard'].text,
                        'sentence_translation': p['gold_standard'].text_en,
                        'num_tokens': len(p['gold_standard'].tokens)
                    }
                    for p in result['predictions']
                ]
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {filename}")

class WolofLLMInterface:
    """Interface for calling LLMs with prompts"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        
    def call_llm(self, prompt: str) -> str:
        """Call LLM with prompt (placeholder implementation)"""
        
        # This is a placeholder - in real implementation you would:
        # 1. Use OpenAI API, Anthropic API, or other LLM service
        # 2. Handle API keys, rate limiting, etc.
        # 3. Parse and return the response
        
        print(f"\n[PLACEHOLDER] Calling {self.model_name} with prompt:")
        print("-" * 50)
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("-" * 50)
        
        # Return mock response
        return """Dependencies:
- Ki → yore (subj)
- wàllu → yore (comp:obj)
- tàggatu → wàllu (mod)
- gi → wàllu (det)
- ca → yore (udep)
- Dakaar → ca (comp:obj)"""
    
    def parse_llm_response(self, response: str, target_sentence: Sentence) -> Sentence:
        """Parse LLM response to extract predicted dependencies"""
        
        # Create copy of target sentence for predictions
        predicted_sentence = Sentence(
            sent_id=target_sentence.sent_id,
            text=target_sentence.text,
            text_en=target_sentence.text_en,
            tokens=[Token(
                id=t.id, form=t.form, lemma=t.lemma, upos=t.upos,
                head=0, deprel='root', gloss=t.gloss
            ) for t in target_sentence.tokens]
        )
        
        # Parse dependencies from response
        # This is a simplified parser - real implementation would be more robust
        
        lines = response.strip().split('\n')
        for line in lines:
            if '→' in line and '(' in line:
                # Extract dependency: "word1 → word2 (relation)"
                match = re.match(r'.*?([^\s]+)\s*→\s*([^\s]+)\s*\(([^)]+)\)', line)
                if match:
                    dependent, head_word, relation = match.groups()
                    
                    # Find tokens by form
                    dep_token = None
                    head_token = None
                    
                    for token in predicted_sentence.tokens:
                        if token.form == dependent:
                            dep_token = token
                        if token.form == head_word:
                            head_token = token
                    
                    if dep_token and head_token:
                        dep_token.head = head_token.id
                        dep_token.deprel = relation
                    elif relation == 'root' and dep_token:
                        dep_token.head = 0
                        dep_token.deprel = 'root'
        
        return predicted_sentence

# Usage and Demo Functions

def demo_data_loading():
    """Demonstrate data loading functionality"""
    print("DEMO: Data Loading")
    print("=" * 40)
    
    # Create sample data
    loader = WolofDataLoader()
    sentences = loader.create_sample_data()
    
    print(f"Loaded {len(sentences)} sentences")
    
    for sentence in sentences:
        print(f"\nSentence ID: {sentence.sent_id}")
        print(f"Wolof: {sentence.text}")
        print(f"English: {sentence.text_en}")
        print(f"Tokens: {len(sentence.tokens)}")
        
        # Show first few tokens
        for token in sentence.tokens[:5]:
            print(f"  {token.id}: {token.form} ({token.upos}) → {token.head} ({token.deprel}) [Gloss: {token.gloss}]")
        if len(sentence.tokens) > 5:
            print(f"  ... and {len(sentence.tokens) - 5} more tokens")
    
    return sentences

def demo_prompt_engineering():
    """Demonstrate prompt engineering functionality"""
    print("\nDEMO: Prompt Engineering")
    print("=" * 40)
    
    # Load data
    loader = WolofDataLoader()
    sentences = loader.create_sample_data()
    
    # Initialize components
    engineer = WolofDependencyPromptEngineer(sentences)
    prompt_generator = WolofPromptGenerator()
    
    # Show relation statistics
    print("Relation Statistics:")
    for rel, stats in engineer.relation_stats.items():
        print(f"  {rel}: {stats['count']} occurrences")
    
    # Demonstrate example selection
    target_sentence = sentences[0]
    
    print(f"\nTarget sentence: {target_sentence.text}")
    
    strategies = ['diverse', 'frequent', 'challenging']
    for strategy in strategies:
        print(f"\n{strategy.upper()} strategy examples:")
        examples = engineer.select_few_shot_examples(k=3, strategy=strategy)
        for i, ex in enumerate(examples, 1):
            print(f"  {i}. {ex['sentence'].text[:50]}...")
    
    # Demonstrate different prompt types
    examples = engineer.select_few_shot_examples(k=3, strategy='diverse')
    
    prompt_types = ['basic', 'step_by_step', 'context_rich', 'interactive']
    for prompt_type in prompt_types:
        print(f"\n{prompt_type.upper()} PROMPT:")
        print("-" * 50)
        
        if prompt_type == 'basic':
            prompt = prompt_generator.create_basic_prompt(target_sentence, examples)
        elif prompt_type == 'step_by_step':
            prompt = prompt_generator.create_step_by_step_prompt(target_sentence, examples)
        elif prompt_type == 'context_rich':
            prompt = prompt_generator.create_context_rich_prompt(target_sentence, examples)
        else:
            prompt = prompt_generator.create_interactive_prompt(target_sentence, examples)
        
        # Show first part of prompt
        print(prompt[:400] + "..." if len(prompt) > 400 else prompt)

def demo_full_experiment():
    """Demonstrate full experimental pipeline"""
    print("\nDEMO: Full Experimental Pipeline")
    print("=" * 50)
    
    # Load data
    loader = WolofDataLoader()
    sentences = loader.create_sample_data()
    
    # Run experiments
    runner = WolofDependencyExperimentRunner(sentences)
    results = runner.run_comprehensive_experiments(max_test_samples=2)
    
    # Analyze results
    runner.analyze_results(results)
    
    # Save results
    runner.save_results(results, "wolof_dependency_experiments.json")
    
    return results

if __name__ == "__main__":
    print("WOLOF DEPENDENCY PARSING FRAMEWORK")
    print("=" * 60)
    
    # Run all demos
    sentences = demo_data_loading()
    demo_prompt_engineering() 
    results = demo_full_experiment()
    
    print(f"\n{'='*60}")
    print("FRAMEWORK READY FOR USE!")
    print(f"{'='*60}")
    print("\nTo use this framework:")
    print("1. Load your Wolof data: loader = WolofDataLoader('your_file.conllu')")
    print("2. Create experiment runner: runner = WolofDependencyExperimentRunner(sentences)")
    print("3. Run experiments: results = runner.run_comprehensive_experiments()")
    print("4. Integrate with your LLM API using WolofLLMInterface")
    print("\nSee demo functions above for detailed examples!")