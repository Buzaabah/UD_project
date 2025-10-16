from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class PromptStyle(Enum):
    BASIC = "basic"
    DETAILED = "detailed"
    CHAIN_OF_THOUGHT = "cot"
    MULTILINGUAL = "multilingual"
    GLOSS_FOCUSED = "gloss_focused"

@dataclass
class PromptTemplate:
    name: str
    template: str
    style: PromptStyle
    description: str

class WolofDependencyPromptGenerator:
    """Generate prompts for Wolof dependency relation prediction"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize all prompt templates"""
        return {
            "basic": PromptTemplate(
                name="Basic Dependency Prediction",
                template="""Task: Predict dependency relations for Wolof sentences.

Given a Wolof sentence with tokens, predict the syntactic dependency relation for each token to its head.

Examples:
{examples}

Now predict the dependency relations for this sentence:

Wolof: {target_text}
English: {target_text_en}

Tokens:
{target_tokens}

Output format: For each token, provide: TOKEN_ID FORM -> HEAD_ID RELATION

Answer:""",
                style=PromptStyle.BASIC,
                description="Simple template for basic dependency prediction"
            ),
            
            "detailed": PromptTemplate(
                name="Detailed Linguistic Analysis",
                template="""Task: Analyze Wolof syntax and predict dependency relations.

You are analyzing Wolof, a West African language. Each token has linguistic annotations including part-of-speech tags and English glosses to help with understanding.

Universal Dependencies Relations Guide:
- subj: subject of a verb
- comp:obj: object complement  
- det: determiner
- mod: modifier (adjective, adverb, etc.)
- conj: conjunction
- punct: punctuation
- flat: flat multiword expression (names)
- cc: coordinating conjunction

Examples:
{examples}

Analysis Task:
Wolof sentence: {target_text}
English translation: {target_text_en}

Tokens with linguistic information:
{target_tokens_detailed}

For each token, analyze:
1. Its grammatical role in the sentence
2. Which word it depends on (head)  
3. The type of dependency relation

Provide your predictions in this format:
TOKEN_ID FORM [POS] -> HEAD_ID RELATION (reasoning)

Answer:""",
                style=PromptStyle.DETAILED,
                description="Detailed template with linguistic context and reasoning"
            ),
            
            "chain_of_thought": PromptTemplate(
                name="Chain of Thought Dependency Analysis",
                template="""Task: Step-by-step dependency relation prediction for Wolof.

Let's analyze Wolof dependency relations step by step, using the English glosses and translation to understand the meaning.

Examples with reasoning:
{examples_with_reasoning}

Now let's analyze this sentence step by step:

Wolof: {target_text}
English: {target_text_en}

Step 1: Identify the main verb and root of the sentence
Step 2: Find the subject and object relationships  
Step 3: Identify modifiers, determiners, and other relations
Step 4: Handle coordination and subordination
Step 5: Assign final dependency relations

Tokens:
{target_tokens_detailed}

Let me work through this step by step:

Step 1 - Main verb identification:
[Think about which word is the main predicate]

Step 2 - Core arguments:
[Identify subject, object relationships]

Step 3 - Modifiers and determiners:
[Find words that modify or determine others]

Step 4 - Complex relations:
[Handle coordination, subordination, etc.]

Step 5 - Final predictions:
[Format: TOKEN_ID FORM -> HEAD_ID RELATION]""",
                style=PromptStyle.CHAIN_OF_THOUGHT,
                description="Step-by-step reasoning approach"
            ),
            
            "multilingual": PromptTemplate(
                name="Multilingual Cross-lingual Analysis",
                template="""Task: Cross-lingual dependency parsing for Wolof using English alignment.

You will analyze Wolof syntax by leveraging both the Wolof text and its English translation. Use the word-level glosses to understand correspondences between languages.

Key principles:
- Wolof may have different syntactic patterns than English
- Use glosses to understand semantic roles
- Consider that Wolof may have different word order patterns
- Some relations may not have direct English equivalents

Examples:
{examples_crosslingual}

Target Analysis:

Wolof text: {target_text}
English text: {target_text_en}

Word-by-word analysis:
{target_alignment}

Cross-lingual insights:
- Compare Wolof and English syntactic patterns
- Use English structure as a guide but respect Wolof grammar
- Pay attention to word order differences
- Consider morphological differences

Dependency predictions:
[For each Wolof token, predict head and relation based on both languages]

TOKEN_ID WOLOF_FORM (Gloss: ENGLISH_GLOSS) -> HEAD_ID RELATION

Answer:""",
                style=PromptStyle.MULTILINGUAL,
                description="Cross-lingual template using English alignment"
            ),
            
            "gloss_focused": PromptTemplate(
                name="Gloss-Guided Dependency Prediction",
                template="""Task: Use English glosses to guide Wolof dependency relation prediction.

The English glosses provide word-level translations that help understand the semantic role of each Wolof word. Use these glosses to infer syntactic relationships.

Gloss interpretation guide:
- Articles (the, a) usually indicate determiners (det)
- Prepositions (in, on, of) often show oblique relations (comp:obj, udep)
- Conjunctions (and, or) indicate coordination (cc, conj)
- Pronouns often serve as subjects or objects (subj, comp:obj)

Examples:
{examples_gloss_focused}

Target sentence analysis:

Wolof: {target_text}
English: {target_text_en}

Gloss-guided token analysis:
{target_gloss_analysis}

Using the glosses, predict dependency relations:

Instructions:
1. Read each gloss to understand the word's meaning
2. Determine how that meaning relates to other words
3. Assign the appropriate syntactic relation
4. Consider Wolof-specific patterns from the examples

Predictions:
TOKEN_ID FORM (gloss) -> HEAD_ID RELATION

Answer:""",
                style=PromptStyle.GLOSS_FOCUSED,
                description="Template emphasizing gloss information"
            ),
            
            "few_shot_adaptive": PromptTemplate(
                name="Adaptive Few-Shot Learning",
                template="""Task: Learn Wolof dependency patterns from examples and apply to new sentence.

You are learning to parse Wolof dependency relations. Study the patterns in the examples carefully, noting:
- Common dependency relations in Wolof
- Typical word order patterns  
- How different parts of speech relate to each other
- Role of glosses in understanding meaning

Learning examples:
{adaptive_examples}

Pattern observations from examples:
[Based on the examples above, note key patterns you observe]

Target sentence to parse:
Wolof: {target_text}
English: {target_text_en}

Tokens: {target_tokens}

Apply the patterns you learned:
TOKEN_ID FORM -> HEAD_ID RELATION

Answer:""",
                style=PromptStyle.BASIC,
                description="Adaptive learning template"
            )
        }
    
    def format_example_basic(self, sentence) -> str:
        """Format a sentence as a basic example"""
        example = f"Wolof: {sentence.text}\n"
        example += f"English: {sentence.text_en}\n"
        
        for token in sentence.tokens:
            if token.id > 0:  # Skip special tokens
                example += f"{token.id} {token.form} -> {token.head} {token.deprel}\n"
        
        return example + "\n"
    
    def format_example_detailed(self, sentence) -> str:
        """Format a sentence with detailed linguistic information"""
        example = f"Wolof: {sentence.text}\n"
        example += f"English: {sentence.text_en}\n"
        example += "Detailed analysis:\n"
        
        for token in sentence.tokens:
            if token.id > 0:
                gloss_part = f" (gloss: {token.gloss})" if token.gloss else ""
                example += f"{token.id} {token.form} [{token.upos}]{gloss_part} -> {token.head} {token.deprel}\n"
        
        return example + "\n"
    
    def format_example_with_reasoning(self, sentence) -> str:
        """Format example with step-by-step reasoning"""
        example = f"Sentence: {sentence.text} ({sentence.text_en})\n"
        
        # Find main verb (root)
        root_token = None
        for token in sentence.tokens:
            if token.deprel == 'root' or token.head == 0:
                root_token = token
                break
        
        example += f"Analysis:\n"
        if root_token:
            example += f"- Main verb: '{root_token.form}' (gloss: {root_token.gloss})\n"
        
        # Show some key relations
        subjects = [t for t in sentence.tokens if 'subj' in t.deprel]
        objects = [t for t in sentence.tokens if 'obj' in t.deprel]
        
        if subjects:
            example += f"- Subject(s): {', '.join([f'{t.form}' for t in subjects])}\n"
        if objects:
            example += f"- Object(s): {', '.join([f'{t.form}' for t in objects])}\n"
        
        example += "Dependencies:\n"
        for token in sentence.tokens:
            if token.id > 0:
                example += f"{token.id} {token.form} -> {token.head} {token.deprel}\n"
        
        return example + "\n"
    
    def format_target_tokens_basic(self, sentence) -> str:
        """Format target tokens in basic format"""
        tokens_str = ""
        for token in sentence.tokens:
            if token.id > 0:
                tokens_str += f"{token.id} {token.form} [{token.upos}]\n"
        return tokens_str
    
    def format_target_tokens_detailed(self, sentence) -> str:
        """Format target tokens with detailed information"""
        tokens_str = ""
        for token in sentence.tokens:
            if token.id > 0:
                gloss_part = f" (gloss: '{token.gloss}')" if token.gloss else ""
                tokens_str += f"{token.id} {token.form} [{token.upos}]{gloss_part}\n"
        return tokens_str
    
    def format_crosslingual_alignment(self, sentence) -> str:
        """Format word alignment between Wolof and English"""
        alignment = ""
        for token in sentence.tokens:
            if token.id > 0:
                gloss = token.gloss if token.gloss else "?"
                alignment += f"{token.id}. {token.form} ↔ {gloss}\n"
        return alignment
    
    def format_gloss_analysis(self, sentence) -> str:
        """Format detailed gloss analysis"""
        analysis = ""
        for token in sentence.tokens:
            if token.id > 0:
                gloss = token.gloss if token.gloss else "no gloss"
                pos_info = f"[{token.upos}]" if token.upos != '*' else ""
                analysis += f"{token.id}. {token.form} {pos_info} → gloss: '{gloss}'\n"
        return analysis
    
    def generate_prompt(self, 
                       template_name: str,
                       examples: List,
                       target_sentence,
                       **kwargs) -> str:
        """Generate a complete prompt using specified template"""
        
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        
        # Format examples based on template style
        if template.style == PromptStyle.BASIC:
            examples_str = "\n".join([self.format_example_basic(ex) for ex in examples])
        elif template.style == PromptStyle.DETAILED:
            examples_str = "\n".join([self.format_example_detailed(ex) for ex in examples])
        elif template.style == PromptStyle.CHAIN_OF_THOUGHT:
            examples_str = "\n".join([self.format_example_with_reasoning(ex) for ex in examples])
        else:
            examples_str = "\n".join([self.format_example_detailed(ex) for ex in examples])
        
        # Prepare template variables
        template_vars = {
            'examples': examples_str,
            'examples_with_reasoning': "\n".join([self.format_example_with_reasoning(ex) for ex in examples]),
            'examples_crosslingual': "\n".join([self.format_example_detailed(ex) for ex in examples]),
            'examples_gloss_focused': "\n".join([self.format_example_detailed(ex) for ex in examples]),
            'adaptive_examples': "\n".join([self.format_example_detailed(ex) for ex in examples]),
            'target_text': target_sentence.text,
            'target_text_en': target_sentence.text_en,
            'target_tokens': self.format_target_tokens_basic(target_sentence),
            'target_tokens_detailed': self.format_target_tokens_detailed(target_sentence),
            'target_alignment': self.format_crosslingual_alignment(target_sentence),
            'target_gloss_analysis': self.format_gloss_analysis(target_sentence),
        }
        
        # Add any additional kwargs
        template_vars.update(kwargs)
        
        return template.template.format(**template_vars)
    
    def get_available_templates(self) -> Dict[str, str]:
        """Get list of available templates with descriptions"""
        return {name: template.description for name, template in self.templates.items()}
    
    def create_custom_template(self, name: str, template_str: str, style: PromptStyle, description: str):
        """Add a custom template"""
        self.templates[name] = PromptTemplate(name, template_str, style, description)

# Example usage and testing
def demo_prompt_generation():
    """Demonstrate prompt generation"""
    generator = WolofDependencyPromptGenerator()
    
    # Mock sentence data for demonstration
    class MockToken:
        def __init__(self, id, form, upos, head, deprel, gloss=""):
            self.id = id
            self.form = form
            self.upos = upos
            self.head = head
            self.deprel = deprel
            self.gloss = gloss
    
    class MockSentence:
        def __init__(self, text, text_en, tokens):
            self.text = text
            self.text_en = text_en
            self.tokens = tokens
    
    # Example sentence (simplified)
    example_tokens = [
        MockToken(1, "Ki", "PRON", 2, "subj", "the one who"),
        MockToken(2, "yore", "VERB", 0, "root", "be in charge"),
        MockToken(3, "wàllu", "NOUN", 2, "comp:obj", "part of")
    ]
    
    target_tokens = [
        MockToken(1, "Mu", "PRON", 2, "subj", "he"),
        MockToken(2, "dem", "VERB", 0, "root", "go"),
        MockToken(3, "ca", "ADP", 2, "udep", "to")
    ]
    
    example_sentence = MockSentence(
        "Ki yore wàllu",
        "The one who is in charge of",
        example_tokens
    )
    
    target_sentence = MockSentence(
        "Mu dem ca",
        "He goes to",
        target_tokens
    )
    
    print("Available Prompt Templates:")
    print("=" * 50)
    for name, desc in generator.get_available_templates().items():
        print(f"- {name}: {desc}")
    
    print("\n" + "="*50)
    print("SAMPLE PROMPT - Basic Template:")
    print("="*50)
    prompt = generator.generate_prompt("basic", [example_sentence], target_sentence)
    print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)

if __name__ == "__main__":
    demo_prompt_generation()