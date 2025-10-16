"""
debug_parser.py - Debug script to check LLM response parsing issues
"""

import sys
import os
import re
sys.path.append('.')

try:
    from LLM_testing import WolofExperimentPipeline, LLMClient
    from evaluation import LLMResponseParser
    from prompts import WolofDependencyPromptGenerator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all component files are in the same directory")
    sys.exit(1)

def debug_parser_patterns():
    """Test the parser with various response formats"""
    
    print("🔍 DEBUGGING LLM RESPONSE PARSING")
    print("=" * 50)
    
    parser = LLMResponseParser()
    
    # Sample responses that the model might generate
    sample_responses = [
        # Format 1: Basic
        "1 Ki -> 2 subj\n2 yore -> 0 root\n3 wàllu -> 2 comp:obj",
        
        # Format 2: With POS tags
        "1 Ki [PRON] -> 2 subj\n2 yore [VERB] -> 0 root\n3 wàllu [NOUN] -> 2 comp:obj",
        
        # Format 3: With explanations
        "1. Ki (the one who) -> 2 subj\n2. yore (be in charge) -> 0 root\n3. wàllu (part of) -> 2 comp:obj",
        
        # Format 4: Verbose format
        "Token 1: Ki depends on token 2 with relation subj\nToken 2: yore is the root (relation: root)\nToken 3: wàllu depends on token 2 with relation comp:obj",
        
        # Format 5: Table format
        "ID | Form | Head | Relation\n1  | Ki   | 2    | subj\n2  | yore | 0    | root\n3  | wàllu| 2    | comp:obj",
        
        # Format 6: Natural language
        """Based on the analysis:
        
        1. Ki (gloss: the one who) -> 2 subj
        2. yore (gloss: be in charge) -> 0 root  
        3. wàllu (gloss: part of) -> 2 comp:obj
        
        The sentence structure shows Ki as subject of yore.""",
        
        # Format 7: What ChatGPT might actually output
        """For each token, here are the dependency relations:

        TOKEN_ID FORM -> HEAD_ID RELATION

        1 Ki -> 2 subj
        2 yore -> 0 root
        3 wàllu -> 2 comp:obj"""
    ]
    
    print("Testing parser with different response formats:\n")
    
    for i, response in enumerate(sample_responses, 1):
        print(f"🧪 Test {i}:")
        print(f"Input: '{response[:100]}{'...' if len(response) > 100 else ''}'")
        
        parsed = parser.parse_response(response, f"test_{i}")
        
        if parsed:
            print(f"✅ Parsed {len(parsed.predictions)} predictions:")
            for pred in parsed.predictions[:3]:  # Show first 3
                print(f"   {pred.token_id} {pred.form} -> {pred.predicted_head} {pred.predicted_relation}")
        else:
            print("❌ Failed to parse")
            
            # Try manual pattern matching to see what might work
            print("  🔧 Testing individual patterns:")
            patterns = {
                'basic': r'(\d+)\s+(\S+)\s*->\s*(\d+)\s+(\S+)',
                'with_arrows': r'(\d+)\s+(\S+).*?->\s*(\d+)\s+(\S+)',
                'flexible': r'(\d+).*?(\w+).*?(\d+).*?(\w+)',
            }
            
            for pattern_name, pattern in patterns.items():
                matches = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
                if matches:
                    print(f"    ✅ {pattern_name}: {len(matches)} matches")
                else:
                    print(f"    ❌ {pattern_name}: No matches")
        print()

def test_with_actual_llm():
    """Run a real experiment to see actual LLM responses"""
    
    print("🧪 TESTING WITH REAL LLM RESPONSES")
    print("=" * 50)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return
    
    try:
        # Initialize pipeline
        pipeline = WolofExperimentPipeline("annotated_data")
        
        if not pipeline.train_sentences:
            print("❌ No training data loaded")
            return
        
        if not (pipeline.test_sentences or pipeline.dev_sentences):
            print("❌ No test data available")
            return
        
        # Setup LLM client
        llm_client = LLMClient(
            model_name="gpt-4o-mini",
            max_tokens=1500,
            temperature=0.0
        )
        
        # Get a target sentence
        target = pipeline.test_sentences[0] if pipeline.test_sentences else pipeline.dev_sentences[0]
        print(f"🎯 Target sentence: {target.text}")
        print(f"   English: {target.text_en}")
        print(f"   Tokens: {len([t for t in target.tokens if t.id > 0])}")
        
        # Get some examples
        examples = pipeline.example_selector.random_selection(2)
        print(f"📚 Using {len(examples)} example sentences")
        
        # Test different templates
        templates = ["basic", "gloss_focused"]
        
        for template_name in templates:
            print(f"\n📝 Testing template: {template_name}")
            print("-" * 40)
            
            try:
                # Generate prompt
                prompt = pipeline.prompt_generator.generate_prompt(
                    template_name=template_name,
                    examples=examples,
                    target_sentence=target
                )
                
                print(f"Prompt length: {len(prompt)} chars")
                print(f"Prompt preview:\n{prompt[:300]}...")
                
                # Query LLM
                print(f"\n🤖 Querying LLM...")
                response, time_taken = llm_client.query(prompt)
                
                print(f"⏱️ Response time: {time_taken:.2f}s")
                print(f"📄 Response length: {len(response)} chars")
                
                print(f"\n🔍 RAW LLM RESPONSE:")
                print("=" * 60)
                print(response)
                print("=" * 60)
                
                # Try to parse
                parser = LLMResponseParser()
                parsed = parser.parse_response(response, target.sent_id)
                
                if parsed:
                    print(f"\n✅ Successfully parsed {len(parsed.predictions)} predictions:")
                    for pred in parsed.predictions:
                        print(f"   {pred.token_id} {pred.form} -> {pred.predicted_head} {pred.predicted_relation}")
                else:
                    print(f"\n❌ PARSING FAILED!")
                    print("This is why your evaluation is crashing.")
                    
                    # Analyze why it failed
                    print(f"\n🔧 DEBUGGING PARSING FAILURE:")
                    print(f"Looking for patterns in the response...")
                    
                    # Test various patterns
                    test_patterns = {
                        'basic_arrow': r'(\d+)\s+(\w+)\s*->\s*(\d+)\s+(\w+)',
                        'with_colon': r'(\d+)[.:]\s*(\w+).*?(\d+)[.:]\s*(\w+)',
                        'token_word': r'[Tt]oken\s+(\d+)[.:]\s*(\w+).*?(\d+).*?(\w+)',
                        'parentheses': r'(\d+)[.)]\s*(\w+).*?(\d+).*?(\w+)',
                        'any_numbers': r'(\d+).*?(\w+).*?(\d+).*?(\w+)',
                    }
                    
                    for pattern_name, pattern in test_patterns.items():
                        matches = re.findall(pattern, response, re.IGNORECASE)
                        if matches:
                            print(f"   ✅ Pattern '{pattern_name}': {len(matches)} matches")
                            for match in matches[:3]:
                                print(f"      {match}")
                        else:
                            print(f"   ❌ Pattern '{pattern_name}': No matches")
                
                print(f"\n" + "="*60)
                
            except Exception as e:
                print(f"❌ Error with template {template_name}: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def analyze_failed_experiment():
    """Analyze the output from a failed experiment"""
    
    print("📊 ANALYZING FAILED EXPERIMENT OUTPUT")
    print("=" * 50)
    
    print("Based on your error, here's what happened:")
    print("1. The LLM generated responses for each test sentence")
    print("2. The parser tried to extract dependency predictions")
    print("3. Parser found 0 predictions (empty list)")
    print("4. Evaluator tried to compare 200 gold relations vs 0 predictions")
    print("5. Sklearn's confusion_matrix crashed due to mismatched lengths")
    
    print(f"\n🔧 SOLUTIONS:")
    print("1. Check what format the LLM is actually outputting")
    print("2. Update the parser patterns to match that format")
    print("3. Add better error handling in evaluation")
    
    print(f"\n💡 LIKELY CAUSES:")
    print("- LLM responds in natural language instead of structured format")
    print("- Gloss-focused template produces verbose explanations")
    print("- Parser patterns don't match actual LLM output style")

def main():
    """Main function to run different debug options"""
    
    print("🐛 WOLOF DEPENDENCY PARSING - DEBUG TOOL")
    print("=" * 60)
    
    print("What would you like to debug?")
    print("1. Test parser with sample response formats")
    print("2. Run real LLM experiment and see actual responses") 
    print("3. Analyze the failed experiment error")
    print("4. All of the above")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        debug_parser_patterns()
    elif choice == "2":
        test_with_actual_llm()
    elif choice == "3":
        analyze_failed_experiment()
    elif choice == "4":
        debug_parser_patterns()
        print("\n" + "="*80 + "\n")
        test_with_actual_llm()
        print("\n" + "="*80 + "\n")
        analyze_failed_experiment()
    else:
        print("Invalid choice. Running parser pattern test...")
        debug_parser_patterns()

if __name__ == "__main__":
    main()