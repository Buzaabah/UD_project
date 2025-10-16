"""
LLM_testing.py - Complete Integrated Wolof Few-Shot Dependency Parsing Pipeline
Updated for actual file names: wol.Wolof.{train,dev,test}.conllu
Supports OpenAI and Anthropic models with comprehensive experimentation framework
"""

import os
import json
import time
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from enum import Enum
import numpy as np
from collections import Counter, defaultdict
import traceback

# Import components from separate files
try:
    from fewShort_selection import ConlluParser, ExampleSelector, Token, Sentence, load_wolof_data
    from prompts import WolofDependencyPromptGenerator
    from evaluation import DependencyEvaluator, EvaluationReporter, ExperimentResults
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all component files are in the same directory:")
    print("- fewShort_selection.py (with updated file handling)")
    print("- prompts.py") 
    print("- evaluation.py")
    print("\nTrying to continue with basic functionality...")
    
    # Create minimal fallback classes if imports fail
    class Token:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class Sentence:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"

@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    model_name: str
    api_key: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout: int = 60

class LLMClient:
    """Unified client for OpenAI and Anthropic models with enhanced error handling"""
    
    def __init__(self, 
                 model_name: str,
                 api_key: Optional[str] = None,
                 max_tokens: int = 2000,
                 temperature: float = 0.1):
        
        self.config = LLMConfig(
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        self.provider = self._detect_provider()
        self.client = None
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self._setup_client()
    
    def _detect_provider(self) -> LLMProvider:
        """Detect LLM provider based on model name"""
        model_lower = self.config.model_name.lower()
        
        if "claude" in model_lower:
            return LLMProvider.ANTHROPIC
        elif any(name in model_lower for name in ["gpt", "openai", "davinci", "o1"]):
            return LLMProvider.OPENAI
        else:
            # Default to OpenAI for unknown models
            print(f"⚠️  Unknown model '{model_lower}', defaulting to OpenAI")
            return LLMProvider.OPENAI
    
    def _setup_client(self):
        """Setup the appropriate client based on provider"""
        try:
            if self.provider == LLMProvider.ANTHROPIC:
                import anthropic
                api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
                self.client = anthropic.Client(api_key=api_key)
                print(f"🤖 Anthropic client initialized: {self.config.model_name}")
                
            elif self.provider == LLMProvider.OPENAI:
                import openai
                api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                self.client = openai.Client(api_key=api_key)
                print(f"🤖 OpenAI client initialized: {self.config.model_name}")
                
        except ImportError as e:
            if "anthropic" in str(e):
                raise ImportError(f"Install anthropic: pip install anthropic") from e
            elif "openai" in str(e):
                raise ImportError(f"Install openai: pip install openai") from e
        except Exception as e:
            print(f"❌ Failed to setup {self.provider.value} client: {e}")
            raise
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def estimate_cost(self, input_tokens: int, output_tokens: int = 500) -> float:
        """Estimate API cost based on token usage"""
        # Cost per 1K tokens (approximate, as of 2024)
        costs = {
            # OpenAI models
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            # Anthropic models (rough estimates)
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
        }
        
        model_key = self.config.model_name.lower()
        for key in costs.keys():
            if key in model_key:
                cost_info = costs[key]
                break
        else:
            # Default to mid-range pricing
            cost_info = {"input": 0.002, "output": 0.006}
        
        cost = (input_tokens * cost_info["input"] / 1000 + 
                output_tokens * cost_info["output"] / 1000)
        return cost
    
    def query(self, prompt: str) -> Tuple[str, float]:
        """Query the LLM with enhanced error handling and cost tracking"""
        start_time = time.time()
        
        # Estimate cost before query
        estimated_input_tokens = self.estimate_tokens(prompt)
        estimated_cost = self.estimate_cost(estimated_input_tokens)
        
        try:
            if self.provider == LLMProvider.ANTHROPIC:
                response = self._query_anthropic(prompt)
            elif self.provider == LLMProvider.OPENAI:
                response = self._query_openai(prompt)
            else:
                raise NotImplementedError(f"Provider {self.provider} not implemented")
            
            processing_time = time.time() - start_time
            
            # Update cost tracking
            estimated_output_tokens = self.estimate_tokens(response)
            self.total_tokens_used += estimated_input_tokens + estimated_output_tokens
            self.total_cost += estimated_cost
            
            return response, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"LLM_ERROR: {type(e).__name__}: {str(e)}"
            print(f"❌ LLM Query failed: {error_msg}")
            return error_msg, processing_time
    
    def _query_anthropic(self, prompt: str) -> str:
        """Query Anthropic Claude models"""
        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            if hasattr(response, 'content') and response.content:
                if isinstance(response.content, list) and len(response.content) > 0:
                    return response.content[0].text
                elif hasattr(response.content, 'text'):
                    return response.content.text
                else:
                    return str(response.content)
            else:
                return "ERROR: Empty response from Anthropic"
        
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI GPT models"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or "ERROR: Empty response"
            else:
                return "ERROR: No choices in response"
        
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def get_usage_stats(self) -> Dict[str, float]:
        """Get usage statistics"""
        return {
            "total_tokens": self.total_tokens_used,
            "estimated_cost": self.total_cost,
            "model": self.config.model_name,
            "provider": self.provider.value
        }

class WolofExperimentPipeline:
    """Complete integrated pipeline for Wolof dependency parsing experiments"""
    
    def __init__(self, data_directory: str = "../data/annotated_data/wol"):
        self.data_dir = Path(data_directory)
        
        # Initialize components
        try:
            self.parser = ConlluParser()
            self.prompt_generator = WolofDependencyPromptGenerator()
            self.evaluator = DependencyEvaluator()
            self.reporter = EvaluationReporter()
        except:
            print("⚠️  Some components failed to initialize. Basic functionality available.")
            self.parser = None
            self.prompt_generator = None
            self.evaluator = None
            self.reporter = None
        
        self.llm_client = None
        
        # Data containers
        self.train_sentences = []
        self.dev_sentences = []
        self.test_sentences = []
        
        # Load data with actual Wolof file names
        self.load_wolof_data()
        
        # Initialize example selector
        if self.train_sentences:
            try:
                self.example_selector = ExampleSelector(self.train_sentences)
                print(f"✅ Example selector initialized with {len(self.train_sentences)} training sentences")
            except:
                print("⚠️  Example selector failed to initialize")
                self.example_selector = None
    
    def load_wolof_data(self):
        """Load Wolof data using the dedicated function from fewShort_selection.py"""
        try:
            print(f"📁 Loading Wolof data from {self.data_dir}")
            self.train_sentences, self.dev_sentences, self.test_sentences = load_wolof_data(str(self.data_dir))
            
            total = len(self.train_sentences) + len(self.dev_sentences) + len(self.test_sentences)
            if total == 0:
                print("❌ No data loaded! Please check:")
                print("   1. Directory '../data/annotated_data/wol' exists")
                print("   2. Files: wol.Wolof.train.conllu, wol.Wolof.dev.conllu, wol.Wolof.test.conllu")
                print("   3. Files are valid CoNLL-U format")
            else:
                print(f"✅ Successfully loaded {total} total sentences")
                
        except Exception as e:
            print(f"❌ Error loading Wolof data: {e}")
            print("Trying basic file loading...")
            
            # Fallback: basic file loading
            try:
                files = {
                    "train": self.data_dir / "wol.Wolof.train.conllu",
                    "dev": self.data_dir / "wol.Wolof.dev.conllu",
                    "test": self.data_dir / "wol.Wolof.test.conllu"
                }
                
                for split, filepath in files.items():
                    if filepath.exists():
                        print(f"📄 Found: {filepath.name}")
                    else:
                        print(f"❌ Missing: {filepath.name}")
            except Exception as e2:
                print(f"❌ Fallback loading failed: {e2}")
    
    def setup_llm(self, model_config: Dict):
        """Setup LLM client with enhanced configuration"""
        try:
            self.llm_client = LLMClient(
                model_name=model_config['model'],
                api_key=model_config.get('api_key'),
                max_tokens=model_config.get('max_tokens', 2000),
                temperature=model_config.get('temperature', 0.1)
            )
            print(f"✅ LLM Client ready: {model_config['model']}")
            
            # Test connection with minimal query
            try:
                test_response, test_time = self.llm_client.query("Hello")
                if "ERROR" not in test_response:
                    print(f"✅ Connection test passed ({test_time:.1f}s)")
                else:
                    print(f"⚠️  Connection test warning: {test_response}")
            except Exception as test_error:
                print(f"⚠️  Connection test failed: {test_error}")
                
        except Exception as e:
            print(f"❌ Failed to setup LLM: {e}")
            raise
    
    def run_experiment(self, config: Dict) -> Optional[ExperimentResults]:
        """Run a single few-shot learning experiment with comprehensive logging"""
        
        experiment_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"🧪 EXPERIMENT: {config['name']}")
        print(f"{'='*80}")
        
        # Validate configuration
        required_keys = ['llm', 'prompt', 'selection']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing configuration key: {key}")
                return None
        
        # Setup LLM
        try:
            self.setup_llm(config['llm'])
        except Exception as e:
            print(f"❌ LLM setup failed: {e}")
            return None
        
        # Extract experiment parameters
        template_name = config['prompt']['template']
        selection_strategy = config['selection']['strategy']
        num_examples = config['selection']['num_examples']
        test_size = config.get('test_size', 10)
        
        print(f"📋 Configuration:")
        print(f"   Model: {config['llm']['model']}")
        print(f"   Template: {template_name}")
        print(f"   Selection: {selection_strategy}")
        print(f"   Examples: {num_examples}")
        print(f"   Test size: {test_size}")
        print(f"   Temperature: {config['llm'].get('temperature', 0.1)}")
        
        # Validate data availability
        if not self.train_sentences:
            print("❌ No training data available!")
            return None
        
        if not self.example_selector:
            print("❌ Example selector not initialized!")
            return None
        
        # Select examples
        print(f"\n🎯 Selecting {num_examples} examples using '{selection_strategy}' strategy...")
        
        try:
            selection_start_time = time.time()
            
            if selection_strategy == "random":
                selected_examples = self.example_selector.random_selection(
                    num_examples, seed=config['selection'].get('seed', 42)
                )
            elif selection_strategy == "diversity":
                selected_examples = self.example_selector.diversity_based_selection(num_examples)
            elif selection_strategy == "coverage":
                selected_examples = self.example_selector.coverage_based_selection(num_examples)
            elif selection_strategy == "stratified":
                selected_examples = self.example_selector.stratified_selection(num_examples)
            elif selection_strategy == "complexity":
                selected_examples = self.example_selector.complexity_balanced_selection(num_examples)
            else:
                raise ValueError(f"Unknown selection strategy: {selection_strategy}")
            
            selection_time = time.time() - selection_start_time
            print(f"   ✅ Selection completed in {selection_time:.2f}s")
            print(f"   📝 Selected: {[s.sent_id for s in selected_examples[:5]]}{'...' if len(selected_examples) > 5 else ''}")
            
            # Show selection quality
            if hasattr(self, 'test_sentences') and self.test_sentences:
                from fewShort_selection import evaluate_selection_quality
                quality = evaluate_selection_quality(selected_examples, self.test_sentences[:20])
                print(f"   📊 Selection Quality:")
                print(f"      Relation Coverage: {quality.get('relation_coverage', 0):.3f}")
                print(f"      Diversity: {quality.get('relation_diversity', 0)} relations")
                print(f"      Avg Length: {quality.get('avg_sentence_length', 0):.1f}")
            
        except Exception as e:
            print(f"❌ Example selection failed: {e}")
            traceback.print_exc()
            return None
        
        # Prepare test data
        if self.test_sentences:
            test_data = self.test_sentences[:test_size]
        elif self.dev_sentences:
            test_data = self.dev_sentences[:test_size]
            print("⚠️  Using dev data for testing (no test data available)")
        else:
            print("❌ No test data available!")
            return None
            
        print(f"\n🔍 Testing on {len(test_data)} sentences...")
        
        # Estimate total cost
        if hasattr(self, 'llm_client') and self.llm_client:
            sample_prompt_length = 2000  # Rough estimate
            total_estimated_tokens = len(test_data) * sample_prompt_length
            estimated_cost = self.llm_client.estimate_cost(total_estimated_tokens)
            print(f"💰 Estimated cost: ${estimated_cost:.4f}")
        
        # Generate prompts and query LLM
        all_responses = []
        all_prompts = []
        total_llm_time = 0
        successful_queries = 0
        failed_queries = 0
        
        for i, test_sentence in enumerate(test_data):
            print(f"   [{i+1}/{len(test_data)}] Processing: {test_sentence.sent_id}", end=" ")
            
            try:
                # Handle similarity-based selection (dynamic per test sentence)
                current_examples = selected_examples
                if selection_strategy == "similarity":
                    current_examples = self.example_selector.similarity_based_selection(
                        test_sentence, num_examples
                    )
                
                # Generate prompt
                prompt_start_time = time.time()
                
                if not self.prompt_generator:
                    raise Exception("Prompt generator not available")
                
                prompt = self.prompt_generator.generate_prompt(
                    template_name=template_name,
                    examples=current_examples,
                    target_sentence=test_sentence
                )
                
                prompt_time = time.time() - prompt_start_time
                all_prompts.append(prompt)
                
                # Query LLM
                response, llm_time = self.llm_client.query(prompt)
                
                all_responses.append(response)
                total_llm_time += llm_time
                successful_queries += 1
                
                # Check for errors
                if "ERROR" in response or "LLM_ERROR" in response:
                    print(f"⚠️  ({llm_time:.1f}s, API error)")
                    failed_queries += 1
                else:
                    print(f"✅ ({llm_time:.1f}s, {len(response)} chars)")
                
                # Rate limiting to respect API limits
                sleep_time = config.get('sleep_time', 0.5)
                if i < len(test_data) - 1:  # Don't sleep after last query
                    time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                print(f"\n🛑 Experiment interrupted by user at {i+1}/{len(test_data)}")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                all_responses.append(f"PROCESSING_ERROR: {str(e)}")
                failed_queries += 1
                continue
        
        print(f"\n📊 Query Summary:")
        print(f"   ✅ Successful: {successful_queries}/{len(test_data)}")
        print(f"   ❌ Failed: {failed_queries}/{len(test_data)}")
        print(f"   ⏱️  Total LLM time: {total_llm_time:.2f}s")
        print(f"   📊 Avg time per query: {total_llm_time/max(successful_queries,1):.2f}s")
        
        # Show usage stats
        if self.llm_client:
            usage = self.llm_client.get_usage_stats()
            print(f"   💰 Estimated cost: ${usage['estimated_cost']:.4f}")
            print(f"   🎯 Estimated tokens: {usage['total_tokens']:,}")
        
        # Evaluate results
        print(f"\n📈 Evaluating predictions...")
        
        if not self.evaluator:
            print("❌ Evaluator not available!")
            return None
        
        try:
            evaluation_start_time = time.time()
            
            experiment_config_for_eval = {
                'name': config['name'],
                'model': config['llm']['model'],
                'template': template_name,
                'selection': selection_strategy,
                'num_examples': num_examples,
                'total_processing_time': total_llm_time,
                'successful_queries': successful_queries,
                'failed_queries': failed_queries,
                'experiment_duration': time.time() - experiment_start_time
            }
            
            # Only evaluate responses that we got (in case of partial completion)
            actual_test_data = test_data[:len(all_responses)]
            
            results = self.evaluator.evaluate_experiment(
                gold_sentences=actual_test_data,
                predicted_responses=all_responses,
                experiment_config=experiment_config_for_eval
            )
            
            evaluation_time = time.time() - evaluation_start_time
            print(f"✅ Evaluation completed in {evaluation_time:.2f}s!")
            
            # Print immediate results summary
            print(f"\n📊 RESULTS PREVIEW:")
            print(f"   🎯 UAS (Unlabeled Attachment): {results.overall_uas:.3f}")
            print(f"   🏷️  LAS (Labeled Attachment):   {results.overall_las:.3f}")
            print(f"   📝 Label Accuracy:             {results.overall_la:.3f}")
            print(f"   🌳 Root Accuracy:              {results.root_accuracy:.3f}")
            print(f"   ✨ Complete Match Rate:        {results.complete_match_rate:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            traceback.print_exc()
            return None
    
    def run_multiple_experiments(self, experiment_configs: List[Dict]) -> List[ExperimentResults]:
        """Run multiple experiments with progress tracking and error recovery"""
        
        print(f"\n🚀 RUNNING {len(experiment_configs)} EXPERIMENTS")
        print(f"{'='*80}")
        
        all_results = []
        start_time = time.time()
        
        for i, config in enumerate(experiment_configs):
            print(f"\n[{i+1}/{len(experiment_configs)}] Starting: {config['name']}")
            
            try:
                results = self.run_experiment(config)
                if results:
                    all_results.append(results)
                    print(f"✅ Experiment {i+1} completed successfully!")
                    
                    # Save intermediate results
                    if hasattr(self, 'reporter') and self.reporter:
                        try:
                            filename = f"intermediate_{results.experiment_name.replace(' ', '_')}.json"
                            self.reporter.export_results_to_json(results, filename)
                            print(f"💾 Intermediate results saved: {filename}")
                        except Exception as save_error:
                            print(f"⚠️  Failed to save intermediate results: {save_error}")
                else:
                    print(f"❌ Experiment {i+1} failed!")
                    
            except KeyboardInterrupt:
                print(f"\n🛑 Experiments interrupted by user")
                print(f"✅ Completed {len(all_results)}/{len(experiment_configs)} experiments")
                break
            except Exception as e:
                print(f"❌ Unexpected error in experiment {i+1}: {e}")
                traceback.print_exc()
                continue
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total experiment time: {total_time:.1f}s")
        print(f"✅ Successfully completed {len(all_results)}/{len(experiment_configs)} experiments")
        
        return all_results
    
    def generate_comprehensive_report(self, all_results: List[ExperimentResults]):
        """Generate comprehensive analysis with enhanced visualizations and insights"""
        
        print(f"\n📊 COMPREHENSIVE EXPERIMENT ANALYSIS")
        print(f"{'='*80}")
        
        if not all_results:
            print("❌ No results to analyze!")
            return
        
        if not self.reporter:
            print("❌ Reporter not available!")
            return
        
        # Individual experiment reports
        print(f"\n📋 INDIVIDUAL EXPERIMENT RESULTS:")
        print(f"{'-'*80}")
        
        for i, results in enumerate(all_results, 1):
            print(f"\n[{i}] {results.experiment_name}")
            print(f"{'-'*40}")
            
            try:
                summary = self.reporter.generate_summary_report(results)
                # Print condensed summary
                lines = summary.split('\n')
                for line in lines:
                    if any(keyword in line for keyword in ['UAS', 'LAS', 'Label Accuracy', 'Root Accuracy', 'Complete Match']):
                        print(f"   {line.strip()}")
                
                # Save detailed results
                results_filename = f"results_{results.experiment_name.replace(' ', '_')}.json"
                self.reporter.export_results_to_json(results, results_filename)
                
                csv_filename = f"detailed_{results.experiment_name.replace(' ', '_')}.csv"
                self.reporter.export_results_to_csv(results, csv_filename)
                
                print(f"   💾 Saved: {results_filename}, {csv_filename}")
                
            except Exception as e:
                print(f"   ❌ Failed to process results for {results.experiment_name}: {e}")
        
        # Comparative analysis
        if len(all_results) > 1:
            print(f"\n📊 COMPARATIVE ANALYSIS")
            print(f"{'-'*80}")
            
            # Create comparison table
            print(f"{'Experiment':<30} {'Model':<15} {'UAS':<8} {'LAS':<8} {'LA':<8} {'Root':<8} {'Time':<8}")
            print("-" * 95)
            
            best_uas = max(r.overall_uas for r in all_results)
            best_las = max(r.overall_las for r in all_results)
            
            for result in all_results:
                model_short = result.model_name.split('-')[-1] if '-' in result.model_name else result.model_name[:15]
                
                uas_marker = "🏆" if result.overall_uas == best_uas else " "
                las_marker = "🏆" if result.overall_las == best_las else " "
                
                print(f"{result.experiment_name:<30} {model_short:<15} "
                     f"{result.overall_uas:<7.3f}{uas_marker} {result.overall_las:<7.3f}{las_marker} "
                     f"{result.overall_la:<8.3f} {result.root_accuracy:<8.3f} "
                     f"{result.total_processing_time:<8.1f}")
            
            # Save comparison data
            try:
                comparison_data = []
                for result in all_results:
                    comparison_data.append({
                        'Experiment': result.experiment_name,
                        'Model': result.model_name,
                        'Template': result.prompt_template,
                        'Selection': result.selection_strategy,
                        'Examples': result.num_examples,
                        'UAS': result.overall_uas,
                        'LAS': result.overall_las,
                        'LA': result.overall_la,
                        'Root_Accuracy': result.root_accuracy,
                        'Complete_Match': result.complete_match_rate,
                        'Processing_Time': result.total_processing_time
                    })
                
                # Save as JSON for detailed analysis
                with open('experiment_comparison.json', 'w') as f:
                    json.dump(comparison_data, f, indent=2)
                
                print(f"\n💾 Comparison data saved: experiment_comparison.json")
                
                # Try to save CSV if pandas available
                try:
                    import pandas as pd
                    df = pd.DataFrame(comparison_data)
                    df.to_csv('experiment_comparison.csv', index=False)
                    print(f"💾 Comparison table saved: experiment_comparison.csv")
                except ImportError:
                    print("💾 Install pandas for CSV export: pip install pandas")
                
            except Exception as e:
                print(f"❌ Failed to save comparison: {e}")
            
            # Generate insights
            print(f"\n🔍 KEY INSIGHTS:")
            best_overall = max(all_results, key=lambda r: r.overall_las)
            print(f"   🏆 Best Overall Performance: {best_overall.experiment_name}")
            print(f"      LAS: {best_overall.overall_las:.3f}, UAS: {best_overall.overall_uas:.3f}")
            
            # Template comparison
            template_performance = defaultdict(list)
            for result in all_results:
                template_performance[result.prompt_template].append(result.overall_las)
            
            if len(template_performance) > 1:
                print(f"\n   📝 Template Performance:")
                for template, scores in template_performance.items():
                    avg_score = np.mean(scores)
                    print(f"      {template}: {avg_score:.3f} (avg LAS)")
            
            # Selection strategy comparison
            selection_performance = defaultdict(list)
            for result in all_results:
                selection_performance[result.selection_strategy].append(result.overall_las)
            
            if len(selection_performance) > 1:
                print(f"\n   🎯 Selection Strategy Performance:")
                for strategy, scores in selection_performance.items():
                    avg_score = np.mean(scores)
                    print(f"      {strategy}: {avg_score:.3f} (avg LAS)")
            
            # Model comparison
            model_performance = defaultdict(list)
            for result in all_results:
                model_performance[result.model_name].append(result.overall_las)
            
            if len(model_performance) > 1:
                print(f"\n   🤖 Model Performance:")
                for model, scores in model_performance.items():
                    avg_score = np.mean(scores)
                    print(f"      {model}: {avg_score:.3f} (avg LAS)")
        
        # Generate final summary
        print(f"\n📋 EXPERIMENT SUMMARY")
        print(f"{'-'*80}")
        print(f"Total experiments completed: {len(all_results)}")
        print(f"Average UAS: {np.mean([r.overall_uas for r in all_results]):.3f}")
        print(f"Average LAS: {np.mean([r.overall_las for r in all_results]):.3f}")
        print(f"Average processing time: {np.mean([r.total_processing_time for r in all_results]):.1f}s")
        
        if hasattr(self, 'llm_client') and self.llm_client:
            usage = self.llm_client.get_usage_stats()
            print(f"Total estimated cost: ${usage['estimated_cost']:.4f}")
        
        print(f"\n📁 All results and analyses saved in current directory")

def create_experiment_configurations(models: List[str] = None, 
                                  templates: List[str] = None,
                                  strategies: List[str] = None,
                                  shot_counts: List[int] = None) -> List[Dict]:
    """Create experimental configurations for Wolof dependency parsing"""
    
    # Default configurations optimized for different scenarios
    if models is None:
        models = ["gpt-4o-mini"]  # Cost-effective default
    
    if templates is None:
        templates = ["basic", "gloss_focused"]
    
    if strategies is None:
        strategies = ["random", "diversity"]
    
    if shot_counts is None:
        shot_counts = [3, 5]
    
    experiments = []
    
    # Generate all combinations
    for model in models:
        for template in templates:
            for strategy in strategies:
                for k in shot_counts:
                    
                    # Determine base LLM config based on model
                    if "gpt" in model.lower() or "openai" in model.lower():
                        base_config = {
                            'model': model,
                            'max_tokens': 2000,
                            'temperature': 0.1,
                            'api_key': None  # Uses environment variable
                        }
                    else:  # Assume Anthropic
                        base_config = {
                            'model': model,
                            'max_tokens': 2000,
                            'temperature': 0.1,
                            'api_key': None
                        }
                    
                    # Create experiment name
                    model_short = model.replace('-', '_').replace('.', '_')
                    exp_name = f"{model_short}_{template}_{strategy}_{k}shot"
                    
                    experiment = {
                        'name': exp_name,
                        'description': f'{model} with {template} prompts, {strategy} selection, {k} examples',
                        'llm': base_config,
                        'selection': {
                            'strategy': strategy,
                            'num_examples': k,
                            'seed': 42
                        },
                        'prompt': {
                            'template': template
                        },
                        'test_size': 15,  # Reasonable for testing
                        'sleep_time': 0.5  # Respect API limits
                    }
                    
                    experiments.append(experiment)
    
    return experiments

def create_quick_test_config() -> Dict:
    """Create a quick test configuration for development"""
    return {
        'name': 'Quick_Test',
        'description': 'Quick test with minimal API usage',
        'llm': {
            'model': 'gpt-4o-mini',
            'max_tokens': 1000,
            'temperature': 0.0,
            'api_key': None
        },
        'selection': {
            'strategy': 'random',
            'num_examples': 2,
            'seed': 42
        },
        'prompt': {
            'template': 'basic'
        },
        'test_size': 3,
        'sleep_time': 0.2
    }

def create_comprehensive_config() -> List[Dict]:
    """Create comprehensive experiment configuration"""
    return create_experiment_configurations(
        models=["gpt-4o-mini", "gpt-3.5-turbo"],
        templates=["basic", "gloss_focused", "chain_of_thought"],
        strategies=["random", "diversity", "coverage"],
        shot_counts=[3, 5, 7]
    )

def test_pipeline_components():
    """Test pipeline components without making API calls"""
    
    print("🔧 TESTING WOLOF PIPELINE COMPONENTS")
    print("=" * 60)
    
    try:
        # Test data loading
        pipeline = WolofExperimentPipeline("../data/annotated_data/wol")
        
        total_sentences = len(pipeline.train_sentences) + len(pipeline.dev_sentences) + len(pipeline.test_sentences)
        if total_sentences == 0:
            print("❌ No data loaded - check your ../data/annotated_data/wol directory")
            return False
        
        print("✅ Data loading successful")
        print(f"  📚 Train: {len(pipeline.train_sentences)} sentences")
        print(f"  📖 Dev: {len(pipeline.dev_sentences)} sentences")
        print(f"  📄 Test: {len(pipeline.test_sentences)} sentences")
        
        # Test example selection
        if pipeline.example_selector and pipeline.train_sentences:
            print("\n✅ Testing example selection...")
            
            strategies_to_test = ["random", "diversity"]
            for strategy in strategies_to_test:
                try:
                    if strategy == "random":
                        examples = pipeline.example_selector.random_selection(3)
                    elif strategy == "diversity":
                        examples = pipeline.example_selector.diversity_based_selection(3)
                    
                    print(f"  🎯 {strategy}: Selected {len(examples)} examples")
                    for i, ex in enumerate(examples[:2]):  # Show first 2
                        print(f"    {i+1}. {ex.sent_id}: '{ex.text[:40]}...'")
                
                except Exception as e:
                    print(f"  ❌ {strategy} selection failed: {e}")
        
        # Test prompt generation
        if pipeline.prompt_generator and pipeline.train_sentences:
            print("\n✅ Testing prompt generation...")
            
            try:
                examples = pipeline.train_sentences[:2]
                target = pipeline.test_sentences[0] if pipeline.test_sentences else pipeline.dev_sentences[0]
                
                templates_to_test = ["basic", "gloss_focused"]
                for template in templates_to_test:
                    try:
                        prompt = pipeline.prompt_generator.generate_prompt(
                            template_name=template,
                            examples=examples,
                            target_sentence=target
                        )
                        print(f"  📝 {template}: Generated {len(prompt)} character prompt")
                    except Exception as e:
                        print(f"  ❌ {template} prompt generation failed: {e}")
            
            except Exception as e:
                print(f"  ❌ Prompt generation test failed: {e}")
        
        # Test API key availability (without actually calling)
        print("\n✅ Testing API configuration...")
        
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        if openai_key:
            print("  🔑 OpenAI API key: Available")
        else:
            print("  ⚠️  OpenAI API key: Not set")
        
        if anthropic_key:
            print("  🔑 Anthropic API key: Available") 
        else:
            print("  ⚠️  Anthropic API key: Not set")
        
        if not openai_key and not anthropic_key:
            print("  ❌ No API keys available - experiments will fail")
            print("  💡 Set with: export OPENAI_API_KEY='your_key' or export ANTHROPIC_API_KEY='your_key'")
        
        print("\n✅ All component tests completed!")
        print("🚀 Ready to run experiments with: python3 -c \"from LLM_testing import main; main()\"")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main execution function for Wolof dependency parsing experiments"""
    
    print("🇸🇳 WOLOF FEW-SHOT DEPENDENCY PARSING PIPELINE")
    print("=" * 80)
    print("📦 Complete Integration:")
    print("  📁 Data: ../data/annotated_data/wol/ (wol.Wolof.{train,dev,test}.conllu)")
    print("  🎯 Selection: fewShort_selection.py")
    print("  📝 Prompts: prompts.py")  
    print("  📊 Evaluation: evaluation.py")
    print("  🤖 Integration: LLM_testing.py (this file)")
    print("=" * 80)
    
    try:
        # Initialize pipeline
        pipeline = WolofExperimentPipeline("../data/annotated_data/wol")
        
        # Check data availability
        total_sentences = len(pipeline.train_sentences) + len(pipeline.dev_sentences) + len(pipeline.test_sentences)
        if total_sentences == 0:
            print("\n❌ No data available. Please check your ../data/annotated_data/wol directory.")
            print("Expected files:")
            print("  - ../data/annotated_data/wol/wol.Wolof.train.conllu")
            print("  - ../data/annotated_data/wol/wol.Wolof.dev.conllu")
            print("  - ../data/annotated_data/wol/wol.Wolof.test.conllu")
            return
        
        print(f"\n✅ Data loaded successfully: {total_sentences} total sentences")
        
        # Check API keys
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not openai_key and not anthropic_key:
            print(f"\n⚠️  WARNING: No API keys found!")
            print(f"Please set environment variable:")
            print(f"  export OPENAI_API_KEY='your_key_here'")
            print(f"  # or")  
            print(f"  export ANTHROPIC_API_KEY='your_key_here'")
            
            # Ask if user wants to test components without API
            response = input("\nRun component tests without API calls? (y/N): ").strip().lower()
            if response == 'y':
                test_pipeline_components()
            return
        
        # Show available experiment types
        print(f"\n🧪 EXPERIMENT OPTIONS:")
        print(f"1. Quick Test (minimal cost, 2-3 examples, 3 test sentences)")
        print(f"2. Standard Experiment (balanced, 3-5 examples, 10-15 test sentences)")  
        print(f"3. Comprehensive Study (multiple configs, may take 30+ minutes)")
        print(f"4. Custom Configuration")
        print(f"5. Component Test Only (no API calls)")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            # Quick test
            config = create_quick_test_config()
            print(f"\n🚀 Running quick test...")
            results = pipeline.run_experiment(config)
            if results:
                pipeline.generate_comprehensive_report([results])
        
        elif choice == "2":
            # Standard experiment
            configs = create_experiment_configurations(
                models=["gpt-4o-mini"],
                templates=["basic", "gloss_focused"],
                strategies=["random", "diversity"],
                shot_counts=[3, 5]
            )
            print(f"\n🚀 Running {len(configs)} standard experiments...")
            all_results = pipeline.run_multiple_experiments(configs)
            pipeline.generate_comprehensive_report(all_results)
        
        elif choice == "3":
            # Comprehensive study
            configs = create_comprehensive_config()
            print(f"\n🚀 Running {len(configs)} comprehensive experiments...")
            print(f"⚠️  This may take 30+ minutes and cost $5-20 depending on model")
            
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                all_results = pipeline.run_multiple_experiments(configs)
                pipeline.generate_comprehensive_report(all_results)
            else:
                print("Cancelled.")
        
        elif choice == "4":
            # Custom configuration
            print(f"\n⚙️  Custom configuration mode")
            print(f"Available models: gpt-4o-mini, gpt-3.5-turbo, gpt-4o, claude-3-sonnet-20240229")
            print(f"Available templates: basic, detailed, chain_of_thought, multilingual, gloss_focused")
            print(f"Available strategies: random, diversity, coverage, stratified, complexity")
            
            model = input("Model [gpt-4o-mini]: ").strip() or "gpt-4o-mini"
            template = input("Template [gloss_focused]: ").strip() or "gloss_focused"
            strategy = input("Selection [diversity]: ").strip() or "diversity"
            k = int(input("Number of examples [5]: ").strip() or "5")
            test_size = int(input("Test sentences [10]: ").strip() or "10")
            
            custom_config = {
                'name': f'Custom_{model}_{template}_{strategy}_{k}shot',
                'description': f'Custom: {model}, {template}, {strategy}, {k} examples',
                'llm': {
                    'model': model,
                    'max_tokens': 2000,
                    'temperature': 0.1
                },
                'selection': {
                    'strategy': strategy,
                    'num_examples': k,
                    'seed': 42
                },
                'prompt': {
                    'template': template
                },
                'test_size': test_size,
                'sleep_time': 0.5
            }
            
            results = pipeline.run_experiment(custom_config)
            if results:
                pipeline.generate_comprehensive_report([results])
        
        elif choice == "5":
            # Component test
            test_pipeline_components()
        
        else:
            print("Invalid choice. Exiting.")
            return
        
        print(f"\n🎉 EXPERIMENTS COMPLETED!")
        print(f"📁 Results and analyses saved in current directory")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Allow different entry points
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_pipeline_components()
        elif sys.argv[1] == "quick":
            pipeline = WolofExperimentPipeline("../data/annotated_data/wol")
            config = create_quick_test_config()
            results = pipeline.run_experiment(config)
            if results:
                pipeline.generate_comprehensive_report([results])
        else:
            main()
    else:
        main()