import sys
import os
import time
sys.path.append('.')

from LLM_testing import WolofExperimentPipeline

def main():
    # Configuration
    config = {
        'name': 'gpt_4o_mini_gloss_focused_diversity_5shot',
        'description': 'OpenAI experiment: gpt-4o-mini, gloss_focused, diversity, 5 examples',
        'llm': {
            'model': 'gpt-4o-mini',
            'max_tokens': 2000,
            'temperature': 0.0,
            'api_key': None  # Uses environment variable
        },
        'selection': {
            'strategy': 'diversity',
            'num_examples': 5,
            'seed': 42
        },
        'prompt': {
            'template': 'gloss_focused'
        },
        'test_size': 10
    }
    
    print(f"🚀 Starting OpenAI experiment: gpt_4o_mini_gloss_focused_diversity_5shot")
    print(f"⚙️  Model: gpt-4o-mini")
    print(f"📝 Template: gloss_focused") 
    print(f"🎯 Selection: diversity")
    print(f"📊 Examples: 5, Test size: 10")
    
    try:
        # Initialize pipeline
        pipeline = WolofExperimentPipeline('../data/annotated_data/wol')
        
        # Run experiment
        results = pipeline.run_experiment(config)
        
        if results:
            # Save results
            results_dir = 'outputs/wolof/results'
            os.makedirs(results_dir, exist_ok=True)
            
            pipeline.reporter.export_results_to_json(results, f'{results_dir}/gpt_4o_mini_gloss_focused_diversity_5shot_results.json')
            pipeline.reporter.export_results_to_csv(results, f'{results_dir}/gpt_4o_mini_gloss_focused_diversity_5shot_detailed.csv')
            
            # Print summary
            print(f"\n📊 RESULTS SUMMARY for {results.experiment_name}")
            print(f"{'='*50}")
            print(f"UAS (Unlabeled Attachment): {results.overall_uas:.3f}")
            print(f"LAS (Labeled Attachment):   {results.overall_las:.3f}")
            print(f"Label Accuracy:             {results.overall_la:.3f}")
            print(f"Root Accuracy:              {results.root_accuracy:.3f}")
            print(f"Complete Match Rate:        {results.complete_match_rate:.3f}")
            print(f"Processing Time:            {results.total_processing_time:.1f}s")
            
            print(f"\n💾 Results saved:")
            print(f"  JSON: {results_dir}/gpt_4o_mini_gloss_focused_diversity_5shot_results.json")
            print(f"  CSV:  {results_dir}/gpt_4o_mini_gloss_focused_diversity_5shot_detailed.csv")
            
            return True
        else:
            print(f"❌ Experiment failed!")
            return False
            
    except KeyboardInterrupt:
        print(f"\n🛑 Experiment interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Experiment error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
