#!/bin/bash

# Wolof Few-Shot Dependency Parsing with OpenAI Models
# Optimized for OpenAI API usage

set -e  # Exit on any error

# Default configuration optimized for OpenAI
INPUT_DIR="../data/annotated_data/wol"
OUTPUT_DIR="outputs/wolof"
MODEL="gpt-4o-mini"  # Cost-effective default
FEW_SHOT_K=5
TEMPLATE="gloss_focused"
SELECTION_STRATEGY="diversity"
TEST_SIZE=15
TEMPERATURE=0.0
SLEEP_TIME=0.2
VERBOSE=true

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BLUE}$1${NC}"; }

# OpenAI model options with cost info
show_openai_models() {
    cat << EOF
🤖 Available OpenAI Models:

RECOMMENDED FOR EXPERIMENTS:
  gpt-4o-mini         - Most cost-effective, good performance
  gpt-3.5-turbo       - Fast and affordable
  gpt-4o             - Best performance, higher cost
  gpt-4-turbo        - Good balance of cost/performance

COST COMPARISON (approximate):
  gpt-4o-mini:     \$0.00015 / 1K input tokens
  gpt-3.5-turbo:   \$0.0005 / 1K input tokens  
  gpt-4o:          \$0.005 / 1K input tokens
  gpt-4-turbo:     \$0.01 / 1K input tokens

For your experiments, we recommend starting with gpt-4o-mini or gpt-3.5-turbo.
EOF
}

show_usage() {
    cat << EOF
🇸🇳 Wolof Few-Shot Dependency Parsing with OpenAI

Usage: $0 [OPTIONS]

QUICK START:
    # Set your OpenAI API key
    export OPENAI_API_KEY='your_key_here'
    
    # Run basic experiment
    $0
    
    # Or with custom settings
    $0 --model gpt-4o-mini --few_shot_k 3 --test_size 10

OPTIONS:
    --input_dir DIR         Input directory (default: annotated_data)
    --output_dir DIR        Output directory (default: outputs/wolof)
    --model MODEL           OpenAI model (default: gpt-4o-mini)
    --few_shot_k K          Number of examples (default: 5)
    --template TEMPLATE     Prompt template (default: gloss_focused)
                           Options: basic, detailed, chain_of_thought, multilingual, gloss_focused
    --selection STRATEGY    Selection strategy (default: diversity)
                           Options: random, diversity, coverage, stratified
    --test_size SIZE        Test sentences (default: 15)
    --temperature TEMP      Temperature 0.0-1.0 (default: 0.0)
    --sleep TIME           Sleep between calls (default: 0.2)
    
UTILITY:
    --help                 Show this help
    --models              Show OpenAI model options
    --check               Check setup and data
    --test                Test without API calls
    --quick               Run quick test (3 examples, 5 test sentences)
    --full                Run comprehensive experiments

EXAMPLES:
    # Quick test with minimal cost
    $0 --quick
    
    # Cost-effective comprehensive test
    $0 --model gpt-4o-mini --few_shot_k 3 --test_size 10
    
    # High-performance experiment  
    $0 --model gpt-4o --few_shot_k 7 --test_size 20
    
    # Multiple configurations
    $0 --model gpt-4o-mini,gpt-3.5-turbo --few_shot_k 3,5

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir) INPUT_DIR="$2"; shift 2 ;;
        --output_dir|--out_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --few_shot_k) FEW_SHOT_K="$2"; shift 2 ;;
        --template) TEMPLATE="$2"; shift 2 ;;
        --selection) SELECTION_STRATEGY="$2"; shift 2 ;;
        --test_size) TEST_SIZE="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --sleep) SLEEP_TIME="$2"; shift 2 ;;
        --verbose) VERBOSE=true; shift ;;
        --help|-h) show_usage; exit 0 ;;
        --models) show_openai_models; exit 0 ;;
        --quick)
            MODEL="gpt-4o-mini"
            FEW_SHOT_K=3
            TEST_SIZE=5
            TEMPLATE="basic"
            print_status "Quick mode: minimal cost test"
            shift ;;
        --full)
            MODEL="gpt-4o-mini,gpt-3.5-turbo"
            FEW_SHOT_K="3,5"
            TEMPLATE="basic,gloss_focused"
            TEST_SIZE=15
            print_status "Full mode: comprehensive experiments"
            shift ;;
        --check)
            print_header "🔍 CHECKING SETUP"
            # Check data
            if [[ -d "$INPUT_DIR" ]]; then
                print_status "Data directory exists: $INPUT_DIR"
                ls -la "$INPUT_DIR"/wol.Wolof.*.conllu 2>/dev/null || print_warning "No Wolof .conllu files found"
                # Show file details if found
                if ls "$INPUT_DIR"/wol.Wolof.*.conllu >/dev/null 2>&1; then
                    for file in "$INPUT_DIR"/wol.Wolof.*.conllu; do
                        if [[ -f "$file" ]]; then
                            lines=$(wc -l < "$file" 2>/dev/null || echo "0")
                            print_status "  $(basename "$file"): $lines lines"
                        fi
                    done
                fi
            else
                print_error "Data directory not found: $INPUT_DIR"
            fi
            # Check API key
            if [[ -n "$OPENAI_API_KEY" ]]; then
                print_status "OpenAI API key is set"
            else
                print_error "OPENAI_API_KEY not set"
            fi
            # Check Python packages
            python3 -c "import openai; print('OpenAI package: OK')" 2>/dev/null || print_error "Install: pip install openai"
            exit 0 ;;
        --test)
            print_header "🧪 TESTING COMPONENTS"
            python3 -c "
import sys
sys.path.append('.')
try:
    from LLM_testing import test_pipeline_components
    test_pipeline_components()
except Exception as e:
    print(f'Test failed: {e}')
    print('Make sure all .py files are in current directory')
"
            exit 0 ;;
        *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
    esac
done

check_prerequisites() {
    print_header "🔍 CHECKING PREREQUISITES"
    
    # Check OpenAI API key
    if [[ -z "$OPENAI_API_KEY" ]]; then
        print_error "OPENAI_API_KEY not set!"
        print_status "Set with: export OPENAI_API_KEY='your_key_here'"
        exit 1
    fi
    print_status "OpenAI API key: Set ✓"
    
    # Test API key validity
    print_status "Testing API key validity..."
    python3 -c "
import openai
import os
try:
    client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))
    # Test with minimal request
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': 'Hi'}],
        max_tokens=5
    )
    print('API key valid ✓')
except Exception as e:
    print(f'API key test failed: {e}')
    exit(1)
" || exit 1
    
    # Check Python packages
    python3 -c "
packages = ['openai', 'numpy', 'pandas']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'Install missing packages: pip install {\" \".join(missing)}')
    exit(1)
else:
    print('Python packages: OK ✓')
"
    
    # Check data files - Updated for actual Wolof file names
    if [[ ! -d "$INPUT_DIR" ]]; then
        print_error "Data directory not found: $INPUT_DIR"
        print_status "Expected: $INPUT_DIR/wol.Wolof.{train,dev,test}.conllu"
        exit 1
    fi
    
    data_files=("$INPUT_DIR/wol.Wolof.train.conllu" "$INPUT_DIR/wol.Wolof.dev.conllu" "$INPUT_DIR/wol.Wolof.test.conllu")
    found_files=0
    for file in "${data_files[@]}"; do
        if [[ -f "$file" ]]; then
            size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
            kb_size=$(python3 -c "print(f'{$size/1024:.1f}')")
            print_status "Found: $(basename $file) (${kb_size} KB)"
            ((found_files++))
        fi
    done
    
    if [[ $found_files -eq 0 ]]; then
        print_error "No Wolof .conllu files found in $INPUT_DIR"
        print_status "Expected files:"
        print_status "  - $INPUT_DIR/wol.Wolof.train.conllu"
        print_status "  - $INPUT_DIR/wol.Wolof.dev.conllu"
        print_status "  - $INPUT_DIR/wol.Wolof.test.conllu"
        exit 1
    fi
    print_status "Wolof data files: $found_files found ✓"
    
    # Check Python component files
    component_files=("fewShort_selection.py" "prompts.py" "evaluation.py" "LLM_testing.py")
    for file in "${component_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "Component file missing: $file"
            exit 1
        fi
    done
    print_status "Component files: All present ✓"
}

estimate_cost() {
    local model=$1
    local k=$2
    local test_size=$3
    
    # Rough token estimates
    local tokens_per_example=200
    local tokens_per_test=300
    local total_tokens=$(( (k * tokens_per_example + tokens_per_test) * test_size ))
    
    # Cost per 1K tokens (rough estimates)
    local cost_per_1k
    case $model in
        "gpt-4o-mini") cost_per_1k=0.00015 ;;
        "gpt-3.5-turbo") cost_per_1k=0.0005 ;;
        "gpt-4o") cost_per_1k=0.005 ;;
        "gpt-4-turbo") cost_per_1k=0.01 ;;
        *) cost_per_1k=0.001 ;;
    esac
    
    local estimated_cost=$(python3 -c "print(f'{$total_tokens * $cost_per_1k / 1000:.4f}')")
    
    print_status "Estimated cost for $model:"
    print_status "  Tokens: ~$total_tokens"
    print_status "  Cost: ~\$$estimated_cost USD"
}

run_single_experiment() {
    local model=$1
    local k=$2
    local template=$3
    local selection=$4
    
    local exp_name="${model//[.-]/_}_${template}_${selection}_${k}shot"
    print_header "🧪 Running: $exp_name"
    
    # Estimate cost
    estimate_cost "$model" "$k" "$TEST_SIZE"
    
    if [[ "$VERBOSE" == "true" ]]; then
        read -p "Continue with this experiment? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Experiment skipped"
            return 0
        fi
    fi
    
    # Create experiment script
    mkdir -p "$OUTPUT_DIR/scripts"
    local script_file="$OUTPUT_DIR/scripts/run_${exp_name}.py"
    
    cat > "$script_file" << EOF
import sys
import os
import time
sys.path.append('.')

from LLM_testing import WolofExperimentPipeline

def main():
    # Configuration
    config = {
        'name': '$exp_name',
        'description': 'OpenAI experiment: $model, $template, $selection, $k examples',
        'llm': {
            'model': '$model',
            'max_tokens': 2000,
            'temperature': $TEMPERATURE,
            'api_key': None  # Uses environment variable
        },
        'selection': {
            'strategy': '$selection',
            'num_examples': $k,
            'seed': 42
        },
        'prompt': {
            'template': '$template'
        },
        'test_size': $TEST_SIZE
    }
    
    print(f"🚀 Starting OpenAI experiment: $exp_name")
    print(f"⚙️  Model: $model")
    print(f"📝 Template: $template") 
    print(f"🎯 Selection: $selection")
    print(f"📊 Examples: $k, Test size: $TEST_SIZE")
    
    try:
        # Initialize pipeline
        pipeline = WolofExperimentPipeline('$INPUT_DIR')
        
        # Run experiment
        results = pipeline.run_experiment(config)
        
        if results:
            # Save results
            results_dir = '$OUTPUT_DIR/results'
            os.makedirs(results_dir, exist_ok=True)
            
            pipeline.reporter.export_results_to_json(results, f'{results_dir}/${exp_name}_results.json')
            pipeline.reporter.export_results_to_csv(results, f'{results_dir}/${exp_name}_detailed.csv')
            
            # Print summary
            print(f"\\n📊 RESULTS SUMMARY for {results.experiment_name}")
            print(f"{'='*50}")
            print(f"UAS (Unlabeled Attachment): {results.overall_uas:.3f}")
            print(f"LAS (Labeled Attachment):   {results.overall_las:.3f}")
            print(f"Label Accuracy:             {results.overall_la:.3f}")
            print(f"Root Accuracy:              {results.root_accuracy:.3f}")
            print(f"Complete Match Rate:        {results.complete_match_rate:.3f}")
            print(f"Processing Time:            {results.total_processing_time:.1f}s")
            
            print(f"\\n💾 Results saved:")
            print(f"  JSON: {results_dir}/${exp_name}_results.json")
            print(f"  CSV:  {results_dir}/${exp_name}_detailed.csv")
            
            return True
        else:
            print(f"❌ Experiment failed!")
            return False
            
    except KeyboardInterrupt:
        print(f"\\n🛑 Experiment interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Experiment error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
EOF

    # Run the experiment
    print_status "Executing experiment script..."
    python3 "$script_file"
    
    # Sleep between experiments if multiple
    if [[ "$SLEEP_TIME" != "0" ]]; then
        print_status "Sleeping ${SLEEP_TIME}s before next experiment..."
        sleep "$SLEEP_TIME"
    fi
}

run_experiments() {
    print_header "🇸🇳 WOLOF DEPENDENCY PARSING WITH OPENAI"
    print_header "========================================="
    
    # Parse comma-separated values for multiple experiments
    IFS=',' read -ra MODELS <<< "$MODEL"
    IFS=',' read -ra KS <<< "$FEW_SHOT_K"
    IFS=',' read -ra TEMPLATES <<< "$TEMPLATE"
    IFS=',' read -ra SELECTIONS <<< "$SELECTION_STRATEGY"
    
    local total_experiments=$((${#MODELS[@]} * ${#KS[@]} * ${#TEMPLATES[@]} * ${#SELECTIONS[@]}))
    print_status "Planning $total_experiments experiment(s)"
    
    # Show total estimated cost
    local total_cost=0
    for model in "${MODELS[@]}"; do
        for k in "${KS[@]}"; do
            estimate_cost "$model" "$k" "$TEST_SIZE"
        done
    done
    
    echo
    if [[ "$total_experiments" -gt 1 ]]; then
        read -p "Run all $total_experiments experiments? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Experiments cancelled"
            exit 0
        fi
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"/{results,scripts,logs}
    
    # Run experiments
    local current=0
    local successful=0
    
    for model in "${MODELS[@]}"; do
        for k in "${KS[@]}"; do
            for template in "${TEMPLATES[@]}"; do
                for selection in "${SELECTIONS[@]}"; do
                    current=$((current + 1))
                    print_header "[$current/$total_experiments] Experiment Configuration"
                    
                    run_single_experiment "$model" "$k" "$template" "$selection" && successful=$((successful + 1))
                    
                done
            done
        done
    done
    
    print_header "🎉 EXPERIMENTS COMPLETED!"
    print_status "Successful: $successful/$total_experiments"
    print_status "Results directory: $OUTPUT_DIR/results/"
    
    # List result files
    if [[ -d "$OUTPUT_DIR/results" ]]; then
        print_status "Generated files:"
        ls -la "$OUTPUT_DIR/results/"
    fi
}

# Main execution
main() {
    print_header "🇸🇳 WOLOF FEW-SHOT DEPENDENCY PARSING"
    print_status "Optimized for OpenAI API"
    print_status "Current configuration:"
    print_status "  Model(s): $MODEL"
    print_status "  Examples: $FEW_SHOT_K"
    print_status "  Template: $TEMPLATE"
    print_status "  Selection: $SELECTION_STRATEGY" 
    print_status "  Test Size: $TEST_SIZE"
    print_status "  Temperature: $TEMPERATURE"
    
    check_prerequisites
    run_experiments
}

# Execute
main "$@"