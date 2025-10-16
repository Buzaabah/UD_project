#!/bin/bash

# Wolof Few-Shot Dependency Parsing Experiment Runner
# Usage: ./run_wolof_experiments.sh [options]

set -e  # Exit on any error

# Default configuration
INPUT_DIR="data/annotated_data/wol"
OUTPUT_DIR="outputs/wolof"
MODEL="claude-3-sonnet-20240229"
FEW_SHOT_K=5
LANG_HINT="Wolof"
TEMPLATE="gloss_focused"
SELECTION_STRATEGY="diversity"
TEST_SIZE=20
TEMPERATURE=0.1
SLEEP_TIME=0.5
VERBOSE=true
IGNORE_PUNCT=true
KEEP_TOKENIZATION=true

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
🇸🇳 Wolof Few-Shot Dependency Parsing Experiment Runner

Usage: $0 [OPTIONS]

DATA OPTIONS:
    --input_dir DIR         Input directory containing .conllu files (default: annotated_data)
    --output_dir DIR        Output directory for results (default: outputs/wolof)

MODEL OPTIONS:
    --model MODEL           LLM model to use (default: claude-3-sonnet-20240229)
                           Options: claude-3-sonnet-20240229, claude-3-haiku-20240307,
                                   gpt-4, gpt-4-turbo, gpt-3.5-turbo
    --temperature TEMP      Temperature for generation (default: 0.1)

EXPERIMENT OPTIONS:
    --few_shot_k K          Number of few-shot examples (default: 5)
    --template TEMPLATE     Prompt template to use (default: gloss_focused)
                           Options: basic, detailed, chain_of_thought, multilingual, gloss_focused
    --selection STRATEGY    Example selection strategy (default: diversity)
                           Options: random, diversity, coverage, stratified, similarity
    --test_size SIZE        Number of test sentences (default: 20)

PROCESSING OPTIONS:
    --lang_hint LANG        Language hint for model (default: Wolof)
    --keep_tokenization     Keep original tokenization (default: true)
    --ignore_punct         Ignore punctuation in evaluation (default: true)
    --sleep TIME           Sleep time between API calls in seconds (default: 0.5)
    --verbose              Enable verbose output (default: true)

UTILITY OPTIONS:
    --help                 Show this help message
    --test_components      Test pipeline components without API calls
    --check_data          Check data files and show statistics
    --setup_env           Setup Python environment and dependencies

EXAMPLES:
    # Basic experiment with Claude
    $0 --model claude-3-sonnet-20240229 --few_shot_k 3 --test_size 10

    # GPT-4 with chain-of-thought prompting
    $0 --model gpt-4 --template chain_of_thought --few_shot_k 5

    # Multiple experiments (advanced)
    $0 --few_shot_k 3,5,7 --template basic,gloss_focused --selection random,diversity

    # Test setup without API calls
    $0 --test_components

ENVIRONMENT VARIABLES:
    ANTHROPIC_API_KEY      API key for Claude models
    OPENAI_API_KEY         API key for OpenAI models

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output_dir|--out_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --few_shot_k)
            FEW_SHOT_K="$2"
            shift 2
            ;;
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        --selection)
            SELECTION_STRATEGY="$2"
            shift 2
            ;;
        --test_size)
            TEST_SIZE="$2"
            shift 2
            ;;
        --lang_hint)
            LANG_HINT="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --sleep)
            SLEEP_TIME="$2"
            shift 2
            ;;
        --keep_tokenization)
            KEEP_TOKENIZATION="${2:-true}"
            shift 1
            if [[ "$1" == "true" || "$1" == "false" ]]; then
                shift 1
            fi
            ;;
        --ignore_punct)
            IGNORE_PUNCT="${2:-true}"
            shift 1
            if [[ "$1" == "true" || "$1" == "false" ]]; then
                shift 1
            fi
            ;;
        --verbose)
            VERBOSE="${2:-true}"
            shift 1
            if [[ "$1" == "true" || "$1" == "false" ]]; then
                shift 1
            fi
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        --test_components)
            print_header "🔧 TESTING PIPELINE COMPONENTS"
            python3 -c "
import sys
sys.path.append('.')
from LLM_testing import test_pipeline_components
test_pipeline_components()
"
            exit 0
            ;;
        --check_data)
            print_header "📊 ANALYZING WOLOF DATASET"
            python3 -c "
import sys
sys.path.append('.')
from LLM_testing import WolofExperimentPipeline
pipeline = WolofExperimentPipeline('$INPUT_DIR')
pipeline.analyze_dataset_statistics()
"
            exit 0
            ;;
        --setup_env)
            print_header "🐍 SETTING UP PYTHON ENVIRONMENT"
            print_status "Installing required packages..."
            pip3 install anthropic openai numpy pandas matplotlib seaborn scikit-learn
            print_status "Environment setup complete!"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to check prerequisites
check_prerequisites() {
    print_header "🔍 CHECKING PREREQUISITES"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3.7+"
        exit 1
    fi
    print_status "Python 3: $(python3 --version)"
    
    # Check required Python packages
    python3 -c "
import sys
required_packages = ['anthropic', 'openai', 'numpy', 'pandas']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'Missing packages: {missing}')
    print('Run: pip3 install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('All required packages installed ✓')
" || {
        print_error "Missing Python dependencies"
        print_status "Run: $0 --setup_env"
        exit 1
    }
    
    # Check data directory
    if [[ ! -d "$INPUT_DIR" ]]; then
        print_error "Data directory '$INPUT_DIR' not found"
        print_status "Expected structure:"
        print_status "  $INPUT_DIR/"
        print_status "  ├── wolof_train.conllu"
        print_status "  ├── wolof_dev.conllu"
        print_status "  └── wolof_test.conllu"
        exit 1
    fi
    
    # Check for at least one data file
    if [[ ! -f "$INPUT_DIR/wolof_train.conllu" && ! -f "$INPUT_DIR/wolof_dev.conllu" && ! -f "$INPUT_DIR/wolof_test.conllu" ]]; then
        print_error "No Wolof .conllu files found in '$INPUT_DIR'"
        exit 1
    fi
    print_status "Data directory: $INPUT_DIR ✓"
    
    # Check API keys based on model
    if [[ "$MODEL" == *"claude"* ]]; then
        if [[ -z "$ANTHROPIC_API_KEY" ]]; then
            print_warning "ANTHROPIC_API_KEY not set for Claude model"
            print_status "Set with: export ANTHROPIC_API_KEY='your_key_here'"
        else
            print_status "Anthropic API key: Set ✓"
        fi
    elif [[ "$MODEL" == *"gpt"* || "$MODEL" == *"openai"* ]]; then
        if [[ -z "$OPENAI_API_KEY" ]]; then
            print_warning "OPENAI_API_KEY not set for OpenAI model"
            print_status "Set with: export OPENAI_API_KEY='your_key_here'"
        else
            print_status "OpenAI API key: Set ✓"
        fi
    fi
    
    # Check component files
    required_files=("fewShort_selection.py" "prompts.py" "evaluation.py" "LLM_testing.py")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "Required component file '$file' not found"
            exit 1
        fi
    done
    print_status "All component files present ✓"
}

# Function to create output directory
setup_output_dir() {
    print_status "Setting up output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    
    # Create subdirectories
    mkdir -p "$OUTPUT_DIR/results"
    mkdir -p "$OUTPUT_DIR/logs" 
    mkdir -p "$OUTPUT_DIR/plots"
    
    # Create experiment log
    EXPERIMENT_LOG="$OUTPUT_DIR/logs/experiment_$(date +%Y%m%d_%H%M%S).log"
    echo "Wolof Dependency Parsing Experiment" > "$EXPERIMENT_LOG"
    echo "Started: $(date)" >> "$EXPERIMENT_LOG"
    echo "Configuration:" >> "$EXPERIMENT_LOG"
    echo "  Model: $MODEL" >> "$EXPERIMENT_LOG"
    echo "  Few-shot K: $FEW_SHOT_K" >> "$EXPERIMENT_LOG"
    echo "  Template: $TEMPLATE" >> "$EXPERIMENT_LOG"
    echo "  Selection: $SELECTION_STRATEGY" >> "$EXPERIMENT_LOG"
    echo "  Test Size: $TEST_SIZE" >> "$EXPERIMENT_LOG"
    echo "" >> "$EXPERIMENT_LOG"
}

# Function to run single experiment configuration
run_experiment_config() {
    local model=$1
    local k=$2
    local template=$3
    local selection=$4
    
    local exp_name="${model}_${template}_${selection}_${k}shot"
    print_status "Running experiment: $exp_name"
    
    # Create Python script for this specific experiment
    cat > "$OUTPUT_DIR/run_experiment.py" << EOF
import sys
import os
sys.path.append('.')

from LLM_testing import WolofExperimentPipeline, create_experiment_configurations

# Custom experiment configuration
experiment_config = {
    'name': '$exp_name',
    'description': 'Automated experiment: $model with $template template, $selection selection, $k examples',
    'llm': {
        'model': '$model',
        'max_tokens': 2000,
        'temperature': $TEMPERATURE,
        'api_key': None
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

# Initialize pipeline
pipeline = WolofExperimentPipeline('$INPUT_DIR')

# Run experiment
print(f"🚀 Starting experiment: $exp_name")
results = pipeline.run_experiment(experiment_config)

if results:
    # Generate report
    summary = pipeline.reporter.generate_summary_report(results)
    print(summary)
    
    # Save results
    pipeline.reporter.export_results_to_json(results, '$OUTPUT_DIR/results/${exp_name}_results.json')
    pipeline.reporter.export_results_to_csv(results, '$OUTPUT_DIR/results/${exp_name}_detailed.csv')
    
    print(f"✅ Experiment completed: $exp_name")
    print(f"📁 Results saved in: $OUTPUT_DIR/results/")
else:
    print(f"❌ Experiment failed: $exp_name")
    sys.exit(1)
EOF

    # Run the experiment
    cd "$(dirname "$0")"
    python3 "$OUTPUT_DIR/run_experiment.py" 2>&1 | tee -a "$EXPERIMENT_LOG"
    
    # Add sleep between experiments
    if [[ "$SLEEP_TIME" != "0" ]]; then
        print_status "Sleeping for ${SLEEP_TIME}s between experiments..."
        sleep "$SLEEP_TIME"
    fi
}

# Function to run multiple experiments (if comma-separated values provided)
run_experiments() {
    print_header "🧪 RUNNING WOLOF DEPENDENCY PARSING EXPERIMENTS"
    
    # Parse comma-separated values
    IFS=',' read -ra MODELS <<< "$MODEL"
    IFS=',' read -ra KS <<< "$FEW_SHOT_K"  
    IFS=',' read -ra TEMPLATES <<< "$TEMPLATE"
    IFS=',' read -ra SELECTIONS <<< "$SELECTION_STRATEGY"
    
    total_experiments=$((${#MODELS[@]} * ${#KS[@]} * ${#TEMPLATES[@]} * ${#SELECTIONS[@]}))
    current_exp=0
    
    print_status "Planning $total_experiments total experiments"
    
    # Run all combinations
    for model in "${MODELS[@]}"; do
        for k in "${KS[@]}"; do
            for template in "${TEMPLATES[@]}"; do
                for selection in "${SELECTIONS[@]}"; do
                    current_exp=$((current_exp + 1))
                    print_header "[$current_exp/$total_experiments] Experiment Configuration:"
                    print_status "  Model: $model"
                    print_status "  Few-shot K: $k"
                    print_status "  Template: $template"
                    print_status "  Selection: $selection"
                    
                    run_experiment_config "$model" "$k" "$template" "$selection"
                done
            done
        done
    done
}

# Function to generate final report
generate_final_report() {
    print_header "📊 GENERATING FINAL REPORT"
    
    cat > "$OUTPUT_DIR/generate_report.py" << EOF
import sys
import os
import json
import glob
sys.path.append('.')

from LLM_testing import EvaluationReporter
from evaluation import ExperimentResults

# Load all results
results_files = glob.glob('$OUTPUT_DIR/results/*_results.json')
all_results = []

reporter = EvaluationReporter()

print(f"Found {len(results_files)} result files")

for results_file in results_files:
    try:
        with open(results_file, 'r') as f:
            result_data = json.load(f)
            # Convert back to ExperimentResults object (simplified)
            print(f"Loaded: {result_data.get('experiment_name', 'Unknown')}")
    except Exception as e:
        print(f"Error loading {results_file}: {e}")

print("📊 Final experiment summary complete!")
print(f"📁 All results available in: $OUTPUT_DIR/")
EOF

    python3 "$OUTPUT_DIR/generate_report.py"
    
    # Create summary file
    echo "Wolof Dependency Parsing Experiment Summary" > "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "==========================================" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "Completed: $(date)" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "Configuration:" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "  Input Directory: $INPUT_DIR" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "  Output Directory: $OUTPUT_DIR" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "  Models: $MODEL" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "  Few-shot K: $FEW_SHOT_K" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "  Templates: $TEMPLATE" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "  Selection Strategies: $SELECTION_STRATEGY" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "  Test Size: $TEST_SIZE" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    echo "Results Files:" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
    ls -la "$OUTPUT_DIR/results/" >> "$OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
}

# Main execution
main() {
    print_header "🇸🇳 WOLOF FEW-SHOT DEPENDENCY PARSING"
    print_header "======================================"
    
    print_status "Configuration:"
    print_status "  Input Directory: $INPUT_DIR" 
    print_status "  Output Directory: $OUTPUT_DIR"
    print_status "  Model(s): $MODEL"
    print_status "  Few-shot K: $FEW_SHOT_K"
    print_status "  Template(s): $TEMPLATE"
    print_status "  Selection: $SELECTION_STRATEGY"
    print_status "  Test Size: $TEST_SIZE"
    print_status "  Temperature: $TEMPERATURE"
    
    # Check prerequisites
    check_prerequisites
    
    # Setup output directory
    setup_output_dir
    
    # Confirm before running
    if [[ "$VERBOSE" == "true" ]]; then
        echo ""
        read -p "Continue with experiments? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Experiments cancelled by user"
            exit 0
        fi
    fi
    
    # Run experiments
    run_experiments
    
    # Generate final report
    generate_final_report
    
    print_header "🎉 ALL EXPERIMENTS COMPLETED!"
    print_status "Results directory: $OUTPUT_DIR"
    print_status "Experiment log: $EXPERIMENT_LOG"
    print_status "Summary: $OUTPUT_DIR/EXPERIMENT_SUMMARY.txt"
}

# Execute main function
main "$@"