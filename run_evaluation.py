"""
Quick start script - Chạy evaluation và hiển thị kết quả ngay
Hỗ trợ 2 modes: Ollama Local hoặc Cloud API
"""

import os
import sys
import subprocess
import pandas as pd
from datetime import datetime

# Đường dẫn
SCRIPT_DIR = r"C:\vscode\python"
EVAL_SCRIPT_OLLAMA = os.path.join(SCRIPT_DIR, "evaluate_model_robust.py")
EVAL_SCRIPT_CLOUDAPI = os.path.join(SCRIPT_DIR, "evaluate_model_cloudapi.py")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "test_results")

# Current mode
CURRENT_MODE = None
EVAL_SCRIPT = None

def check_ollama():
    """Kiểm tra Ollama đang chạy không"""
    import requests
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        print("✅ Ollama is running!")
        return True
    except:
        print("❌ Ollama is NOT running!")
        print("   Please start Ollama first: ollama serve")
        return False

def run_evaluation(mode):
    """Chạy evaluation script theo mode"""
    global EVAL_SCRIPT, CURRENT_MODE
    
    CURRENT_MODE = mode
    
    if mode == "ollama":
        EVAL_SCRIPT = EVAL_SCRIPT_OLLAMA
        print("\n" + "="*80)
        print("🚀 STARTING EVALUATION - OLLAMA LOCAL")
        print("="*80)
        
        if not check_ollama():
            return False
    
    elif mode == "cloudapi":
        EVAL_SCRIPT = EVAL_SCRIPT_CLOUDAPI
        print("\n" + "="*80)
        print("🚀 STARTING EVALUATION - CLOUD API")
        print("="*80)
    
    else:
        print("❌ Invalid mode!")
        return False
    
    try:
        # Chạy evaluation
        print("\n⏳ Running evaluation (this may take ~10-15 minutes)...\n")
        result = subprocess.run(
            [sys.executable, EVAL_SCRIPT],
            cwd=SCRIPT_DIR,
            capture_output=False
        )
        
        if result.returncode != 0:
            print("❌ Evaluation failed!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False

def show_latest_results(mode=None):
    """Hiển thị kết quả mới nhất"""
    if not os.path.exists(RESULTS_DIR):
        print("❌ No results found!")
        return
    
    # Tìm file mới nhất theo mode
    if mode == "ollama":
        csv_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("evaluation_results_robust_") and f.endswith(".csv")]
    elif mode == "cloudapi":
        csv_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("evaluation_results_cloudapi_") and f.endswith(".csv")]
    else:
        # Show all results
        csv_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("evaluation_results_") and f.endswith(".csv")]
    
    if not csv_files:
        print("❌ No CSV results found!")
        return
    
    latest_csv = os.path.join(RESULTS_DIR, sorted(csv_files)[-1])
    
    print("\n" + "="*80)
    print("📊 LATEST RESULTS")
    print("="*80)
    print(f"\n📁 File: {latest_csv}\n")
    
    # Load và display
    df = pd.read_csv(latest_csv)
    
    # Summary
    total = len(df)
    runnable = (df['runnable'] == 'Yes').sum()
    correct = (df['chart_match'] == 'Yes').sum()
    
    print(f"Total queries:        {total}")
    print(f"Runnable:             {runnable}/{total} ({100*runnable/total:.1f}%)")
    print(f"Chart type correct:   {correct}/{total} ({100*correct/total:.1f}%)")
    
    # Show first 10 results
    print("\n📋 First 10 queries:")
    print(df[['query_id', 'expected_chart', 'detected_chart', 'chart_match', 'runnable']].head(10).to_string(index=False))
    
    # Show failed queries
    failed = df[df['runnable'] == 'No']
    if len(failed) > 0:
        print(f"\n⚠️  Failed queries ({len(failed)}):")
        for idx, row in failed.head(5).iterrows():
            print(f"   {row['query_id']}: {row['query'][:50]}...")
            print(f"      Error: {row['error'][:100]}...")
    
    # Show wrong chart type
    wrong_chart = df[(df['chart_match'] == 'No') & (df['runnable'] == 'Yes')]
    if len(wrong_chart) > 0:
        print(f"\n🎯 Wrong chart type ({len(wrong_chart)}):")
        for idx, row in wrong_chart.head(5).iterrows():
            print(f"   {row['query_id']}: Expected {row['expected_chart']}, Got {row['detected_chart']}")
    
    print("\n✅ Full CSV saved to:", latest_csv)

def select_mode():
    """Select evaluation mode"""
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║  🧪 MODEL EVALUATION - SELECT MODE                   ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    print("\nModes:")
    print("  1. 🏠 Ollama Local (test_api_local_ollama.py logic)")
    print("  2. ☁️  Cloud API (chart_generator.py logic)")
    print("  3. 📊 Compare both results")
    print("  4. ❌ Exit")
    
    choice = input("\nSelect mode (1-4): ").strip()
    
    if choice == "1":
        return "ollama"
    elif choice == "2":
        return "cloudapi"
    elif choice == "3":
        return "compare"
    elif choice == "4":
        return None
    else:
        print("Invalid option!")
        return None

def main():
    """Main"""
    mode = select_mode()
    
    if mode is None:
        print("Goodbye! 👋")
        return
    
    if mode == "compare":
        print("\n📊 Comparing both modes...")
        print("Coming soon! For now, run them separately.")
        return
    
    # Menu
    print("\n╔═══════════════════════════════════════════════════════╗")
    print(f"║  🧪 MODEL EVALUATION - {mode.upper():30}║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    print("\nOptions:")
    print("  1. Run evaluation (100 queries)")
    print("  2. Show latest results")
    print("  3. Run and show results")
    print("  4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        if run_evaluation(mode):
            print("✅ Evaluation completed!")
        else:
            print("❌ Evaluation failed!")
    
    elif choice == "2":
        show_latest_results(mode)
    
    elif choice == "3":
        if run_evaluation(mode):
            show_latest_results(mode)
    
    elif choice == "4":
        print("Goodbye! 👋")
    
    else:
        print("Invalid option!")

if __name__ == "__main__":
    main()
