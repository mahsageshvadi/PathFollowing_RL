#!/usr/bin/env python3
"""
Generate summary table for baseline results.
Reports Dice ↑ and BNE ↓ for each dataset and model.

Output format:
METHOD           | DICE ↑ | BNE ↓
U-Net (DRIVE)
U-Net++ (DRIVE)
U-Net (ISBI12)
U-Net++ (ISBI12)
U-Net (CrackForest)
U-Net++ (CrackForest)
"""
import os
import sys
import pandas as pd
from pathlib import Path

def load_metrics(csv_path):
    """Load metrics CSV and return all mean metrics."""
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        metrics = df.mean(numeric_only=True).to_dict()
        return metrics
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def main():
    script_dir = Path(__file__).parent.absolute()
    metrics_dir = script_dir / "metrics"
    
    # Dataset names for display
    dataset_names = {
        'drive_test': 'DRIVE',
        'isbi12_test': 'ISBI12',
        'crack_test': 'CrackForest'
    }
    
    # Model display names
    model_names = {
        'unet': 'U-Net',
        'unetpp': 'U-Net++'
    }
    
    results = []
    
    # Collect results for each dataset and model
    for dataset_key, dataset_display in dataset_names.items():
        for model in ['unet', 'unetpp']:
            csv_path = metrics_dir / f"{dataset_key}" / model / "metrics.csv"
            
            metrics = load_metrics(csv_path)
            
            if metrics is not None:
                model_display = model_names.get(model, model.upper())
                results.append({
                    'method': f"{model_display} ({dataset_display})",
                    'dataset': dataset_display,
                    'model': model_display,
                    'dice': metrics.get('dice', 0.0),
                    'bne': metrics.get('bne', 0.0),
                    'iou': metrics.get('iou', 0.0),
                    'cldice': metrics.get('cldice', 0.0),
                    'n_samples': len(pd.read_csv(csv_path))
                })
            else:
                print(f"WARNING: Missing metrics for {dataset_key}/{model}")
    
    if not results:
        print("ERROR: No metrics found. Make sure eval_baseline.slurm has completed.")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by dataset, then by model
    df = df.sort_values(['dataset', 'model'])
    
    # Print main table (Dice and BNE for paper)
    print("\n" + "="*80)
    print("BASELINE RESULTS SUMMARY (TEST SETS)")
    print("="*80)
    print(f"{'METHOD':<35} | {'DICE ↑':<12} | {'BNE ↓':<12} | {'SAMPLES':<10}")
    print("-"*80)
    
    for _, row in df.iterrows():
        print(f"{row['method']:<35} | {row['dice']:>11.4f} | {row['bne']:>11.4f} | {row['n_samples']:>9}")
    
    print("="*80)
    
    # Print detailed metrics table
    print("\n" + "="*100)
    print("DETAILED METRICS (ALL)")
    print("="*100)
    print(f"{'METHOD':<35} | {'DICE ↑':<10} | {'IoU ↑':<10} | {'clDice ↑':<12} | {'BNE ↓':<12}")
    print("-"*100)
    
    for _, row in df.iterrows():
        print(f"{row['method']:<35} | {row['dice']:>9.4f} | {row['iou']:>9.4f} | {row['cldice']:>11.4f} | {row['bne']:>11.4f}")
    
    print("="*100)
    
    # Print per-dataset summary
    print("\nPer-Dataset Summary:")
    print("-"*100)
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        print(f"\n{dataset}:")
        for _, row in dataset_df.iterrows():
            print(f"  {row['model']:<15} | DICE: {row['dice']:>7.4f} | IoU: {row['iou']:>7.4f} | clDice: {row['cldice']:>7.4f} | BNE: {row['bne']:>7.4f}")
    
    print("\n" + "="*100)
    print("Legend:")
    print("  DICE ↑:   Higher is better (Dice Similarity Coefficient) - Pixel-wise accuracy")
    print("  IoU ↑:    Higher is better (Intersection over Union)")
    print("  clDice ↑: Higher is better (Centerline Dice) - Skeleton-based metric")
    print("  BNE ↓:    Lower is better (Betti Number Error) - Topological correctness")
    print("="*100)
    print()
    
    # Save to CSV
    output_csv = script_dir / "baseline_results_summary.csv"
    df.to_csv(output_csv, index=False)
    print(f"Summary saved to: {output_csv}")
    
    # Also save LaTeX table format (main table for paper)
    output_latex = script_dir / "baseline_results_table.tex"
    with open(output_latex, 'w') as f:
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("METHOD & DICE $\\uparrow$ & BNE $\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            method = row['method'].replace('_', '\\_')
            f.write(f"{method} & {row['dice']:.4f} & {row['bne']:.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"LaTeX table (main) saved to: {output_latex}")
    
    # Save detailed LaTeX table
    output_latex_detailed = script_dir / "baseline_results_detailed.tex"
    with open(output_latex_detailed, 'w') as f:
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("METHOD & DICE $\\uparrow$ & IoU $\\uparrow$ & clDice $\\uparrow$ & BNE $\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            method = row['method'].replace('_', '\\_')
            f.write(f"{method} & {row['dice']:.4f} & {row['iou']:.4f} & {row['cldice']:.4f} & {row['bne']:.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"LaTeX table (detailed) saved to: {output_latex_detailed}")

if __name__ == "__main__":
    main()
