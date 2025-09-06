#!/usr/bin/env python3
"""
Script para explorar y visualizar los experimentos de MLflow.

Uso:
    python mlflow_explorer.py                    # Lista experimentos
    python mlflow_explorer.py --experiment-id 1  # Muestra runs de un experimento específico
    python mlflow_explorer.py --best-models      # Muestra los mejores modelos
    python mlflow_explorer.py --serve            # Inicia MLflow UI server
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Agregar el directorio model al path
model_dir = Path(__file__).parent / "model"
sys.path.append(str(model_dir))

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from config.mlflow_config import setup_mlflow_tracking
except ImportError:
    print("Error: MLflow not installed. Please install with: pip install mlflow")
    sys.exit(1)


def setup_mlflow():
    """Setup MLflow tracking."""
    return setup_mlflow_tracking()


def list_experiments():
    """List all experiments."""
    print("MLflow Experiments:")
    print("=" * 50)
    
    client = MlflowClient()
    experiments = client.search_experiments()
    
    if not experiments:
        print("No experiments found.")
        return
    
    for experiment in experiments:
        print(f"ID: {experiment.experiment_id}")
        print(f"Name: {experiment.name}")
        print(f"Lifecycle Stage: {experiment.lifecycle_stage}")
        if experiment.tags:
            print(f"Tags: {experiment.tags}")
        print("-" * 30)


def list_runs(experiment_id: str = None):
    """List runs for an experiment."""
    client = MlflowClient()
    
    if experiment_id:
        runs = client.search_runs(experiment_ids=[experiment_id])
        print(f"Runs for Experiment ID {experiment_id}:")
    else:
        # Get the default experiment
        experiment = client.get_experiment_by_name("LifeMiles_Forecasting_Training")
        if experiment:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            print(f"Runs for LifeMiles_Forecasting_Training:")
        else:
            print("No LifeMiles experiment found.")
            return
    
    print("=" * 80)
    
    if not runs:
        print("No runs found.")
        return
    
    # Create a summary table
    data = []
    for run in runs:
        row = {
            'run_id': run.info.run_id[:8],
            'name': run.data.tags.get('mlflow.runName', 'N/A'),
            'status': run.info.status,
            'avg_r2': run.data.metrics.get('avg_r2', 'N/A'),
            'avg_smape': run.data.metrics.get('avg_smape', 'N/A'),
            'successful_models': run.data.metrics.get('successful_models', 'N/A'),
            'start_time': pd.to_datetime(run.info.start_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))


def show_best_models():
    """Show best performing models."""
    client = MlflowClient()
    
    # Get the experiment
    experiment = client.get_experiment_by_name("LifeMiles_Forecasting_Training")
    if not experiment:
        print("No LifeMiles experiment found.")
        return
    
    # Search for runs with metrics
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.avg_r2 > 0.0",
        order_by=["metrics.avg_r2 DESC"]
    )
    
    if not runs:
        print("No runs with R² metrics found.")
        return
    
    print("Top 10 Best Performing Training Runs:")
    print("=" * 80)
    
    data = []
    for i, run in enumerate(runs[:10]):
        row = {
            'rank': i + 1,
            'run_id': run.info.run_id[:8],
            'name': run.data.tags.get('mlflow.runName', 'N/A'),
            'avg_r2': f"{run.data.metrics.get('avg_r2', 0):.3f}",
            'avg_smape': f"{run.data.metrics.get('avg_smape', 0):.1f}%",
            'models': run.data.metrics.get('successful_models', 0),
            'success_rate': f"{run.data.metrics.get('success_rate', 0):.1f}%",
            'date': pd.to_datetime(run.info.start_time, unit='ms').strftime('%Y-%m-%d')
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Show best individual store models
    print("\n" + "=" * 80)
    print("Individual Store Models (Top 5 by R²):")
    print("=" * 80)
    
    # Search for nested runs (individual store models)
    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.store_id != ''",
        order_by=["metrics.test_r2 DESC"]
    )
    
    if all_runs:
        store_data = []
        for i, run in enumerate(all_runs[:5]):
            row = {
                'rank': i + 1,
                'store_id': run.data.tags.get('store_id', 'N/A'),
                'quality': run.data.tags.get('model_quality', 'N/A'),
                'test_r2': f"{run.data.metrics.get('test_r2', 0):.3f}",
                'test_smape': f"{run.data.metrics.get('test_smape', 0):.1f}%",
                'train_samples': run.data.metrics.get('train_samples', 0),
                'features': run.data.metrics.get('n_features', 0)
            }
            store_data.append(row)
        
        store_df = pd.DataFrame(store_data)
        print(store_df.to_string(index=False))


def start_mlflow_ui():
    """Start MLflow UI server."""
    import subprocess
    import os
    
    mlruns_path = setup_mlflow()
    
    print(f"Starting MLflow UI...")
    print(f"MLflow tracking URI: file://{mlruns_path}")
    print(f"Open your browser at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Change to the directory containing mlruns
        os.chdir(Path(mlruns_path).parent)
        subprocess.run(["mlflow", "ui", "--backend-store-uri", f"file://{mlruns_path}"])
    except KeyboardInterrupt:
        print("\nMLflow UI server stopped.")
    except FileNotFoundError:
        print("Error: MLflow command not found. Please install MLflow: pip install mlflow")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MLflow Explorer for LifeMiles Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mlflow_explorer.py                    # Lista experimentos
  python mlflow_explorer.py --experiment-id 1  # Muestra runs de un experimento
  python mlflow_explorer.py --best-models      # Muestra los mejores modelos
  python mlflow_explorer.py --serve            # Inicia MLflow UI server
        """
    )
    
    parser.add_argument(
        '--experiment-id',
        type=str,
        help='Show runs for specific experiment ID'
    )
    
    parser.add_argument(
        '--best-models',
        action='store_true',
        help='Show best performing models'
    )
    
    parser.add_argument(
        '--serve',
        action='store_true',
        help='Start MLflow UI server'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    print("LIFEMILES MLFLOW EXPLORER")
    print("=" * 50)
    
    # Setup MLflow
    setup_mlflow()
    
    if args.serve:
        start_mlflow_ui()
    elif args.best_models:
        show_best_models()
    elif args.experiment_id:
        list_runs(args.experiment_id)
    else:
        list_experiments()
        print("\nTo see runs in an experiment, use --experiment-id")
        print("To see best models, use --best-models")
        print("To start MLflow UI, use --serve")


if __name__ == "__main__":
    main()
