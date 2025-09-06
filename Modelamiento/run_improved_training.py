#!/usr/bin/env python3
"""
Script principal para ejecutar el entrenamiento del modelo LifeMiles.

Uso:
    python run_improved_training.py                # Todas las tiendas
    python run_improved_training.py --n 1         # Solo 1 tienda (testing)
    python run_improved_training.py --n 5         # Solo 5 tiendas (debugging)
    python run_improved_training.py --stores 100  # Máximo 100 tiendas

"""

import sys
import os
import argparse
from pathlib import Path

# Agregar el directorio model al path
model_dir = Path(__file__).parent / "model"
sys.path.append(str(model_dir))

from train_pipeline_improved import run_training_improved


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LifeMiles Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_improved_training.py                # Train all stores
  python run_improved_training.py --n 1         # Train only 1 store (testing)
  python run_improved_training.py --n 5         # Train only 5 stores (debugging)
  python run_improved_training.py --n 50        # Train 50 stores (validation)
        """
    )
    
    parser.add_argument(
        '--n', '--stores',
        type=int,
        default=None,
        help='Number of stores to train (default: all stores with sufficient data)'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=1,
        help='Minimum samples required per store (default: 1 - trains all stores)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Execute the training pipeline."""
    
    # Parse arguments
    args = parse_arguments()
    
    print("LIFEMILES TRAINING PIPELINE")
    print("="*50)
    print("="*50)
    
    if args.n:
        print(f"Training mode: LIMITED to {args.n} stores (testing/debugging)")
    else:
        print(f"Training mode: ALL STORES (production)")
    
    print(f"Minimum samples per store: {args.min_samples}")
    print("="*50)
    
    try:
        # Ejecutar entrenamiento mejorado con parámetros
        results = run_training_improved(
            max_stores=args.n,
            min_samples=args.min_samples,
            verbose=args.verbose
        )
        
        print(f"\nTRAINING COMPLETED SUCCESSFULLY!")
        print(f"Summary:")
        print(f"  - Total stores: {results['total_stores']}")
        print(f"  - Successful models: {results['successful_models']}")
        print(f"  - Success rate: {results['successful_models']/results['total_stores']*100:.1f}%")
        
        successful_results = results['results']
        if successful_results:
            # Estadísticas finales
            r2_scores = [r['test_metrics']['r2'] for r in successful_results]
            smape_scores = [r['test_metrics']['smape'] for r in successful_results]
            
            excellent_models = sum(1 for r in successful_results if r['quality'] == 'EXCELENTE')
            good_models = sum(1 for r in successful_results if r['quality'] == 'BUENO')
            
            print(f"\nFINAL PERFORMANCE:")
            print(f"  - Average R²: {sum(r2_scores)/len(r2_scores):.3f}")
            print(f"  - Average SMAPE: {sum(smape_scores)/len(smape_scores):.1f}%")
            print(f"  - Excellent models: {excellent_models} ({excellent_models/len(successful_results)*100:.1f}%)")
            print(f"  - Good+ models: {excellent_models + good_models} ({(excellent_models + good_models)/len(successful_results)*100:.1f}%)")
            
            # Verificar si se cumplieron los objetivos
            avg_r2 = sum(r2_scores) / len(r2_scores)
            avg_smape = sum(smape_scores) / len(smape_scores)
            
            print(f"\nOBJECTIVES ACHIEVEMENT:")
            if avg_r2 >= 0.8:
                print(f"  R² target (≥0.8): ACHIEVED ({avg_r2:.3f})")
            else:
                print(f"  R² target (≥0.8): NOT ACHIEVED ({avg_r2:.3f})")
                
            if avg_smape <= 10.0:
                print(f"  SMAPE target (≤10%): ACHIEVED ({avg_smape:.1f}%)")
            else:
                print(f"  SMAPE target (≤10%): NOT ACHIEVED ({avg_smape:.1f}%)")
        
        print(f"\nFiles saved:")
        print(f"  - Models: model/trained/")
        print(f"  - Pipeline: model/trained/")
        print(f"  - MLflow logs: mlruns/")
        
        print(f"\nReady for production forecasting!")
        
        return 0
        
    except Exception as e:
        print(f"\nTRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
