#!/usr/bin/env python3
"""
Script para ejecutar predicciones con el modelo mejorado de LifeMiles.

Uso:
    python run_improved_predictions.py [--days N] [--stores partner1_store1,partner2_store2]

Ejemplos:
    python run_improved_predictions.py --days 7
    python run_improved_predictions.py --days 30 --stores COBU3_11475712,ATRCO_12821393
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Agregar el directorio model al path
model_dir = Path(__file__).parent / "model"
sys.path.append(str(model_dir))

from predict_improved import make_bulk_predictions_improved
from processing.data_manager import load_dataset
from config.core import config


def parse_store_list(store_str: str):
    """Parse comma-separated store list."""
    if not store_str:
        return None
    
    stores = []
    for store in store_str.split(','):
        if '_' in store:
            partner, store_code = store.strip().split('_', 1)
            stores.append((partner, store_code))
        else:
            print(f"⚠️ Invalid store format: {store} (expected: PARTNER_STORECODE)")
    
    return stores if stores else None


def main():
    parser = argparse.ArgumentParser(description='Run improved LifeMiles predictions')
    parser.add_argument('--days', type=int, default=28, 
                       help='Number of days to forecast (default: 28)')
    parser.add_argument('--stores', type=str, 
                       help='Comma-separated list of stores (format: PARTNER_STORECODE)')
    parser.add_argument('--output', type=str, default='predictions_improved.csv',
                       help='Output file name (default: predictions_improved.csv)')
    
    args = parser.parse_args()
    
    print("🔮 LIFEMILES IMPROVED PREDICTIONS")
    print("="*50)
    print(f"Forecast days: {args.days}")
    
    try:
        # Cargar datos
        print("📊 Loading data...")
        data = load_dataset(file_name=config.app_config.train_data_file)
        print(f"✅ Data loaded: {data.shape}")
        
        # Parsear lista de tiendas
        store_list = parse_store_list(args.stores)
        if store_list:
            print(f"🏪 Predicting for {len(store_list)} specific stores")
        else:
            print("🏪 Predicting for all available stores")
        
        # Ejecutar predicciones
        print("🚀 Running predictions...")
        results = make_bulk_predictions_improved(
            input_data=data,
            forecast_days=args.days,
            store_list=store_list
        )
        
        # Procesar resultados
        if results['summary']['successful_predictions'] > 0:
            # Crear DataFrame con todas las predicciones
            all_predictions = []
            
            for store_key, prediction in results['store_predictions'].items():
                if prediction['status'] == 'success':
                    for pred in prediction['predictions']:
                        pred['store_key'] = store_key
                        pred['model_quality'] = prediction['model_info']['model_quality']
                        pred['test_r2'] = prediction['model_info']['test_r2']
                        pred['test_smape'] = prediction['model_info']['test_smape']
                        all_predictions.append(pred)
            
            if all_predictions:
                predictions_df = pd.DataFrame(all_predictions)
                predictions_df.to_csv(args.output, index=False)
                
                print(f"\n📊 PREDICTIONS SUMMARY:")
                print(f"  Total stores: {results['summary']['total_stores']}")
                print(f"  Successful: {results['summary']['successful_predictions']}")
                print(f"  Success rate: {results['summary']['success_rate_pct']:.1f}%")
                
                # Estadísticas de calidad
                quality_counts = predictions_df['model_quality'].value_counts()
                print(f"\n🏆 MODEL QUALITY DISTRIBUTION:")
                for quality, count in quality_counts.items():
                    print(f"  {quality}: {count} stores")
                
                # Estadísticas de predicciones
                total_forecast = predictions_df['predicted_billings'].sum()
                avg_daily = predictions_df['predicted_billings'].mean()
                
                print(f"\n📈 FORECAST SUMMARY:")
                print(f"  Total forecast: ${total_forecast:,.2f}")
                print(f"  Average daily: ${avg_daily:.2f}")
                print(f"  Forecast period: {args.days} days")
                
                print(f"\n💾 Results saved to: {args.output}")
                
                # Mostrar ejemplos de top stores
                if 'test_r2' in predictions_df.columns:
                    top_stores = predictions_df.groupby('store_key')['test_r2'].first().sort_values(ascending=False).head(5)
                    print(f"\n🎯 TOP 5 STORES BY MODEL QUALITY (R²):")
                    for store, r2 in top_stores.items():
                        print(f"  {store}: R² = {r2:.3f}")
            
        else:
            print("❌ No successful predictions generated")
            return 1
        
        print(f"\n✅ PREDICTIONS COMPLETED SUCCESSFULLY!")
        return 0
        
    except Exception as e:
        print(f"\n❌ PREDICTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
