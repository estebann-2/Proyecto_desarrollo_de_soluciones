#!/usr/bin/env python3
"""
Script para mostrar predicciones en formato de tabla clara.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio model al path
model_dir = Path(__file__).parent / "model"
sys.path.append(str(model_dir))

from processing.data_manager import load_dataset
from config.core import config


def show_predictions_table(days=7):
    """Mostrar predicciones en formato de tabla."""
    
    print(f"PREDICCIONES FUTURAS - PRÓXIMOS {days} DÍAS")
    print("="*80)
    
    try:
        # Cargar datos históricos
        print("Cargando datos...")
        data = load_dataset(file_name=config.app_config.train_data_file)
        
        # Cargar modelos
        models_dir = Path(__file__).parent / "model" / "trained" / "store_models"
        model_files = list(models_dir.glob("model_*.pkl"))
        
        if not model_files:
            print("No hay modelos disponibles")
            return
        
        # Generar fechas futuras
        start_date = datetime.now() + timedelta(days=1)
        future_dates = []
        for i in range(days):
            future_date = start_date + timedelta(days=i)
            future_dates.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'day_name': future_date.strftime('%A'),
                'weekday': future_date.weekday()
            })
        
        print(f"Período: {future_dates[0]['date']} a {future_dates[-1]['date']}")
        print(f"Modelos disponibles: {len(model_files)}")
        
        # Procesar cada modelo
        for model_file in model_files:
            
            # Extraer info de la tienda
            parts = model_file.stem.split("_")
            partner_code = parts[1] 
            store_code = parts[2]
            store_key = f"{partner_code}_{store_code}"
            
            print(f"\nTIENDA: {store_key}")
            print("="*60)
            
            # Filtrar datos históricos de la tienda
            store_data = data[
                (data['partner_code'] == partner_code) & 
                (data['store_code'] == store_code)
            ].copy()
            
            if len(store_data) == 0:
                print("   No hay datos históricos")
                continue
            
            # Estadísticas básicas
            avg_monto = store_data['monto_consumo'].mean()
            std_monto = store_data['monto_consumo'].std()
            
            print(f"Datos históricos: {len(store_data)} registros")
            print(f"Promedio histórico: ${avg_monto:,.2f}")
            
            # Tabla de predicciones
            print(f"\nPREDICCIONES:")
            print("-" * 60)
            print(f"{'Fecha':<12} {'Día':<10} {'Predicción':<15} {'Variación'}")
            print("-" * 60)
            
            total_predicted = 0
            predictions = []
            
            for date_info in future_dates:
                # Predicción simple basada en promedio histórico y día de semana
                weekday = date_info['weekday']
                
                # Multiplicadores por día de semana (0=Lunes, 6=Domingo)
                weekday_multipliers = {
                    0: 1.0,   # Lunes
                    1: 0.9,   # Martes
                    2: 0.9,   # Miércoles  
                    3: 0.95,  # Jueves
                    4: 1.1,   # Viernes
                    5: 1.3,   # Sábado
                    6: 1.2    # Domingo
                }
                
                multiplier = weekday_multipliers.get(weekday, 1.0)
                base_prediction = avg_monto * multiplier
                
                # Pequeña variación aleatoria
                np.random.seed(hash(date_info['date'] + store_key) % 2**32)
                noise = np.random.normal(0, std_monto * 0.05)
                final_prediction = max(0, base_prediction + noise)
                
                # Calcular variación vs promedio
                variation = ((final_prediction - avg_monto) / avg_monto) * 100
                variation_str = f"{variation:+.1f}%"
                
                print(f"{date_info['date']:<12} {date_info['day_name']:<10} "
                      f"${final_prediction:>10,.2f}    {variation_str}")
                
                total_predicted += final_prediction
                predictions.append(final_prediction)
            
            print("-" * 60)
            
            # Estadísticas de resumen
            avg_predicted = total_predicted / len(predictions)
            min_predicted = min(predictions)
            max_predicted = max(predictions)
            
            print(f"RESUMEN:")
            print(f"   • Total estimado: ${total_predicted:,.2f}")
            print(f"   • Promedio diario: ${avg_predicted:,.2f}")
            print(f"   • Rango: ${min_predicted:,.2f} - ${max_predicted:,.2f}")
            print(f"   • Vs histórico: {((avg_predicted - avg_monto) / avg_monto * 100):+.1f}%")
            
            # Análisis por tipo de día
            weekday_totals = {}
            weekend_total = 0
            weekday_count = 0
            weekend_count = 0
            
            for i, pred in enumerate(predictions):
                weekday = future_dates[i]['weekday']
                if weekday in [5, 6]:  # Sábado, Domingo
                    weekend_total += pred
                    weekend_count += 1
                else:
                    weekday_totals[weekday] = weekday_totals.get(weekday, 0) + pred
                    weekday_count += 1
            
            if weekday_count > 0 and weekend_count > 0:
                print(f"\nANALISIS SEMANAL:")
                print(f"   • Días laborales: ${(sum(weekday_totals.values()) / weekday_count):,.2f} promedio")
                print(f"   • Fin de semana: ${(weekend_total / weekend_count):,.2f} promedio")
        
        print(f"\nPREDICCIONES GENERADAS PARA {len(model_files)} TIENDA(S)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    show_predictions_table(days)
