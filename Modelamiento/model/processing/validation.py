from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, validator

from model.config.core import config


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    
    validated_data = input_data.copy()
    errors = None
    
    try:
        # Validaciones específicas para datos de LifeMiles
        
        # 1. Verificar que existen las columnas requeridas
        required_columns = [
            config.model_config.date_column,
            config.model_config.target,
            config.model_config.store_id_column,
            config.model_config.partner_id_column
        ]
        
        missing_columns = [col for col in required_columns if col not in validated_data.columns]
        if missing_columns:
            errors = f"Missing required columns: {missing_columns}"
            return validated_data, errors
        
        # 2. Validar formato de fecha
        try:
            validated_data[config.model_config.date_column] = pd.to_datetime(
                validated_data[config.model_config.date_column]
            )
        except Exception as e:
            errors = f"Invalid date format in {config.model_config.date_column}: {str(e)}"
            return validated_data, errors
        
        # 3. Validar que los valores de billings son numéricos y no negativos
        if validated_data[config.model_config.target].dtype not in ['int64', 'float64']:
            try:
                validated_data[config.model_config.target] = pd.to_numeric(
                    validated_data[config.model_config.target], errors='coerce'
                )
            except Exception as e:
                errors = f"Cannot convert {config.model_config.target} to numeric: {str(e)}"
                return validated_data, errors
        
        # 4. Rellenar valores nulos en target con 0 (asumiendo que significa no hubo ventas)
        validated_data[config.model_config.target] = validated_data[config.model_config.target].fillna(0)
        
        # 5. Asegurar que no hay valores negativos en billings
        if (validated_data[config.model_config.target] < 0).any():
            validated_data[config.model_config.target] = validated_data[config.model_config.target].clip(lower=0)
        
        # 6. Validar que hay datos suficientes
        if len(validated_data) < 30:
            errors = f"Insufficient data: {len(validated_data)} rows. Minimum 30 required."
            return validated_data, errors
        
        # 7. Ordenar por fecha para asegurar secuencia temporal
        validated_data = validated_data.sort_values([
            config.model_config.partner_id_column,
            config.model_config.store_id_column,
            config.model_config.date_column
        ]).reset_index(drop=True)
        
    except Exception as e:
        errors = f"Validation error: {str(e)}"
    
    return validated_data, errors


class ForecastInputSchema(BaseModel):
    """Schema for forecast input validation."""
    
    Process_Date: str
    Partner_Code: str
    Store_Code: str
    Billings: float
    Partner_Name: Optional[str] = None
    Store_Name: Optional[str] = None
    
    @validator('Process_Date')
    def validate_date(cls, v):
        try:
            pd.to_datetime(v)
            return v
        except Exception:
            raise ValueError('Invalid date format')
    
    @validator('Billings')
    def validate_billings(cls, v):
        if v < 0:
            raise ValueError('Billings cannot be negative')
        return v
    
    @validator('Partner_Code', 'Store_Code')
    def validate_codes(cls, v):
        if not v or v.strip() == '':
            raise ValueError('Code cannot be empty')
        return v.strip()


class MultipleForecastInputs(BaseModel):
    """Schema for multiple forecast inputs."""
    inputs: List[ForecastInputSchema]


def validate_forecast_input(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Validate input data for forecasting."""
    
    try:
        # Convertir a formato esperado para validación
        data_dict = input_data.to_dict('records')
        
        # Validar usando Pydantic
        MultipleForecastInputs(inputs=data_dict)
        
        # Si la validación pasa, aplicar validaciones adicionales
        return validate_inputs(input_data=input_data)
        
    except ValidationError as e:
        return input_data, f"Input validation error: {e.json()}"
    except Exception as e:
        return input_data, f"Unexpected validation error: {str(e)}"
