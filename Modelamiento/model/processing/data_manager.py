import typing as t
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Load dataset from CSV file and standardize column names."""
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    
    # Mapeo de columnas para estandarizar nombres
    column_mapping = {
        'Partner Code': 'partner_code',
        'Partner Name': 'partner_name', 
        'Store Code': 'store_code',
        'Store Name': 'store_name',
        'Process Date': 'date',
        'Billings': 'billings',
        'Miles': 'miles'
    }
    
    # Renombrar columnas si existen
    columns_to_rename = {k: v for k, v in column_mapping.items() if k in dataframe.columns}
    if columns_to_rename:
        dataframe = dataframe.rename(columns=columns_to_rename)
        print(f"Columnas renombradas: {list(columns_to_rename.keys())} -> {list(columns_to_rename.values())}")
    
    # Asegurar tipos correctos y crear store_id si las columnas existen
    if 'partner_code' in dataframe.columns and 'store_code' in dataframe.columns:
        # Convertir ambos a string para evitar problemas de concatenación
        dataframe['partner_code'] = dataframe['partner_code'].astype(str)
        dataframe['store_code'] = dataframe['store_code'].astype(str)
        # Crear store_id combinado
        dataframe['store_id'] = dataframe['partner_code'] + '_' + dataframe['store_code']
    
    return dataframe


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the preprocessing pipeline.
    Saves the versioned pipeline, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained pipeline that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}_preprocessing_{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def save_store_models(*, models_data: t.List[t.Dict]) -> None:
    """Save individual store models and metadata."""
    
    # Prepare models directory
    models_dir = TRAINED_MODEL_DIR / "store_models"
    models_dir.mkdir(exist_ok=True)
    
    # Save metadata
    metadata = []
    
    for model_data in models_data:
        store_code = model_data['store_code']
        partner_code = model_data['partner_code']
        model = model_data['model']
        
        # Create unique filename
        model_filename = f"model_{partner_code}_{store_code}_{_version}.pkl"
        model_path = models_dir / model_filename
        
        # Save model
        joblib.dump(model, model_path)
        
        # Add to metadata
        metadata.append({
            'store_code': store_code,
            'partner_code': partner_code,
            'model_file': model_filename,
            'mae': model_data['mae'],
            'rmse': model_data['rmse'],
            'features': model_data['features'],
            'train_samples': model_data['train_samples'],
            'val_samples': model_data['val_samples'],
            'version': _version
        })
    
    # Save metadata
    metadata_file = TRAINED_MODEL_DIR / f"models_metadata_{_version}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def load_store_model(*, partner_code: str, store_code: str) -> t.Optional[Pipeline]:
    """Load a specific store model."""
    
    # Load metadata to find the correct model file
    metadata_file = TRAINED_MODEL_DIR / f"models_metadata_{_version}.json"
    
    if not metadata_file.exists():
        print(f"Metadata file not found: {metadata_file}")
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Find model for specific store and partner
    for model_info in metadata:
        if (model_info['store_code'] == store_code and 
            model_info['partner_code'] == partner_code):
            
            model_file = model_info['model_file']
            model_path = TRAINED_MODEL_DIR / "store_models" / model_file
            
            if model_path.exists():
                return joblib.load(model_path)
            else:
                print(f"Model file not found: {model_path}")
                return None
    
    print(f"No model found for Partner {partner_code}, Store {store_code}")
    return None


def get_available_stores() -> t.List[t.Dict[str, str]]:
    """Get list of stores with trained models."""
    
    metadata_file = TRAINED_MODEL_DIR / f"models_metadata_{_version}.json"
    
    if not metadata_file.exists():
        return []
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return [
        {
            'partner_code': model_info['partner_code'],
            'store_code': model_info['store_code'],
            'mae': model_info['mae'],
            'rmse': model_info['rmse']
        }
        for model_info in metadata
    ]


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py", "store_models"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            if model_file.is_file():
                model_file.unlink()
            elif model_file.is_dir() and model_file.name not in do_not_delete:
                # Remove old store models directory if needed
                import shutil
                shutil.rmtree(model_file)
