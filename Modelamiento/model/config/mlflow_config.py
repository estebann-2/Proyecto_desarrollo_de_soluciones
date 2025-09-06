"""
MLflow configuration for LifeMiles forecasting project.
"""

import mlflow
import os
from pathlib import Path


def setup_mlflow_tracking():
    """
    Setup MLflow tracking configuration.
    """
    # Configurar el directorio donde se guardan los experimentos
    current_dir = Path(__file__).parent.parent.parent  # Retrocede a Modelamiento/
    mlruns_dir = current_dir / "mlruns"
    
    # Crear directorio si no existe
    mlruns_dir.mkdir(exist_ok=True)
    
    # Configurar MLflow tracking URI
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    
    print(f"MLflow tracking configured:")
    print(f"  - Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"  - Artifacts location: {mlruns_dir}")
    
    return str(mlruns_dir)


def create_experiment_if_not_exists(experiment_name: str):
    """
    Create MLflow experiment if it doesn't exist.
    
    Args:
        experiment_name (str): Name of the experiment
    
    Returns:
        str: Experiment ID
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
        
        return experiment_id
    except Exception as e:
        print(f"Error setting up MLflow experiment: {e}")
        raise


def get_run_name(prefix: str = "LifeMiles_Training") -> str:
    """
    Generate a unique run name with timestamp.
    
    Args:
        prefix (str): Prefix for the run name
    
    Returns:
        str: Generated run name
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


# Configuración por defecto de MLflow
MLFLOW_CONFIG = {
    "experiment_name": "LifeMiles_Forecasting_Training",
    "artifact_location": None,  # Se configura automáticamente
    "tags": {
        "project": "LifeMiles_Forecasting",
        "version": "1.0",
        "model_type": "Random_Forest",
        "framework": "scikit-learn"
    }
}


def initialize_mlflow_for_training():
    """
    Initialize MLflow for training pipeline.
    This function should be called at the beginning of training.
    """
    # Setup tracking
    mlruns_path = setup_mlflow_tracking()
    
    # Create or get experiment
    experiment_id = create_experiment_if_not_exists(MLFLOW_CONFIG["experiment_name"])
    
    # Set experiment
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
    
    return {
        "mlruns_path": mlruns_path,
        "experiment_id": experiment_id,
        "experiment_name": MLFLOW_CONFIG["experiment_name"]
    }


if __name__ == "__main__":
    # Test the configuration
    print("Testing MLflow configuration...")
    config_info = initialize_mlflow_for_training()
    print("MLflow configuration test completed successfully!")
    print(f"Configuration: {config_info}")
