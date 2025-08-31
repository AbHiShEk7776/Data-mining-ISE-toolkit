from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
import json
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Import your ML models
from .ml_models.regressor import LogisticRegressionClassifier
from .ml_models.knn import KNNClassifier
from .ml_models.decision_tree import DecisionTreeClassifier
from .ml_models.bayes import NaiveBayesClassifier
from .ml_models.ann import NNClassifier
from .ml_models.rule_based import RuleBasedClassifier
from .ml_models.classification_metrics import get_classification_metrics
from .models import Dataset, PreprocessedDataset

MODEL_MAPPING = {
    'logistic_regression': LogisticRegressionClassifier,
    'knn': KNNClassifier,
    'decision_tree': DecisionTreeClassifier,
    'naive_bayes': NaiveBayesClassifier,
    'neural_network': NNClassifier,
    'rule_based': RuleBasedClassifier
}

trained_models = {}

def validate_and_clean_data(df, target_column, min_samples_per_class=2):
    """Validate and clean dataset by removing classes with insufficient samples"""
    print(f"Original dataset shape: {df.shape}")
    print(f"Original class distribution:\n{df[target_column].value_counts()}")
    
    class_counts = df[target_column].value_counts()
    insufficient_classes = class_counts[class_counts < min_samples_per_class].index.tolist()
    
    if insufficient_classes:
        print(f"Removing classes with < {min_samples_per_class} samples: {insufficient_classes}")
        df_cleaned = df[~df[target_column].isin(insufficient_classes)].copy()
        print(f"Cleaned dataset shape: {df_cleaned.shape}")
        print(f"Cleaned class distribution:\n{df_cleaned[target_column].value_counts()}")
        return df_cleaned, insufficient_classes
    
    return df, []

@csrf_exempt
@require_http_methods(['POST'])
def train_model(request):
    """Train a machine learning model with proper error handling"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        model_type = data.get('model_type')
        features = data.get('features')
        target = data.get('target')
        parameters = data.get('parameters', {})

        # Validation
        if not all([dataset_id, model_type, features, target]):
            return JsonResponse({
                'error': 'Missing required parameters: dataset_id, model_type, features, target'
            }, status=400)

        # Get dataset
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            processed = PreprocessedDataset.objects.filter(original_dataset=dataset).last()
            file_path = processed.processed_file_path if processed else dataset.file_path
        except Dataset.DoesNotExist:
            return JsonResponse({'error': 'Dataset not found'}, status=404)

        # Load data
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return JsonResponse({'error': 'Unsupported file format'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Error loading dataset: {str(e)}'}, status=400)

        # Validate columns exist
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            return JsonResponse({
                'error': f'Missing columns in dataset: {missing_cols}'
            }, status=400)

        # **FIX 1: Clean data to remove classes with insufficient samples**
        df_cleaned, removed_classes = validate_and_clean_data(df, target, min_samples_per_class=2)
        
        if len(df_cleaned) < 4:
            return JsonResponse({
                'error': f'Insufficient data after cleaning. Need at least 4 samples. '
                         f'Got {len(df_cleaned)} samples. Removed classes: {removed_classes}',
                'removed_classes': removed_classes
            }, status=400)

        # **FIX 2: Proper data preparation and label encoding**
        try:
            # Extract features (ensure numeric)
            X = df_cleaned[features].to_numpy(dtype=np.float64)
            y_raw = df_cleaned[target].to_numpy()
            
            print(f"Raw labels sample: {y_raw[:10] if len(y_raw) >= 10 else y_raw}")
            print(f"Raw labels dtype: {y_raw.dtype}")
            
            # **CRITICAL: Encode labels to integers**
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_raw)
            
            # **CRITICAL: Ensure integer dtype**
            y = y_encoded.astype(np.int32)
            
            print(f"Encoded labels sample: {y[:10] if len(y) >= 10 else y}")
            print(f"Encoded labels dtype: {y.dtype}")
            print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
            
        except Exception as e:
            return JsonResponse({'error': f'Error preparing data: {str(e)}'}, status=400)

        # **FIX 3: Smart train/test split**
        try:
            class_counts = Counter(y)
            min_class_count = min(class_counts.values())
            
            if min_class_count >= 2:
                # Use stratified split
                print("Using stratified train/test split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                split_type = "stratified"
            else:
                # Use regular split
                print("Using regular (non-stratified) train/test split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                split_type = "non-stratified"
                
        except Exception as e:
            return JsonResponse({'error': f'Error splitting data: {str(e)}'}, status=400)

        # Initialize model
        ModelClass = MODEL_MAPPING.get(model_type)
        if not ModelClass:
            return JsonResponse({'error': f'Unknown model type: {model_type}'}, status=400)

        try:
            # **FIX 4: Proper model initialization with correct parameters**
            if model_type == 'neural_network':
                layer_sizes = parameters.get('hidden_layers', [64, 32])
                n_classes = len(label_encoder.classes_)
                n_features = X.shape[1]
                
                # Build architecture: [input, hidden layers, output]
                full_layer_sizes = [n_features] + layer_sizes + [n_classes]
                learning_rate = parameters.get('learning_rate', 0.01)
                epochs = parameters.get('epochs', 100)
                
                print(f"Neural Network Architecture: {full_layer_sizes}")
                print(f"Number of classes: {n_classes}")
                print(f"Number of features: {n_features}")
                
                model = NNClassifier(
                    layer_sizes=full_layer_sizes,
                    lr=learning_rate,
                    epochs=epochs
                )
            elif model_type == 'logistic_regression':
                model = LogisticRegressionClassifier(
                    learning_rate=parameters.get('learning_rate', 0.01),
                    n_iterations=parameters.get('n_iterations', 1000)
                )
            elif model_type == 'knn':
                model = KNNClassifier(k=parameters.get('k', 3))
            elif model_type == 'decision_tree':
                model = DecisionTreeClassifier(
                    max_depth=parameters.get('max_depth', 5),
                    min_samples_split=parameters.get('min_samples_split', 2),
                    criterion=parameters.get('criterion', 'gini')
                )
            elif model_type == 'naive_bayes':
                model = NaiveBayesClassifier()
            else:
                model = RuleBasedClassifier()

            # **FIX 5: Training with proper data types**
            start_time = time.time()
            
            print(f"Training {model_type} with:")
            print(f"  X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
            print(f"  y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
            print(f"  y_train sample: {y_train[:10] if len(y_train) >= 10 else y_train}")
            
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Evaluate model
            y_pred = model.predict(X_test)
            
            print(f"Predictions shape: {y_pred.shape}, dtype: {y_pred.dtype}")
            print(f"Predictions sample: {y_pred[:10] if len(y_pred) >= 10 else y_pred}")
            
            # Calculate metrics
            metrics = get_classification_metrics(y_test, y_pred)

            # Generate model ID and store
            model_id = f"{model_type}_{dataset_id}_{int(time.time())}"
            
            trained_models[model_id] = {
                'model': model,
                'label_encoder': label_encoder,
                'features': features,
                'target': target,
                'model_type': model_type,
                'parameters': parameters,
                'training_time': training_time,
                'metrics': metrics,
                'removed_classes': removed_classes
            }

            return JsonResponse({
                'success': True,
                'model_id': model_id,
                'metrics': metrics,
                'training_time': training_time,
                'n_classes': len(label_encoder.classes_),
                'class_labels': label_encoder.classes_.tolist(),
                'split_type': split_type,
                'removed_classes': removed_classes,
                'message': f'{model_type.title()} model trained successfully'
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': f'Error training model: {str(e)}'}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)

@csrf_exempt
@require_http_methods(['POST'])
def make_prediction(request):
    """Make predictions with proper label decoding"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        model_id = data.get('model_id')

        if not all([dataset_id, model_id]):
            return JsonResponse({'error': 'Missing dataset_id or model_id'}, status=400)

        # Get model info
        model_info = trained_models.get(model_id)
        if not model_info:
            return JsonResponse({'error': 'Model not found'}, status=404)

        model = model_info['model']
        label_encoder = model_info['label_encoder']
        features = model_info['features']

        # Get dataset
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            processed = PreprocessedDataset.objects.filter(original_dataset=dataset).last()
            file_path = processed.processed_file_path if processed else dataset.file_path
        except Dataset.DoesNotExist:
            return JsonResponse({'error': 'Dataset not found'}, status=404)

        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)

        # Prepare features
        X = df[features].to_numpy(dtype=np.float64)

        # Make predictions (returns integer indices)
        predictions_encoded = model.predict(X)
        
        # **FIX 6: Decode predictions back to original labels**
        predictions_decoded = label_encoder.inverse_transform(predictions_encoded)
        
        # Format results
        results = []
        for i, (encoded, decoded) in enumerate(zip(predictions_encoded, predictions_decoded)):
            results.append({
                'index': i,
                'prediction': str(decoded),
                'prediction_encoded': int(encoded),
                'confidence': 0.85  # Default confidence
            })

        return JsonResponse({
            'success': True,
            'predictions': results[:100],  # Limit to first 100
            'total_predictions': len(results),
            'class_labels': label_encoder.classes_.tolist()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': f'Prediction error: {str(e)}'}, status=500)

@csrf_exempt
@require_http_methods(['GET'])
def list_models(request):
    """List all trained models"""
    models = []
    for model_id, info in trained_models.items():
        models.append({
            'model_id': model_id,
            'model_type': info['model_type'],
            'features': info['features'],
            'target': info['target'],
            'training_time': info['training_time'],
            'accuracy': info['metrics'].get('accuracy', 0),
            'f1_score': info['metrics'].get('f1_score', 0),
            'class_labels': info['label_encoder'].classes_.tolist()
        })
    
    return JsonResponse({
        'models': models,
        'total_models': len(models)
    })

@csrf_exempt
@require_http_methods(['DELETE'])
def delete_model(request, model_id):
    """Delete a trained model"""
    if model_id in trained_models:
        del trained_models[model_id]
        return JsonResponse({'message': 'Model deleted successfully'})
    else:
        return JsonResponse({'error': 'Model not found'}, status=404)

@csrf_exempt
@require_http_methods(['POST'])
def export_model(request):
    """Export a trained model"""
    try:
        data = json.loads(request.body)
        model_id = data.get('model_id')
        
        model_info = trained_models.get(model_id)
        if not model_info:
            return JsonResponse({'error': 'Model not found'}, status=404)

        # Create export data
        export_data = {
            'model_id': model_id,
            'model_type': model_info['model_type'],
            'features': model_info['features'],
            'target': model_info['target'],
            'parameters': model_info['parameters'],
            'metrics': model_info['metrics'],
            'training_time': model_info['training_time'],
            'class_labels': model_info['label_encoder'].classes_.tolist()
        }

        return JsonResponse(export_data)

    except Exception as e:
        return JsonResponse({'error': f'Export error: {str(e)}'}, status=500)
