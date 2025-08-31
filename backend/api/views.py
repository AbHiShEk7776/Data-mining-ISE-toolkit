from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import json
import os
import uuid
from .models import Dataset, PreprocessedDataset, StatisticalAnalysis
from .preprocessing import DataPreprocessor
from .visualization import DataVisualizer

@csrf_exempt
@require_http_methods(["POST"])
def upload_dataset(request):
    """Handle dataset upload"""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file provided'}, status=400)
        
        uploaded_file = request.FILES['file']
        
        # Validate file type
        allowed_extensions = ['.csv', '.xlsx', '.xls']
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension not in allowed_extensions:
            return JsonResponse({'error': 'Invalid file type. Only CSV and Excel files are allowed.'}, status=400)
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(settings.MEDIA_ROOT, 'datasets', unique_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save file
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        # Process file and get statistics
        preprocessor = DataPreprocessor(file_path)
        stats = preprocessor.get_basic_statistics()
        
        if not stats:
            if os.path.exists(file_path):
                os.remove(file_path)
            return JsonResponse({'error': 'Could not process file'}, status=400)
        
        # Create database record
        dataset = Dataset.objects.create(
            name=uploaded_file.name,
            file_path=file_path,
            rows=stats.get('total_rows', 0),
            columns=stats.get('total_columns', 0),
            file_size=uploaded_file.size
        )
        
        return JsonResponse({
            'success': True,
            'dataset_id': str(dataset.id),
            'message': 'Dataset uploaded successfully',
            'statistics': stats  # Include initial statistics
        })
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def list_datasets(request):
    """List all datasets"""
    datasets = Dataset.objects.all().values(
        'id', 'name', 'rows', 'columns', 'uploaded_at'
    )
    return JsonResponse(list(datasets), safe=False)

@require_http_methods(["GET"])
def get_statistics(request, dataset_id):
    """Get statistical analysis for a dataset"""
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Check if there's a processed version
        processed = PreprocessedDataset.objects.filter(original_dataset=dataset).last()
        file_path = processed.processed_file_path if processed else dataset.file_path
        
        preprocessor = DataPreprocessor(file_path)
        statistics = preprocessor.get_basic_statistics()
        
        return JsonResponse({
            'statistics': statistics,
            'is_processed': bool(processed),
            'processing_steps': processed.preprocessing_steps if processed else {}
        })
        
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def preprocess_data(request):
    """Apply preprocessing to dataset and return updated statistics"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        options = data.get('options', {})
        
        dataset = Dataset.objects.get(id=dataset_id)
        preprocessor = DataPreprocessor(dataset.file_path)
        
        # Apply preprocessing steps
        results = {}
        
        if options.get('removeDuplicates'):
            result = preprocessor.remove_duplicates()
            results['duplicates'] = result
        
        if options.get('handleMissing'):
            result = preprocessor.handle_missing_values(method=options['handleMissing'])
            results['missing_values'] = result
        
        if options.get('normalizationMethod'):
            result = preprocessor.normalize_data(method=options['normalizationMethod'])
            results['normalization'] = result
        
        # Get updated statistics after preprocessing
        updated_statistics = preprocessor.get_basic_statistics()
        
        # Save processed dataset
        processed_filename = f"processed_{uuid.uuid4()}.csv"
        processed_path = os.path.join(settings.MEDIA_ROOT, 'processed', processed_filename)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        
        if preprocessor.save_processed_data(processed_path):
            # Remove old processed versions
            PreprocessedDataset.objects.filter(original_dataset=dataset).delete()
            
            # Create new processed dataset record
            PreprocessedDataset.objects.create(
                original_dataset=dataset,
                processed_file_path=processed_path,
                preprocessing_steps=options
            )
        
        return JsonResponse({
            'success': True,
            'results': results,
            'statistics': updated_statistics,  # Return updated statistics
            'message': 'Preprocessing completed successfully'
        })
        
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def revert_data(request, dataset_id):
    """Revert dataset to original state"""
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Remove all processed versions
        processed_datasets = PreprocessedDataset.objects.filter(original_dataset=dataset)
        for processed in processed_datasets:
            # Remove processed files
            if os.path.exists(processed.processed_file_path):
                os.remove(processed.processed_file_path)
        processed_datasets.delete()
        
        # Get original statistics
        preprocessor = DataPreprocessor(dataset.file_path)
        original_statistics = preprocessor.get_basic_statistics()
        
        return JsonResponse({
            'success': True,
            'message': 'Data reverted to original state',
            'statistics': original_statistics,
            'is_processed': False
        })
        
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_correlation(request, dataset_id):
    """Calculate correlation matrix"""
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Use processed version if available
        processed = PreprocessedDataset.objects.filter(original_dataset=dataset).last()
        file_path = processed.processed_file_path if processed else dataset.file_path
        
        preprocessor = DataPreprocessor(file_path)
        correlation = preprocessor.calculate_correlation()
        
        if correlation is None:
            return JsonResponse({'error': 'Insufficient numerical data for correlation analysis'}, status=400)
        
        return JsonResponse({
            'correlation_matrix': correlation,
            'success': True,
            'message': 'Correlation analysis completed successfully'
        })
        
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def chi_square_test(request, dataset_id):
    """Perform chi-square test"""
    try:
        data = json.loads(request.body)
        col1 = data.get('column1')
        col2 = data.get('column2')
        
        if not col1 or not col2:
            return JsonResponse({'error': 'Both column1 and column2 are required'}, status=400)
        
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Use processed version if available
        processed = PreprocessedDataset.objects.filter(original_dataset=dataset).last()
        file_path = processed.processed_file_path if processed else dataset.file_path
        
        preprocessor = DataPreprocessor(file_path)
        result = preprocessor.chi_square_test(col1, col2)
        
        if 'error' in result:
            return JsonResponse(result, status=400)
        
        result['success'] = True
        return JsonResponse(result)
        
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_columns(request, dataset_id):
    """Get column names for a dataset"""
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Use processed version if available
        processed = PreprocessedDataset.objects.filter(original_dataset=dataset).last()
        file_path = processed.processed_file_path if processed else dataset.file_path
        
        preprocessor = DataPreprocessor(file_path)
        
        return JsonResponse({
            'columns': list(preprocessor.df.columns),
            'is_processed': bool(processed)
        })
        
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def create_visualization(request):
    """Create data visualization"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        chart_type = data.get('chart_type')
        columns = data.get('columns', [])
        
        if not dataset_id or not chart_type or not columns:
            return JsonResponse({'error': 'Missing required parameters'}, status=400)
        
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Use processed version if available
        processed = PreprocessedDataset.objects.filter(original_dataset=dataset).last()
        file_path = processed.processed_file_path if processed else dataset.file_path
        
        visualizer = DataVisualizer(file_path)
        chart_data = visualizer.generate_chart(chart_type, columns)
        
        if chart_data:
            return JsonResponse({
                'success': True,
                'chart_data': chart_data,
                'message': 'Visualization generated successfully'
            })
        else:
            return JsonResponse({'error': 'Could not generate chart'}, status=400)
        
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# Add this new endpoint for sample data
@require_http_methods(["GET"])
def get_sample_data(request, dataset_id):
    """Get sample data for visualization"""
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Use processed version if available
        processed = PreprocessedDataset.objects.filter(original_dataset=dataset).last()
        file_path = processed.processed_file_path if processed else dataset.file_path
        
        visualizer = DataVisualizer(file_path)
        n_samples = int(request.GET.get('samples', 100))
        sample_data = visualizer.get_sample_data(n_samples)
        
        if sample_data:
            return JsonResponse({
                'success': True,
                'sample_data': sample_data,
                'total_rows': len(visualizer.df) if visualizer.df is not None else 0
            })
        else:
            return JsonResponse({'error': 'Could not get sample data'}, status=400)
        

        
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
