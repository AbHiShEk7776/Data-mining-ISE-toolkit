from django.urls import path
from . import views
from . import ml_views

urlpatterns = [
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('datasets/', views.list_datasets, name='list_datasets'),
    path('statistics/<uuid:dataset_id>/', views.get_statistics, name='get_statistics'),
    path('preprocess/', views.preprocess_data, name='preprocess_data'),
    path('revert/<uuid:dataset_id>/', views.revert_data, name='revert_data'),  # New endpoint
    path('correlation/<uuid:dataset_id>/', views.get_correlation, name='get_correlation'),
    path('chi-square/<uuid:dataset_id>/', views.chi_square_test, name='chi_square_test'),
    path('columns/<uuid:dataset_id>/', views.get_columns, name='get_columns'),
    path('visualize/', views.create_visualization, name='create_visualization'),
    path('ml/train/', ml_views.train_model, name='ml_train'),
    path('ml/predict/', ml_views.make_prediction, name='ml_predict'),
    path('ml/models/', ml_views.list_models, name='ml_list_models'),
    # path('ml/models/<str:model_id>/', ml_views.get_model_info, name='ml_model_info'),
    path('ml/models/<str:model_id>/delete/', ml_views.delete_model, name='ml_delete_model'),
    path('ml/export/', ml_views.export_model, name='ml_export_model'),
]
