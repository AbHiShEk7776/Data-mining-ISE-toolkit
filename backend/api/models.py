from django.db import models
from django.contrib.auth.models import User
import uuid

class Dataset(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    rows = models.IntegerField(default=0)
    columns = models.IntegerField(default=0)
    file_size = models.BigIntegerField(default=0)
    
    def __str__(self):
        return self.name

class PreprocessedDataset(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    processed_file_path = models.CharField(max_length=500)
    preprocessing_steps = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Processed_{self.original_dataset.name}"

class StatisticalAnalysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    analysis_type = models.CharField(max_length=100)
    results = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.analysis_type}_{self.dataset.name}"
