import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

// Declare Plotly for confusion matrix visualization
declare var Plotly: any;

interface Dataset {
  id: string;
  name: string;
  rows: number;
  columns: number;
}

interface MLModel {
  name: string;
  type: string;
  parameters: any;
  trained: boolean;
  metrics?: any;
}

interface TrainingResult {
  success: boolean;
  model_id: string;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    confusion_matrix: number[][];
  };
  training_time: number;
}

@Component({
  selector: 'app-ml-models',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './ml-models.component.html',
  styleUrls: ['./ml-models.component.css']
})
export class MlModelsComponent implements OnInit {
  // Data and Models
  datasets: Dataset[] = [];
  selectedDatasetId: string = '';
  selectedDataset: Dataset | null = null;
  availableColumns: string[] = [];
  selectedFeatures: string[] = [];
  targetColumn: string = '';
  
  // Model Configuration
  selectedModelType: string = 'logistic_regression';
  trainedModels: MLModel[] = [];
  currentModel: MLModel | null = null;
  
  // UI States
  loading = false;
  training = false;
  error: string | null = null;
  successMessage: string | null = null;
  
  // Results
  trainingResults: TrainingResult | null = null;
  predictions: any[] = [];
  
  // API URL
  private apiUrl = 'http://localhost:8000/api';
  
  // Available Models
  modelTypes = [
    {
      value: 'logistic_regression',
      label: 'Logistic Regression',
      description: 'Linear classifier for binary/multiclass problems',
      icon: 'fas fa-chart-line'
    },
    {
      value: 'knn',
      label: 'K-Nearest Neighbors',
      description: 'Instance-based learning algorithm',
      icon: 'fas fa-users'
    },
    {
      value: 'decision_tree',
      label: 'Decision Tree',
      description: 'Tree-based classifier with interpretable rules',
      icon: 'fas fa-sitemap'
    },
    {
      value: 'naive_bayes',
      label: 'Naive Bayes',
      description: 'Probabilistic classifier based on Bayes theorem',
      icon: 'fas fa-brain'
    },
    {
      value: 'neural_network',
      label: 'Neural Network',
      description: 'Multi-layer perceptron for complex patterns',
      icon: 'fas fa-project-diagram'
    }
  ];
  
  // Model Parameters
  modelParameters = {
    logistic_regression: {
      learning_rate: 0.01,
      n_iterations: 1000
    },
    knn: {
      k: 3
    },
    decision_tree: {
      max_depth: 5,
      min_samples_split: 2,
      criterion: 'gini'
    },
    naive_bayes: {},
    neural_network: {
      hidden_layers: [64, 32],
      learning_rate: 0.01,
      epochs: 100,
      activation: 'relu'
    }
  };

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.loadDatasets();
    this.loadPlotlyScript();
  }

  private loadPlotlyScript(): void {
    if (typeof Plotly !== 'undefined') return;
    
    const script = document.createElement('script');
    script.src = 'https://cdn.plot.ly/plotly-latest.min.js';
    script.onload = () => console.log('Plotly loaded for ML visualization');
    document.head.appendChild(script);
  }

  loadDatasets(): void {
    this.loading = true;
    this.error = null;
    
    this.http.get<Dataset[]>(`${this.apiUrl}/datasets/`).subscribe({
      next: (data) => {
        this.datasets = data;
        this.loading = false;
      },
      error: (error) => {
        console.error('Error loading datasets:', error);
        this.error = 'Failed to load datasets';
        this.loading = false;
      }
    });
  }

  onDatasetChange(): void {
    if (this.selectedDatasetId === '') {
      this.selectedDataset = null;
      this.availableColumns = [];
    } else {
      this.selectedDataset = this.datasets.find(d => d.id === this.selectedDatasetId) || null;
    }
    
    this.selectedFeatures = [];
    this.targetColumn = '';
    this.error = null;
    
    if (this.selectedDataset) {
      this.loadColumns();
    }
  }

  loadColumns(): void {
    if (!this.selectedDataset) return;
    
    this.loading = true;
    
    this.http.get<{columns: string[]}>(`${this.apiUrl}/columns/${this.selectedDataset.id}/`).subscribe({
      next: (data) => {
        this.availableColumns = data.columns;
        this.loading = false;
      },
      error: (error) => {
        console.error('Error loading columns:', error);
        this.error = 'Failed to load columns';
        this.loading = false;
      }
    });
  }

  onFeatureSelectionChange(column: string, event: Event): void {
    const target = event.target as HTMLInputElement;
    
    if (target.checked) {
      if (!this.selectedFeatures.includes(column)) {
        this.selectedFeatures.push(column);
      }
    } else {
      this.selectedFeatures = this.selectedFeatures.filter(f => f !== column);
    }
  }

  isFeatureSelected(column: string): boolean {
    return this.selectedFeatures.includes(column);
  }

  trainModel(): void {
    if (!this.selectedDataset || this.selectedFeatures.length === 0 || !this.targetColumn) {
      this.error = 'Please select dataset, features, and target column';
      return;
    }

    this.training = true;
    this.error = null;
    this.successMessage = null;

    const trainingData = {
      dataset_id: this.selectedDataset.id,
      model_type: this.selectedModelType,
      features: this.selectedFeatures,
      target: this.targetColumn,
      parameters: this.modelParameters[this.selectedModelType as keyof typeof this.modelParameters]
    };

    console.log('Training model with data:', trainingData);

    this.http.post<TrainingResult>(`${this.apiUrl}/ml/train/`, trainingData).subscribe({
      next: (result) => {
        console.log('Training completed:', result);
        this.trainingResults = result;
        this.training = false;
        this.successMessage = 'Model trained successfully!';
        
        // Add trained model to list
        const trainedModel: MLModel = {
          name: `${this.getModelLabel()} - ${new Date().toLocaleString()}`,
          type: this.selectedModelType,
          parameters: { ...this.modelParameters[this.selectedModelType as keyof typeof this.modelParameters] },
          trained: true,
          metrics: result.metrics
        };
        
        this.trainedModels.push(trainedModel);
        this.currentModel = trainedModel;
        
        // Render confusion matrix
        setTimeout(() => this.renderConfusionMatrix(), 500);
      },
      error: (error) => {
        console.error('Training error:', error);
        this.error = 'Failed to train model: ' + (error.error?.error || error.message);
        this.training = false;
      }
    });
  }

  makePredict(): void {
    if (!this.currentModel || !this.selectedDataset) {
      this.error = 'Please train a model first';
      return;
    }

    this.loading = true;
    this.error = null;

    const predictionData = {
      dataset_id: this.selectedDataset.id,
      features: this.selectedFeatures
    };

    this.http.post<any>(`${this.apiUrl}/ml/predict/`, predictionData).subscribe({
      next: (result) => {
        console.log('Predictions:', result);
        this.predictions = result.predictions;
        this.loading = false;
      },
      error: (error) => {
        console.error('Prediction error:', error);
        this.error = 'Failed to make predictions';
        this.loading = false;
      }
    });
  }

  renderConfusionMatrix(): void {
    if (!this.trainingResults?.metrics?.confusion_matrix || typeof Plotly === 'undefined') {
      return;
    }

    const matrix = this.trainingResults.metrics.confusion_matrix;
    const size = matrix.length;
    
    // Create labels
    const labels = Array.from({length: size}, (_, i) => `Class ${i}`);

    const data = [{
      z: matrix,
      x: labels,
      y: labels,
      type: 'heatmap',
      colorscale: 'Blues',
      showscale: true,
      text: matrix.map(row => row.map(val => val.toString())),
      texttemplate: '%{text}',
      textfont: { size: 12 }
    }];

    const layout = {
      title: 'Confusion Matrix',
      xaxis: { title: 'Predicted Class' },
      yaxis: { title: 'True Class' },
      width: 500,
      height: 400
    };

    const plotDiv = document.getElementById('confusionMatrix');
    if (plotDiv) {
      Plotly.newPlot('confusionMatrix', data, layout, { responsive: true });
    }
  }

  getModelLabel(): string {
    const model = this.modelTypes.find(m => m.value === this.selectedModelType);
    return model ? model.label : this.selectedModelType;
  }

  getModelIcon(modelType: string): string {
    const model = this.modelTypes.find(m => m.value === modelType);
    return model ? model.icon : 'fas fa-cog';
  }

  clearResults(): void {
    this.trainingResults = null;
    this.predictions = [];
    this.currentModel = null;
    this.error = null;
    this.successMessage = null;
  }

  removeModel(index: number): void {
    this.trainedModels.splice(index, 1);
    if (this.currentModel === this.trainedModels[index]) {
      this.currentModel = null;
    }
  }

  exportModel(): void {
    if (!this.currentModel) {
      this.error = 'No model to export';
      return;
    }

    const modelData = {
      model: this.currentModel,
      results: this.trainingResults
    };

    const blob = new Blob([JSON.stringify(modelData, null, 2)], {
      type: 'application/json'
    });

    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${this.currentModel.type}_model.json`;
    link.click();
    window.URL.revokeObjectURL(url);
  }

  getParameterKeys(): string[] {
    return Object.keys(this.modelParameters[this.selectedModelType as keyof typeof this.modelParameters] || {});
  }
}
