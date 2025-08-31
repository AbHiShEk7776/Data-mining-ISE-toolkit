import { Component, OnInit, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';

// Declare Plotly for TypeScript
declare var Plotly: any;

interface Dataset {
  id: string;
  name: string;
  rows: number;
  columns: number;
  uploaded_at: string;
}

interface PreprocessingOptions {
  handleMissing: string;
  normalizationMethod: string;
  discretizationBins: number;
  removeDuplicates: boolean;
}

interface NumericColumn {
  name: string;
  mean: number;
  median: number;
  mode: number;
  std: number;
  variance: number;
  range: number;
  min: number;
  max: number;
}

interface CategoricalColumn {
  name: string;
  unique_values: number;
  most_frequent: string;
  value_counts: Record<string, number>;
}

// Union type for columns
type Column = NumericColumn | CategoricalColumn;

interface Statistics {
  total_rows: number;
  total_columns: number;
  missing_values: Record<string, number>;
  data_types: Record<string, string>;
  numerical_columns: NumericColumn[];
  categorical_columns: CategoricalColumn[];
}

interface ChiSquareResult {
  chi2_statistic: number;
  p_value: number;
  degrees_of_freedom: number;
  contingency_table: Record<string, Record<string, number>>;
}

interface CorrelationResult {
  correlation_matrix: Record<string, Record<string, number>>;
}

@Component({
  selector: 'app-preprocessing',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './preprocessing.component.html',
  styleUrls: ['./preprocessing.component.css']
})
export class PreprocessingComponent implements OnInit, AfterViewInit {
  datasets: Dataset[] = [];
  selectedDatasetId: string = '';
  selectedDataset: Dataset | null = null;
  statistics: Statistics | null = null;
  originalStatistics: Statistics | null = null;
  
  // Combined columns for iteration
  allColumns: Column[] = [];
  
  preprocessingOptions: PreprocessingOptions = {
    handleMissing: 'drop',
    normalizationMethod: 'minmax',
    discretizationBins: 5,
    removeDuplicates: true
  };
  
  processing = false;
  loading = false;
  error: string | null = null;
  successMessage: string | null = null;
  isProcessed = false;
  
  // Analysis results
  chiSquareResult: ChiSquareResult | null = null;
  correlationResult: CorrelationResult | null = null;
  selectedColumns: string[] = [];
  
  constructor(private http: HttpClient, private router: Router) {}
  
  ngOnInit(): void {
    this.loadDatasets();
  }

  ngAfterViewInit(): void {
    this.loadPlotlyScript();
  }

  private loadPlotlyScript(): void {
    if (typeof Plotly !== 'undefined') {
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://cdn.plot.ly/plotly-latest.min.js';
    script.onload = () => {
      console.log('Plotly loaded successfully');
    };
    document.head.appendChild(script);
  }
  
  loadDatasets(): void {
    this.loading = true;
    this.error = null;
    
    this.http.get<Dataset[]>('http://localhost:8000/api/datasets/').subscribe({
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
      this.statistics = null;
      this.originalStatistics = null;
      this.allColumns = [];
      this.isProcessed = false;
    } else {
      this.selectedDataset = this.datasets.find(dataset => dataset.id === this.selectedDatasetId) || null;
    }
    
    if (this.selectedDataset) {
      this.loadStatistics();
    }
  }
  
  loadStatistics(): void {
    if (!this.selectedDataset) return;
    
    this.loading = true;
    this.error = null;
    
    this.http.get<{statistics: Statistics, is_processed: boolean}>(`http://localhost:8000/api/statistics/${this.selectedDataset.id}/`).subscribe({
      next: (response) => {
        this.statistics = response.statistics;
        this.isProcessed = response.is_processed;
        
        // Store original statistics for revert functionality
        if (!this.originalStatistics || !this.isProcessed) {
          this.originalStatistics = JSON.parse(JSON.stringify(response.statistics));
        }
        
        // Create combined columns array
        this.updateAllColumns();
        
        this.loading = false;
        setTimeout(() => this.createStatisticalCharts(), 500);
      },
      error: (error) => {
        console.error('Error loading statistics:', error);
        this.error = 'Failed to load statistics';
        this.loading = false;
      }
    });
  }

  private updateAllColumns(): void {
    if (this.statistics) {
      this.allColumns = [
        ...this.statistics.numerical_columns,
        ...this.statistics.categorical_columns
      ];
    } else {
      this.allColumns = [];
    }
  }
  
  preprocessData(): void {
    if (!this.selectedDataset) return;
    
    this.processing = true;
    this.error = null;
    this.successMessage = null;
    
    const requestData = {
      dataset_id: this.selectedDataset.id,
      options: this.preprocessingOptions
    };
    
    this.http.post<{success: boolean, statistics: Statistics, results: any, message: string}>('http://localhost:8000/api/preprocess/', requestData).subscribe({
      next: (result) => {
        console.log('Preprocessing completed:', result);
        this.processing = false;
        this.isProcessed = true;
        this.successMessage = result.message || 'Data preprocessing completed successfully!';
        
        // Update statistics with returned data
        if (result.statistics) {
          this.statistics = result.statistics;
          this.updateAllColumns();
          setTimeout(() => this.createStatisticalCharts(), 500);
        }
      },
      error: (error) => {
        console.error('Preprocessing error:', error);
        this.error = 'Preprocessing failed: ' + (error.error?.error || error.message);
        this.processing = false;
      }
    });
  }

  revertData(): void {
    if (!this.selectedDataset) return;
    
    this.processing = true;
    this.error = null;
    this.successMessage = null;
    
    this.http.post<{success: boolean, statistics: Statistics, message: string}>(`http://localhost:8000/api/revert/${this.selectedDataset.id}/`, {}).subscribe({
      next: (result) => {
        console.log('Data reverted:', result);
        this.processing = false;
        this.isProcessed = false;
        this.statistics = result.statistics;
        this.updateAllColumns();
        this.successMessage = result.message || 'Data reverted to original state!';
        setTimeout(() => this.createStatisticalCharts(), 500);
      },
      error: (error) => {
        console.error('Revert error:', error);
        this.error = 'Failed to revert data: ' + (error.error?.error || error.message);
        this.processing = false;
      }
    });
  }
  
  performChiSquareTest(): void {
    if (!this.selectedDataset || this.selectedColumns.length !== 2) {
      this.error = 'Please select exactly 2 columns for Chi-Square test';
      return;
    }
    
    const requestData = {
      column1: this.selectedColumns[0],
      column2: this.selectedColumns[1]
    };
    
    this.http.post<ChiSquareResult>(`http://localhost:8000/api/chi-square/${this.selectedDataset.id}/`, requestData).subscribe({
      next: (result) => {
        console.log('Chi-square test result:', result);
        this.chiSquareResult = result;
        this.error = null;
        setTimeout(() => this.createChiSquareChart(), 100);
      },
      error: (error) => {
        console.error('Chi-square test error:', error);
        this.error = 'Chi-square test failed: ' + (error.error?.error || error.message);
      }
    });
  }
  
  calculateCorrelation(): void {
    if (!this.selectedDataset) return;
    
    this.http.get<CorrelationResult>(`http://localhost:8000/api/correlation/${this.selectedDataset.id}/`).subscribe({
      next: (result) => {
        console.log('Correlation result:', result);
        this.correlationResult = result;
        this.error = null;
        setTimeout(() => this.createCorrelationHeatmap(), 100);
      },
      error: (error) => {
        console.error('Correlation calculation error:', error);
        this.error = 'Correlation calculation failed: ' + (error.error?.error || error.message);
      }
    });
  }

  onColumnSelectionChange(column: string, event: Event): void {
    const target = event.target as HTMLInputElement;
    
    if (target.checked) {
      if (!this.selectedColumns.includes(column)) {
        this.selectedColumns.push(column);
      }
    } else {
      this.selectedColumns = this.selectedColumns.filter(c => c !== column);
    }
  }

  isColumnSelected(column: string): boolean {
    return this.selectedColumns.includes(column);
  }

  createStatisticalCharts(): void {
    if (!this.statistics || typeof Plotly === 'undefined') return;

    setTimeout(() => {
      this.createNumericalHistograms();
      this.createCategoricalCharts();
    }, 500);
  }

  createNumericalHistograms(): void {
    if (!this.statistics?.numerical_columns) return;

    this.statistics.numerical_columns.forEach((column, index) => {
      const chartId = `numerical-chart-${index}`;
      const chartDiv = document.getElementById(chartId);
      
      if (chartDiv) {
        const data = [{
          x: ['Mean', 'Median', 'Mode'],
          y: [column.mean, column.median, column.mode],
          type: 'bar',
          name: column.name,
          marker: { color: ['#2563eb', '#10b981', '#f59e0b'] }
        }];

        const layout = {
          title: `${column.name} - Central Tendency`,
          xaxis: { title: 'Measures' },
          yaxis: { title: 'Values' },
          height: 300
        };

        Plotly.newPlot(chartId, data, layout, { responsive: true });
      }
    });
  }

  createCategoricalCharts(): void {
    if (!this.statistics?.categorical_columns) return;

    this.statistics.categorical_columns.forEach((column, index) => {
      const chartId = `categorical-chart-${index}`;
      const chartDiv = document.getElementById(chartId);
      
      if (chartDiv) {
        const labels = Object.keys(column.value_counts);
        const values = Object.values(column.value_counts);

        const data = [{
          labels: labels,
          values: values,
          type: 'pie',
          hole: 0.3
        }];

        const layout = {
          title: `${column.name} - Distribution`,
          height: 300
        };

        Plotly.newPlot(chartId, data, layout, { responsive: true });
      }
    });
  }

  createChiSquareChart(): void {
    // Safe null checks
    if (!this.chiSquareResult || typeof Plotly === 'undefined') return;

    setTimeout(() => {
      const chartDiv = document.getElementById('chi-square-chart');
      if (chartDiv && this.chiSquareResult) {
        const table = this.chiSquareResult.contingency_table;
        const xLabels = Object.keys(table);
        const yLabels = Object.keys(table[xLabels[0]] || {});
        
        const z = yLabels.map(yLabel => 
          xLabels.map(xLabel => table[xLabel][yLabel] || 0)
        );

        const data = [{
          z: z,
          x: xLabels,
          y: yLabels,
          type: 'heatmap',
          colorscale: 'Blues'
        }];

        // Safe property access
        const chi2Value = this.chiSquareResult.chi2_statistic?.toFixed(4) || 'N/A';
        const pValue = this.chiSquareResult.p_value?.toFixed(6) || 'N/A';

        const layout = {
          title: `Chi-Square Test - Contingency Table<br>χ² = ${chi2Value}, p-value = ${pValue}`,
          height: 400
        };

        Plotly.newPlot('chi-square-chart', data, layout, { responsive: true });
      }
    }, 100);
  }

  createCorrelationHeatmap(): void {
    // Safe null checks
    if (!this.correlationResult || typeof Plotly === 'undefined') return;

    setTimeout(() => {
      const chartDiv = document.getElementById('correlation-chart');
      if (chartDiv && this.correlationResult) {
        const matrix = this.correlationResult.correlation_matrix;
        const variables = Object.keys(matrix);
        
        const z = variables.map(var1 => 
          variables.map(var2 => matrix[var1][var2])
        );

        const data = [{
          z: z,
          x: variables,
          y: variables,
          type: 'heatmap',
          colorscale: 'RdBu',
          zmid: 0
        }];

        const layout = {
          title: 'Correlation Matrix',
          height: 500
        };

        Plotly.newPlot('correlation-chart', data, layout, { responsive: true });
      }
    }, 100);
  }

  navigateToVisualization(): void {
    this.router.navigate(['/visualization']);
  }

  trackByDataset(index: number, dataset: Dataset): string {
    return dataset.id;
  }

  trackByColumn(index: number, column: Column): string {
    return column.name;
  }
}
