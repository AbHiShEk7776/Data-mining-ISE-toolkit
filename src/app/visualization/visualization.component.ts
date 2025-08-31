import { Component, OnInit, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

// Declare Plotly for TypeScript
declare var Plotly: any;

interface Dataset {
  id: string;
  name: string;
  rows: number;
  columns: number;
}

interface ChartType {
  value: string;
  label: string;
}

@Component({
  selector: 'app-visualization',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './visualization.component.html',
  styleUrls: ['./visualization.component.css']
})
export class VisualizationComponent implements OnInit, AfterViewInit {
  datasets: Dataset[] = [];
  selectedDatasetId: string = '';
  selectedDataset: Dataset | null = null;
  chartType = 'histogram';
  selectedColumns: string[] = [];
  availableColumns: string[] = [];
  loading = false;
  error: string | null = null;
  plotlyLoaded = false;
  
  chartTypes: ChartType[] = [
    { value: 'histogram', label: 'Histogram' },
    { value: 'boxplot', label: 'Box Plot' },
    { value: 'scatter', label: 'Scatter Plot' },
    { value: 'bar', label: 'Bar Chart' },
    { value: 'correlation_heatmap', label: 'Correlation Heatmap' }
  ];
  
  constructor(private http: HttpClient) {}
  
  ngOnInit(): void {
    this.loadDatasets();
  }

  ngAfterViewInit(): void {
    this.loadPlotlyScript();
  }

  private loadPlotlyScript(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (typeof Plotly !== 'undefined') {
        this.plotlyLoaded = true;
        resolve();
        return;
      }

      const script = document.createElement('script');
      script.src = 'https://cdn.plot.ly/plotly-latest.min.js';
      script.onload = () => {
        console.log('Plotly loaded successfully');
        this.plotlyLoaded = true;
        resolve();
      };
      script.onerror = () => {
        console.error('Failed to load Plotly');
        this.error = 'Failed to load Plotly library';
        reject();
      };
      document.head.appendChild(script);
    });
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
      this.availableColumns = [];
    } else {
      this.selectedDataset = this.datasets.find(dataset => dataset.id === this.selectedDatasetId) || null;
    }
    
    this.selectedColumns = [];
    if (this.selectedDataset) {
      this.loadColumns();
    }
  }
  
  loadColumns(): void {
    if (!this.selectedDataset) return;
    
    this.loading = true;
    
    this.http.get<{columns: string[]}>(`http://localhost:8000/api/columns/${this.selectedDataset.id}/`).subscribe({
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
  
  generateVisualization(): void {
    if (!this.selectedDataset || this.selectedColumns.length === 0) {
      this.error = 'Please select a dataset and at least one column';
      return;
    }

    if (!this.plotlyLoaded) {
      this.error = 'Plotly library is not loaded yet. Please wait and try again.';
      return;
    }
    
    this.loading = true;
    this.error = null;
    
    const requestData = {
      dataset_id: this.selectedDataset.id,
      chart_type: this.chartType,
      columns: this.selectedColumns
    };
    
    this.http.post<any>('http://localhost:8000/api/visualize/', requestData).subscribe({
      next: (result) => {
        console.log('Visualization result:', result);
        this.loading = false;
        
        if (result.chart_data) {
          this.renderChart(result.chart_data);
        } else {
          // If backend doesn't return proper chart data, create it ourselves
          this.createChartFromData(result);
        }
      },
      error: (error) => {
        console.error('Visualization error:', error);
        this.error = 'Failed to generate visualization: ' + (error.error?.error || error.message);
        this.loading = false;
      }
    });
  }

  createChartFromData(data: any): void {
    if (!this.selectedDataset || this.selectedColumns.length === 0) return;

    // Get sample data from the dataset
    this.http.get<any>(`http://localhost:8000/api/sample-data/${this.selectedDataset.id}/?samples=100`).subscribe({
      next: (response) => {
        const sampleData = response.sample_data || response;
        this.createChart(sampleData);
      },
      error: (error) => {
        console.error('Error getting sample data:', error);
        this.error = 'Failed to get sample data for visualization';
      }
    });
  }

  createChart(data: any[]): void {
    if (!data || data.length === 0) {
      this.error = 'No data available for visualization';
      return;
    }

    const column = this.selectedColumns[0];
    const values = data.map(row => row[column]).filter(val => val != null);

    let chartData: any;
    let layout: any;

    switch (this.chartType) {
      case 'histogram':
        chartData = [{
          x: values,
          type: 'histogram',
          name: column,
          marker: { color: '#2563eb' }
        }];
        layout = {
          title: `Histogram of ${column}`,
          xaxis: { title: column },
          yaxis: { title: 'Frequency' }
        };
        break;

      case 'boxplot':
        chartData = [{
          y: values,
          type: 'box',
          name: column,
          marker: { color: '#10b981' }
        }];
        layout = {
          title: `Box Plot of ${column}`,
          yaxis: { title: column }
        };
        break;

      case 'scatter':
        if (this.selectedColumns.length >= 2) {
          const column2 = this.selectedColumns[28];
          const values2 = data.map(row => row[column2]).filter(val => val != null);
          
          chartData = [{
            x: values,
            y: values2,
            mode: 'markers',
            type: 'scatter',
            name: `${column} vs ${column2}`,
            marker: { color: '#f59e0b' }
          }];
          layout = {
            title: `${column} vs ${column2}`,
            xaxis: { title: column },
            yaxis: { title: column2 }
          };
        } else {
          this.error = 'Scatter plot requires at least 2 columns';
          return;
        }
        break;

      case 'bar':
        // For bar chart, count occurrences
        const counts: Record<string, number> = {};
        values.forEach(val => {
          const key = String(val);
          counts[key] = (counts[key] || 0) + 1;
        });
        
        chartData = [{
          x: Object.keys(counts),
          y: Object.values(counts),
          type: 'bar',
          name: column,
          marker: { color: '#ef4444' }
        }];
        layout = {
          title: `Bar Chart of ${column}`,
          xaxis: { title: column },
          yaxis: { title: 'Count' }
        };
        break;

      case 'correlation_heatmap':
        // Get correlation data
        this.http.get<any>(`http://localhost:8000/api/correlation/${this.selectedDataset!.id}/`).subscribe({
          next: (correlationResult) => {
            if (correlationResult.correlation_matrix) {
              this.renderCorrelationHeatmap(correlationResult.correlation_matrix);
            } else {
              this.error = 'No correlation data available';
            }
          },
          error: (error) => {
            this.error = 'Failed to get correlation data';
          }
        });
        return;

      default:
        this.error = 'Unknown chart type';
        return;
    }

    this.renderChart({ data: chartData, layout: layout });
  }

  renderCorrelationHeatmap(correlationMatrix: Record<string, Record<string, number>>): void {
    const variables = Object.keys(correlationMatrix);
    const z = variables.map(var1 => 
      variables.map(var2 => correlationMatrix[var1][var2])
    );

    const chartData = [{
      z: z,
      x: variables,
      y: variables,
      type: 'heatmap',
      colorscale: 'RdBu',
      zmid: 0
    }];

    const layout = {
      title: 'Correlation Heatmap',
      xaxis: { title: 'Variables' },
      yaxis: { title: 'Variables' }
    };

    this.renderChart({ data: chartData, layout: layout });
  }
  
  renderChart(chartData: any): void {
    const plotDiv = document.getElementById('plotDiv');
    if (!plotDiv) {
      this.error = 'Chart container not found';
      return;
    }

    if (typeof Plotly === 'undefined') {
      this.error = 'Plotly library not loaded';
      return;
    }
    
    const config = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
      displaylogo: false
    };
    
    try {
      // Clear previous plot
      Plotly.purge('plotDiv');
      
      // Create new plot
      Plotly.newPlot('plotDiv', chartData.data, chartData.layout, config);
      
      console.log('Chart rendered successfully');
    } catch (error) {
      console.error('Error rendering chart:', error);
      this.error = 'Failed to render chart: ' + error;
    }
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

  clearSelection(): void {
    this.selectedColumns = [];
  }

  getChartTypeLabel(): string {
    const chartType = this.chartTypes.find(type => type.value === this.chartType);
    return chartType ? chartType.label : '';
  }

  getChartIcon(chartType: string): string {
    const icons: {[key: string]: string} = {
      'histogram': 'fas fa-chart-bar',
      'boxplot': 'fas fa-square',
      'scatter': 'fas fa-braille',
      'bar': 'fas fa-chart-column',
      'correlation_heatmap': 'fas fa-th'
    };
    return icons[chartType] || 'fas fa-chart-line';
  }

  getChartDescription(chartType: string): string {
    const descriptions: {[key: string]: string} = {
      'histogram': 'Show data distribution',
      'boxplot': 'Display statistical summary',
      'scatter': 'Explore relationships',
      'bar': 'Compare categories',
      'correlation_heatmap': 'Show correlations'
    };
    return descriptions[chartType] || '';
  }

  trackByColumn(index: number, column: string): string {
    return column;
  }

  trackByDataset(index: number, dataset: Dataset): string {
    return dataset.id;
  }

  trackByChartType(index: number, chartType: ChartType): string {
    return chartType.value;
  }
}
