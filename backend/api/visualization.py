import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np
from decimal import Decimal

class DataVisualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.load_data()

    def load_data(self):
        """Load and clean data from file"""
        try:
            if self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            elif self.file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(self.file_path)
            
            if self.df is not None:
                # Critical fix: Proper data cleaning
                self.df = self.df.replace([np.inf, -np.inf], np.nan)
                
                # Convert object columns that are actually numeric
                for col in self.df.columns:
                    if self.df[col].dtype == 'object':
                        # Try converting to numeric
                        numeric_series = pd.to_numeric(self.df[col], errors='coerce')
                        if numeric_series.notna().sum() > 0:  # If some values converted
                            self.df[col] = numeric_series
                
                print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
                
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def _clean_numeric_data(self, column):
        """Helper method to clean numeric data properly"""
        if column not in self.df.columns:
            return None
        
        # Convert to numeric, replacing errors with NaN
        numeric_data = pd.to_numeric(self.df[column], errors='coerce')
        
        # Remove NaN, inf, -inf
        clean_data = numeric_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Convert to standard Python types (critical for JSON serialization)
        clean_data = clean_data.astype(float)
        
        print(f"Column {column}: {len(self.df[column])} -> {len(clean_data)} clean values")
        
        return clean_data if len(clean_data) > 0 else None

    def generate_histogram(self, column):
        """Generate histogram with proper data handling"""
        print(f"=== HISTOGRAM: {column} ===")
        
        try:
            clean_data = self._clean_numeric_data(column)
            
            if clean_data is None or len(clean_data) < 2:
                print(f"Insufficient numeric data for histogram: {column}")
                return None
            
            # Critical fix: Convert to list of Python floats
            data_list = clean_data.tolist()
            
            # Create figure with explicit data
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data_list,
                name=column,
                nbinsx=min(30, max(5, len(set(data_list))))
            ))
            
            # Critical fix: Proper layout configuration
            fig.update_layout(
                title=f"Histogram of {column}",
                xaxis=dict(title=column),
                yaxis=dict(title="Frequency"),
                width=800,
                height=400,
                margin=dict(l=50, r=50, t=80, b=50),
                template="plotly_white"
            )
            
            # Serialize properly
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Histogram error for {column}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def generate_boxplot(self, columns):
        """Generate boxplot with proper data handling"""
        print(f"=== BOXPLOT: {columns} ===")
        
        try:
            fig = go.Figure()
            valid_columns = 0
            
            for col in columns:
                clean_data = self._clean_numeric_data(col)
                
                if clean_data is not None and len(clean_data) >= 2:
                    # Critical fix: Convert to list
                    data_list = clean_data.tolist()
                    
                    fig.add_trace(go.Box(
                        y=data_list,
                        name=col,
                        boxpoints='outliers'
                    ))
                    valid_columns += 1
            
            if valid_columns == 0:
                print("No valid columns for boxplot")
                return None
            
            # Critical fix: Proper layout
            fig.update_layout(
                title="Box Plot Comparison",
                yaxis=dict(title="Values"),
                width=800,
                height=400,
                margin=dict(l=50, r=50, t=80, b=50),
                template="plotly_white"
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Boxplot error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def generate_scatterplot(self, x_column, y_column):
        """Generate scatter plot with proper data handling"""
        print(f"=== SCATTER: {x_column} vs {y_column} ===")
        
        try:
            x_clean = self._clean_numeric_data(x_column)
            y_clean = self._clean_numeric_data(y_column)
            
            if x_clean is None or y_clean is None:
                print(f"Insufficient data for scatter plot")
                return None
            
            # Align data - keep only rows where both columns have valid data
            combined_df = pd.DataFrame({
                x_column: pd.to_numeric(self.df[x_column], errors='coerce'),
                y_column: pd.to_numeric(self.df[y_column], errors='coerce')
            }).dropna()
            
            if len(combined_df) < 2:
                print(f"Insufficient paired data: {len(combined_df)} points")
                return None
            
            # Critical fix: Convert to lists
            x_list = combined_df[x_column].tolist()
            y_list = combined_df[y_column].tolist()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_list,
                y=y_list,
                mode='markers',
                name=f'{x_column} vs {y_column}',
                marker=dict(size=8, opacity=0.7)
            ))
            
            # Critical fix: Proper layout
            fig.update_layout(
                title=f"Scatter Plot: {x_column} vs {y_column}",
                xaxis=dict(title=x_column),
                yaxis=dict(title=y_column),
                width=800,
                height=400,
                margin=dict(l=50, r=50, t=80, b=50),
                template="plotly_white"
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Scatter plot error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def generate_barchart(self, column):
        """Generate bar chart with proper data handling"""
        print(f"=== BAR CHART: {column} ===")
        
        try:
            if column not in self.df.columns:
                print(f"Column {column} not found")
                return None
            
            # Get value counts
            value_counts = self.df[column].value_counts().head(15)
            
            if value_counts.empty:
                print(f"No data for bar chart: {column}")
                return None
            
            # Critical fix: Ensure proper data types
            categories = [str(cat) for cat in value_counts.index]
            values = [int(val) for val in value_counts.values]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=categories,
                y=values,
                name=column
            ))
            
            # Critical fix: Proper layout
            fig.update_layout(
                title=f"Bar Chart of {column}",
                xaxis=dict(title=column, tickangle=-45 if len(categories) > 5 else 0),
                yaxis=dict(title="Count"),
                width=800,
                height=400,
                margin=dict(l=50, r=50, t=80, b=80),
                template="plotly_white"
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Bar chart error for {column}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def generate_correlation_heatmap(self):
        """Generate correlation heatmap with proper data handling"""
        print("=== CORRELATION HEATMAP ===")
        
        try:
            # Get only numeric columns
            numeric_cols = []
            for col in self.df.columns:
                clean_data = self._clean_numeric_data(col)
                if clean_data is not None and len(clean_data) > 10:  # Need sufficient data
                    numeric_cols.append(col)
            
            if len(numeric_cols) < 2:
                print(f"Need at least 2 numeric columns, found {len(numeric_cols)}")
                return None
            
            # Create correlation matrix
            corr_data = {}
            for col in numeric_cols:
                corr_data[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            corr_df = pd.DataFrame(corr_data)
            correlation_matrix = corr_df.corr()
            
            # Critical fix: Handle NaN in correlation matrix
            correlation_matrix = correlation_matrix.fillna(0)
            
            # Critical fix: Convert to lists for JSON serialization
            z_values = correlation_matrix.values.tolist()
            x_labels = correlation_matrix.columns.tolist()
            y_labels = correlation_matrix.index.tolist()
            
            # Create text annotations
            text_values = np.around(correlation_matrix.values, decimals=2).tolist()
            
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=z_values,
                x=x_labels,
                y=y_labels,
                text=text_values,
                texttemplate='%{text}',
                textfont=dict(size=10),
                colorscale='RdBu',
                zmid=0,
                hoverongaps=False
            ))
            
            # Critical fix: Proper layout
            fig.update_layout(
                title="Correlation Heatmap",
                width=600,
                height=500,
                margin=dict(l=80, r=50, t=80, b=80),
                template="plotly_white"
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Correlation heatmap error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def generate_chart(self, chart_type, columns):
        """Main dispatcher with enhanced logging"""
        print(f"\n=== CHART REQUEST: {chart_type} ===")
        print(f"Columns: {columns}")
        print(f"Dataset shape: {self.df.shape if self.df is not None else 'No data'}")
        
        if self.df is None:
            print("ERROR: No dataset loaded")
            return None
        
        try:
            result = None
            
            if chart_type == 'histogram' and len(columns) >= 1:
                result = self.generate_histogram(columns[0])
                
            elif chart_type == 'boxplot' and len(columns) >= 1:
                result = self.generate_boxplot(columns)
                
            elif chart_type == 'scatter' and len(columns) >= 2:
                result = self.generate_scatterplot(columns[0], columns[1])
                
            elif chart_type == 'bar' and len(columns) >= 1:
                result = self.generate_barchart(columns[0])
                
            elif chart_type == 'correlation_heatmap':
                result = self.generate_correlation_heatmap()
                
            else:
                print(f"ERROR: Invalid chart type or insufficient columns")
                return None
            
            if result is not None:
                print(f"SUCCESS: {chart_type} chart generated")
                # Validate the result structure
                if 'data' in result and 'layout' in result:
                    print(f"Chart data structure valid")
                else:
                    print(f"WARNING: Invalid chart data structure")
            else:
                print(f"FAILED: {chart_type} chart generation returned None")
            
            return result
            
        except Exception as e:
            print(f"ERROR in chart generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
