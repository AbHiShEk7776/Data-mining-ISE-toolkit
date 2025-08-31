import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import chi2_contingency
import os
import json
import copy

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.original_df = None  # Store original data
        self.load_data()
    
    def load_data(self):
        """Load data from CSV or Excel file"""
        try:
            if self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            elif self.file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Store original data for revert functionality
            self.original_df = self.df.copy()
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def revert_to_original(self):
        """Revert dataset to original state"""
        if self.original_df is not None:
            self.df = self.original_df.copy()
            return True
        return False
    
    def get_basic_statistics(self):
        """Calculate basic statistical measures"""
        if self.df is None:
            return None
        
        stats_dict = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict(),
            'numerical_columns': [],
            'categorical_columns': []
        }
        
        # Analyze numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if not self.df[col].empty:
                col_stats = {
                    'name': col,
                    'mean': float(self.df[col].mean()) if pd.notna(self.df[col].mean()) else 0,
                    'median': float(self.df[col].median()) if pd.notna(self.df[col].median()) else 0,
                    'mode': float(self.df[col].mode().iloc[0]) if not self.df[col].mode().empty else 0,
                    'std': float(self.df[col].std()) if pd.notna(self.df[col].std()) else 0,
                    'variance': float(self.df[col].var()) if pd.notna(self.df[col].var()) else 0,
                    'range': float(self.df[col].max() - self.df[col].min()) if pd.notna(self.df[col].max()) and pd.notna(self.df[col].min()) else 0,
                    'min': float(self.df[col].min()) if pd.notna(self.df[col].min()) else 0,
                    'max': float(self.df[col].max()) if pd.notna(self.df[col].max()) else 0
                }
                stats_dict['numerical_columns'].append(col_stats)
        
        # Analyze categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if not self.df[col].empty:
                value_counts = self.df[col].value_counts().head(10)
                col_stats = {
                    'name': col,
                    'unique_values': int(self.df[col].nunique()),
                    'most_frequent': str(self.df[col].mode().iloc[0]) if not self.df[col].mode().empty else '',
                    'value_counts': {str(k): int(v) for k, v in value_counts.to_dict().items()}
                }
                stats_dict['categorical_columns'].append(col_stats)
        
        return stats_dict
    
    def handle_missing_values(self, method='drop'):
        """Handle missing values in the dataset"""
        if self.df is None:
            return False
        
        initial_rows = len(self.df)
        
        try:
            if method == 'drop':
                self.df = self.df.dropna()
            elif method == 'mean':
                numerical_cols = self.df.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    if self.df[col].isnull().any():
                        self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif method == 'median':
                numerical_cols = self.df.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    if self.df[col].isnull().any():
                        self.df[col] = self.df[col].fillna(self.df[col].median())
            elif method == 'mode':
                for col in self.df.columns:
                    if self.df[col].isnull().any():
                        mode_value = self.df[col].mode()
                        if not mode_value.empty:
                            self.df[col] = self.df[col].fillna(mode_value.iloc[0])
            
            final_rows = len(self.df)
            return {'initial_rows': initial_rows, 'final_rows': final_rows, 'method': method}
        
        except Exception as e:
            print(f"Error handling missing values: {str(e)}")
            return False
    
    def normalize_data(self, method='minmax', columns=None):
        """Normalize numerical data"""
        if self.df is None:
            return False
        
        try:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
            
            # Only process columns that exist and have numeric data
            valid_columns = [col for col in columns if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']]
            
            if not valid_columns:
                return {'method': method, 'columns_processed': 0}
            
            if method == 'minmax':
                scaler = MinMaxScaler()
                self.df[valid_columns] = scaler.fit_transform(self.df[valid_columns])
            elif method == 'zscore':
                scaler = StandardScaler()
                self.df[valid_columns] = scaler.fit_transform(self.df[valid_columns])
            elif method == 'decimal':
                for col in valid_columns:
                    max_abs = abs(self.df[col]).max()
                    if max_abs > 0:
                        power = len(str(int(max_abs)))
                        self.df[col] = self.df[col] / (10 ** power)
            
            return {'method': method, 'columns_processed': len(valid_columns)}
        
        except Exception as e:
            print(f"Error normalizing data: {str(e)}")
            return False
    
    def discretize_data(self, column, bins=5, method='equal_width'):
        """Discretize continuous data into bins"""
        if self.df is None or column not in self.df.columns:
            return False
        
        try:
            if method == 'equal_width':
                self.df[f'{column}_binned'] = pd.cut(self.df[column], bins=bins, labels=False)
            elif method == 'equal_frequency':
                self.df[f'{column}_binned'] = pd.qcut(self.df[column], q=bins, labels=False, duplicates='drop')
            
            return True
        except Exception as e:
            print(f"Error discretizing data: {str(e)}")
            return False
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        if self.df is None:
            return False
        
        try:
            initial_rows = len(self.df)
            self.df = self.df.drop_duplicates()
            final_rows = len(self.df)
            
            return {'initial_rows': initial_rows, 'final_rows': final_rows}
        except Exception as e:
            print(f"Error removing duplicates: {str(e)}")
            return False
    
    def calculate_correlation(self):
        """Calculate correlation matrix for numerical columns"""
        if self.df is None:
            return None
        
        try:
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) < 2:
                return None
            
            correlation_matrix = self.df[numerical_cols].corr()
            
            # Convert to serializable format
            result = {}
            for col1 in correlation_matrix.columns:
                result[col1] = {}
                for col2 in correlation_matrix.columns:
                    value = correlation_matrix.loc[col1, col2]
                    result[col1][col2] = float(value) if pd.notna(value) else 0.0
            
            return result
        except Exception as e:
            print(f"Error calculating correlation: {str(e)}")
            return None
    
    def calculate_covariance(self):
        """Calculate covariance matrix for numerical columns"""
        if self.df is None:
            return None
        
        try:
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) < 2:
                return None
            
            covariance_matrix = self.df[numerical_cols].cov()
            
            # Convert to serializable format
            result = {}
            for col1 in covariance_matrix.columns:
                result[col1] = {}
                for col2 in covariance_matrix.columns:
                    value = covariance_matrix.loc[col1, col2]
                    result[col1][col2] = float(value) if pd.notna(value) else 0.0
            
            return result
        except Exception as e:
            print(f"Error calculating covariance: {str(e)}")
            return None
    
    def chi_square_test(self, col1, col2):
        """Perform chi-square test of independence"""
        if self.df is None or col1 not in self.df.columns or col2 not in self.df.columns:
            return {'error': 'Invalid columns specified'}
        
        try:
            # Remove rows with missing values for these columns
            clean_df = self.df[[col1, col2]].dropna()
            
            if clean_df.empty:
                return {'error': 'No valid data after removing missing values'}
            
            contingency_table = pd.crosstab(clean_df[col1], clean_df[col2])
            
            if contingency_table.empty or contingency_table.size < 4:
                return {'error': 'Insufficient data for chi-square test'}
            
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Convert contingency table to serializable format
            contingency_dict = {}
            for row_idx in contingency_table.index:
                contingency_dict[str(row_idx)] = {}
                for col_idx in contingency_table.columns:
                    contingency_dict[str(row_idx)][str(col_idx)] = int(contingency_table.loc[row_idx, col_idx])
            
            return {
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'contingency_table': contingency_dict,
                'columns': [col1, col2]
            }
        except Exception as e:
            return {'error': f'Chi-square test failed: {str(e)}'}
    
    def save_processed_data(self, output_path):
        """Save processed data to file"""
        if self.df is None:
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if output_path.endswith('.csv'):
                self.df.to_csv(output_path, index=False)
            elif output_path.endswith(('.xlsx', '.xls')):
                self.df.to_excel(output_path, index=False)
            return True
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return False
    
    def get_data_for_visualization(self, columns=None):
        """Get data formatted for visualization"""
        if self.df is None:
            return None
        
        try:
            if columns:
                available_columns = [col for col in columns if col in self.df.columns]
                if available_columns:
                    return self.df[available_columns].to_dict('records')
            return self.df.to_dict('records')
        except Exception as e:
            print(f"Error getting visualization data: {str(e)}")
            return None
    
    def get_sample_data(self, n_samples=100):
        """Get sample data for visualization (performance optimization)"""
        if self.df is None:
            return None
        
        try:
            if len(self.df) > n_samples:
                return self.df.sample(n=n_samples).to_dict('records')
            return self.df.to_dict('records')
        except Exception as e:
            print(f"Error getting sample data: {str(e)}")
            return None
