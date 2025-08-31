# Data Mining Toolkit

A Django + Angular web application for data analysis and machine learning, built as a student project.

## Features

- Upload CSV/Excel datasets
- Data preprocessing and cleaning
- Interactive visualizations (charts, plots)
- Train ML models (Logistic Regression, KNN, Decision Tree, Naive Bayes, Neural Network)
- Model evaluation and comparison

## Technologies

- **Backend**: Django, Python, Pandas, NumPy, scikit-learn
- **Frontend**: Angular, TypeScript, Plotly.js
- **Database**: SQLite (default)

## Quick Setup

### Backend

git clone <repository-url>
cd datamining-toolkit/backend
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver



### Frontend

cd ../frontend
npm install
ng serve



Access: http://localhost:4200

## Project Structure

backend/ 
├── api/ 
│ ├── ml_models/ # Custom ML implementations
│ ├── views.py # API endpoints
│ └── models.py # Database models
frontend/
├── src/app/
│ ├── upload/ # File upload component
│ ├── preprocessing/ # Data preprocessing
│ ├── visualization/ # Charts and plots
│ └── ml-models/ # ML training interface



## API Endpoints

- `POST /api/upload/` - Upload dataset
- `POST /api/preprocess/` - Preprocess data
- `POST /api/visualize/` - Generate charts
- `POST /api/ml/train/` - Train ML models
- `GET /api/ml/models/` - List trained models

## Requirements

### Backend (requirements.txt)

Django==4.2.0
djangorestframework==3.14.0
django-cors-headers==4.0.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
plotly==5.15.0
openpyxl==3.1.0



### Frontend (package.json)

{
"dependencies": {
"@angular/core": "^15.0.0",
"@angular/common": "^15.0.0",
"@angular/forms": "^15.0.0",
"@angular/router": "^15.0.0",
"plotly.js-dist": "^2.24.0"
}
}



## Usage

1. **Upload**: Select CSV/Excel file
2. **Preprocess**: Clean data, handle nulls
3. **Visualize**: Create charts and plots
4. **Train**: Select ML algorithm and features
5. **Evaluate**: View metrics and confusion matrix
