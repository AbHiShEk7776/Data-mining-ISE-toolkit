import { bootstrapApplication } from '@angular/platform-browser';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { AppComponent } from './app/app.component';

import type { Routes } from '@angular/router';

// Define routes
const routes: Routes = [
  { path: '', redirectTo: '/upload', pathMatch: 'full' },
  { 
    path: 'upload', 
    loadComponent: () => import('./app/upload/upload.component').then(m => m.UploadComponent)
  },
  { 
    path: 'preprocessing', 
    loadComponent: () => import('./app/preprocessing/preprocessing.component').then(m => m.PreprocessingComponent)
  },
  { 
    path: 'visualization', 
    loadComponent: () => import('./app/visualization/visualization.component').then(m => m.VisualizationComponent)
  },
  { 
    path: 'ml-models', 
    loadComponent: () => import('./app/ml-models/ml-models.component').then(m => m.MlModelsComponent)
  }
];

bootstrapApplication(AppComponent, {
  providers: [
    provideRouter(routes),
    provideHttpClient()
  ]
}).catch(err => console.error(err));
