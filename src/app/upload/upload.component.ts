import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpEventType } from '@angular/common/http';
import { Router } from '@angular/router';
import * as XLSX from 'xlsx';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.css']
})
export class UploadComponent implements OnInit {
  // private http = HttpClient.prototype;
  // private router = Router.prototype;

  selectedFile: File | null = null;
  uploadProgress = 0;
  isUploading = false;
  dataPreview: any[] = [];
  columnHeaders: string[] = [];
  fileName = '';
  fileSize = 0;
  isDragOver = false;
  error: string | null = null;
  
 constructor(private http: HttpClient, private router: Router) {}
  
  ngOnInit(): void {}
  // Add these methods to the existing UploadComponent class


  onFileSelected(event: Event): void {
    const target = event.target as HTMLInputElement;
    const file = target.files?.[0];
    if (file) {
      this.selectedFile = file;
      this.fileName = file.name;
      this.fileSize = file.size;
      this.previewFile(file);
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver = false;
    
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (this.isValidFileType(file)) {
        this.selectedFile = file;
        this.fileName = file.name;
        this.fileSize = file.size;
        this.previewFile(file);
      }
    }
  }

  private isValidFileType(file: File): boolean {
    const validTypes = [
      'text/csv',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/vnd.ms-excel'
    ];
    const validExtensions = ['.csv', '.xlsx', '.xls'];
    
    return validTypes.includes(file.type) || 
           validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
  }
  
  previewFile(file: File): void {
    const reader = new FileReader();
    
    if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
      reader.onload = (e: ProgressEvent<FileReader>) => {
        const csvData = e.target?.result as string;
        this.parseCSV(csvData);
      };
      reader.readAsText(file);
    } else if (this.isExcelFile(file)) {
      reader.onload = (e: ProgressEvent<FileReader>) => {
        const data = new Uint8Array(e.target?.result as ArrayBuffer);
        const workbook = XLSX.read(data, { type: 'array' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const jsonData: any[][] = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
        this.processExcelData(jsonData);
      };
      reader.readAsArrayBuffer(file);
    }
  }

  private isExcelFile(file: File): boolean {
    return file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' || 
           file.name.endsWith('.xlsx') || 
           file.name.endsWith('.xls');
  }
  
  parseCSV(csvData: string): void {
    const lines = csvData.split('\n').filter(line => line.trim());
    if (lines.length > 0) {
      this.columnHeaders = this.parseCSVLine(lines[0]);
      this.dataPreview = [];
      
      for (let i = 1; i < Math.min(6, lines.length); i++) {
        const values = this.parseCSVLine(lines[i]);
        const row: Record<string, string> = {};
        this.columnHeaders.forEach((header, index) => {
          row[header] = values[index] || '';
        });
        this.dataPreview.push(row);
      }
    }
  }

  private parseCSVLine(line: string): string[] {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        result.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    
    result.push(current.trim());
    return result;
  }
  
  processExcelData(data: any[][]): void {
    if (data.length > 0) {
      this.columnHeaders = data[0].map(header => String(header || ''));
      this.dataPreview = [];
      
      for (let i = 1; i < Math.min(6, data.length); i++) {
        const row: Record<string, string> = {};
        this.columnHeaders.forEach((header, index) => {
          row[header] = String(data[i][index] || '');
        });
        this.dataPreview.push(row);
      }
    }
  }
  
  uploadFile(): void {
    if (!this.selectedFile) return;
    
    const formData = new FormData();
    formData.append('file', this.selectedFile);
    
    this.isUploading = true;
    this.uploadProgress = 0;
    
    this.http.post('http://localhost:8000/api/upload/', formData, {
      reportProgress: true,
      observe: 'events'
    }).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          this.uploadProgress = Math.round(100 * event.loaded / event.total);
        } else if (event.type === HttpEventType.Response) {
          console.log('Upload successful!', event.body);
          this.isUploading = false;
          this.router.navigate(['/preprocessing']);
        }
      },
      error: (error) => {
        console.error('Upload error:', error);
        this.error = 'Upload failed. Please try again.';
        this.isUploading = false;
        this.uploadProgress = 0;
      }
    });
  }

  removeFile(): void {
    this.selectedFile = null;
    this.fileName = '';
    this.fileSize = 0;
    this.dataPreview = [];
    this.columnHeaders = [];
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  trackByHeader(index: number, header: string): string {
    return header;
  }

  trackByRow(index: number, row: any): number {
    return index;
  }
}
