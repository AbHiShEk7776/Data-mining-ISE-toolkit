import { Component, OnInit } from '@angular/core';
import { Router, NavigationEnd, RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { CommonModule } from '@angular/common';
import { filter } from 'rxjs/operators';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'Data Mining Toolkit';
  currentRoute = '';
  isLoading = false;

  constructor(private router: Router) {}

  ngOnInit(): void {
    this.router.events
      .pipe(filter(event => event instanceof NavigationEnd))
      .subscribe((event: NavigationEnd) => {
        this.currentRoute = event.url;
      });
  }

  // Navigation methods
  navigateToUpload(): void {
    this.router.navigate(['/upload']);
  }

  navigateToPreprocessing(): void {
    this.router.navigate(['/preprocessing']);
  }

  navigateToVisualization(): void {
    this.router.navigate(['/visualization']);
  }

  navigateToMlModels(): void {
    this.router.navigate(['/ml-models']);
  }

  isRouteActive(route: string): boolean {
    return this.currentRoute === route;
  }

  getCurrentPageTitle(): string {
    const titles: { [key: string]: string } = {
      '/upload': 'Upload Dataset',
      '/preprocessing': 'Data Preprocessing',
      '/visualization': 'Data Visualization',
      '/ml-models': 'Machine Learning Models'
    };
    return titles[this.currentRoute] || 'Data Mining Toolkit';
  }

  getCurrentPageIcon(): string {
    const icons: { [key: string]: string } = {
      '/upload': 'fas fa-cloud-upload-alt',
      '/preprocessing': 'fas fa-cogs',
      '/visualization': 'fas fa-chart-bar',
      '/ml-models': 'fas fa-robot'
    };
    return icons[this.currentRoute] || 'fas fa-chart-line';
  }
}
