/**
 * Performance Monitoring Chart Integration
 * © 2025 Negative Space Imaging, Inc. - CONFIDENTIAL
 *
 * JavaScript module for integrating performance monitoring charts with backend API.
 */

/**
 * PerformanceMonitor class for managing real-time performance charts
 */
class PerformanceMonitor {
    /**
     * Create a new PerformanceMonitor instance
     * @param {Object} options - Configuration options
     * @param {number} options.refreshInterval - Data refresh interval in milliseconds
     * @param {boolean} options.autoStart - Whether to start monitoring automatically
     */
    constructor(options = {}) {
        this.options = {
            refreshInterval: 5000, // 5 seconds default
            autoStart: true,
            ...options
        };

        this.charts = {};
        this.metrics = {};
        this.dataPoints = 60; // Number of data points to display
        this.refreshTimer = null;
        this.isRunning = false;

        // Initialize when document is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initialize());
        } else {
            this.initialize();
        }
    }

    /**
     * Initialize the performance monitor
     */
    initialize() {
        this.initializeCharts();

        if (this.options.autoStart) {
            this.start();
        }
    }

    /**
     * Initialize all charts
     */
    initializeCharts() {
        // CPU Usage Chart
        this.initChart('cpu-chart', 'CPU Usage (%)', 'rgb(75, 192, 192)');

        // Memory Usage Chart
        this.initChart('memory-chart', 'Memory Usage (MB)', 'rgb(255, 99, 132)');

        // Thread Count Chart
        this.initChart('thread-chart', 'Thread Count', 'rgb(54, 162, 235)');

        // Cache Hit Ratio Chart
        this.initChart('cache-chart', 'Cache Hit Ratio (%)', 'rgb(153, 102, 255)');
    }

    /**
     * Initialize a specific chart
     * @param {string} canvasId - The ID of the canvas element
     * @param {string} label - The label for the dataset
     * @param {string} color - The color for the dataset
     */
    initChart(canvasId, label, color) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(this.dataPoints).fill(''),
                datasets: [{
                    label: label,
                    data: Array(this.dataPoints).fill(null),
                    borderColor: color,
                    backgroundColor: color.replace(')', ', 0.2)').replace('rgb', 'rgba'),
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        display: true,
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: label
                        }
                    }
                },
                animation: {
                    duration: 500
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                }
            }
        });
    }

    /**
     * Start monitoring performance metrics
     */
    start() {
        if (this.isRunning) return;

        this.isRunning = true;
        this.fetchData();

        this.refreshTimer = setInterval(() => {
            this.fetchData();
        }, this.options.refreshInterval);

        console.log('Performance monitoring started');
    }

    /**
     * Stop monitoring performance metrics
     */
    stop() {
        if (!this.isRunning) return;

        this.isRunning = false;
        clearInterval(this.refreshTimer);

        console.log('Performance monitoring stopped');
    }

    /**
     * Fetch performance data from API
     */
    fetchData() {
        Promise.all([
            this.fetchMetrics(),
            this.fetchMemoryUsage(),
            this.fetchCacheStats()
        ])
        .then(() => {
            this.updateCharts();
        })
        .catch(error => {
            console.error('Error fetching performance data:', error);
        });
    }

    /**
     * Fetch general metrics
     * @returns {Promise} Promise that resolves when metrics are fetched
     */
    fetchMetrics() {
        return fetch('/performance/api/metrics')
            .then(response => response.json())
            .then(data => {
                this.metrics.cpu = data.cpu_usage;
                this.metrics.thread = data.thread_count;

                // Add timestamp
                const now = new Date();
                this.metrics.timestamp = now.toLocaleTimeString();
            });
    }

    /**
     * Fetch memory usage
     * @returns {Promise} Promise that resolves when memory usage is fetched
     */
    fetchMemoryUsage() {
        return fetch('/performance/api/memory')
            .then(response => response.json())
            .then(data => {
                this.metrics.memory = data.rss || data.memory_usage;
            });
    }

    /**
     * Fetch cache statistics
     * @returns {Promise} Promise that resolves when cache stats are fetched
     */
    fetchCacheStats() {
        return fetch('/performance/api/cache')
            .then(response => response.json())
            .then(data => {
                if (data.hits !== undefined && data.misses !== undefined) {
                    const total = data.hits + data.misses;
                    this.metrics.cacheHitRatio = total > 0 ? (data.hits / total) * 100 : 0;
                } else {
                    this.metrics.cacheHitRatio = data.hit_ratio * 100 || 0;
                }
            });
    }

    /**
     * Update all charts with new data
     */
    updateCharts() {
        this.updateChart('cpu-chart', this.metrics.cpu);
        this.updateChart('memory-chart', this.metrics.memory);
        this.updateChart('thread-chart', this.metrics.thread);
        this.updateChart('cache-chart', this.metrics.cacheHitRatio);

        // Update any display elements
        this.updateDisplayElements();
    }

    /**
     * Update a specific chart with new data
     * @param {string} chartId - The ID of the chart canvas
     * @param {number} value - The new data value
     */
    updateChart(chartId, value) {
        const chart = this.charts[chartId];
        if (!chart) return;

        // Add new data point
        chart.data.labels.push(this.metrics.timestamp);
        chart.data.datasets[0].data.push(value);

        // Remove oldest data point if we have more than dataPoints
        if (chart.data.labels.length > this.dataPoints) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        // Update chart
        chart.update();
    }

    /**
     * Update any display elements with current metrics
     */
    updateDisplayElements() {
        // Update CPU usage display
        const cpuElement = document.getElementById('cpu-value');
        if (cpuElement) cpuElement.textContent = `${this.metrics.cpu.toFixed(1)}%`;

        // Update memory usage display
        const memoryElement = document.getElementById('memory-value');
        if (memoryElement) memoryElement.textContent = `${this.metrics.memory.toFixed(1)} MB`;

        // Update thread count display
        const threadElement = document.getElementById('thread-value');
        if (threadElement) threadElement.textContent = this.metrics.thread;

        // Update cache hit ratio display
        const cacheElement = document.getElementById('cache-value');
        if (cacheElement) cacheElement.textContent = `${this.metrics.cacheHitRatio.toFixed(1)}%`;

        // Update timestamp
        const timestampElement = document.getElementById('last-updated');
        if (timestampElement) timestampElement.textContent = this.metrics.timestamp;
    }

    /**
     * Set the refresh interval
     * @param {number} interval - Refresh interval in milliseconds
     */
    setRefreshInterval(interval) {
        this.options.refreshInterval = interval;

        // Restart if running
        if (this.isRunning) {
            this.stop();
            this.start();
        }
    }
}

// Create global instance
window.performanceMonitor = new PerformanceMonitor();

// Export class for module usage
export default PerformanceMonitor;
