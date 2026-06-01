/**
 * Docker Health Check System - Node.js Implementation
 *
 * Production-grade health monitoring module for Express.js applications
 * Monitors service connectivity, API endpoints, and database health
 *
 * @author DevOps Team
 * @date 2025-11-08
 * @version 1.0.0
 */

const http = require('http');
const https = require('https');
const { createConnection } = require('net');
const { promisify } = require('util');
const fs = require('fs');
const path = require('path');

/**
 * Configuration for health checks
 */
class HealthCheckConfig {
  constructor() {
    this.services = {
      postgresql: {
        host: process.env.POSTGRES_HOST || 'localhost',
        port: parseInt(process.env.POSTGRES_PORT) || 5432,
        timeout: 5000,
      },
      redis: {
        host: process.env.REDIS_HOST || 'localhost',
        port: parseInt(process.env.REDIS_PORT) || 6379,
        timeout: 5000,
      },
      nodejs: {
        host: process.env.NODE_HOST || 'localhost',
        port: parseInt(process.env.NODE_PORT) || 3000,
        endpoint: '/api/health',
        timeout: 5000,
      },
      python: {
        host: process.env.PYTHON_HOST || 'localhost',
        port: parseInt(process.env.PYTHON_PORT) || 5000,
        endpoint: '/health',
        timeout: 5000,
      },
      react: {
        host: process.env.REACT_HOST || 'localhost',
        port: parseInt(process.env.REACT_PORT) || 3001,
        endpoint: '/',
        timeout: 5000,
      },
    };

    this.thresholds = {
      cpu: 80,
      memory: 85,
      disk: 90,
      restarts: 5,
    };

    this.logDir = process.env.LOG_DIR || path.join(__dirname, '../logs');
    this.reportFormat = process.env.REPORT_FORMAT || 'json';
  }
}

/**
 * Core health check engine
 */
class HealthChecker {
  constructor(config = new HealthCheckConfig()) {
    this.config = config;
    this.results = {};
    this.overallStatus = 'HEALTHY';
    this.failures = 0;
    this.warnings = 0;
    this.startTime = Date.now();
  }

  /**
   * Test TCP port connectivity
   * @param {string} host - Target host
   * @param {number} port - Target port
   * @param {number} timeout - Timeout in milliseconds
   * @returns {Promise<boolean>}
   */
  async testPortOpen(host, port, timeout = 5000) {
    return new Promise((resolve) => {
      const socket = createConnection({ host, port });

      const onConnect = () => {
        socket.destroy();
        resolve(true);
      };

      const onError = () => {
        socket.destroy();
        resolve(false);
      };

      const onTimeout = () => {
        socket.destroy();
        resolve(false);
      };

      socket.once('connect', onConnect);
      socket.once('error', onError);
      socket.setTimeout(timeout, onTimeout);
    });
  }

  /**
   * Test HTTP/HTTPS endpoint
   * @param {string} url - Target URL
   * @param {number} timeout - Timeout in milliseconds
   * @returns {Promise<{status: number, success: boolean}>}
   */
  async testHttpEndpoint(url, timeout = 5000) {
    return new Promise((resolve) => {
      const protocol = url.startsWith('https') ? https : http;

      const request = protocol.get(url, { timeout }, (response) => {
        let data = '';

        response.on('data', chunk => {
          data += chunk;
        });

        response.on('end', () => {
          request.abort();
          resolve({
            status: response.statusCode,
            success: response.statusCode >= 200 && response.statusCode < 300,
            headers: response.headers,
          });
        });
      });

      request.on('error', (error) => {
        request.abort();
        resolve({
          status: 0,
          success: false,
          error: error.message,
        });
      });

      request.on('timeout', () => {
        request.abort();
        resolve({
          status: 0,
          success: false,
          error: 'Request timeout',
        });
      });
    });
  }

  /**
   * Update service health status
   * @param {string} service - Service name
   * @param {string} status - Health status (HEALTHY, WARNING, CRITICAL)
   * @param {string} details - Status details
   */
  updateStatus(service, status, details = '') {
    this.results[service] = { status, details, timestamp: new Date().toISOString() };

    if (status === 'CRITICAL') {
      this.failures++;
      this.overallStatus = 'UNHEALTHY';
    } else if (status === 'WARNING') {
      this.warnings++;
    }

    const symbol = {
      HEALTHY: '✓',
      WARNING: '⚠',
      CRITICAL: '✗',
    }[status];

    const message = `${symbol} ${service}: ${status}${details ? ` - ${details}` : ''}`;
    console.log(message);

    return message;
  }

  /**
   * Check PostgreSQL connectivity
   */
  async checkPostgreSQL() {
    console.log('Checking PostgreSQL...');
    const cfg = this.config.services.postgresql;

    try {
      const isOpen = await this.testPortOpen(cfg.host, cfg.port, cfg.timeout);

      if (!isOpen) {
        this.updateStatus('PostgreSQL', 'CRITICAL', 'Port not responding');
        return;
      }

      this.updateStatus('PostgreSQL', 'HEALTHY');
    } catch (error) {
      this.updateStatus('PostgreSQL', 'CRITICAL', error.message);
    }
  }

  /**
   * Check Redis connectivity
   */
  async checkRedis() {
    console.log('Checking Redis...');
    const cfg = this.config.services.redis;

    try {
      const isOpen = await this.testPortOpen(cfg.host, cfg.port, cfg.timeout);

      if (!isOpen) {
        this.updateStatus('Redis', 'CRITICAL', 'Port not responding');
        return;
      }

      this.updateStatus('Redis', 'HEALTHY');
    } catch (error) {
      this.updateStatus('Redis', 'CRITICAL', error.message);
    }
  }

  /**
   * Check Node.js backend
   */
  async checkNodeJS() {
    console.log('Checking Node.js Backend...');
    const cfg = this.config.services.nodejs;

    try {
      const isOpen = await this.testPortOpen(cfg.host, cfg.port, cfg.timeout);

      if (!isOpen) {
        this.updateStatus('Node.js', 'CRITICAL', 'Port not responding');
        return;
      }

      const url = `http://${cfg.host}:${cfg.port}${cfg.endpoint}`;
      const response = await this.testHttpEndpoint(url, cfg.timeout);

      if (response.success) {
        this.updateStatus('Node.js', 'HEALTHY');
      } else {
        this.updateStatus('Node.js', 'CRITICAL', `HTTP ${response.status}`);
      }
    } catch (error) {
      this.updateStatus('Node.js', 'CRITICAL', error.message);
    }
  }

  /**
   * Check Python service
   */
  async checkPython() {
    console.log('Checking Python Service...');
    const cfg = this.config.services.python;

    try {
      const isOpen = await this.testPortOpen(cfg.host, cfg.port, cfg.timeout);

      if (!isOpen) {
        this.updateStatus('Python', 'CRITICAL', 'Port not responding');
        return;
      }

      const url = `http://${cfg.host}:${cfg.port}${cfg.endpoint}`;
      const response = await this.testHttpEndpoint(url, cfg.timeout);

      if (response.success) {
        this.updateStatus('Python', 'HEALTHY');
      } else {
        this.updateStatus('Python', 'CRITICAL', `HTTP ${response.status}`);
      }
    } catch (error) {
      this.updateStatus('Python', 'CRITICAL', error.message);
    }
  }

  /**
   * Check React frontend
   */
  async checkReact() {
    console.log('Checking React Frontend...');
    const cfg = this.config.services.react;

    try {
      const isOpen = await this.testPortOpen(cfg.host, cfg.port, cfg.timeout);

      if (!isOpen) {
        this.updateStatus('React', 'CRITICAL', 'Port not responding');
        return;
      }

      const url = `http://${cfg.host}:${cfg.port}${cfg.endpoint}`;
      const response = await this.testHttpEndpoint(url, cfg.timeout);

      if (response.success) {
        this.updateStatus('React', 'HEALTHY');
      } else {
        this.updateStatus('React', 'WARNING', `HTTP ${response.status}`);
      }
    } catch (error) {
      this.updateStatus('React', 'CRITICAL', error.message);
    }
  }

  /**
   * Check system resources
   */
  checkSystemResources() {
    console.log('\nSystem Resources:');

    try {
      const osUtils = require('os');
      const os = osUtils;

      // CPU Usage (as percentage of used memory)
      const totalMem = os.totalmem();
      const freeMem = os.freemem();
      const usedMem = totalMem - freeMem;
      const memPercent = ((usedMem / totalMem) * 100).toFixed(2);

      if (memPercent > this.config.thresholds.memory) {
        this.updateStatus('Memory Usage', 'WARNING', `${memPercent}%`);
      } else {
        this.updateStatus('Memory Usage', 'HEALTHY', `${memPercent}%`);
      }

      // Load average
      const loadAverage = os.loadavg();
      const cpuCount = os.cpus().length;
      const loadPercent = ((loadAverage[0] / cpuCount) * 100).toFixed(2);

      if (loadPercent > this.config.thresholds.cpu) {
        this.updateStatus('CPU Load', 'WARNING', `${loadPercent}%`);
      } else {
        this.updateStatus('CPU Load', 'HEALTHY', `${loadPercent}%`);
      }
    } catch (error) {
      console.error(`Resource check error: ${error.message}`);
    }
  }

  /**
   * Generate JSON report
   */
  generateReport() {
    const duration = Date.now() - this.startTime;

    const report = {
      timestamp: new Date().toISOString(),
      timestamp_unix: Math.floor(Date.now() / 1000),
      overall_status: this.overallStatus,
      check_duration_ms: duration,
      summary: {
        total_services: Object.keys(this.results).length,
        healthy: Object.values(this.results).filter(r => r.status === 'HEALTHY').length,
        warnings: this.warnings,
        failures: this.failures,
      },
      services: this.results,
    };

    return report;
  }

  /**
   * Save report to file
   */
  async saveReport(report) {
    try {
      if (!fs.existsSync(this.config.logDir)) {
        fs.mkdirSync(this.config.logDir, { recursive: true });
      }

      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filepath = path.join(this.config.logDir, `health-report-${timestamp}.json`);

      fs.writeFileSync(filepath, JSON.stringify(report, null, 2));
      console.log(`\nReport saved: ${filepath}`);

      return filepath;
    } catch (error) {
      console.error(`Failed to save report: ${error.message}`);
    }
  }

  /**
   * Display summary
   */
  displaySummary() {
    console.log('\n' + '='.repeat(80));
    console.log('DOCKER HEALTH CHECK SUMMARY');
    console.log('='.repeat(80));
    console.log(`Overall Status: ${this.overallStatus}`);
    console.log(`Total Services: ${Object.keys(this.results).length}`);
    console.log(`Healthy: ${Object.values(this.results).filter(r => r.status === 'HEALTHY').length}`);
    console.log(`Warnings: ${this.warnings}`);
    console.log(`Failures: ${this.failures}`);
    console.log('='.repeat(80));
  }

  /**
   * Run all health checks
   */
  async runAll() {
    console.log('Starting Docker Health Check System...\n');

    await Promise.all([
      this.checkPostgreSQL(),
      this.checkRedis(),
      this.checkNodeJS(),
      this.checkPython(),
      this.checkReact(),
    ]);

    this.checkSystemResources();

    const report = this.generateReport();
    await this.saveReport(report);

    this.displaySummary();

    return {
      status: this.overallStatus,
      code: this.overallStatus === 'HEALTHY' ? 0 : 1,
      report,
    };
  }
}

/**
 * Express.js middleware for health check endpoint
 */
function healthCheckMiddleware(checker) {
  return async (req, res) => {
    const report = checker.generateReport();
    res.status(report.overall_status === 'HEALTHY' ? 200 : 503).json(report);
  };
}

// Export for use in Express applications
module.exports = {
  HealthChecker,
  HealthCheckConfig,
  healthCheckMiddleware,
};

// Run directly if called as main module
if (require.main === module) {
  (async () => {
    const checker = new HealthChecker();
    const result = await checker.runAll();
    process.exit(result.code);
  })();
}
