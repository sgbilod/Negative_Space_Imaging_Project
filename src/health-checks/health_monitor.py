"""
Docker Health Check System - Python Implementation

Production-grade health monitoring module for multi-service applications.
Monitors service connectivity, API endpoints, and system resources.

Author: DevOps Team
Date: 2025-11-08
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path
import socket
import subprocess

import aiohttp
import psutil


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ServiceConfig:
    """Service configuration"""
    host: str
    port: int
    timeout: float = 5.0
    endpoint: Optional[str] = None


class HealthCheckConfig:
    """Central configuration for health checks"""

    def __init__(self):
        self.services: Dict[str, ServiceConfig] = {
            'postgresql': ServiceConfig(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', '5432')),
            ),
            'redis': ServiceConfig(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', '6379')),
            ),
            'nodejs': ServiceConfig(
                host=os.getenv('NODE_HOST', 'localhost'),
                port=int(os.getenv('NODE_PORT', '3000')),
                endpoint='/api/health',
            ),
            'python': ServiceConfig(
                host=os.getenv('PYTHON_HOST', 'localhost'),
                port=int(os.getenv('PYTHON_PORT', '5000')),
                endpoint='/health',
            ),
            'react': ServiceConfig(
                host=os.getenv('REACT_HOST', 'localhost'),
                port=int(os.getenv('REACT_PORT', '3001')),
                endpoint='/',
            ),
        }

        self.thresholds = {
            'cpu': 80,
            'memory': 85,
            'disk': 90,
            'restarts': 5,
        }

        self.log_dir = Path(os.getenv('LOG_DIR', './logs'))
        self.log_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # File handler
        log_file = self.log_dir / f"health-check-{datetime.now():%Y%m%d}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


# ============================================================================
# Health Check Engine
# ============================================================================

class AsyncHealthChecker:
    """Asynchronous health check engine"""

    def __init__(self, config: Optional[HealthCheckConfig] = None):
        self.config = config or HealthCheckConfig()
        self.results: Dict[str, Dict] = {}
        self.overall_status = 'HEALTHY'
        self.failures = 0
        self.warnings = 0
        self.start_time = time.time()

    def update_status(
        self,
        service: str,
        status: str,
        details: str = '',
    ) -> None:
        """
        Update service health status

        Args:
            service: Service name
            status: Health status (HEALTHY, WARNING, CRITICAL)
            details: Status details
        """
        self.results[service] = {
            'status': status,
            'details': details,
            'timestamp': datetime.utcnow().isoformat(),
        }

        if status == 'CRITICAL':
            self.failures += 1
            self.overall_status = 'UNHEALTHY'
        elif status == 'WARNING':
            self.warnings += 1

        symbol = {'HEALTHY': '✓', 'WARNING': '⚠', 'CRITICAL': '✗'}.get(status, '?')
        message = f"{symbol} {service}: {status}"
        if details:
            message += f" - {details}"

        self.config.logger.info(message)

    async def test_port_open(
        self,
        host: str,
        port: int,
        timeout: float = 5.0,
    ) -> bool:
        """
        Test TCP port connectivity

        Args:
            host: Target host
            port: Target port
            timeout: Timeout in seconds

        Returns:
            True if port is open, False otherwise
        """
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, OSError, ConnectionRefusedError):
            return False

    async def test_http_endpoint(
        self,
        url: str,
        timeout: float = 5.0,
    ) -> Tuple[int, bool]:
        """
        Test HTTP endpoint

        Args:
            url: Target URL
            timeout: Timeout in seconds

        Returns:
            Tuple of (status_code, success)
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    return response.status, response.status in (200, 201, 202, 204)
        except (asyncio.TimeoutError, aiohttp.ClientError):
            return 0, False

    async def check_postgresql(self) -> None:
        """Check PostgreSQL connectivity"""
        self.config.logger.debug("Checking PostgreSQL...")
        cfg = self.config.services['postgresql']

        if not await self.test_port_open(cfg.host, cfg.port, cfg.timeout):
            self.update_status('PostgreSQL', 'CRITICAL', 'Port not responding')
            return

        self.update_status('PostgreSQL', 'HEALTHY')

    async def check_redis(self) -> None:
        """Check Redis connectivity"""
        self.config.logger.debug("Checking Redis...")
        cfg = self.config.services['redis']

        if not await self.test_port_open(cfg.host, cfg.port, cfg.timeout):
            self.update_status('Redis', 'CRITICAL', 'Port not responding')
            return

        self.update_status('Redis', 'HEALTHY')

    async def check_nodejs(self) -> None:
        """Check Node.js backend"""
        self.config.logger.debug("Checking Node.js Backend...")
        cfg = self.config.services['nodejs']

        if not await self.test_port_open(cfg.host, cfg.port, cfg.timeout):
            self.update_status('Node.js', 'CRITICAL', 'Port not responding')
            return

        url = f"http://{cfg.host}:{cfg.port}{cfg.endpoint}"
        status_code, success = await self.test_http_endpoint(url, cfg.timeout)

        if success:
            self.update_status('Node.js', 'HEALTHY')
        else:
            self.update_status('Node.js', 'CRITICAL', f'HTTP {status_code}')

    async def check_python(self) -> None:
        """Check Python service"""
        self.config.logger.debug("Checking Python Service...")
        cfg = self.config.services['python']

        if not await self.test_port_open(cfg.host, cfg.port, cfg.timeout):
            self.update_status('Python', 'CRITICAL', 'Port not responding')
            return

        url = f"http://{cfg.host}:{cfg.port}{cfg.endpoint}"
        status_code, success = await self.test_http_endpoint(url, cfg.timeout)

        if success:
            self.update_status('Python', 'HEALTHY')
        else:
            self.update_status('Python', 'CRITICAL', f'HTTP {status_code}')

    async def check_react(self) -> None:
        """Check React frontend"""
        self.config.logger.debug("Checking React Frontend...")
        cfg = self.config.services['react']

        if not await self.test_port_open(cfg.host, cfg.port, cfg.timeout):
            self.update_status('React', 'CRITICAL', 'Port not responding')
            return

        url = f"http://{cfg.host}:{cfg.port}{cfg.endpoint}"
        status_code, success = await self.test_http_endpoint(url, cfg.timeout)

        if success:
            self.update_status('React', 'HEALTHY')
        else:
            self.update_status('React', 'WARNING', f'HTTP {status_code}')

    def check_system_resources(self) -> None:
        """Check system resource usage"""
        self.config.logger.info("Checking system resources...")

        # CPU Usage
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config.thresholds['cpu']:
                self.update_status('CPU Usage', 'WARNING', f'{cpu_percent:.1f}%')
            else:
                self.update_status('CPU Usage', 'HEALTHY', f'{cpu_percent:.1f}%')
        except Exception as e:
            self.config.logger.warning(f"Unable to check CPU: {e}")

        # Memory Usage
        try:
            memory = psutil.virtual_memory()
            if memory.percent > self.config.thresholds['memory']:
                self.update_status('Memory Usage', 'WARNING', f'{memory.percent:.1f}%')
            else:
                self.update_status('Memory Usage', 'HEALTHY', f'{memory.percent:.1f}%')
        except Exception as e:
            self.config.logger.warning(f"Unable to check memory: {e}")

        # Disk Usage
        try:
            disk = psutil.disk_usage('/')
            if disk.percent > self.config.thresholds['disk']:
                self.update_status('Disk Usage', 'WARNING', f'{disk.percent:.1f}%')
            else:
                self.update_status('Disk Usage', 'HEALTHY', f'{disk.percent:.1f}%')
        except Exception as e:
            self.config.logger.warning(f"Unable to check disk: {e}")

    def generate_report(self) -> Dict:
        """
        Generate health check report

        Returns:
            Dictionary containing report data
        """
        duration = time.time() - self.start_time

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'timestamp_unix': int(time.time()),
            'overall_status': self.overall_status,
            'check_duration_seconds': round(duration, 2),
            'summary': {
                'total_services': len(self.results),
                'healthy': sum(1 for r in self.results.values() if r['status'] == 'HEALTHY'),
                'warnings': self.warnings,
                'failures': self.failures,
            },
            'services': self.results,
        }

    async def save_report(self, report: Dict) -> str:
        """
        Save report to JSON file

        Args:
            report: Report dictionary

        Returns:
            Path to saved report
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.config.log_dir / f'health-report-{timestamp}.json'

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)

            self.config.logger.info(f"Report saved: {filepath}")
            return str(filepath)
        except Exception as e:
            self.config.logger.error(f"Failed to save report: {e}")
            raise

    def display_summary(self) -> None:
        """Display health check summary"""
        print()
        print('=' * 80)
        print('DOCKER HEALTH CHECK SUMMARY')
        print('=' * 80)
        print(f"Overall Status: {self.overall_status}")
        print(f"Total Services: {len(self.results)}")
        print(f"Healthy: {sum(1 for r in self.results.values() if r['status'] == 'HEALTHY')}")
        print(f"Warnings: {self.warnings}")
        print(f"Failures: {self.failures}")
        print('=' * 80)
        print()

    async def run_all(self) -> Dict:
        """
        Run all health checks

        Returns:
            Dictionary with results and exit code
        """
        self.config.logger.info("Starting Docker Health Check System...")
        print()

        # Run all checks concurrently
        await asyncio.gather(
            self.check_postgresql(),
            self.check_redis(),
            self.check_nodejs(),
            self.check_python(),
            self.check_react(),
        )

        # Check system resources
        print()
        self.check_system_resources()

        # Generate and save report
        print()
        report = self.generate_report()
        await self.save_report(report)

        # Display summary
        self.display_summary()

        return {
            'status': self.overall_status,
            'code': 0 if self.overall_status == 'HEALTHY' else 1,
            'report': report,
        }


# ============================================================================
# Flask Integration
# ============================================================================

def create_flask_blueprint(checker: Optional[AsyncHealthChecker] = None):
    """
    Create Flask blueprint for health check endpoint

    Args:
        checker: AsyncHealthChecker instance

    Returns:
        Flask Blueprint
    """
    try:
        from flask import Blueprint, jsonify

        bp = Blueprint('health', __name__)
        checker_instance = checker or AsyncHealthChecker()

        @bp.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            report = checker_instance.generate_report()
            status_code = 200 if report['overall_status'] == 'HEALTHY' else 503
            return jsonify(report), status_code

        return bp
    except ImportError:
        raise ImportError("Flask not installed. Install with: pip install flask")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main execution"""
    try:
        checker = AsyncHealthChecker()
        result = await checker.run_all()
        sys.exit(result['code'])
    except KeyboardInterrupt:
        print("\nHealth check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(2)


if __name__ == '__main__':
    asyncio.run(main())
