
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
let errorRate = new Rate('errors');
let responseTime = new Trend('response_time');

// Test configuration
export let options = {
  stages: [
    { duration: '1m', target: 100 },   // Ramp up to 100 users
    { duration: '2m', target: 300 },   // Ramp up to 300 users
    { duration: '3m', target: 500 },   // Ramp up to 500 users
    { duration: '5m', target: 750 },   // Ramp up to 750 users
    { duration: '10m', target: 1000 }, // Ramp up to 1000 users
    { duration: '30m', target: 1000 }, // Sustained load at 1000 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests should be below 500ms
    http_req_failed: ['rate<0.1'],    // Error rate should be below 10%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_TOKEN = __ENV.API_TOKEN || 'test-token';

export default function () {
  let headers = {
    'Authorization': `Bearer ${API_TOKEN}`,
    'Content-Type': 'application/json',
  };

  // Health check - 20% of requests
  if (Math.random() < 0.2) {
    let response = http.get(`${BASE_URL}/api/health`, { headers });
    check(response, { 'health status is 200': (r) => r.status === 200 });
    errorRate.add(response.status !== 200);
    responseTime.add(response.timings.duration);
  }

  // Get images - 30% of requests
  else if (Math.random() < 0.3) {
    let response = http.get(`${BASE_URL}/api/images?limit=10&offset=${Math.floor(Math.random() * 100)}`, { headers });
    check(response, { 'images status is 200': (r) => r.status === 200 });
    errorRate.add(response.status !== 200);
    responseTime.add(response.timings.duration);
  }

  // Process image - 25% of requests
  else if (Math.random() < 0.25) {
    let payload = JSON.stringify({
      image_url: `https://example.com/image_${Math.floor(Math.random() * 1000) + 1}.jpg`,
      processing_options: {
        enhancement: true,
        negative_space_detection: true,
        quality: 'high'
      }
    });
    let response = http.post(`${BASE_URL}/api/process`, payload, { headers });
    check(response, { 'process status is 200': (r) => r.status === 200 });
    errorRate.add(response.status !== 200);
    responseTime.add(response.timings.duration);
  }

  // Analytics - 15% of requests
  else if (Math.random() < 0.15) {
    let response = http.get(`${BASE_URL}/api/analytics?period=24h`, { headers });
    check(response, { 'analytics status is 200': (r) => r.status === 200 });
    errorRate.add(response.status !== 200);
    responseTime.add(response.timings.duration);
  }

  // Auth login - 10% of requests
  else {
    let payload = JSON.stringify({
      username: `user_${Math.floor(Math.random() * 100) + 1}`,
      password: 'test_password'
    });
    let response = http.post(`${BASE_URL}/api/auth/login`, payload, { headers });
    check(response, { 'login status is 200': (r) => r.status === 200 });
    errorRate.add(response.status !== 200);
    responseTime.add(response.timings.duration);
  }

  sleep(Math.random() * 2 + 1); // Random sleep between 1-3 seconds
}

export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'load_test_results.json': JSON.stringify(data),
  };
}
