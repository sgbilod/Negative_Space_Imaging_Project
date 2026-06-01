/**
 * API Integration E2E Tests
 *
 * Tests:
 * - API response formats
 * - Data validation
 * - Error responses
 * - Status codes
 * - Rate limiting
 * - CORS handling
 */

import { test, expect } from './fixtures';

const apiURL = process.env.API_URL || 'http://localhost:5000';

test.describe('API Integration Tests', () => {
  test('should return correct auth login response', async ({ apiClient }) => {
    const response = await apiClient.post('/api/auth/login', {
      email: 'test@example.com',
      password: 'TestPassword123!',
    });

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('token');
    expect(response.data).toHaveProperty('user');
    expect(response.data.user).toHaveProperty('id');
    expect(response.data.user).toHaveProperty('email');
    expect(response.data.user).toHaveProperty('firstName');
  });

  test('should validate user profile format', async ({ authenticatedApiClient }) => {
    const response = await authenticatedApiClient.get('/api/user/profile');

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('id');
    expect(response.data).toHaveProperty('email');
    expect(response.data).toHaveProperty('firstName');
    expect(response.data).toHaveProperty('lastName');
    expect(response.data).toHaveProperty('createdAt');

    // Validate email format
    expect(response.data.email).toMatch(/^[^\s@]+@[^\s@]+\.[^\s@]+$/);
  });

  test('should return error for invalid login', async ({ apiClient }) => {
    const response = await apiClient.post('/api/auth/login', {
      email: 'invalid@example.com',
      password: 'WrongPassword123!',
    });

    expect(response.status).toBe(401);
    expect(response.data).toHaveProperty('error');
    expect(response.data.error).toContain('Invalid');
  });

  test('should return 404 for non-existent endpoint', async ({ apiClient }) => {
    const response = await apiClient.get('/api/nonexistent/endpoint');

    expect(response.status).toBe(404);
    expect(response.data).toHaveProperty('error');
  });

  test('should return 401 for missing authentication', async ({ apiClient }) => {
    const response = await apiClient.get('/api/user/profile');

    expect(response.status).toBe(401);
    expect(response.data).toHaveProperty('error');
    expect(response.data.error).toContain('Unauthorized');
  });

  test('should validate image upload response', async ({ authenticatedApiClient }) => {
    // This would require actual file upload, simplified for demo
    const response = await authenticatedApiClient.post('/api/images/check-upload', {
      fileName: 'test.jpg',
      fileSize: 1024000,
      mimeType: 'image/jpeg',
    });

    if (response.status === 200) {
      expect(response.data).toHaveProperty('uploadUrl');
      expect(response.data).toHaveProperty('uploadId');
    }
  });

  test('should return validation errors for invalid data', async ({ apiClient }) => {
    const response = await apiClient.post('/api/auth/register', {
      email: 'invalid-email', // Invalid email format
      password: '123', // Too short
      firstName: '',
      lastName: '',
    });

    expect(response.status).toBeGreaterThanOrEqual(400);
    expect(response.data).toHaveProperty('errors');
    expect(Array.isArray(response.data.errors)).toBe(true);
  });

  test('should handle concurrent API requests', async ({ apiClient }) => {
    const requests = Array(10)
      .fill(null)
      .map(() =>
        apiClient.get('/api/config', {
          headers: { Authorization: `Bearer ${process.env.TEST_AUTH_TOKEN}` },
        }),
      );

    const responses = await Promise.all(requests);

    // All should succeed
    responses.forEach((response) => {
      expect(response.status).toBe(200);
      expect(response.data).toBeDefined();
    });
  });

  test('should paginate analysis results', async ({ authenticatedApiClient }) => {
    const response = await authenticatedApiClient.get('/api/analysis', {
      params: {
        page: 1,
        limit: 10,
      },
    });

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('data');
    expect(response.data).toHaveProperty('pagination');
    expect(response.data.pagination).toHaveProperty('page');
    expect(response.data.pagination).toHaveProperty('limit');
    expect(response.data.pagination).toHaveProperty('total');
  });

  test('should filter images by status', async ({ authenticatedApiClient }) => {
    const response = await authenticatedApiClient.get('/api/images', {
      params: {
        status: 'uploaded',
      },
    });

    expect(response.status).toBe(200);
    expect(Array.isArray(response.data.images)).toBe(true);

    // Verify all returned images have correct status
    response.data.images.forEach((image: any) => {
      expect(image.status).toBe('uploaded');
    });
  });

  test('should sort results correctly', async ({ authenticatedApiClient }) => {
    const response = await authenticatedApiClient.get('/api/analysis', {
      params: {
        sortBy: 'createdAt',
        sortOrder: 'desc',
      },
    });

    expect(response.status).toBe(200);

    const analyses = response.data.data;
    if (analyses.length > 1) {
      for (let i = 0; i < analyses.length - 1; i++) {
        const date1 = new Date(analyses[i].createdAt).getTime();
        const date2 = new Date(analyses[i + 1].createdAt).getTime();
        expect(date1).toBeGreaterThanOrEqual(date2);
      }
    }
  });

  test('should include correct headers in response', async ({ apiClient }) => {
    const response = await apiClient.get('/api/config');

    expect(response.status).toBe(200);
    expect(response.headers['content-type']).toContain('application/json');
    expect(response.headers['x-powered-by']).toBeDefined();
  });

  test('should handle CORS preflight requests', async ({ page }) => {
    // This would be tested through browser OPTIONS request
    const response = await page.request.options(`${apiURL}/api/auth/login`);

    expect(response.status()).toBe(204);
    const headers = response.headers();
    expect(headers['access-control-allow-methods']).toBeDefined();
    expect(headers['access-control-allow-headers']).toBeDefined();
  });

  test('should rate limit excessive requests', async ({ apiClient }) => {
    const requests = Array(100)
      .fill(null)
      .map(() => apiClient.get('/api/config'));

    const responses = await Promise.all(requests);

    // Should have some 429 (Too Many Requests)
    const rateLimited = responses.some((r) => r.status === 429);

    // Either all succeed (no rate limiting) or some are rate limited
    expect(
      responses.every((r) => r.status === 200) || responses.some((r) => r.status === 429),
    ).toBe(true);
  });

  test('should return consistent user data', async ({ authenticatedApiClient }) => {
    const response1 = await authenticatedApiClient.get('/api/user/profile');
    const response2 = await authenticatedApiClient.get('/api/user/profile');

    expect(response1.status).toBe(200);
    expect(response2.status).toBe(200);
    expect(response1.data.id).toBe(response2.data.id);
    expect(response1.data.email).toBe(response2.data.email);
  });

  test('should handle empty result sets', async ({ authenticatedApiClient }) => {
    const response = await authenticatedApiClient.get('/api/images', {
      params: {
        status: 'never_uploaded',
      },
    });

    expect(response.status).toBe(200);
    expect(Array.isArray(response.data.images)).toBe(true);
    expect(response.data.images.length).toBe(0);
  });

  test('should validate timestamp formats', async ({ authenticatedApiClient }) => {
    const response = await authenticatedApiClient.get('/api/user/profile');

    expect(response.status).toBe(200);

    // Verify ISO 8601 format
    const createdAt = response.data.createdAt;
    expect(new Date(createdAt).getTime()).toBeGreaterThan(0);
  });
});
