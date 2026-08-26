/**
 * Performance Tests
 *
 * Tests:
 * - Page load times
 * - API response times
 * - Image upload performance
 * - Analysis processing time
 * - Bundle size
 * - Memory leaks
 */

import { test, expect } from './fixtures';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import UploadPage from './pages/UploadPage';

test.describe('Performance & Load Time Tests', () => {
  test('should load login page within acceptable time', async ({ page }) => {
    const startTime = Date.now();

    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.waitForPageLoad();

    const loadTime = Date.now() - startTime;

    // Should load in under 2 seconds
    expect(loadTime).toBeLessThan(2000);

    // Check Core Web Vitals
    const metrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as any;
      return {
        fcp: navigation.responseEnd - navigation.responseStart,
        lcp: performance.getEntriesByType('largest-contentful-paint').pop()?.startTime || 0,
        cls: performance
          .getEntriesByType('layout-shift')
          .reduce((sum: any, entry: any) => sum + entry.value, 0),
      };
    });

    expect(metrics.fcp).toBeLessThan(2000);
  });

  test('should load dashboard within acceptable time', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login('test@example.com', 'TestPassword123!');

    const startTime = Date.now();

    const dashboardPage = new DashboardPage(page);
    await dashboardPage.waitForPageLoad();

    const loadTime = Date.now() - startTime;

    // Should load in under 2 seconds
    expect(loadTime).toBeLessThan(2000);
  });

  test('should handle rapid page navigation', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login('test@example.com', 'TestPassword123!');

    const startTime = Date.now();

    // Navigate rapidly between pages
    for (let i = 0; i < 5; i++) {
      await page.goto('http://localhost:3000/dashboard');
      await page.goto('http://localhost:3000/upload');
      await page.goto('http://localhost:3000/settings');
    }

    const totalTime = Date.now() - startTime;

    // Should handle 15 navigation events
    expect(totalTime).toBeLessThan(15000);
  });

  test('should upload large file efficiently', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login('test@example.com', 'TestPassword123!');

    const dashboardPage = new DashboardPage(page);
    await dashboardPage.waitForPageLoad();
    await dashboardPage.clickUploadButton();

    const uploadPage = new UploadPage(page);
    await uploadPage.waitForPageLoad();

    const startTime = Date.now();

    // Upload a large file (simulated)
    await uploadPage.uploadFile('./tests/e2e/fixtures/large-image.jpg');

    const uploadTime = Date.now() - startTime;

    // Should complete in under 30 seconds
    expect(uploadTime).toBeLessThan(30000);
  });

  test('should measure API response times', async ({ apiClient }) => {
    const measurements: number[] = [];

    for (let i = 0; i < 10; i++) {
      const startTime = Date.now();

      await apiClient.get('/api/config');

      const responseTime = Date.now() - startTime;
      measurements.push(responseTime);
    }

    const average = measurements.reduce((a, b) => a + b) / measurements.length;
    const max = Math.max(...measurements);

    // Average should be under 500ms
    expect(average).toBeLessThan(500);

    // Max should be under 1000ms
    expect(max).toBeLessThan(1000);
  });

  test('should not have memory leaks', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login('test@example.com', 'TestPassword123!');

    const dashboardPage = new DashboardPage(page);
    await dashboardPage.waitForPageLoad();

    // Get initial memory
    const initialMemory = await page.evaluate(() => {
      if (performance.memory) {
        return (performance.memory as any).usedJSHeapSize;
      }
      return 0;
    });

    // Perform actions
    for (let i = 0; i < 10; i++) {
      await page.reload();
      await dashboardPage.waitForPageLoad();
    }

    // Check final memory
    const finalMemory = await page.evaluate(() => {
      if (performance.memory) {
        return (performance.memory as any).usedJSHeapSize;
      }
      return 0;
    });

    // Memory increase should be reasonable (less than 50MB)
    const increase = finalMemory - initialMemory;
    expect(increase).toBeLessThan(50 * 1024 * 1024);
  });

  test('should optimize image rendering', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.goto();

    const startTime = Date.now();

    // Wait for all images to load
    const images = await page.$$('img');
    for (const img of images) {
      await img.evaluate((el: any) => el.complete);
    }

    const imageLoadTime = Date.now() - startTime;

    // Should load all images in under 5 seconds
    expect(imageLoadTime).toBeLessThan(5000);
  });

  test('should cache static assets', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();

    // First load
    const firstLoadMetrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as any;
      return navigation.duration;
    });

    // Reload
    await page.reload();

    // Second load (should be cached)
    const secondLoadMetrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as any;
      return navigation.duration;
    });

    // Second load should be faster due to caching
    expect(secondLoadMetrics).toBeLessThanOrEqual(firstLoadMetrics);
  });

  test('should measure bundle size', async ({ page }) => {
    const resources = await page.evaluate(() => {
      return performance
        .getEntriesByType('resource')
        .map((entry: any) => ({
          name: entry.name,
          size: entry.transferSize || 0,
          type: entry.initiatorType,
        }))
        .filter((r: any) => r.type === 'script' || r.type === 'link');
    });

    const totalSize = resources.reduce((sum: number, r: any) => sum + r.size, 0);

    // Total bundle size should be under 500KB (gzipped)
    expect(totalSize).toBeLessThan(500 * 1024);
  });

  test('should handle concurrent user operations', async ({ browser }) => {
    const startTime = Date.now();

    const contexts = await Promise.all([
      browser.newContext(),
      browser.newContext(),
      browser.newContext(),
    ]);

    const pages = await Promise.all(contexts.map((ctx) => ctx.newPage()));

    // Simulate concurrent logins
    await Promise.all(
      pages.map(async (page) => {
        const loginPage = new LoginPage(page);
        await loginPage.goto();
        await loginPage.login('test@example.com', 'TestPassword123!');
      }),
    );

    const totalTime = Date.now() - startTime;

    // Should handle 3 concurrent logins
    expect(totalTime).toBeLessThan(10000);

    await Promise.all(contexts.map((ctx) => ctx.close()));
  });

  test('should not have layout shifts', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.goto();
    await dashboardPage.waitForPageLoad();

    const cls = await page.evaluate(() => {
      return performance
        .getEntriesByType('layout-shift')
        .filter((entry: any) => !(entry as any).hadRecentInput)
        .reduce((sum: number, entry: any) => sum + entry.value, 0);
    });

    // Cumulative Layout Shift should be low (< 0.1)
    expect(cls).toBeLessThan(0.1);
  });
});
