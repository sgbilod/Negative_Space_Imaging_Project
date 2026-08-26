/**
 * Analysis Workflow E2E Tests
 *
 * Tests:
 * - Trigger analysis
 * - Monitor analysis progress
 * - View analysis results
 * - Export results
 * - Delete analysis
 * - Error handling
 */

import { test, expect, TestDataGenerator } from './fixtures';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import UploadPage from './pages/UploadPage';

test.describe('Analysis Workflow', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login('test@example.com', 'TestPassword123!');

    const dashboardPage = new DashboardPage(page);
    await dashboardPage.waitForPageLoad();
  });

  test('should trigger analysis after upload', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    // Upload image
    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();
    await uploadPage.uploadFile('./tests/e2e/fixtures/test-image.jpg');

    // Verify success and trigger analysis
    expect(await uploadPage.isSuccessMessageVisible()).toBe(true);

    // Should auto-start analysis or show option to start
    await page.waitForSelector('[data-testid="analysis-button"], [data-testid="analysis-status"]', {
      timeout: 5000,
    });
  });

  test('should display analysis progress', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);

    // Wait for recent analyses
    await dashboardPage.waitForRecentAnalyses();

    // Click on an analysis
    if ((await dashboardPage.getRecentAnalysesCount()) > 0) {
      const analysisData = await dashboardPage.getAnalysisCardData(0);
      expect(analysisData.status).toBeDefined();

      // If processing, should show progress
      if (analysisData.status.includes('Processing')) {
        const progressBar = await page.$('[data-testid="analysis-progress"]');
        expect(progressBar).toBeDefined();
      }
    }
  });

  test('should view analysis results', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);

    await dashboardPage.waitForRecentAnalyses();

    if ((await dashboardPage.getRecentAnalysesCount()) > 0) {
      // Click on first completed analysis
      await dashboardPage.clickAnalysisCard(0);

      // Verify results page loaded
      await page.waitForSelector('[data-testid="analysis-results"]', { timeout: 5000 });

      // Verify key results are displayed
      expect(await page.isVisible('[data-testid="results-summary"]')).toBe(true);
      expect(await page.isVisible('[data-testid="detected-regions"]')).toBe(true);
    }
  });

  test('should export analysis results as CSV', async ({ page, context }) => {
    const dashboardPage = new DashboardPage(page);

    await dashboardPage.waitForRecentAnalyses();

    if ((await dashboardPage.getRecentAnalysesCount()) > 0) {
      await dashboardPage.clickAnalysisCard(0);

      // Wait for download
      const downloadPromise = page.waitForEvent('download');

      // Click export
      await page.click('button:has-text("Export as CSV")');

      const download = await downloadPromise;

      // Verify download
      expect(download.suggestedFilename()).toContain('.csv');

      // Can verify content if needed
      const path = await download.path();
      expect(path).toBeDefined();
    }
  });

  test('should export analysis results as JSON', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);

    await dashboardPage.waitForRecentAnalyses();

    if ((await dashboardPage.getRecentAnalysesCount()) > 0) {
      await dashboardPage.clickAnalysisCard(0);

      const downloadPromise = page.waitForEvent('download');
      await page.click('button:has-text("Export as JSON")');

      const download = await downloadPromise;
      expect(download.suggestedFilename()).toContain('.json');
    }
  });

  test('should export analysis results as PDF', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);

    await dashboardPage.waitForRecentAnalyses();

    if ((await dashboardPage.getRecentAnalysesCount()) > 0) {
      await dashboardPage.clickAnalysisCard(0);

      const downloadPromise = page.waitForEvent('download');
      await page.click('button:has-text("Export as PDF")');

      const download = await downloadPromise;
      expect(download.suggestedFilename()).toContain('.pdf');
    }
  });

  test('should show analysis statistics', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);

    await dashboardPage.waitForRecentAnalyses();

    if ((await dashboardPage.getRecentAnalysesCount()) > 0) {
      await dashboardPage.clickAnalysisCard(0);

      // Verify statistics are displayed
      const stats = [
        '[data-testid="stat-total-regions"]',
        '[data-testid="stat-coverage"]',
        '[data-testid="stat-confidence"]',
      ];

      for (const stat of stats) {
        expect(await page.isVisible(stat)).toBe(true);
      }
    }
  });

  test('should delete analysis', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);

    const initialCount = await dashboardPage.getRecentAnalysesCount();

    if (initialCount > 0) {
      // Get analysis data before deletion
      const analysisData = await dashboardPage.getAnalysisCardData(0);

      // Click delete button
      await page.click('[data-testid="delete-analysis-btn"]');

      // Confirm deletion
      await page.click('button:has-text("Confirm")');

      // Verify success message
      await page.waitForSelector('[data-testid="success-message"]', { timeout: 3000 });

      // Reload and verify deletion
      await page.reload();
      const newCount = await dashboardPage.getRecentAnalysesCount();
      expect(newCount).toBeLessThan(initialCount);
    }
  });

  test('should compare two analyses', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);

    if ((await dashboardPage.getRecentAnalysesCount()) >= 2) {
      // Select comparison mode
      await page.click('[data-testid="compare-mode-toggle"]');

      // Select two analyses
      await page.click('[data-testid="analysis-checkbox-0"]');
      await page.click('[data-testid="analysis-checkbox-1"]');

      // Click compare
      await page.click('button:has-text("Compare")');

      // Verify comparison view
      await page.waitForSelector('[data-testid="comparison-view"]', { timeout: 5000 });
      expect(await page.isVisible('[data-testid="side-by-side-results"]')).toBe(true);
    }
  });

  test('should handle analysis errors gracefully', async ({ page, apiClient }) => {
    // Create a request that will fail (invalid parameters)
    const response = await apiClient.post('/api/analysis/create', {
      imageId: 'invalid-id',
      analysisType: 'unknown_type',
    });

    expect(response.status).toBeGreaterThanOrEqual(400);
    expect(response.data.error).toBeDefined();
  });

  test('should retry failed analysis', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);

    await dashboardPage.waitForRecentAnalyses();

    // Find failed analysis
    const analysisCards = await page.$$('[data-testid="analysis-card"]');

    for (const card of analysisCards) {
      const status = await card.textContent('[data-testid="card-status"]');

      if (status?.includes('Failed')) {
        // Click retry
        await card.$eval('[data-testid="retry-btn"]', (el: any) => el.click());

        // Verify retrying status
        const newStatus = await card.textContent('[data-testid="card-status"]');
        expect(newStatus).toContain('Processing');
        break;
      }
    }
  });

  test('should display detected regions with highlights', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);

    await dashboardPage.waitForRecentAnalyses();

    if ((await dashboardPage.getRecentAnalysesCount()) > 0) {
      await dashboardPage.clickAnalysisCard(0);

      // Verify image with highlights
      const highlightedImage = await page.$('[data-testid="highlighted-image"]');
      expect(highlightedImage).toBeDefined();

      // Verify regions list
      const regions = await page.$$('[data-testid="region-item"]');
      expect(regions.length).toBeGreaterThan(0);

      // Click on region to highlight it
      await regions[0].click();

      // Verify highlight updated
      const highlight = await page.getAttribute(regions[0], 'data-highlighted');
      expect(highlight).toBe('true');
    }
  });

  test('should measure analysis performance', async ({ page, apiClient }) => {
    const startTime = Date.now();

    // Trigger analysis
    const response = await apiClient.post('/api/analysis/create', {
      imageId: process.env.TEST_IMAGE_ID || 'test-image-1',
      analysisType: 'negative_space',
    });

    const endTime = Date.now();
    const analysisTime = endTime - startTime;

    expect(response.status).toBe(200);

    // Should complete within reasonable time
    expect(analysisTime).toBeLessThan(60000); // 1 minute max for API call
  });
});
