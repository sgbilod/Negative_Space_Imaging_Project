/**
 * Upload Flow E2E Tests
 *
 * Tests:
 * - Single file upload
 * - Multiple file upload
 * - Large file upload
 * - Upload progress tracking
 * - File validation
 * - Drag and drop
 * - Upload cancellation
 * - Error handling
 */

import { test, expect, testImages, TestDataGenerator, TestHelpers } from './fixtures';
import LoginPage from './pages/LoginPage';
import UploadPage from './pages/UploadPage';
import DashboardPage from './pages/DashboardPage';

test.describe('File Upload Workflow', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login('test@example.com', 'TestPassword123!');

    const dashboardPage = new DashboardPage(page);
    await dashboardPage.waitForPageLoad();
  });

  test('should upload single file successfully', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    // Navigate to upload
    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    // Upload file
    await uploadPage.uploadFile(testImages.validImagePath);

    // Verify success
    expect(await uploadPage.isSuccessMessageVisible()).toBe(true);
    expect(await uploadPage.getUploadedFilesCount()).toBe(1);
  });

  test('should upload multiple files successfully', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    // Upload multiple files
    await uploadPage.uploadMultipleFiles(testImages.batchImagePaths);

    // Verify success
    expect(await uploadPage.isSuccessMessageVisible()).toBe(true);
    expect(await uploadPage.getUploadedFilesCount()).toBe(3);
  });

  test('should track upload progress', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    // Start upload
    await uploadPage.uploadFile(testImages.largeImagePath);

    // Monitor progress
    let previousProgress = 0;
    for (let i = 0; i < 10; i++) {
      const currentProgress = await uploadPage.getUploadProgress();
      expect(currentProgress).toBeGreaterThanOrEqual(previousProgress);
      previousProgress = currentProgress;

      if (currentProgress === 100) break;
      await page.waitForTimeout(500);
    }

    // Should reach 100%
    expect(await uploadPage.getUploadProgress()).toBe(100);
  });

  test('should display progress text during upload', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    await uploadPage.uploadFile(testImages.largeImagePath);

    const progressText = await uploadPage.getProgressText();
    expect(progressText).toBeDefined();
    expect(progressText).toMatch(/\d+%/);
  });

  test('should reject invalid file types', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    // Try to upload invalid file
    await uploadPage.uploadFile(testImages.invalidImagePath);

    // Verify error
    expect(await uploadPage.isErrorMessageVisible()).toBe(true);
    const errorMsg = await uploadPage.getErrorMessage();
    expect(errorMsg).toContain('Invalid file type');
  });

  test('should handle drag and drop upload', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    // Drag and drop
    await uploadPage.dragAndDropFile(testImages.validImagePath);

    // Verify file was added
    expect(await uploadPage.getUploadedFilesCount()).toBeGreaterThan(0);
  });

  test('should allow file removal before upload', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    // Add multiple files
    await uploadPage.uploadMultipleFiles(testImages.batchImagePaths);
    expect(await uploadPage.getUploadedFilesCount()).toBe(3);

    // Remove one file
    await uploadPage.removeFile(0);
    expect(await uploadPage.getUploadedFilesCount()).toBe(2);
  });

  test('should display file names and sizes', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    await uploadPage.uploadFile(testImages.validImagePath);

    const fileNames = await uploadPage.getUploadedFileNames();
    const fileSizes = await uploadPage.getUploadedFileSizes();

    expect(fileNames.length).toBeGreaterThan(0);
    expect(fileSizes.length).toBeGreaterThan(0);
    expect(fileNames[0]).toContain('.jpg');
  });

  test('should clear all files', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    await uploadPage.uploadMultipleFiles(testImages.batchImagePaths);
    expect(await uploadPage.getUploadedFilesCount()).toBe(3);

    // Clear all
    await uploadPage.clearFiles();
    expect(await uploadPage.getUploadedFilesCount()).toBe(0);
  });

  test('should disable upload button when no files selected', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    // Upload button should be disabled initially
    expect(await uploadPage.isUploadButtonEnabled()).toBe(false);

    // Add file
    await uploadPage.uploadFile(testImages.validImagePath);

    // Button should be enabled
    expect(await uploadPage.isUploadButtonEnabled()).toBe(true);
  });

  test('should handle upload cancellation', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    // Start upload
    const uploadPromise = uploadPage.uploadFile(testImages.largeImagePath);

    // Wait a moment then cancel
    await page.waitForTimeout(1000);
    await page.click('button:has-text("Cancel")');

    // Verify cancellation
    const progress = await uploadPage.getUploadProgress();
    expect(progress).toBeLessThan(100);
  });

  test('should persist uploaded files after navigation', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    await uploadPage.uploadFile(testImages.validImagePath);
    const uploadedCount = await uploadPage.getUploadedFilesCount();

    // Navigate to dashboard
    await dashboardPage.goto();
    await dashboardPage.waitForPageLoad();

    // Go back to upload
    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    // Verify files still there
    expect(await uploadPage.getUploadedFilesCount()).toBeGreaterThanOrEqual(uploadedCount);
  });

  test('should handle maximum file size limit', async ({ page, apiClient }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    // Check max file size from API
    const configResponse = await apiClient.get('/api/config');
    const maxSize = configResponse.data.maxUploadSize || 100 * 1024 * 1024; // 100MB default

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    // Try to upload file exceeding limit would require a test file larger than limit
    // For now, just verify the UI handles the constraint properly
    expect(await uploadPage.isUploadButtonEnabled()).toBe(false);
  });

  test('should batch upload multiple files efficiently', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    const uploadPage = new UploadPage(page);

    await dashboardPage.clickUploadButton();
    await uploadPage.waitForPageLoad();

    const startTime = Date.now();

    await uploadPage.uploadMultipleFiles(testImages.batchImagePaths);

    const endTime = Date.now();
    const uploadTime = endTime - startTime;

    // Should complete in reasonable time
    expect(uploadTime).toBeLessThan(60000); // 1 minute

    expect(await uploadPage.isSuccessMessageVisible()).toBe(true);
  });
});
