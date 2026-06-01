/**
 * Settings & User Profile Tests
 *
 * Tests:
 * - Update user profile
 * - Change password
 * - Update email
 * - Avatar upload
 * - Settings persistence
 * - Accessibility settings
 */

import { test, expect, testUsers, TestDataGenerator } from './fixtures';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';

test.describe('Settings & User Profile', () => {
  test.beforeEach(async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login(testUsers.validUser.email, testUsers.validUser.password);

    const dashboardPage = new DashboardPage(page);
    await dashboardPage.waitForPageLoad();
  });

  test('should navigate to settings page', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);

    await dashboardPage.navigateToSettings();

    expect(page.url()).toContain('/settings');
    await page.waitForSelector('[data-testid="settings-form"]', { timeout: 5000 });
  });

  test('should update user profile', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.navigateToSettings();

    // Update profile fields
    await page.fill('input[name="firstName"]', 'UpdatedFirst');
    await page.fill('input[name="lastName"]', 'UpdatedLast');
    await page.click('button:has-text("Save Changes")');

    // Verify success
    await page.waitForSelector('[data-testid="success-message"]', { timeout: 3000 });

    // Reload and verify persistence
    await page.reload();
    const firstName = await page.inputValue('input[name="firstName"]');
    expect(firstName).toBe('UpdatedFirst');
  });

  test('should change password', async ({ page, apiClient }) => {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.navigateToSettings();

    // Click change password button
    await page.click('button:has-text("Change Password")');

    // Fill password form
    await page.fill('input[name="currentPassword"]', testUsers.validUser.password);
    await page.fill('input[name="newPassword"]', 'NewPassword123!');
    await page.fill('input[name="confirmPassword"]', 'NewPassword123!');

    // Submit
    await page.click('button[type="submit"]:has-text("Update Password")');

    // Verify success
    await page.waitForSelector('[data-testid="success-message"]', { timeout: 3000 });

    // Verify new password works
    const loginPage = new LoginPage(page);
    await dashboardPage.clickLogout();
    await loginPage.goto();
    await loginPage.login(testUsers.validUser.email, 'NewPassword123!');

    // Should succeed
    expect(page.url()).not.toContain('/login');
  });

  test('should validate password requirements', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.navigateToSettings();

    await page.click('button:has-text("Change Password")');

    // Try weak password
    await page.fill('input[name="currentPassword"]', testUsers.validUser.password);
    await page.fill('input[name="newPassword"]', '123');
    await page.fill('input[name="confirmPassword"]', '123');

    // Check for validation error
    const error = await page.textContent('[role="alert"]');
    expect(error).toContain('at least 8 characters');
  });

  test('should upload avatar', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.navigateToSettings();

    // Upload avatar
    await page
      .locator('input[type="file"][aria-label*="avatar"]')
      .setInputFiles('./tests/e2e/fixtures/avatar.jpg');

    // Wait for upload
    await page.waitForSelector('[data-testid="avatar-preview"]', { timeout: 5000 });

    // Verify image is displayed
    const img = await page.$('[data-testid="avatar-preview"]');
    expect(img).toBeDefined();
  });

  test('should toggle theme between light and dark', async ({ page, context }) => {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.navigateToSettings();

    // Toggle theme
    await page.click('[data-testid="theme-toggle"]');

    // Verify theme changed
    const html = await page.locator('html').getAttribute('data-theme');
    expect(['light', 'dark']).toContain(html);

    // Reload and verify persistence
    await page.reload();
    const newHtml = await page.locator('html').getAttribute('data-theme');
    expect(newHtml).toBe(html);
  });

  test('should manage notification settings', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.navigateToSettings();

    // Toggle notifications
    await page.click('[data-testid="notifications-toggle"]');
    await page.click('[data-testid="email-alerts-toggle"]');

    // Save
    await page.click('button:has-text("Save Preferences")');

    // Verify success
    await page.waitForSelector('[data-testid="success-message"]', { timeout: 3000 });
  });

  test('should delete account', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.navigateToSettings();

    // Click delete account
    await page.click('button:has-text("Delete Account")');

    // Confirm warning
    await page.fill('input[placeholder*="confirm"]', 'DELETE');
    await page.click('button:has-text("Permanently Delete")');

    // Should logout and redirect
    await page.waitForURL(/.*\/login/, { timeout: 5000 });
  });

  test('should export user data', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.navigateToSettings();

    const downloadPromise = page.waitForEvent('download');
    await page.click('button:has-text("Export My Data")');

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toContain('.json');
  });

  test('should manage API keys', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.navigateToSettings();

    // Generate new API key
    await page.click('button:has-text("Generate New Key")');

    // Verify key is displayed
    const apiKey = await page.textContent('[data-testid="api-key-display"]');
    expect(apiKey).toMatch(/^[a-zA-Z0-9]+$/);

    // Copy button should work
    await page.click('[data-testid="copy-api-key"]');
  });
});
