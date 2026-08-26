/**
 * Authentication Flow E2E Tests
 *
 * Tests:
 * - User registration
 * - Email verification
 * - Login with valid credentials
 * - Login with invalid credentials
 * - Session persistence
 * - Logout
 * - Token refresh
 * - Forgot password flow
 */

import { test, expect, testUsers, testImages, TestDataGenerator, TestHelpers } from './fixtures';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';

test.describe('Authentication Flows', () => {
  test('should register a new user successfully', async ({ page, apiClient }) => {
    const registerPage = new RegisterPage(page);
    const newUser = TestDataGenerator.generateUser();

    await registerPage.goto();
    await registerPage.fillRegistrationForm(newUser);
    await registerPage.clickRegisterButton();

    // Verify success message
    const successMsg = await registerPage.getSuccessMessage();
    expect(successMsg).toContain('Registration successful');

    // Verify email verification email was sent (in test env)
    const response = await apiClient.post('/api/test/get-emails', {
      to: newUser.email,
      type: 'verification',
    });
    expect(response.status).toBe(200);
    expect(response.data.emails.length).toBeGreaterThan(0);
  });

  test('should prevent registration with existing email', async ({ page }) => {
    const registerPage = new RegisterPage(page);

    await registerPage.goto();
    await registerPage.fillRegistrationForm(testUsers.validUser);
    await registerPage.clickRegisterButton();

    // Verify error message
    const errorMsg = await registerPage.getErrorMessage();
    expect(errorMsg).toContain('User already exists');
  });

  test('should validate registration form', async ({ page }) => {
    const registerPage = new RegisterPage(page);

    await registerPage.goto();

    // Try to submit empty form
    await registerPage.clickRegisterButton();

    // Check for validation errors
    const errors = await registerPage.getValidationErrors();
    expect(errors.length).toBeGreaterThan(0);
  });

  test('should login with valid credentials', async ({ page, context }) => {
    const loginPage = new LoginPage(page);
    const dashboardPage = new DashboardPage(page);

    await loginPage.goto();
    await loginPage.login(testUsers.validUser.email, testUsers.validUser.password);

    // Verify redirect to dashboard
    await dashboardPage.waitForPageLoad();
    expect(page.url()).toContain('/');

    // Verify auth token is stored
    const cookies = await context.cookies();
    const authCookie = cookies.find((c) => c.name === 'auth_token');
    expect(authCookie).toBeDefined();
  });

  test('should reject login with invalid password', async ({ page }) => {
    const loginPage = new LoginPage(page);

    await loginPage.goto();
    await loginPage.login(testUsers.validUser.email, 'WrongPassword123!');

    // Verify error message
    const errorMsg = await loginPage.getErrorMessage();
    expect(errorMsg).toContain('Invalid email or password');

    // Verify still on login page
    expect(page.url()).toContain('/login');
  });

  test('should reject login with non-existent user', async ({ page }) => {
    const loginPage = new LoginPage(page);

    await loginPage.goto();
    await loginPage.login('nonexistent@example.com', 'Password123!');

    const errorMsg = await loginPage.getErrorMessage();
    expect(errorMsg).toContain('Invalid email or password');
  });

  test('should persist session after page reload', async ({ page, context }) => {
    const loginPage = new LoginPage(page);
    const dashboardPage = new DashboardPage(page);

    // Login
    await loginPage.goto();
    await loginPage.login(testUsers.validUser.email, testUsers.validUser.password);
    await dashboardPage.waitForPageLoad();

    // Reload page
    await page.reload();
    await dashboardPage.waitForPageLoad();

    // Verify still logged in
    expect(page.url()).not.toContain('/login');
    expect(await dashboardPage.isDashboardDisplayed()).toBe(true);
  });

  test('should logout successfully', async ({ page, context }) => {
    const loginPage = new LoginPage(page);
    const dashboardPage = new DashboardPage(page);

    // Login first
    await loginPage.goto();
    await loginPage.login(testUsers.validUser.email, testUsers.validUser.password);
    await dashboardPage.waitForPageLoad();

    // Logout
    await dashboardPage.clickLogout();
    await loginPage.waitForPageLoad();

    // Verify redirected to login
    expect(page.url()).toContain('/login');

    // Verify auth cookie is cleared
    const cookies = await context.cookies();
    const authCookie = cookies.find((c) => c.name === 'auth_token');
    expect(authCookie).toBeUndefined();
  });

  test('should handle token refresh', async ({ page, apiClient, authenticatedApiClient }) => {
    const token = process.env.TEST_AUTH_TOKEN;
    expect(token).toBeDefined();

    // Use token to make authenticated request
    const response = await authenticatedApiClient.get('/api/user/profile');
    expect(response.status).toBe(200);

    // Simulate token expiration and refresh
    const refreshResponse = await apiClient.post('/api/auth/refresh', {
      refreshToken: process.env.TEST_REFRESH_TOKEN,
    });

    expect(refreshResponse.status).toBe(200);
    expect(refreshResponse.data.token).toBeDefined();
  });

  test('should protect routes without authentication', async ({ page }) => {
    // Try to access dashboard without logging in
    await page.goto('http://localhost:3000/');

    // Should redirect to login
    await page.waitForURL(/.*\/login/);
    expect(page.url()).toContain('/login');
  });

  test('should handle remember me functionality', async ({ page, context }) => {
    const loginPage = new LoginPage(page);

    await loginPage.goto();
    await loginPage.toggleRememberMe();
    expect(await loginPage.isRememberMeChecked()).toBe(true);

    await loginPage.login(testUsers.validUser.email, testUsers.validUser.password);

    // Verify remember me token is stored
    const cookies = await context.cookies();
    const rememberMeCookie = cookies.find((c) => c.name === 'remember_me');
    expect(rememberMeCookie).toBeDefined();
  });

  test('should initiate forgot password flow', async ({ page, apiClient }) => {
    const loginPage = new LoginPage(page);

    await loginPage.goto();
    await loginPage.clickForgotPasswordLink();

    // Fill forgot password form
    await page.fill('input[type="email"]', testUsers.validUser.email);
    await page.click('button:has-text("Send Reset Link")');

    // Verify success message
    await page.waitForSelector('text=/reset link sent/i');

    // Verify reset email was sent (in test env)
    const emailResponse = await apiClient.post('/api/test/get-emails', {
      to: testUsers.validUser.email,
      type: 'reset_password',
    });
    expect(emailResponse.data.emails.length).toBeGreaterThan(0);
  });

  test('should handle concurrent login attempts', async ({ browser }) => {
    const context1 = await browser.newContext();
    const context2 = await browser.newContext();

    const page1 = await context1.newPage();
    const page2 = await context2.newPage();

    const loginPage1 = new LoginPage(page1);
    const loginPage2 = new LoginPage(page2);

    await loginPage1.goto();
    await loginPage2.goto();

    // Try to login from both pages simultaneously
    await Promise.all([
      loginPage1.login(testUsers.validUser.email, testUsers.validUser.password),
      loginPage2.login(testUsers.validUser.email, testUsers.validUser.password),
    ]);

    // Both should succeed
    expect(page1.url()).not.toContain('/login');
    expect(page2.url()).not.toContain('/login');

    await context1.close();
    await context2.close();
  });
});
