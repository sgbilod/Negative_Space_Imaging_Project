/**
 * Page Object: Login Page
 *
 * Encapsulates all interactions with the login page
 * to provide a clean, reusable interface for tests
 */

import { Page } from '@playwright/test';
import { BasePage } from './fixtures';

export class LoginPage extends BasePage {
  // Selectors
  private emailInput = 'input[type="email"]';
  private passwordInput = 'input[type="password"]';
  private loginButton = 'button[type="submit"]';
  private registerLink = 'a:has-text("Register")';
  private forgotPasswordLink = 'a:has-text("Forgot Password")';
  private errorMessage = '[role="alert"]';
  private rememberMeCheckbox = 'input[type="checkbox"]';
  private loadingSpinner = '[data-testid="loading-spinner"]';
  private successMessage = '.Mui-success';

  constructor(page: Page) {
    super(page);
  }

  /**
   * Navigate to login page
   */
  async goto() {
    await super.goto('/login');
    await this.waitForPageLoad();
  }

  /**
   * Wait for login page to fully load
   */
  async waitForPageLoad() {
    await this.page.waitForSelector(this.emailInput, { timeout: 5000 });
    // Wait for form to be interactive
    await this.page.waitForLoadState('networkidle');
  }

  /**
   * Enter email address
   */
  async enterEmail(email: string) {
    await this.page.fill(this.emailInput, email);
  }

  /**
   * Enter password
   */
  async enterPassword(password: string) {
    await this.page.fill(this.passwordInput, password);
  }

  /**
   * Click login button
   */
  async clickLogin() {
    await this.page.click(this.loginButton);
  }

  /**
   * Login with email and password
   */
  async login(email: string, password: string) {
    await this.enterEmail(email);
    await this.enterPassword(password);
    await this.clickLogin();
    // Wait for navigation or error message
    await Promise.race([
      this.page.waitForNavigation().catch(() => {}),
      this.page.waitForSelector(this.errorMessage).catch(() => {}),
    ]);
  }

  /**
   * Get error message text
   */
  async getErrorMessage(): Promise<string> {
    const element = await this.page.$(this.errorMessage);
    if (!element) return '';
    return await element.textContent();
  }

  /**
   * Check if error message is displayed
   */
  async isErrorMessageVisible(): Promise<boolean> {
    try {
      await this.page.waitForSelector(this.errorMessage, { timeout: 2000 });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Toggle remember me checkbox
   */
  async toggleRememberMe() {
    await this.page.click(this.rememberMeCheckbox);
  }

  /**
   * Check if remember me is checked
   */
  async isRememberMeChecked(): Promise<boolean> {
    return await this.page.isChecked(this.rememberMeCheckbox);
  }

  /**
   * Navigate to register page
   */
  async clickRegisterLink() {
    await this.page.click(this.registerLink);
    await this.page.waitForNavigation();
  }

  /**
   * Navigate to forgot password page
   */
  async clickForgotPasswordLink() {
    await this.page.click(this.forgotPasswordLink);
    await this.page.waitForNavigation();
  }

  /**
   * Check if login button is enabled
   */
  async isLoginButtonEnabled(): Promise<boolean> {
    return !(await this.page.isDisabled(this.loginButton));
  }

  /**
   * Check if loading spinner is visible
   */
  async isLoadingSpinnerVisible(): Promise<boolean> {
    try {
      await this.page.waitForSelector(this.loadingSpinner, { timeout: 1000 });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get all input field values
   */
  async getFormValues() {
    return {
      email: await this.page.inputValue(this.emailInput),
      password: await this.page.inputValue(this.passwordInput),
      rememberMe: await this.isRememberMeChecked(),
    };
  }

  /**
   * Clear all form fields
   */
  async clearForm() {
    await this.page.fill(this.emailInput, '');
    await this.page.fill(this.passwordInput, '');
  }
}

export default LoginPage;
