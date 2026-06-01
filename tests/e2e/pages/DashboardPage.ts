/**
 * Page Object: Dashboard Page
 *
 * Encapsulates dashboard interactions:
 * - Navigation
 * - File uploads
 * - Recent analyses
 * - User settings access
 */

import { Page } from '@playwright/test';
import { BasePage } from './fixtures';

export class DashboardPage extends BasePage {
  // Selectors
  private welcomeText = 'text=/Welcome/i';
  private uploadButton = 'button:has-text("Upload Image")';
  private recentAnalysesSection = '[data-testid="recent-analyses"]';
  private analysisCard = '[data-testid="analysis-card"]';
  private userMenu = '[data-testid="user-menu"]';
  private logoutButton = 'button:has-text("Logout")';
  private settingsLink = 'a:has-text("Settings")';
  private navBar = '[data-testid="navbar"]';
  private sidebar = '[data-testid="sidebar"]';
  private userAvatar = '[data-testid="user-avatar"]';

  constructor(page: Page) {
    super(page);
  }

  /**
   * Navigate to dashboard
   */
  async goto() {
    await super.goto('/');
    await this.waitForPageLoad();
  }

  /**
   * Wait for dashboard to load
   */
  async waitForPageLoad() {
    await this.page.waitForSelector(this.uploadButton, { timeout: 5000 });
    await this.page.waitForLoadState('networkidle');
  }

  /**
   * Check if dashboard is displayed
   */
  async isDashboardDisplayed(): Promise<boolean> {
    try {
      await this.page.waitForSelector(this.uploadButton, { timeout: 2000 });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Click upload button
   */
  async clickUploadButton() {
    await this.page.click(this.uploadButton);
  }

  /**
   * Get number of recent analyses
   */
  async getRecentAnalysesCount(): Promise<number> {
    const cards = await this.page.$$(this.analysisCard);
    return cards.length;
  }

  /**
   * Get analysis card data
   */
  async getAnalysisCardData(index: number) {
    const cards = await this.page.$$(this.analysisCard);
    if (index >= cards.length) throw new Error('Card not found');

    const card = cards[index];
    const title = await card.$eval('[data-testid="card-title"]', (el: any) => el.textContent);
    const status = await card.$eval('[data-testid="card-status"]', (el: any) => el.textContent);
    const date = await card.$eval('[data-testid="card-date"]', (el: any) => el.textContent);

    return { title, status, date };
  }

  /**
   * Click on analysis card
   */
  async clickAnalysisCard(index: number) {
    const cards = await this.page.$$(this.analysisCard);
    if (index >= cards.length) throw new Error('Card not found');
    await cards[index].click();
    await this.page.waitForNavigation();
  }

  /**
   * Open user menu
   */
  async openUserMenu() {
    await this.page.click(this.userMenu);
    await this.page.waitForSelector(this.logoutButton, { timeout: 3000 });
  }

  /**
   * Click logout
   */
  async clickLogout() {
    await this.openUserMenu();
    await this.page.click(this.logoutButton);
    await this.page.waitForNavigation();
  }

  /**
   * Navigate to settings
   */
  async navigateToSettings() {
    await this.openUserMenu();
    await this.page.click(this.settingsLink);
    await this.page.waitForNavigation();
  }

  /**
   * Check if user avatar is visible
   */
  async isUserAvatarVisible(): Promise<boolean> {
    return await this.page.isVisible(this.userAvatar);
  }

  /**
   * Check if navbar is visible
   */
  async isNavbarVisible(): Promise<boolean> {
    return await this.page.isVisible(this.navBar);
  }

  /**
   * Check if sidebar is visible (on desktop)
   */
  async isSidebarVisible(): Promise<boolean> {
    try {
      return await this.page.isVisible(this.sidebar);
    } catch {
      return false;
    }
  }

  /**
   * Get welcome text
   */
  async getWelcomeText(): Promise<string | null> {
    return await this.page.textContent(this.welcomeText);
  }

  /**
   * Wait for recent analyses to load
   */
  async waitForRecentAnalyses(timeout = 5000) {
    await this.page.waitForSelector(this.recentAnalysesSection, { timeout });
  }

  /**
   * Check if any analyses are displayed
   */
  async hasAnalyses(): Promise<boolean> {
    const count = await this.getRecentAnalysesCount();
    return count > 0;
  }

  /**
   * Export analysis data
   */
  async exportAnalysisData(format: 'csv' | 'json' = 'csv') {
    const exportButton = `button:has-text("Export as ${format.toUpperCase()}")`;
    await this.page.click(exportButton);
    // Wait for download
    await this.page.waitForEvent('download');
  }
}

export default DashboardPage;
