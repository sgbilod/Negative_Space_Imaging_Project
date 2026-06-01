/**
 * Page Object: Upload Page
 *
 * Handles file upload interactions:
 * - Single/multiple file uploads
 * - Progress tracking
 * - Upload status
 * - File validation
 */

import { Page } from '@playwright/test';
import { BasePage } from './fixtures';

export class UploadPage extends BasePage {
  // Selectors
  private fileInput = 'input[type="file"]';
  private dropZone = '[data-testid="drop-zone"]';
  private uploadButton = 'button:has-text("Upload")';
  private progressBar = '[data-testid="progress-bar"]';
  private progressText = '[data-testid="progress-text"]';
  private uploadedFileList = '[data-testid="uploaded-files"]';
  private fileItem = '[data-testid="file-item"]';
  private removeFileButton = 'button:has-text("Remove")';
  private successMessage = '[data-testid="success-message"]';
  private errorMessage = '[data-testid="error-message"]';
  private fileNameDisplay = '[data-testid="file-name"]';
  private fileSizeDisplay = '[data-testid="file-size"]';

  constructor(page: Page) {
    super(page);
  }

  /**
   * Navigate to upload page
   */
  async goto() {
    await super.goto('/upload');
    await this.waitForPageLoad();
  }

  /**
   * Wait for upload page to load
   */
  async waitForPageLoad() {
    await this.page.waitForSelector(this.dropZone, { timeout: 5000 });
    await this.page.waitForLoadState('networkidle');
  }

  /**
   * Upload single file
   */
  async uploadFile(filePath: string) {
    await this.page.locator(this.fileInput).setInputFiles(filePath);
    await this.page.click(this.uploadButton);
    await this.waitForUploadComplete();
  }

  /**
   * Upload multiple files
   */
  async uploadMultipleFiles(filePaths: string[]) {
    await this.page.locator(this.fileInput).setInputFiles(filePaths);
    await this.page.click(this.uploadButton);
    await this.waitForUploadComplete();
  }

  /**
   * Drag and drop file
   */
  async dragAndDropFile(filePath: string) {
    // In Playwright, we simulate drag-drop through the file input
    await this.page.locator(this.fileInput).setInputFiles(filePath);
    // Trigger drop event
    await this.page.dispatchEvent(this.dropZone, 'drop');
  }

  /**
   * Wait for upload to complete
   */
  async waitForUploadComplete(timeout = 30000) {
    await Promise.race([
      this.page.waitForSelector(this.successMessage, { timeout }).catch(() => {}),
      this.page.waitForFunction(
        () => {
          const progressBar = document.querySelector('[data-testid="progress-bar"]');
          return progressBar ? progressBar.getAttribute('aria-valuenow') === '100' : false;
        },
        { timeout },
      ),
    ]);
  }

  /**
   * Get upload progress percentage
   */
  async getUploadProgress(): Promise<number> {
    const value = await this.page.getAttribute(this.progressBar, 'aria-valuenow');
    return value ? parseInt(value) : 0;
  }

  /**
   * Get progress text
   */
  async getProgressText(): Promise<string | null> {
    return await this.page.textContent(this.progressText);
  }

  /**
   * Get uploaded files count
   */
  async getUploadedFilesCount(): Promise<number> {
    const items = await this.page.$$(this.fileItem);
    return items.length;
  }

  /**
   * Get file names
   */
  async getUploadedFileNames(): Promise<string[]> {
    const items = await this.page.$$(this.fileItem);
    const names: string[] = [];

    for (const item of items) {
      const name = await item.$eval(this.fileNameDisplay, (el: any) => el.textContent);
      if (name) names.push(name);
    }

    return names;
  }

  /**
   * Get file sizes
   */
  async getUploadedFileSizes(): Promise<string[]> {
    const items = await this.page.$$(this.fileItem);
    const sizes: string[] = [];

    for (const item of items) {
      const size = await item.$eval(this.fileSizeDisplay, (el: any) => el.textContent);
      if (size) sizes.push(size);
    }

    return sizes;
  }

  /**
   * Remove uploaded file
   */
  async removeFile(index: number) {
    const items = await this.page.$$(this.fileItem);
    if (index >= items.length) throw new Error('File not found');

    const item = items[index];
    const removeBtn = await item.$('[data-testid="remove-file-btn"]');
    if (removeBtn) await removeBtn.click();
  }

  /**
   * Check if success message is displayed
   */
  async isSuccessMessageVisible(): Promise<boolean> {
    try {
      await this.page.waitForSelector(this.successMessage, { timeout: 2000 });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get success message text
   */
  async getSuccessMessage(): Promise<string | null> {
    return await this.page.textContent(this.successMessage);
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
   * Get error message text
   */
  async getErrorMessage(): Promise<string | null> {
    return await this.page.textContent(this.errorMessage);
  }

  /**
   * Check if upload button is enabled
   */
  async isUploadButtonEnabled(): Promise<boolean> {
    return !(await this.page.isDisabled(this.uploadButton));
  }

  /**
   * Clear all selected files
   */
  async clearFiles() {
    const removeButtons = await this.page.$$(this.removeFileButton);
    for (const btn of removeButtons) {
      await btn.click();
    }
  }

  /**
   * Get file input value
   */
  async getSelectedFiles(): Promise<string[]> {
    return await this.page.evaluate(() => {
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      if (!input || !input.files) return [];
      return Array.from(input.files).map((f) => f.name);
    });
  }
}

export default UploadPage;
