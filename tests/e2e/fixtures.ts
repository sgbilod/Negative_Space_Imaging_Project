/**
 * Shared Test Fixtures
 *
 * Provides:
 * - Test data (users, images, credentials)
 * - API clients with authentication
 * - Database utilities
 * - Common test helpers
 */

import { test as base, expect } from '@playwright/test';
import axios, { AxiosInstance } from 'axios';

const apiURL = process.env.API_URL || 'http://localhost:5000';
const baseURL = process.env.BASE_URL || 'http://localhost:3000';

/**
 * Test User Data
 */
export const testUsers = {
  validUser: {
    email: 'test@example.com',
    password: 'TestPassword123!',
    firstName: 'Test',
    lastName: 'User',
  },
  adminUser: {
    email: 'admin@example.com',
    password: 'AdminPassword123!',
    firstName: 'Admin',
    lastName: 'User',
  },
  newUser: {
    email: `test-${Date.now()}@example.com`,
    password: 'NewPassword123!',
    firstName: 'New',
    lastName: 'User',
  },
  invalidUser: {
    email: 'invalid@example.com',
    password: 'WrongPassword123!',
  },
};

/**
 * Test Image Data
 */
export const testImages = {
  validImagePath: './tests/e2e/fixtures/test-image.jpg',
  largeImagePath: './tests/e2e/fixtures/large-image.jpg',
  invalidImagePath: './tests/e2e/fixtures/invalid.txt',
  batchImagePaths: [
    './tests/e2e/fixtures/image-1.jpg',
    './tests/e2e/fixtures/image-2.jpg',
    './tests/e2e/fixtures/image-3.jpg',
  ],
};

/**
 * API Error Messages
 */
export const apiErrors = {
  invalidCredentials: 'Invalid email or password',
  userExists: 'User already exists',
  unauthorized: 'Unauthorized',
  notFound: 'Not found',
  serverError: 'Internal server error',
};

/**
 * Custom Test Fixture with API Client
 */
export type TestFixtures = {
  apiClient: AxiosInstance;
  authenticatedApiClient: AxiosInstance;
  authToken: string | null;
};

export const test = base.extend<TestFixtures>({
  apiClient: async ({}, use) => {
    const client = axios.create({
      baseURL: apiURL,
      timeout: 10000,
      validateStatus: () => true, // Don't throw on any status
    });

    await use(client);
  },

  authenticatedApiClient: async ({}, use) => {
    const token = process.env.TEST_AUTH_TOKEN || '';
    const client = axios.create({
      baseURL: apiURL,
      timeout: 10000,
      headers: {
        Authorization: `Bearer ${token}`,
      },
      validateStatus: () => true,
    });

    await use(client);
  },

  authToken: async ({}, use) => {
    await use(process.env.TEST_AUTH_TOKEN || null);
  },
});

/**
 * Test Data Generator
 */
export class TestDataGenerator {
  static generateUser() {
    const timestamp = Date.now();
    return {
      email: `test-${timestamp}@example.com`,
      password: 'TestPassword123!',
      firstName: `First${timestamp}`,
      lastName: `Last${timestamp}`,
    };
  }

  static generateImage() {
    return {
      name: `test-image-${Date.now()}.jpg`,
      size: 1024 * 100, // 100KB
      format: 'jpg',
      uploadDate: new Date().toISOString(),
    };
  }

  static generateAnalysisRequest() {
    return {
      imageId: 'test-image-id',
      analysisType: 'negative_space',
      parameters: {
        threshold: 0.5,
        minRegionSize: 100,
      },
    };
  }
}

/**
 * Common Test Helpers
 */
export class TestHelpers {
  static async waitForElement(page: any, selector: string, timeout = 5000) {
    await page.waitForSelector(selector, { timeout });
  }

  static async fillForm(page: any, formData: Record<string, string>) {
    for (const [selector, value] of Object.entries(formData)) {
      await page.fill(selector, value);
    }
  }

  static async screenshot(page: any, name: string) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    await page.screenshot({ path: `test-results/screenshots/${name}-${timestamp}.png` });
  }

  static generateSessionId(): string {
    return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  static formatPhoneNumber(num: string): string {
    return num.replace(/(\d{3})(\d{3})(\d{4})/, '($1) $2-$3');
  }

  static parseAuthToken(token: string) {
    const payload = token.split('.')[1];
    return JSON.parse(Buffer.from(payload, 'base64').toString('utf-8'));
  }
}

/**
 * Page Object Base Class
 */
export class BasePage {
  protected page: any;
  protected baseURL: string;

  constructor(page: any) {
    this.page = page;
    this.baseURL = baseURL;
  }

  async goto(path: string) {
    await this.page.goto(`${this.baseURL}${path}`);
  }

  async waitForNavigation() {
    await this.page.waitForNavigation();
  }

  async screenshot(name: string) {
    await TestHelpers.screenshot(this.page, name);
  }

  async getTitle(): Promise<string> {
    return await this.page.title();
  }

  async getURL(): Promise<string> {
    return this.page.url();
  }
}

export { expect };
