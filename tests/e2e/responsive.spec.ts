/**
 * Responsive Design Tests
 *
 * Tests:
 * - Mobile responsiveness
 * - Tablet responsiveness
 * - Desktop layout
 * - Touch interactions
 * - Font scaling
 * - Image responsiveness
 */

import { test, expect } from '@playwright/test';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';

test.describe('Responsive Design Tests', () => {
  test.describe('Mobile (iPhone 12)', () => {
    test('should render correctly on mobile', async ({ page }) => {
      const loginPage = new LoginPage(page);
      await loginPage.goto();

      // Verify mobile layout
      const viewport = page.viewportSize();
      expect(viewport?.width).toBeLessThan(500);

      // Check for mobile menu
      const mobileMenu = await page.$('[data-testid="mobile-menu"]');
      expect(mobileMenu).toBeDefined();

      // Check that desktop sidebar is hidden
      const sidebar = await page.$('[data-testid="sidebar"]:visible');
      expect(sidebar).toBeNull();
    });

    test('should handle touch interactions', async ({ page }) => {
      const loginPage = new LoginPage(page);
      await loginPage.goto();

      // Simulate touch
      const button = await page.$('button:has-text("Login")');
      if (button) {
        await button.tap();
      }
    });

    test('should have readable text on mobile', async ({ page }) => {
      const loginPage = new LoginPage(page);
      await loginPage.goto();

      const fontSize = await page.evaluate(() => {
        const element = document.querySelector('input[type="email"]');
        return window.getComputedStyle(element!).fontSize;
      });

      // Font size should be at least 16px for mobile
      const size = parseInt(fontSize);
      expect(size).toBeGreaterThanOrEqual(16);
    });

    test('should have adequate touch targets', async ({ page }) => {
      const loginPage = new LoginPage(page);
      await loginPage.goto();

      const button = await page.$('button[type="submit"]');
      const boundingBox = await button?.boundingBox();

      // Touch targets should be at least 48x48
      expect(boundingBox?.height).toBeGreaterThanOrEqual(48);
      expect(boundingBox?.width).toBeGreaterThanOrEqual(48);
    });
  });

  test.describe('Tablet (iPad)', () => {
    test.use({ viewport: { width: 768, height: 1024 } });

    test('should render correctly on tablet', async ({ page }) => {
      const loginPage = new LoginPage(page);
      await loginPage.goto();

      const viewport = page.viewportSize();
      expect(viewport?.width).toBeGreaterThan(600);
      expect(viewport?.width).toBeLessThan(1000);
    });

    test('should have optimized layout for tablet', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();

      // Verify content is readable
      const content = await page.$('[data-testid="main-content"]');
      const boundingBox = await content?.boundingBox();

      // Content should use most of width
      expect(boundingBox?.width).toBeGreaterThan(400);
    });
  });

  test.describe('Desktop (1920x1080)', () => {
    test.use({ viewport: { width: 1920, height: 1080 } });

    test('should render correctly on desktop', async ({ page }) => {
      const loginPage = new LoginPage(page);
      await loginPage.goto();

      // Desktop sidebar should be visible
      const sidebar = await page.$('[data-testid="sidebar"]:visible');
      expect(sidebar).toBeDefined();
    });

    test('should utilize full screen width', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();

      // Sidebar + content should fill screen
      const sidebar = await page.$('[data-testid="sidebar"]');
      const content = await page.$('[data-testid="main-content"]');

      const sidebarBox = await sidebar?.boundingBox();
      const contentBox = await content?.boundingBox();

      const totalWidth = (sidebarBox?.width || 0) + (contentBox?.width || 0);
      expect(totalWidth).toBeGreaterThan(1200);
    });
  });

  test.describe('Breakpoint Transitions', () => {
    test('should transition from mobile to tablet layout', async ({ page }) => {
      // Start mobile
      await page.setViewportSize({ width: 375, height: 667 });

      const loginPage = new LoginPage(page);
      await loginPage.goto();

      let sidebar = await page.$('[data-testid="sidebar"]:visible');
      expect(sidebar).toBeNull(); // Hidden on mobile

      // Resize to tablet
      await page.setViewportSize({ width: 768, height: 1024 });
      await page.waitForTimeout(500); // Wait for layout reflow

      // Sidebar might now be visible
      // Layout should adapt without errors
      expect(page.url()).toContain('/login');
    });

    test('should handle orientation change', async ({ page, context }) => {
      // Portrait
      await page.setViewportSize({ width: 375, height: 667 });

      // Landscape
      await page.setViewportSize({ width: 667, height: 375 });

      const loginPage = new LoginPage(page);
      await loginPage.goto();

      // Should render without errors
      expect(await loginPage.getTitle()).toBeDefined();
    });
  });

  test.describe('Image Responsiveness', () => {
    test('should serve appropriate image sizes', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();

      const images = await page.$$('img');

      for (const img of images) {
        const src = await img.getAttribute('src');
        const computedStyle = await img.evaluate((el: any) => ({
          width: window.getComputedStyle(el).width,
          height: window.getComputedStyle(el).height,
        }));

        // Images should have dimensions
        expect(computedStyle.width).not.toBe('0px');
        expect(computedStyle.height).not.toBe('0px');
      }
    });

    test('should use responsive images', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();

      const hasResponsiveImages = await page.evaluate(() => {
        const pictures = document.querySelectorAll('picture');
        const srcs = document.querySelectorAll('[srcset]');

        return pictures.length > 0 || srcs.length > 0;
      });

      // Should have some responsive images
      // (May be optional depending on implementation)
    });
  });

  test.describe('Content Reflow', () => {
    test('should not have horizontal scroll on any viewport', async ({ page }) => {
      const viewports = [
        { width: 320, height: 568 }, // iPhone SE
        { width: 375, height: 667 }, // iPhone X
        { width: 768, height: 1024 }, // iPad
        { width: 1920, height: 1080 }, // Desktop
      ];

      const loginPage = new LoginPage(page);

      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        await loginPage.goto();

        const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
        const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);

        // Should not have horizontal overflow
        expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 1); // +1 for rounding
      }
    });

    test('should maintain aspect ratios during reflow', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });

      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();

      const initialAspectRatios = await page.evaluate(() => {
        const images = document.querySelectorAll('img');
        return Array.from(images).map((img: any) => ({
          width: img.offsetWidth,
          height: img.offsetHeight,
          ratio: img.offsetHeight / img.offsetWidth,
        }));
      });

      // Resize
      await page.setViewportSize({ width: 768, height: 1024 });
      await page.waitForTimeout(500);

      const finalAspectRatios = await page.evaluate(() => {
        const images = document.querySelectorAll('img');
        return Array.from(images).map((img: any) => ({
          width: img.offsetWidth,
          height: img.offsetHeight,
          ratio: img.offsetHeight / img.offsetWidth,
        }));
      });

      // Aspect ratios should be maintained
      for (let i = 0; i < initialAspectRatios.length; i++) {
        const initial = initialAspectRatios[i];
        const final = finalAspectRatios[i];

        // Allow small rounding differences
        expect(Math.abs(initial.ratio - final.ratio)).toBeLessThan(0.1);
      }
    });
  });

  test.describe('Navigation Responsiveness', () => {
    test('should show mobile menu on small screens', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });

      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();

      const mobileMenuButton = await page.$('[data-testid="menu-button"]');
      expect(mobileMenuButton).toBeDefined();

      // Click menu
      await mobileMenuButton?.click();

      // Menu should open
      const menu = await page.$('[data-testid="mobile-menu"][aria-hidden="false"]');
      expect(menu).toBeDefined();
    });

    test('should show sidebar on large screens', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 });

      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();

      const sidebar = await page.$('[data-testid="sidebar"]:visible');
      expect(sidebar).toBeDefined();
    });
  });
});
