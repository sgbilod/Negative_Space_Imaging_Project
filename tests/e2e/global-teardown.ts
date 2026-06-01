/**
 * Global Test Teardown
 *
 * Runs after all tests:
 * - Clean up test data
 * - Close database connections
 * - Generate final report
 */

import { FullConfig } from '@playwright/test';
import axios from 'axios';

const apiURL = process.env.API_URL || 'http://localhost:5000';

async function globalTeardown(config: FullConfig) {
  console.log('\n🧹 Cleaning up test environment...');

  try {
    // Clean up test data
    await axios.post(`${apiURL}/api/test/cleanup`, {
      headers: {
        Authorization: `Bearer ${process.env.TEST_AUTH_TOKEN}`,
      },
    });

    console.log('✅ Test data cleaned up');
  } catch (error) {
    console.warn('⚠️ Cleanup warning:', error);
    // Don't fail on cleanup errors
  }

  console.log('✅ Test environment cleaned');
}

export default globalTeardown;
