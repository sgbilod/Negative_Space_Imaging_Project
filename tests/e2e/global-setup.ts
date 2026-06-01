/**
 * Global Test Setup
 *
 * Runs before all tests:
 * - Initialize test database
 * - Seed test data
 * - Create test users
 * - Reset API state
 */

import { chromium, FullConfig } from '@playwright/test';
import axios from 'axios';

const apiURL = process.env.API_URL || 'http://localhost:5000';

async function globalSetup(config: FullConfig) {
  console.log('🔧 Setting up test environment...');

  try {
    // Wait for API to be ready
    let retries = 0;
    const maxRetries = 10;
    let apiReady = false;

    while (retries < maxRetries && !apiReady) {
      try {
        await axios.get(`${apiURL}/health`, { timeout: 5000 });
        apiReady = true;
        console.log('✅ API is ready');
      } catch (error) {
        retries++;
        if (retries < maxRetries) {
          console.log(`⏳ Waiting for API... (${retries}/${maxRetries})`);
          await new Promise((resolve) => setTimeout(resolve, 2000));
        }
      }
    }

    if (!apiReady) {
      throw new Error('API failed to start');
    }

    // Seed test database
    console.log('🌱 Seeding test database...');
    await axios.post(`${apiURL}/api/test/seed`, {
      includeUsers: true,
      includeImages: true,
    });

    console.log('✅ Test environment ready');
  } catch (error) {
    console.error('❌ Setup failed:', error);
    process.exit(1);
  }

  // Store auth token for tests
  try {
    const loginResponse = await axios.post(`${apiURL}/api/auth/login`, {
      email: 'test@example.com',
      password: 'TestPassword123!',
    });

    process.env.TEST_AUTH_TOKEN = loginResponse.data.token;
    process.env.TEST_USER_ID = loginResponse.data.user.id;

    console.log('✅ Test authentication token stored');
  } catch (error) {
    console.error('❌ Failed to create test auth token:', error);
  }
}

export default globalSetup;
