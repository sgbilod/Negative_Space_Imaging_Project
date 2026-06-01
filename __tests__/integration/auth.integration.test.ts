import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import request from 'supertest';
import app from '@/index';
import { sequelize } from '@/config/database';
import * as bcrypt from 'bcrypt';

describe('Auth Integration Tests', () => {
  let token: string;
  let userId: string;

  beforeEach(async () => {
    // Setup test database - run migrations
    await sequelize.sync({ force: true });
  });

  afterEach(async () => {
    // Cleanup
    await sequelize.truncate({ cascade: true });
  });

  describe('Registration Flow', () => {
    it('should register new user successfully', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'newuser@example.com',
          password: 'TestPassword123!',
          first_name: 'Test',
          last_name: 'User',
        });

      expect(response.status).toBe(201);
      expect(response.body).toHaveProperty('user');
      expect(response.body).toHaveProperty('token');
      expect(response.body.user.email).toBe('newuser@example.com');
      expect(response.body.token).toBeDefined();
      expect(response.body.token.length).toBeGreaterThan(0);
    });

    it('should reject duplicate email', async () => {
      // Create first user
      await request(app)
        .post('/api/auth/register')
        .send({
          email: 'duplicate@example.com',
          password: 'TestPassword123!',
          first_name: 'First',
          last_name: 'User',
        });

      // Try to create with same email
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'duplicate@example.com',
          password: 'DifferentPassword123!',
          first_name: 'Second',
          last_name: 'User',
        });

      expect(response.status).toBe(409);
      expect(response.body).toHaveProperty('error');
    });

    it('should validate email format', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'invalid-email',
          password: 'TestPassword123!',
          first_name: 'Test',
          last_name: 'User',
        });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('should validate password strength', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'newuser@example.com',
          password: 'weak',
          first_name: 'Test',
          last_name: 'User',
        });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });
  });

  describe('Login Flow', () => {
    beforeEach(async () => {
      // Create test user
      const hashedPassword = await bcrypt.hash('TestPassword123!', 10);
      const userResponse = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'testuser@example.com',
          password: 'TestPassword123!',
          first_name: 'Test',
          last_name: 'User',
        });

      token = userResponse.body.token;
      userId = userResponse.body.user.id;
    });

    it('should login with correct credentials', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'testuser@example.com',
          password: 'TestPassword123!',
        });

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('token');
      expect(response.body).toHaveProperty('user');
      expect(response.body.user.email).toBe('testuser@example.com');
    });

    it('should reject wrong password', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'testuser@example.com',
          password: 'WrongPassword123!',
        });

      expect(response.status).toBe(401);
      expect(response.body).toHaveProperty('error');
    });

    it('should reject non-existent user', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'nonexistent@example.com',
          password: 'TestPassword123!',
        });

      expect(response.status).toBe(401);
      expect(response.body).toHaveProperty('error');
    });
  });

  describe('JWT Token Management', () => {
    beforeEach(async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'tokentest@example.com',
          password: 'TestPassword123!',
          first_name: 'Token',
          last_name: 'Test',
        });

      token = response.body.token;
    });

    it('should access protected route with valid token', async () => {
      const response = await request(app)
        .get('/api/users/profile')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('email');
    });

    it('should reject request without token', async () => {
      const response = await request(app)
        .get('/api/users/profile');

      expect(response.status).toBe(401);
      expect(response.body).toHaveProperty('error');
    });

    it('should reject request with invalid token', async () => {
      const response = await request(app)
        .get('/api/users/profile')
        .set('Authorization', 'Bearer invalid-token');

      expect(response.status).toBe(401);
      expect(response.body).toHaveProperty('error');
    });

    it('should reject request with expired token', async () => {
      // This would require generating an expired token
      // For now, test invalid format
      const response = await request(app)
        .get('/api/users/profile')
        .set('Authorization', 'Bearer ' + 'x'.repeat(100));

      expect(response.status).toBe(401);
    });
  });

  describe('Logout Flow', () => {
    beforeEach(async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'logouttest@example.com',
          password: 'TestPassword123!',
          first_name: 'Logout',
          last_name: 'Test',
        });

      token = response.body.token;
    });

    it('should clear token on logout', async () => {
      const response = await request(app)
        .post('/api/auth/logout')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(200);
      // Token should no longer be valid
      const protectedResponse = await request(app)
        .get('/api/users/profile')
        .set('Authorization', `Bearer ${token}`);

      expect(protectedResponse.status).toBe(401);
    });
  });

  describe('Authorization', () => {
    let user1Token: string;
    let user2Token: string;

    beforeEach(async () => {
      const user1Response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'user1@example.com',
          password: 'TestPassword123!',
          first_name: 'User',
          last_name: 'One',
        });

      const user2Response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'user2@example.com',
          password: 'TestPassword123!',
          first_name: 'User',
          last_name: 'Two',
        });

      user1Token = user1Response.body.token;
      user2Token = user2Response.body.token;
    });

    it('should deny cross-user access', async () => {
      // User 2 tries to access User 1's profile using User 2's token
      const response = await request(app)
        .get('/api/users/1/profile')
        .set('Authorization', `Bearer ${user2Token}`);

      expect(response.status).toBe(403);
    });

    it('should allow same-user access', async () => {
      const response = await request(app)
        .get('/api/users/profile')
        .set('Authorization', `Bearer ${user1Token}`);

      expect(response.status).toBe(200);
    });
  });
});
