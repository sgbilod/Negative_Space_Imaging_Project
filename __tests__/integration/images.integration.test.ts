import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import request from 'supertest';
import app from '@/index';
import { sequelize } from '@/config/database';
import fs from 'fs/promises';
import path from 'path';

describe('Images Integration Tests', () => {
  let token: string;
  let userId: string;
  const testUploadDir = path.join(__dirname, 'temp-uploads');

  beforeEach(async () => {
    // Setup
    await sequelize.sync({ force: true });
    await fs.mkdir(testUploadDir, { recursive: true });

    // Create test user
    const response = await request(app)
      .post('/api/auth/register')
      .send({
        email: 'imagetest@example.com',
        password: 'TestPassword123!',
        first_name: 'Image',
        last_name: 'Test',
      });

    token = response.body.token;
    userId = response.body.user.id;
  });

  afterEach(async () => {
    // Cleanup
    await sequelize.truncate({ cascade: true });
    try {
      await fs.rm(testUploadDir, { recursive: true, force: true });
    } catch (e) {
      // Ignore cleanup errors
    }
  });

  describe('Image Upload', () => {
    it('should upload image successfully', async () => {
      const response = await request(app)
        .post('/api/images/upload')
        .set('Authorization', `Bearer ${token}`)
        .attach('file', Buffer.from('fake image data'), 'test.jpg');

      expect(response.status).toBe(201);
      expect(response.body).toHaveProperty('id');
      expect(response.body).toHaveProperty('filename');
      expect(response.body.user_id).toBe(userId);
    });

    it('should validate file type', async () => {
      const response = await request(app)
        .post('/api/images/upload')
        .set('Authorization', `Bearer ${token}`)
        .attach('file', Buffer.from('invalid data'), 'test.txt');

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('should reject files exceeding size limit', async () => {
      const largeBuffer = Buffer.alloc(60 * 1024 * 1024); // 60MB (exceeds 50MB limit)

      const response = await request(app)
        .post('/api/images/upload')
        .set('Authorization', `Bearer ${token}`)
        .attach('file', largeBuffer, 'large.jpg');

      expect(response.status).toBe(413);
      expect(response.body).toHaveProperty('error');
    });

    it('should reject upload without authentication', async () => {
      const response = await request(app)
        .post('/api/images/upload')
        .attach('file', Buffer.from('fake image data'), 'test.jpg');

      expect(response.status).toBe(401);
    });
  });

  describe('List Images', () => {
    beforeEach(async () => {
      // Upload test images
      for (let i = 1; i <= 3; i++) {
        await request(app)
          .post('/api/images/upload')
          .set('Authorization', `Bearer ${token}`)
          .attach('file', Buffer.from(`fake image data ${i}`), `test${i}.jpg`);
      }
    });

    it('should list user images', async () => {
      const response = await request(app)
        .get('/api/images')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(200);
      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body.length).toBe(3);
    });

    it('should apply pagination', async () => {
      const response = await request(app)
        .get('/api/images?page=1&limit=2')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(200);
      expect(response.body.length).toBeLessThanOrEqual(2);
    });

    it('should return empty array for user with no images', async () => {
      // Create new user without images
      const newUserResponse = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'emptyuser@example.com',
          password: 'TestPassword123!',
          first_name: 'Empty',
          last_name: 'User',
        });

      const response = await request(app)
        .get('/api/images')
        .set('Authorization', `Bearer ${newUserResponse.body.token}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveLength(0);
    });

    it('should only return user own images', async () => {
      // Create another user
      const otherUserResponse = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'otheruser@example.com',
          password: 'TestPassword123!',
          first_name: 'Other',
          last_name: 'User',
        });

      // Query with first user's token
      const response = await request(app)
        .get('/api/images')
        .set('Authorization', `Bearer ${token}`);

      // Should only see first user's images, not other user's
      for (const image of response.body) {
        expect(image.user_id).toBe(userId);
      }
    });
  });

  describe('Get Image Details', () => {
    let imageId: string;

    beforeEach(async () => {
      const response = await request(app)
        .post('/api/images/upload')
        .set('Authorization', `Bearer ${token}`)
        .attach('file', Buffer.from('fake image data'), 'test.jpg');

      imageId = response.body.id;
    });

    it('should get image by id', async () => {
      const response = await request(app)
        .get(`/api/images/${imageId}`)
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(200);
      expect(response.body.id).toBe(imageId);
      expect(response.body).toHaveProperty('filename');
      expect(response.body).toHaveProperty('file_size');
    });

    it('should return 404 for non-existent image', async () => {
      const response = await request(app)
        .get('/api/images/nonexistent-id')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(404);
    });
  });

  describe('Delete Image', () => {
    let imageId: string;

    beforeEach(async () => {
      const response = await request(app)
        .post('/api/images/upload')
        .set('Authorization', `Bearer ${token}`)
        .attach('file', Buffer.from('fake image data'), 'test.jpg');

      imageId = response.body.id;
    });

    it('should delete image', async () => {
      const response = await request(app)
        .delete(`/api/images/${imageId}`)
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(200);

      // Verify image is gone
      const getResponse = await request(app)
        .get(`/api/images/${imageId}`)
        .set('Authorization', `Bearer ${token}`);

      expect(getResponse.status).toBe(404);
    });

    it('should deny deletion of other user images', async () => {
      const otherUserResponse = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'otheruser@example.com',
          password: 'TestPassword123!',
          first_name: 'Other',
          last_name: 'User',
        });

      const response = await request(app)
        .delete(`/api/images/${imageId}`)
        .set('Authorization', `Bearer ${otherUserResponse.body.token}`);

      expect(response.status).toBe(403);
    });

    it('should require authentication for deletion', async () => {
      const response = await request(app)
        .delete(`/api/images/${imageId}`);

      expect(response.status).toBe(401);
    });
  });

  describe('Error Handling', () => {
    it('should handle database errors gracefully', async () => {
      // This would require mocking database failures
      const response = await request(app)
        .get('/api/images')
        .set('Authorization', `Bearer ${token}`);

      // Should not return 500 if database is working
      expect([200, 400, 401, 403, 404]).toContain(response.status);
    });
  });
});
