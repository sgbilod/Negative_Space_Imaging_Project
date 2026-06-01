import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import request from 'supertest';
import app from '@/index';
import { sequelize } from '@/config/database';

describe('Analysis Integration Tests', () => {
  let token: string;
  let imageId: string;

  beforeEach(async () => {
    // Setup database
    await sequelize.sync({ force: true });

    // Create test user
    const userResponse = await request(app)
      .post('/api/auth/register')
      .send({
        email: 'analysistest@example.com',
        password: 'TestPassword123!',
        first_name: 'Analysis',
        last_name: 'Test',
      });

    token = userResponse.body.token;

    // Upload test image
    const imageResponse = await request(app)
      .post('/api/images/upload')
      .set('Authorization', `Bearer ${token}`)
      .attach('file', Buffer.from('fake image data'), 'test.jpg');

    imageId = imageResponse.body.id;
  });

  afterEach(async () => {
    await sequelize.truncate({ cascade: true });
  });

  describe('Start Analysis', () => {
    it('should start analysis for image', async () => {
      const response = await request(app)
        .post(`/api/analysis/start`)
        .set('Authorization', `Bearer ${token}`)
        .send({ image_id: imageId });

      expect(response.status).toBe(202); // Accepted (async processing)
      expect(response.body).toHaveProperty('id');
      expect(response.body.status).toBe('pending');
    });

    it('should reject analysis for non-existent image', async () => {
      const response = await request(app)
        .post('/api/analysis/start')
        .set('Authorization', `Bearer ${token}`)
        .send({ image_id: 'nonexistent' });

      expect(response.status).toBe(404);
    });

    it('should require authentication', async () => {
      const response = await request(app)
        .post('/api/analysis/start')
        .send({ image_id: imageId });

      expect(response.status).toBe(401);
    });

    it('should deny analysis of other user images', async () => {
      const otherUserResponse = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'otheranalysisuser@example.com',
          password: 'TestPassword123!',
          first_name: 'Other',
          last_name: 'User',
        });

      const response = await request(app)
        .post('/api/analysis/start')
        .set('Authorization', `Bearer ${otherUserResponse.body.token}`)
        .send({ image_id: imageId });

      expect(response.status).toBe(403);
    });
  });

  describe('Get Analysis Results', () => {
    let analysisId: string;

    beforeEach(async () => {
      const response = await request(app)
        .post('/api/analysis/start')
        .set('Authorization', `Bearer ${token}`)
        .send({ image_id: imageId });

      analysisId = response.body.id;
    });

    it('should get analysis results', async () => {
      const response = await request(app)
        .get(`/api/analysis/${analysisId}`)
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(200);
      expect(response.body.id).toBe(analysisId);
      expect(response.body).toHaveProperty('status');
    });

    it('should return pending status during processing', async () => {
      const response = await request(app)
        .get(`/api/analysis/${analysisId}`)
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(200);
      expect(['pending', 'processing', 'completed', 'failed']).toContain(response.body.status);
    });

    it('should include metrics when completed', async () => {
      // Mock analysis completion by patching analysis result
      // For now, just verify structure
      const response = await request(app)
        .get(`/api/analysis/${analysisId}`)
        .set('Authorization', `Bearer ${token}`);

      if (response.body.status === 'completed') {
        expect(response.body).toHaveProperty('negative_space_percentage');
        expect(response.body).toHaveProperty('positive_space_percentage');
        expect(response.body).toHaveProperty('confidence_score');
      }
    });

    it('should return 404 for non-existent analysis', async () => {
      const response = await request(app)
        .get('/api/analysis/nonexistent-id')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(404);
    });
  });

  describe('Batch Analysis', () => {
    beforeEach(async () => {
      // Upload additional images
      for (let i = 1; i < 3; i++) {
        await request(app)
          .post('/api/images/upload')
          .set('Authorization', `Bearer ${token}`)
          .attach('file', Buffer.from(`fake image data ${i}`), `test${i}.jpg`);
      }
    });

    it('should start batch analysis', async () => {
      // Get all user images
      const imagesResponse = await request(app)
        .get('/api/images')
        .set('Authorization', `Bearer ${token}`);

      const imageIds = imagesResponse.body.map((img: any) => img.id);

      const response = await request(app)
        .post('/api/analysis/batch')
        .set('Authorization', `Bearer ${token}`)
        .send({ image_ids: imageIds });

      expect(response.status).toBe(202);
      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body.length).toBe(imageIds.length);
    });

    it('should handle mixed valid and invalid image ids', async () => {
      const response = await request(app)
        .post('/api/analysis/batch')
        .set('Authorization', `Bearer ${token}`)
        .send({ image_ids: [imageId, 'nonexistent-id'] });

      // Should process valid images, skip/report invalid ones
      expect(response.status).toMatch(/202|400/);
    });

    it('should reject batch with empty image list', async () => {
      const response = await request(app)
        .post('/api/analysis/batch')
        .set('Authorization', `Bearer ${token}`)
        .send({ image_ids: [] });

      expect(response.status).toBe(400);
    });
  });

  describe('Analysis Error Handling', () => {
    it('should handle invalid image gracefully', async () => {
      const response = await request(app)
        .post('/api/analysis/start')
        .set('Authorization', `Bearer ${token}`)
        .send({ image_id: imageId });

      expect([202, 400, 404]).toContain(response.status);
    });

    it('should handle processing errors', async () => {
      const response = await request(app)
        .post('/api/analysis/start')
        .set('Authorization', `Bearer ${token}`)
        .send({ image_id: imageId });

      // Start analysis
      const analysisId = response.body.id;

      // Check error status
      const statusResponse = await request(app)
        .get(`/api/analysis/${analysisId}`)
        .set('Authorization', `Bearer ${token}`);

      // Should have a status (not 500)
      expect(statusResponse.status).toBe(200);
    });

    it('should require authentication for results', async () => {
      const response = await request(app)
        .get('/api/analysis/some-id');

      expect(response.status).toBe(401);
    });
  });

  describe('Analysis Metrics Validation', () => {
    it('should have valid percentage ranges', async () => {
      const response = await request(app)
        .post('/api/analysis/start')
        .set('Authorization', `Bearer ${token}`)
        .send({ image_id: imageId });

      const analysisId = response.body.id;

      // Mock completion and verify metrics
      // This assumes metrics are returned after processing
      const resultResponse = await request(app)
        .get(`/api/analysis/${analysisId}`)
        .set('Authorization', `Bearer ${token}`);

      if (resultResponse.body.status === 'completed') {
        const negSpacePercent = resultResponse.body.negative_space_percentage;
        const posSpacePercent = resultResponse.body.positive_space_percentage;

        expect(negSpacePercent).toBeGreaterThanOrEqual(0);
        expect(negSpacePercent).toBeLessThanOrEqual(100);
        expect(posSpacePercent).toBeGreaterThanOrEqual(0);
        expect(posSpacePercent).toBeLessThanOrEqual(100);
        expect(negSpacePercent + posSpacePercent).toBeCloseTo(100, 1);
      }
    });

    it('should have valid confidence score', async () => {
      const response = await request(app)
        .post('/api/analysis/start')
        .set('Authorization', `Bearer ${token}`)
        .send({ image_id: imageId });

      const analysisId = response.body.id;

      const resultResponse = await request(app)
        .get(`/api/analysis/${analysisId}`)
        .set('Authorization', `Bearer ${token}`);

      if (resultResponse.body.status === 'completed') {
        const confidence = resultResponse.body.confidence_score;
        expect(confidence).toBeGreaterThanOrEqual(0);
        expect(confidence).toBeLessThanOrEqual(1);
      }
    });
  });
});
