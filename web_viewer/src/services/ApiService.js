/**
 * API Integration Service
 *
 * Handles communication between the OHIF web viewer and the Python backend
 * for quantum processing, segmentation, and analysis operations.
 */

import axios from 'axios';

class ApiService {
  constructor(baseURL = 'http://localhost:8000/api') {
    this.client = axios.create({
      baseURL,
      timeout: 30000, // 30 second timeout for quantum operations
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error);
        return Promise.reject(error);
      },
    );
  }

  /**
   * Process image with quantum analysis
   * @param {File|Blob} imageFile - DICOM or image file
   * @param {Object} options - Processing options
   * @returns {Promise<Object>} Quantum analysis results
   */
  async processQuantumAnalysis(imageFile, options = {}) {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);

      // Add processing options
      Object.keys(options).forEach((key) => {
        formData.append(key, options[key]);
      });

      const response = await this.client.post('/quantum/process', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Quantum analysis failed:', error);
      throw new Error(`Quantum processing failed: ${error.message}`);
    }
  }

  /**
   * Perform segmentation analysis
   * @param {File|Blob} imageFile - Image file to segment
   * @param {Object} options - Segmentation options
   * @returns {Promise<Object>} Segmentation results
   */
  async processSegmentation(imageFile, options = {}) {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);

      // Add segmentation options
      Object.keys(options).forEach((key) => {
        formData.append(key, options[key]);
      });

      const response = await this.client.post('/segmentation/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Segmentation analysis failed:', error);
      throw new Error(`Segmentation failed: ${error.message}`);
    }
  }

  /**
   * Analyze negative space patterns
   * @param {File|Blob} imageFile - Image file to analyze
   * @param {Object} measurementData - Measurement coordinates and data
   * @returns {Promise<Object>} Negative space analysis results
   */
  async analyzeNegativeSpace(imageFile, measurementData) {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('measurement', JSON.stringify(measurementData));

      const response = await this.client.post('/negative-space/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Negative space analysis failed:', error);
      throw new Error(`Negative space analysis failed: ${error.message}`);
    }
  }

  /**
   * Get processing status
   * @param {string} jobId - Job ID to check
   * @returns {Promise<Object>} Job status
   */
  async getProcessingStatus(jobId) {
    try {
      const response = await this.client.get(`/status/${jobId}`);
      return response.data;
    } catch (error) {
      console.error('Status check failed:', error);
      throw new Error(`Status check failed: ${error.message}`);
    }
  }

  /**
   * Batch process multiple images
   * @param {Array<File|Blob>} imageFiles - Array of image files
   * @param {Object} options - Batch processing options
   * @returns {Promise<Object>} Batch processing results
   */
  async batchProcess(imageFiles, options = {}) {
    try {
      const formData = new FormData();

      // Add all image files
      imageFiles.forEach((file, index) => {
        formData.append(`image_${index}`, file);
      });

      // Add batch options
      formData.append('options', JSON.stringify(options));

      const response = await this.client.post('/batch/process', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Batch processing failed:', error);
      throw new Error(`Batch processing failed: ${error.message}`);
    }
  }

  /**
   * Get system health status
   * @returns {Promise<Object>} System health information
   */
  async getSystemHealth() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      return {
        status: 'unhealthy',
        error: error.message,
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Get available processing models
   * @returns {Promise<Object>} Available models information
   */
  async getAvailableModels() {
    try {
      const response = await this.client.get('/models');
      return response.data;
    } catch (error) {
      console.error('Failed to get models:', error);
      throw new Error(`Failed to get models: ${error.message}`);
    }
  }

  /**
   * Validate API connectivity
   * @returns {Promise<boolean>} Connection status
   */
  async validateConnection() {
    try {
      await this.getSystemHealth();
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Configure processing parameters
   * @param {Object} config - Configuration parameters
   * @returns {Promise<Object>} Configuration response
   */
  async configureProcessing(config) {
    try {
      const response = await this.client.post('/config', config);
      return response.data;
    } catch (error) {
      console.error('Configuration failed:', error);
      throw new Error(`Configuration failed: ${error.message}`);
    }
  }
}

// Create singleton instance
const apiService = new ApiService();

// Export both the class and the singleton instance
export { ApiService };
export default apiService;
