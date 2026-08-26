/**
 * NegativeSpaceAPI.js
 * Main API bridge for communicating with the Python backend
 */

import { NativeModules } from 'react-native';
const { NegativeSpaceNativeModule } = NativeModules;

class NegativeSpaceAPI {
  /**
   * Initialize the API with server settings
   * @param {string} serverUrl - URL of the Python backend server
   * @param {Object} options - Additional configuration options
   */
  constructor(serverUrl, options = {}) {
    this.serverUrl = serverUrl;
    this.options = {
      useWebsocket: true,
      useNativeBindings: true,
      offlineMode: false,
      ...options
    };
    
    // Initialize connection
    this.isConnected = false;
    this.websocket = null;
    
    if (this.options.useNativeBindings && NegativeSpaceNativeModule) {
      this.nativeModule = NegativeSpaceNativeModule;
    } else {
      console.warn('Native module not available, falling back to REST API');
      this.nativeModule = null;
    }
  }

  /**
   * Connect to the Python backend server
   * @returns {Promise<boolean>} - Connection status
   */
  async connect() {
    if (this.options.offlineMode) {
      console.log('Operating in offline mode');
      return true;
    }
    
    try {
      if (this.options.useWebsocket) {
        return this._connectWebsocket();
      } else {
        return this._testRestConnection();
      }
    } catch (error) {
      console.error('Connection error:', error);
      return false;
    }
  }

  /**
   * Establish WebSocket connection
   * @private
   */
  _connectWebsocket() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`ws://${this.serverUrl}/ws`);
      
      ws.onopen = () => {
        this.websocket = ws;
        this.isConnected = true;
        resolve(true);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };
      
      ws.onmessage = (event) => {
        this._handleWebsocketMessage(event.data);
      };
      
      ws.onclose = () => {
        this.isConnected = false;
        this.websocket = null;
      };
    });
  }

  /**
   * Test REST API connection
   * @private
   */
  async _testRestConnection() {
    try {
      const response = await fetch(`http://${this.serverUrl}/api/status`);
      if (response.ok) {
        this.isConnected = true;
        return true;
      }
      return false;
    } catch (error) {
      console.error('REST API connection error:', error);
      return false;
    }
  }

  /**
   * Handle incoming WebSocket messages
   * @private
   */
  _handleWebsocketMessage(data) {
    try {
      const message = JSON.parse(data);
      
      // Route message to appropriate handler
      switch (message.type) {
        case 'signature_result':
          this._handleSignatureResult(message.data);
          break;
        case 'verification_result':
          this._handleVerificationResult(message.data);
          break;
        case 'blockchain_result':
          this._handleBlockchainResult(message.data);
          break;
        default:
          console.warn('Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('Error processing WebSocket message:', error);
    }
  }

  /**
   * Extract a negative space signature from an image
   * @param {Object} imageData - Image data from camera
   * @returns {Promise<Object>} - Extracted signature data
   */
  async extractSignature(imageData) {
    if (this.nativeModule) {
      // Use native module for better performance
      return this.nativeModule.extractSignature(imageData);
    } else if (this.isConnected) {
      // Use REST API
      return this._extractSignatureREST(imageData);
    } else {
      throw new Error('Not connected to backend and native module not available');
    }
  }

  /**
   * Extract signature using REST API
   * @private
   */
  async _extractSignatureREST(imageData) {
    const formData = new FormData();
    formData.append('image', {
      uri: imageData.uri,
      type: 'image/jpeg',
      name: 'image.jpg'
    });
    
    const response = await fetch(`http://${this.serverUrl}/api/extract_signature`, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  }

  /**
   * Verify a negative space signature against the blockchain
   * @param {Object} signatureData - Signature data to verify
   * @returns {Promise<Object>} - Verification result
   */
  async verifySignature(signatureData) {
    if (!this.options.offlineMode && !this.isConnected) {
      throw new Error('Not connected to backend');
    }
    
    if (this.options.offlineMode) {
      return this._offlineVerification(signatureData);
    }
    
    if (this.websocket) {
      return this._verifySignatureWS(signatureData);
    } else {
      return this._verifySignatureREST(signatureData);
    }
  }

  /**
   * Verify signature using WebSocket
   * @private
   */
  _verifySignatureWS(signatureData) {
    return new Promise((resolve, reject) => {
      const requestId = Date.now().toString();
      
      const responseHandler = (event) => {
        try {
          const response = JSON.parse(event.data);
          if (response.requestId === requestId) {
            this.websocket.removeEventListener('message', responseHandler);
            resolve(response.result);
          }
        } catch (error) {
          console.error('Error parsing WebSocket response:', error);
        }
      };
      
      this.websocket.addEventListener('message', responseHandler);
      
      this.websocket.send(JSON.stringify({
        type: 'verify_signature',
        requestId,
        data: signatureData
      }));
      
      // Set timeout to prevent hanging
      setTimeout(() => {
        this.websocket.removeEventListener('message', responseHandler);
        reject(new Error('WebSocket verification timeout'));
      }, 30000);
    });
  }

  /**
   * Verify signature using REST API
   * @private
   */
  async _verifySignatureREST(signatureData) {
    const response = await fetch(`http://${this.serverUrl}/api/verify_signature`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(signatureData)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  }

  /**
   * Offline verification of signatures
   * @private
   */
  _offlineVerification(signatureData) {
    if (this.nativeModule) {
      return this.nativeModule.verifySignature(signatureData);
    } else {
      throw new Error('Offline verification requires native module');
    }
  }

  /**
   * Get blockchain registration status for a signature
   * @param {string} signatureId - ID of the signature to check
   * @returns {Promise<Object>} - Blockchain status
   */
  async getBlockchainStatus(signatureId) {
    if (this.options.offlineMode) {
      throw new Error('Blockchain status not available in offline mode');
    }
    
    const response = await fetch(
      `http://${this.serverUrl}/api/blockchain/status/${signatureId}`
    );
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  }

  /**
   * Register a signature on the blockchain
   * @param {Object} signatureData - Signature data to register
   * @param {Object} authData - Authentication data
   * @returns {Promise<Object>} - Registration result
   */
  async registerOnBlockchain(signatureData, authData) {
    if (this.options.offlineMode) {
      throw new Error('Blockchain registration not available in offline mode');
    }
    
    const response = await fetch(`http://${this.serverUrl}/api/blockchain/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authData.token}`
      },
      body: JSON.stringify(signatureData)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  }

  /**
   * Get the visualization data for AR display
   * @param {Object} signatureData - Signature data to visualize
   * @returns {Promise<Object>} - Visualization data
   */
  async getVisualizationData(signatureData) {
    if (this.nativeModule) {
      return this.nativeModule.getVisualizationData(signatureData);
    }
    
    const response = await fetch(`http://${this.serverUrl}/api/visualization`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(signatureData)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  }

  /**
   * Close the connection to the server
   */
  disconnect() {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    this.isConnected = false;
  }
}

export default NegativeSpaceAPI;
