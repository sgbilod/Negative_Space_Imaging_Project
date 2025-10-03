/**
 * signatureUtils.js
 * Utility functions for working with negative space signatures
 */

/**
 * Calculate similarity between two signatures
 * @param {Array} signature1 - First signature feature vector
 * @param {Array} signature2 - Second signature feature vector
 * @returns {number} Similarity score between 0 and 1
 */
export const calculateSimilarity = (signature1, signature2) => {
  if (!Array.isArray(signature1) || !Array.isArray(signature2)) {
    throw new Error('Signatures must be arrays');
  }
  
  if (signature1.length !== signature2.length) {
    throw new Error('Signatures must have the same length');
  }
  
  // Calculate cosine similarity
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  for (let i = 0; i < signature1.length; i++) {
    dotProduct += signature1[i] * signature2[i];
    norm1 += signature1[i] * signature1[i];
    norm2 += signature2[i] * signature2[i];
  }
  
  if (norm1 === 0 || norm2 === 0) {
    return 0;
  }
  
  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
};

/**
 * Generate a unique signature ID
 * @returns {string} Unique signature ID
 */
export const generateSignatureId = () => {
  return 'neg-sig-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
};

/**
 * Format signature data for display
 * @param {Object} signatureData - Raw signature data
 * @returns {Object} Formatted signature data
 */
export const formatSignatureData = (signatureData) => {
  if (!signatureData) {
    return null;
  }
  
  return {
    id: signatureData.signature_id || 'Unknown',
    confidence: signatureData.confidence 
      ? `${(signatureData.confidence * 100).toFixed(2)}%` 
      : 'Unknown',
    timestamp: signatureData.timestamp 
      ? new Date(signatureData.timestamp).toLocaleString() 
      : 'Unknown',
    featureCount: signatureData.features?.length || 0,
    metadata: signatureData.metadata || {}
  };
};

/**
 * Check if a signature meets the minimum quality threshold
 * @param {Object} signatureData - Signature data to check
 * @param {number} threshold - Minimum confidence threshold (0-1)
 * @returns {boolean} Whether the signature meets the quality threshold
 */
export const meetsQualityThreshold = (signatureData, threshold = 0.7) => {
  if (!signatureData || typeof signatureData.confidence !== 'number') {
    return false;
  }
  
  return signatureData.confidence >= threshold;
};

/**
 * Extract signature metadata for display
 * @param {Object} signatureData - Signature data
 * @returns {Array} Array of metadata key-value pairs
 */
export const extractMetadata = (signatureData) => {
  if (!signatureData || !signatureData.metadata) {
    return [];
  }
  
  return Object.entries(signatureData.metadata).map(([key, value]) => ({
    key: key.replace(/_/g, ' '),
    value: String(value)
  }));
};

/**
 * Parse blockchain verification result
 * @param {Object} blockchainData - Blockchain verification data
 * @returns {Object} Parsed blockchain data
 */
export const parseBlockchainData = (blockchainData) => {
  if (!blockchainData) {
    return {
      registered: false,
      status: 'Not registered',
      details: null
    };
  }
  
  const registered = blockchainData.registered || false;
  
  return {
    registered,
    status: registered ? 'Registered' : 'Not registered',
    blockNumber: blockchainData.block_number,
    timestamp: blockchainData.timestamp 
      ? new Date(blockchainData.timestamp).toLocaleString() 
      : null,
    transactionHash: blockchainData.transaction_hash,
    explorerUrl: blockchainData.transaction_hash 
      ? `https://etherscan.io/tx/${blockchainData.transaction_hash}`
      : null
  };
};

/**
 * Generate mock signature data for testing
 * @returns {Object} Mock signature data
 */
export const generateMockSignature = () => {
  return {
    signature_id: generateSignatureId(),
    features: Array.from({ length: 128 }, () => Math.random()),
    confidence: 0.75 + Math.random() * 0.2,
    timestamp: new Date().toISOString(),
    metadata: {
      capture_device: 'Mobile Camera',
      capture_mode: 'Test',
      lighting: Math.random() > 0.5 ? 'Natural' : 'Flash',
      version: '0.1.0'
    }
  };
};

/**
 * Calculate verification confidence based on multiple factors
 * @param {Object} verificationData - Verification result data
 * @returns {Object} Verification confidence data
 */
export const calculateVerificationConfidence = (verificationData) => {
  if (!verificationData) {
    return {
      overall: 0,
      factors: {}
    };
  }
  
  const signatureSimilarity = verificationData.confidence || 0;
  const matchCount = verificationData.matches?.length || 0;
  const highestMatch = verificationData.matches?.length 
    ? Math.max(...verificationData.matches.map(m => m.similarity || 0))
    : 0;
  
  // Calculate weighted confidence
  const weights = {
    signatureSimilarity: 0.6,
    matchCount: 0.2,
    highestMatch: 0.2
  };
  
  const factors = {
    signatureSimilarity,
    matchCount: Math.min(matchCount / 5, 1), // Normalize to 0-1
    highestMatch
  };
  
  const overall = Object.keys(weights).reduce(
    (sum, key) => sum + (factors[key] * weights[key]),
    0
  );
  
  return {
    overall,
    factors,
    threshold: 0.7, // Default threshold
    verified: overall >= 0.7
  };
};

/**
 * Create a shareable summary of verification results
 * @param {Object} signatureData - Signature data
 * @param {Object} verificationResult - Verification result
 * @param {Object} blockchainData - Blockchain data
 * @returns {string} Shareable summary text
 */
export const createShareableSummary = (signatureData, verificationResult, blockchainData) => {
  if (!signatureData) {
    return 'No signature data available';
  }
  
  const formattedData = formatSignatureData(signatureData);
  const verified = verificationResult?.verified ? 'VERIFIED' : 'NOT VERIFIED';
  const blockchainStatus = blockchainData?.registered 
    ? `Registered on blockchain (Block #${blockchainData.block_number})`
    : 'Not registered on blockchain';
  
  return `
Negative Space Signature Verification

ID: ${formattedData.id}
Status: ${verified}
Confidence: ${verificationResult?.confidence ? (verificationResult.confidence * 100).toFixed(2) + '%' : 'Unknown'}
Timestamp: ${formattedData.timestamp}
Blockchain: ${blockchainStatus}

Verified with Negative Space Mobile
  `.trim();
};
