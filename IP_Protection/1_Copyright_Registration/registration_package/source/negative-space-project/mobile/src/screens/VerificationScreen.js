/**
 * VerificationScreen.js
 * Screen for verifying negative space signatures
 */

import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Alert
} from 'react-native';

/**
 * Verification screen component
 */
const VerificationScreen = ({ navigation, route, api }) => {
  const [signature, setSignature] = useState(null);
  const [verificationResult, setVerificationResult] = useState(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [blockchainInfo, setBlockchainInfo] = useState(null);
  const [isLoadingBlockchain, setIsLoadingBlockchain] = useState(false);

  // Get signature data from route params if available
  useEffect(() => {
    if (route.params?.signatureData) {
      setSignature(route.params.signatureData);
    }
  }, [route.params]);

  /**
   * Verify the signature
   */
  const verifySignature = async () => {
    if (!signature) {
      Alert.alert(
        'No Signature',
        'No signature data to verify. Please capture a negative space first.',
        [{ text: 'OK' }]
      );
      return;
    }

    try {
      setIsVerifying(true);

      // In a real implementation, you would verify using the API:
      // const result = await api.verifySignature(signature);

      // Mock implementation - simulate verification delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      const mockResult = {
        verified: Math.random() > 0.3, // 70% chance of success
        confidence: 0.75 + Math.random() * 0.2,
        matches: [
          {
            id: 'orig-sig-12345',
            similarity: 0.85 + Math.random() * 0.1,
            metadata: {
              origin: 'Original registration',
              registration_date: '2023-10-15T08:30:00Z',
              registered_by: 'Verified authority'
            }
          }
        ],
        verification_time: new Date().toISOString()
      };

      setVerificationResult(mockResult);

      // If verified, check blockchain status
      if (mockResult.verified && !api.options.offlineMode) {
        getBlockchainStatus(signature.signature_id);
      }
    } catch (error) {
      console.error('Verification error:', error);
      Alert.alert(
        'Verification Failed',
        'An error occurred during verification. Please try again.',
        [{ text: 'OK' }]
      );
    } finally {
      setIsVerifying(false);
    }
  };

  /**
   * Get blockchain status for the signature
   */
  const getBlockchainStatus = async (signatureId) => {
    try {
      setIsLoadingBlockchain(true);

      // In a real implementation, you would get blockchain info using the API:
      // const blockchainData = await api.getBlockchainStatus(signatureId);

      // Mock implementation - simulate blockchain query delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      const mockBlockchainData = {
        registered: Math.random() > 0.2, // 80% chance of being registered
        block_number: Math.floor(15000000 + Math.random() * 1000000),
        timestamp: new Date(Date.now() - Math.floor(Math.random() * 30 * 24 * 60 * 60 * 1000)).toISOString(),
        transaction_hash: `0x${Array.from({ length: 64 }, () => '0123456789abcdef'[Math.floor(Math.random() * 16)]).join('')}`
      };

      setBlockchainInfo(mockBlockchainData);
    } catch (error) {
      console.error('Blockchain status error:', error);
      Alert.alert(
        'Blockchain Query Failed',
        'Could not retrieve blockchain information for this signature.',
        [{ text: 'OK' }]
      );
    } finally {
      setIsLoadingBlockchain(false);
    }
  };

  /**
   * View details on the blockchain
   */
  const viewBlockchainDetails = () => {
    if (!blockchainInfo) {
      return;
    }

    navigation.navigate('Blockchain', {
      signatureId: signature.signature_id,
      blockchainInfo
    });
  };

  /**
   * Reset all verification data
   */
  const resetVerification = () => {
    setVerificationResult(null);
    setBlockchainInfo(null);
  };

  /**
   * Capture a new negative space
   */
  const captureNew = () => {
    navigation.navigate('Camera');
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Signature information */}
      {signature ? (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Signature Information</Text>
          <View style={styles.infoItem}>
            <Text style={styles.infoLabel}>Signature ID:</Text>
            <Text style={styles.infoValue}>{signature.signature_id}</Text>
          </View>
          <View style={styles.infoItem}>
            <Text style={styles.infoLabel}>Timestamp:</Text>
            <Text style={styles.infoValue}>
              {new Date(signature.timestamp).toLocaleString()}
            </Text>
          </View>
          <View style={styles.infoItem}>
            <Text style={styles.infoLabel}>Confidence:</Text>
            <Text style={styles.infoValue}>
              {(signature.confidence * 100).toFixed(2)}%
            </Text>
          </View>
        </View>
      ) : (
        <View style={styles.noSignatureContainer}>
          <Text style={styles.noSignatureText}>
            No signature data available. Please capture a negative space first.
          </Text>
          <TouchableOpacity style={styles.button} onPress={captureNew}>
            <Text style={styles.buttonText}>Capture Negative Space</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Verification controls */}
      {signature && !verificationResult && (
        <View style={styles.verifyContainer}>
          <TouchableOpacity
            style={[styles.button, isVerifying && styles.buttonDisabled]}
            onPress={verifySignature}
            disabled={isVerifying}
          >
            {isVerifying ? (
              <>
                <ActivityIndicator size="small" color="#fff" />
                <Text style={[styles.buttonText, styles.loadingButtonText]}>
                  Verifying...
                </Text>
              </>
            ) : (
              <Text style={styles.buttonText}>Verify Signature</Text>
            )}
          </TouchableOpacity>
        </View>
      )}

      {/* Verification results */}
      {verificationResult && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Verification Results</Text>
          <View style={[
            styles.resultBox,
            verificationResult.verified ? styles.resultSuccess : styles.resultFailure
          ]}>
            <Text style={styles.resultStatus}>
              {verificationResult.verified ? 'VERIFIED' : 'NOT VERIFIED'}
            </Text>
            <Text style={styles.resultConfidence}>
              Confidence: {(verificationResult.confidence * 100).toFixed(2)}%
            </Text>
          </View>

          {verificationResult.matches && verificationResult.matches.length > 0 && (
            <View style={styles.matchesContainer}>
              <Text style={styles.matchesTitle}>Matched Signatures:</Text>
              {verificationResult.matches.map((match, index) => (
                <View key={index} style={styles.matchItem}>
                  <Text style={styles.matchId}>ID: {match.id}</Text>
                  <Text style={styles.matchSimilarity}>
                    Similarity: {(match.similarity * 100).toFixed(2)}%
                  </Text>
                  {match.metadata && (
                    <View style={styles.matchMetadata}>
                      {Object.entries(match.metadata).map(([key, value], i) => (
                        <Text key={i} style={styles.metadataItem}>
                          {key.replace(/_/g, ' ')}: {value}
                        </Text>
                      ))}
                    </View>
                  )}
                </View>
              ))}
            </View>
          )}
        </View>
      )}

      {/* Blockchain information */}
      {isLoadingBlockchain && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Blockchain Verification</Text>
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#4a90e2" />
            <Text style={styles.loadingText}>
              Checking blockchain records...
            </Text>
          </View>
        </View>
      )}

      {blockchainInfo && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Blockchain Verification</Text>
          <View style={[
            styles.resultBox,
            blockchainInfo.registered ? styles.resultSuccess : styles.resultWarning
          ]}>
            <Text style={styles.resultStatus}>
              {blockchainInfo.registered ? 'REGISTERED ON BLOCKCHAIN' : 'NOT FOUND ON BLOCKCHAIN'}
            </Text>
            {blockchainInfo.registered && (
              <>
                <Text style={styles.blockchainDetail}>
                  Block: {blockchainInfo.block_number}
                </Text>
                <Text style={styles.blockchainDetail}>
                  Timestamp: {new Date(blockchainInfo.timestamp).toLocaleString()}
                </Text>
                <TouchableOpacity 
                  style={styles.viewDetailsButton} 
                  onPress={viewBlockchainDetails}
                >
                  <Text style={styles.viewDetailsText}>View Full Details</Text>
                </TouchableOpacity>
              </>
            )}
          </View>
        </View>
      )}

      {/* Action buttons */}
      {verificationResult && (
        <View style={styles.actionsContainer}>
          <TouchableOpacity style={styles.button} onPress={resetVerification}>
            <Text style={styles.buttonText}>Reset</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={captureNew}>
            <Text style={styles.buttonText}>Capture New</Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  contentContainer: {
    padding: 16,
  },
  section: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 16,
    marginBottom: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  infoItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  infoLabel: {
    fontSize: 14,
    color: '#666',
  },
  infoValue: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
    maxWidth: '60%',
  },
  noSignatureContainer: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 24,
    marginBottom: 16,
    alignItems: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  noSignatureText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 16,
  },
  verifyContainer: {
    alignItems: 'center',
    marginBottom: 16,
  },
  button: {
    backgroundColor: '#4a90e2',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 200,
  },
  buttonDisabled: {
    backgroundColor: '#a0c0e8',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '500',
  },
  loadingButtonText: {
    marginLeft: 8,
  },
  resultBox: {
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  resultSuccess: {
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
    borderWidth: 1,
    borderColor: '#4CAF50',
  },
  resultWarning: {
    backgroundColor: 'rgba(255, 152, 0, 0.1)',
    borderWidth: 1,
    borderColor: '#FF9800',
  },
  resultFailure: {
    backgroundColor: 'rgba(244, 67, 54, 0.1)',
    borderWidth: 1,
    borderColor: '#F44336',
  },
  resultStatus: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  resultConfidence: {
    fontSize: 14,
    marginBottom: 8,
  },
  loadingContainer: {
    alignItems: 'center',
    padding: 24,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  matchesContainer: {
    marginTop: 16,
  },
  matchesTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#333',
  },
  matchItem: {
    backgroundColor: '#f9f9f9',
    padding: 12,
    borderRadius: 6,
    marginBottom: 8,
  },
  matchId: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 4,
  },
  matchSimilarity: {
    fontSize: 14,
    marginBottom: 8,
  },
  matchMetadata: {
    borderTopWidth: 1,
    borderTopColor: '#eee',
    paddingTop: 8,
  },
  metadataItem: {
    fontSize: 12,
    color: '#666',
    marginBottom: 2,
  },
  blockchainDetail: {
    fontSize: 14,
    marginBottom: 4,
  },
  viewDetailsButton: {
    marginTop: 8,
    paddingVertical: 8,
    paddingHorizontal: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    borderRadius: 4,
  },
  viewDetailsText: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
  },
  actionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 24,
  },
});

export default VerificationScreen;
