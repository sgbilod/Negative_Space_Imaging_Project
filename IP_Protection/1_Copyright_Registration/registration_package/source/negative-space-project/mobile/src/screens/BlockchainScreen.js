/**
 * BlockchainScreen.js
 * Screen for viewing blockchain details of a negative space signature
 */

import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  Share,
  Linking,
  Alert
} from 'react-native';

/**
 * Blockchain details screen component
 */
const BlockchainScreen = ({ navigation, route }) => {
  const [showRawData, setShowRawData] = useState(false);
  
  // Get signature and blockchain info from route params
  const signatureId = route.params?.signatureId || 'Unknown';
  const blockchainInfo = route.params?.blockchainInfo || null;
  
  // If no blockchain info is available, show error message
  if (!blockchainInfo) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorText}>
          No blockchain information available for this signature.
        </Text>
        <TouchableOpacity 
          style={styles.button} 
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.buttonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }
  
  /**
   * Format date from ISO string
   */
  const formatDate = (isoString) => {
    return new Date(isoString).toLocaleString();
  };
  
  /**
   * Open transaction in block explorer
   */
  const openInBlockExplorer = () => {
    // This would be the block explorer URL for your specific blockchain
    // For Ethereum, it would typically be Etherscan
    const url = `https://etherscan.io/tx/${blockchainInfo.transaction_hash}`;
    
    Linking.canOpenURL(url).then(supported => {
      if (supported) {
        Linking.openURL(url);
      } else {
        Alert.alert(
          'Cannot Open Link',
          'Unable to open the block explorer. Please check your internet connection.',
          [{ text: 'OK' }]
        );
      }
    });
  };
  
  /**
   * Share transaction details
   */
  const shareTransaction = async () => {
    try {
      const shareMessage = `
Negative Space Signature Verification
ID: ${signatureId}
Transaction: ${blockchainInfo.transaction_hash}
Block: ${blockchainInfo.block_number}
Timestamp: ${formatDate(blockchainInfo.timestamp)}
Verified on Ethereum blockchain
      `.trim();
      
      await Share.share({
        message: shareMessage,
        title: 'Negative Space Signature Verification'
      });
    } catch (error) {
      Alert.alert('Error', 'Could not share blockchain information');
    }
  };
  
  /**
   * Toggle between formatted and raw data view
   */
  const toggleDataView = () => {
    setShowRawData(!showRawData);
  };
  
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Blockchain Verification</Text>
        <View style={styles.registrationBadge}>
          <Text style={styles.registrationText}>REGISTERED</Text>
        </View>
      </View>
      
      {/* Signature ID */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Signature</Text>
        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>ID:</Text>
          <Text style={styles.infoValue}>{signatureId}</Text>
        </View>
      </View>
      
      {/* Transaction Details */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Transaction Details</Text>
        
        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>Transaction Hash:</Text>
          <Text style={styles.infoValue} numberOfLines={1} ellipsizeMode="middle">
            {blockchainInfo.transaction_hash}
          </Text>
        </View>
        
        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>Block Number:</Text>
          <Text style={styles.infoValue}>{blockchainInfo.block_number}</Text>
        </View>
        
        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>Timestamp:</Text>
          <Text style={styles.infoValue}>{formatDate(blockchainInfo.timestamp)}</Text>
        </View>
        
        {/* Additional details would be included here */}
        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>Status:</Text>
          <Text style={[styles.infoValue, styles.confirmedText]}>Confirmed</Text>
        </View>
      </View>
      
      {/* Raw Data (togglable) */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Blockchain Data</Text>
          <TouchableOpacity 
            style={styles.toggleButton} 
            onPress={toggleDataView}
          >
            <Text style={styles.toggleButtonText}>
              {showRawData ? 'Show Formatted' : 'Show Raw'}
            </Text>
          </TouchableOpacity>
        </View>
        
        {showRawData ? (
          <View style={styles.rawDataContainer}>
            <Text style={styles.rawDataText}>
              {JSON.stringify(blockchainInfo, null, 2)}
            </Text>
          </View>
        ) : (
          <View>
            <Text style={styles.dataDescription}>
              This signature has been securely registered on the Ethereum blockchain. 
              The registration is immutable and can be independently verified.
            </Text>
            
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>Contract Address:</Text>
              <Text style={styles.infoValue} numberOfLines={1}>
                0x7890abcdef1234567890abcdef123456789abcde
              </Text>
            </View>
            
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>Method:</Text>
              <Text style={styles.infoValue}>registerSignature(bytes32, bytes)</Text>
            </View>
          </View>
        )}
      </View>
      
      {/* Actions */}
      <View style={styles.actionsContainer}>
        <TouchableOpacity 
          style={styles.actionButton} 
          onPress={openInBlockExplorer}
        >
          <Text style={styles.actionButtonText}>View in Block Explorer</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={styles.actionButton} 
          onPress={shareTransaction}
        >
          <Text style={styles.actionButtonText}>Share</Text>
        </TouchableOpacity>
      </View>
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
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  errorText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 24,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
  },
  registrationBadge: {
    backgroundColor: '#4CAF50',
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 16,
  },
  registrationText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 12,
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
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
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
    flex: 1,
  },
  infoValue: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
    flex: 2,
    textAlign: 'right',
  },
  confirmedText: {
    color: '#4CAF50',
  },
  toggleButton: {
    backgroundColor: '#f0f0f0',
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 4,
  },
  toggleButtonText: {
    fontSize: 12,
    color: '#666',
  },
  rawDataContainer: {
    backgroundColor: '#f8f8f8',
    padding: 12,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#eee',
  },
  rawDataText: {
    fontFamily: 'monospace',
    fontSize: 12,
    color: '#333',
  },
  dataDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
    lineHeight: 20,
  },
  actionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  actionButton: {
    backgroundColor: '#4a90e2',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    flex: 1,
    marginHorizontal: 4,
    alignItems: 'center',
  },
  actionButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
  },
  button: {
    backgroundColor: '#4a90e2',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '500',
  },
});

export default BlockchainScreen;
