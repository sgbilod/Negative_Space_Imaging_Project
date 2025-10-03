/**
 * SignatureCard.js
 * Component for displaying signature information
 */

import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image
} from 'react-native';
import { formatSignatureData, extractMetadata } from '../utils/signatureUtils';

/**
 * SignatureCard component
 * @param {Object} props - Component props
 * @param {Object} props.signatureData - The signature data to display
 * @param {Function} props.onPress - Function to call when card is pressed
 * @param {boolean} props.showDetails - Whether to show detailed information
 * @param {Object} props.verificationResult - Optional verification result
 * @param {Object} props.style - Additional style for the card
 */
const SignatureCard = ({
  signatureData,
  onPress,
  showDetails = false,
  verificationResult = null,
  style
}) => {
  const [expanded, setExpanded] = useState(showDetails);
  
  // Format the signature data for display
  const formattedData = formatSignatureData(signatureData);
  
  // Extract metadata for display
  const metadata = extractMetadata(signatureData);
  
  if (!formattedData) {
    return null;
  }
  
  /**
   * Toggle expanded state
   */
  const toggleExpanded = () => {
    setExpanded(!expanded);
  };
  
  /**
   * Handle card press
   */
  const handlePress = () => {
    if (onPress) {
      onPress(signatureData);
    } else {
      toggleExpanded();
    }
  };
  
  return (
    <TouchableOpacity
      style={[styles.container, style]}
      onPress={handlePress}
      activeOpacity={0.7}
    >
      {/* Card Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.title} numberOfLines={1} ellipsizeMode="middle">
            {formattedData.id}
          </Text>
          <Text style={styles.subtitle}>
            {formattedData.timestamp}
          </Text>
        </View>
        
        {verificationResult && (
          <View style={[
            styles.verificationBadge,
            verificationResult.verified ? styles.verifiedBadge : styles.notVerifiedBadge
          ]}>
            <Text style={styles.verificationText}>
              {verificationResult.verified ? 'VERIFIED' : 'NOT VERIFIED'}
            </Text>
          </View>
        )}
      </View>
      
      {/* Card Content */}
      <View style={styles.content}>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Confidence:</Text>
          <Text style={styles.infoValue}>{formattedData.confidence}</Text>
        </View>
        
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Features:</Text>
          <Text style={styles.infoValue}>{formattedData.featureCount}</Text>
        </View>
        
        {/* Expanded content */}
        {expanded && (
          <View style={styles.expandedContent}>
            {metadata.length > 0 && (
              <>
                <Text style={styles.sectionTitle}>Metadata</Text>
                {metadata.map((item, index) => (
                  <View key={index} style={styles.metadataRow}>
                    <Text style={styles.metadataLabel}>{item.key}:</Text>
                    <Text style={styles.metadataValue}>{item.value}</Text>
                  </View>
                ))}
              </>
            )}
            
            {verificationResult && verificationResult.matches && (
              <>
                <Text style={styles.sectionTitle}>Matches</Text>
                {verificationResult.matches.map((match, index) => (
                  <View key={index} style={styles.matchItem}>
                    <Text style={styles.matchId} numberOfLines={1} ellipsizeMode="middle">
                      {match.id}
                    </Text>
                    <Text style={styles.matchSimilarity}>
                      Similarity: {(match.similarity * 100).toFixed(2)}%
                    </Text>
                  </View>
                ))}
              </>
            )}
          </View>
        )}
      </View>
      
      {/* Expand/Collapse button */}
      <TouchableOpacity style={styles.expandButton} onPress={toggleExpanded}>
        <Text style={styles.expandButtonText}>
          {expanded ? 'Show Less' : 'Show More'}
        </Text>
      </TouchableOpacity>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
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
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  headerLeft: {
    flex: 1,
    marginRight: 8,
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 12,
    color: '#666',
  },
  verificationBadge: {
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 4,
  },
  verifiedBadge: {
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
    borderWidth: 1,
    borderColor: '#4CAF50',
  },
  notVerifiedBadge: {
    backgroundColor: 'rgba(244, 67, 54, 0.1)',
    borderWidth: 1,
    borderColor: '#F44336',
  },
  verificationText: {
    fontSize: 10,
    fontWeight: 'bold',
  },
  content: {
    marginBottom: 12,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 4,
  },
  infoLabel: {
    fontSize: 14,
    color: '#666',
  },
  infoValue: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
  },
  expandedContent: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#333',
  },
  metadataRow: {
    flexDirection: 'row',
    paddingVertical: 3,
  },
  metadataLabel: {
    fontSize: 12,
    color: '#666',
    width: '40%',
  },
  metadataValue: {
    fontSize: 12,
    color: '#333',
    width: '60%',
  },
  matchItem: {
    backgroundColor: '#f9f9f9',
    padding: 8,
    borderRadius: 4,
    marginBottom: 6,
  },
  matchId: {
    fontSize: 12,
    fontWeight: '500',
    marginBottom: 2,
  },
  matchSimilarity: {
    fontSize: 12,
    color: '#666',
  },
  expandButton: {
    alignItems: 'center',
    paddingVertical: 8,
  },
  expandButtonText: {
    fontSize: 12,
    color: '#4a90e2',
    fontWeight: '500',
  }
});

export default SignatureCard;
