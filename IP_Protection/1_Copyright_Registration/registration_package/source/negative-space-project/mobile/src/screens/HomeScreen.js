  /**
   * Navigate to the field verification screen
   */
  const navigateToFieldVerification = () => {
    navigation.navigate('FieldVerification');
  };
/**
 * HomeScreen.js
 * Main dashboard screen for the Negative Space Mobile Application
 */

import React from 'react';
import {
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  ScrollView,
  Alert
} from 'react-native';

/**
 * Home screen component
 */
const HomeScreen = ({ navigation, api, offlineMode }) => {
  /**
   * Navigate to the camera screen
   */
  const navigateToCamera = () => {
    navigation.navigate('Camera');
  };

  /**
   * Navigate to the verification screen
   */
  const navigateToVerification = () => {
    navigation.navigate('Verification');
  };

  /**
   * Navigate to the blockchain screen
   * Checks if offline mode is enabled first
   */
  const navigateToBlockchain = () => {
    if (offlineMode) {
      Alert.alert(
        'Offline Mode',
        'Blockchain features are not available in offline mode. Please connect to a server to use blockchain features.',
        [{ text: 'OK' }]
      );
      return;
    }
    
    navigation.navigate('Blockchain');
  };

  /**
   * Navigate to the AR visualization screen
   */
  const navigateToAR = () => {
    navigation.navigate('AR');
  };

  /**
   * Navigate to the settings screen
   */
  const navigateToSettings = () => {
    navigation.navigate('Settings');
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Negative Space</Text>
          <Text style={styles.subtitle}>Signature Verification</Text>
          
          {offlineMode && (
            <View style={styles.offlineBadge}>
              <Text style={styles.offlineText}>OFFLINE MODE</Text>
            </View>
          )}
        </View>
        
        {/* Main menu buttons */}
        <View style={styles.buttonGrid}>
          <TouchableOpacity 
            style={styles.menuButton} 
            onPress={navigateToCamera}
          >
            <View style={styles.buttonContent}>
              <View style={styles.iconContainer}>
                {/* Replace with actual icon */}
                <Text style={styles.iconText}>📷</Text>
              </View>
              <Text style={styles.buttonText}>Capture Negative Space</Text>
            </View>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.menuButton} 
            onPress={navigateToVerification}
          >
            <View style={styles.buttonContent}>
              <View style={styles.iconContainer}>
                <Text style={styles.iconText}>✓</Text>
              </View>
              <Text style={styles.buttonText}>Verify Signature</Text>
            </View>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.menuButton} 
            onPress={navigateToBlockchain}
          >
            <View style={styles.buttonContent}>
              <View style={styles.iconContainer}>
                <Text style={styles.iconText}>🔗</Text>
              </View>
              <Text style={styles.buttonText}>Blockchain Records</Text>
              {offlineMode && <Text style={styles.disabledText}>(Unavailable Offline)</Text>}
            </View>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.menuButton} 
            onPress={navigateToAR}
          >
            <View style={styles.buttonContent}>
              <View style={styles.iconContainer}>
                <Text style={styles.iconText}>🕶️</Text>
              </View>
              <Text style={styles.buttonText}>AR Visualization</Text>
            </View>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.menuButton} 
            onPress={navigateToFieldVerification}
          >
            <View style={styles.buttonContent}>
              <View style={styles.iconContainer}>
                <Text style={styles.iconText}>📝</Text>
              </View>
              <Text style={styles.buttonText}>Field Verification</Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity 
            style={styles.menuButton} 
            onPress={navigateToSettings}
          >
            <View style={styles.buttonContent}>
              <View style={styles.iconContainer}>
                <Text style={styles.iconText}>⚙️</Text>
              </View>
              <Text style={styles.buttonText}>Settings</Text>
            </View>
          </TouchableOpacity>
        </View>
        
        {/* Status information */}
        <View style={styles.statusContainer}>
          <Text style={styles.statusTitle}>Status</Text>
          <View style={styles.statusItem}>
            <Text style={styles.statusLabel}>Server Connection:</Text>
            <Text style={[
              styles.statusValue, 
              offlineMode ? styles.statusValueOffline : styles.statusValueOnline
            ]}>
              {offlineMode ? 'Offline' : 'Online'}
            </Text>
          </View>
        </View>
        
        {/* Information section */}
        <View style={styles.infoContainer}>
          <Text style={styles.infoTitle}>About Negative Space Imaging</Text>
          <Text style={styles.infoText}>
            This application allows you to capture, verify, and manage negative space signatures. 
            Negative space signatures provide a cryptographically secure way to verify the 
            authenticity of physical objects by analyzing the empty spaces between objects.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContent: {
    padding: 16,
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
  },
  subtitle: {
    fontSize: 18,
    color: '#666',
    marginTop: 4,
  },
  offlineBadge: {
    backgroundColor: '#ff9800',
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 16,
    marginTop: 12,
  },
  offlineText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 12,
  },
  buttonGrid: {
    marginBottom: 24,
  },
  menuButton: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  buttonContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#4a90e2',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  iconText: {
    fontSize: 20,
    color: 'white',
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  disabledText: {
    fontSize: 12,
    color: '#999',
    marginTop: 4,
  },
  statusContainer: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 16,
    marginBottom: 24,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  statusTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  statusItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  statusLabel: {
    fontSize: 14,
    color: '#666',
  },
  statusValue: {
    fontSize: 14,
    fontWeight: '500',
  },
  statusValueOnline: {
    color: '#4caf50',
  },
  statusValueOffline: {
    color: '#ff9800',
  },
  infoContainer: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  infoTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
});

export default HomeScreen;
