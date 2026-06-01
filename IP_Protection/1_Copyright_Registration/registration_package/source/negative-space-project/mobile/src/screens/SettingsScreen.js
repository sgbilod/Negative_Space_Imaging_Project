/**
 * SettingsScreen.js
 * Settings screen for the Negative Space Mobile Application
 */

import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  Switch,
  TextInput,
  Alert,
  ActivityIndicator
} from 'react-native';

/**
 * Settings screen component
 */
const SettingsScreen = ({ 
  navigation, 
  api, 
  offlineMode, 
  setOfflineMode,
  reconnect
}) => {
  const [serverUrl, setServerUrl] = useState(api?.serverUrl || '192.168.1.100:8080');
  const [useWebsocket, setUseWebsocket] = useState(api?.options?.useWebsocket || true);
  const [useNativeBindings, setUseNativeBindings] = useState(api?.options?.useNativeBindings || true);
  const [isConnecting, setIsConnecting] = useState(false);
  
  /**
   * Save settings and attempt to connect
   */
  const saveSettings = async () => {
    if (!serverUrl.trim()) {
      Alert.alert(
        'Invalid Server URL',
        'Please enter a valid server URL',
        [{ text: 'OK' }]
      );
      return;
    }
    
    try {
      setIsConnecting(true);
      
      // Apply settings and reconnect
      if (offlineMode) {
        // If currently in offline mode, attempt to reconnect
        reconnect();
      } else {
        // Update API settings
        if (api) {
          api.serverUrl = serverUrl;
          api.options = {
            ...api.options,
            useWebsocket,
            useNativeBindings,
            offlineMode: false
          };
          
          // Try to connect with new settings
          const connected = await api.connect();
          
          if (connected) {
            Alert.alert(
              'Connection Successful',
              'Successfully connected to server with new settings',
              [{ text: 'OK' }]
            );
          } else {
            promptOfflineMode();
          }
        }
      }
    } catch (error) {
      console.error('Settings update error:', error);
      Alert.alert(
        'Connection Error',
        'Failed to connect with the provided settings',
        [{ text: 'OK' }]
      );
    } finally {
      setIsConnecting(false);
    }
  };
  
  /**
   * Prompt user to enable offline mode
   */
  const promptOfflineMode = () => {
    Alert.alert(
      'Connection Failed',
      'Could not connect to the server. Would you like to work in offline mode?',
      [
        {
          text: 'No',
          style: 'cancel'
        },
        {
          text: 'Yes',
          onPress: enableOfflineMode
        }
      ]
    );
  };
  
  /**
   * Enable offline mode
   */
  const enableOfflineMode = () => {
    if (api) {
      api.options = {
        ...api.options,
        offlineMode: true,
        useWebsocket: false
      };
      setOfflineMode(true);
    }
  };
  
  /**
   * Toggle offline mode
   */
  const toggleOfflineMode = (value) => {
    if (value) {
      // Turning on offline mode
      enableOfflineMode();
    } else {
      // Turning off offline mode, attempt to connect
      reconnect();
    }
  };
  
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Server Connection Settings */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Server Connection</Text>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>Server URL:</Text>
          <TextInput
            style={styles.input}
            value={serverUrl}
            onChangeText={setServerUrl}
            placeholder="e.g., 192.168.1.100:8080"
            editable={!offlineMode}
          />
        </View>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>Use WebSocket:</Text>
          <Switch
            value={useWebsocket}
            onValueChange={setUseWebsocket}
            disabled={offlineMode}
          />
        </View>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>Use Native Bindings:</Text>
          <Switch
            value={useNativeBindings}
            onValueChange={setUseNativeBindings}
          />
        </View>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>Offline Mode:</Text>
          <Switch
            value={offlineMode}
            onValueChange={toggleOfflineMode}
          />
        </View>
        
        {offlineMode && (
          <View style={styles.offlineNotice}>
            <Text style={styles.offlineNoticeText}>
              In offline mode, blockchain features and remote verification are not available.
              Only local signature extraction and verification will work.
            </Text>
          </View>
        )}
        
        <TouchableOpacity
          style={[styles.button, (isConnecting || offlineMode) && styles.buttonDisabled]}
          onPress={saveSettings}
          disabled={isConnecting || offlineMode}
        >
          {isConnecting ? (
            <>
              <ActivityIndicator size="small" color="#fff" />
              <Text style={[styles.buttonText, styles.loadingButtonText]}>
                Connecting...
              </Text>
            </>
          ) : (
            <Text style={styles.buttonText}>
              {offlineMode ? 'Reconnect to Server' : 'Save and Connect'}
            </Text>
          )}
        </TouchableOpacity>
      </View>
      
      {/* Camera Settings */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Camera Settings</Text>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>Default Flash Mode:</Text>
          <View style={styles.segmentedControl}>
            <TouchableOpacity style={[styles.segment, styles.segmentActive]}>
              <Text style={styles.segmentTextActive}>Auto</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.segment}>
              <Text style={styles.segmentText}>On</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.segment}>
              <Text style={styles.segmentText}>Off</Text>
            </TouchableOpacity>
          </View>
        </View>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>High Resolution:</Text>
          <Switch value={true} />
        </View>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>Stabilization:</Text>
          <Switch value={true} />
        </View>
      </View>
      
      {/* Verification Settings */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Verification Settings</Text>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>Verification Strictness:</Text>
          <View style={styles.segmentedControl}>
            <TouchableOpacity style={styles.segment}>
              <Text style={styles.segmentText}>Low</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.segment, styles.segmentActive]}>
              <Text style={styles.segmentTextActive}>Medium</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.segment}>
              <Text style={styles.segmentText}>High</Text>
            </TouchableOpacity>
          </View>
        </View>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>Auto-Verify After Capture:</Text>
          <Switch value={true} />
        </View>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>Cache Verification Results:</Text>
          <Switch value={true} />
        </View>
      </View>
      
      {/* About */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>About</Text>
        
        <View style={styles.aboutContainer}>
          <Text style={styles.appName}>Negative Space Mobile</Text>
          <Text style={styles.appVersion}>Version 0.1.0</Text>
          <Text style={styles.appCopyright}>© 2023 Negative Space Imaging Project</Text>
          
          <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.linkButton}>
              <Text style={styles.linkButtonText}>Privacy Policy</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.linkButton}>
              <Text style={styles.linkButtonText}>Terms of Service</Text>
            </TouchableOpacity>
          </View>
        </View>
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
    marginBottom: 16,
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  settingLabel: {
    fontSize: 16,
    color: '#333',
  },
  input: {
    flex: 1,
    height: 40,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 4,
    paddingHorizontal: 8,
    marginLeft: 16,
    fontSize: 14,
  },
  segmentedControl: {
    flexDirection: 'row',
    borderWidth: 1,
    borderColor: '#4a90e2',
    borderRadius: 4,
    overflow: 'hidden',
  },
  segment: {
    paddingVertical: 6,
    paddingHorizontal: 12,
  },
  segmentActive: {
    backgroundColor: '#4a90e2',
  },
  segmentText: {
    color: '#4a90e2',
    fontSize: 14,
  },
  segmentTextActive: {
    color: 'white',
    fontSize: 14,
  },
  offlineNotice: {
    backgroundColor: 'rgba(255, 152, 0, 0.1)',
    borderWidth: 1,
    borderColor: '#FF9800',
    borderRadius: 4,
    padding: 12,
    marginTop: 16,
    marginBottom: 16,
  },
  offlineNoticeText: {
    color: '#FF9800',
    fontSize: 14,
  },
  button: {
    backgroundColor: '#4a90e2',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 16,
    flexDirection: 'row',
    justifyContent: 'center',
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
  aboutContainer: {
    alignItems: 'center',
    paddingVertical: 16,
  },
  appName: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  appVersion: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  appCopyright: {
    fontSize: 12,
    color: '#999',
    marginTop: 16,
    marginBottom: 16,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    marginTop: 8,
  },
  linkButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  linkButtonText: {
    color: '#4a90e2',
    fontSize: 14,
  },
});

export default SettingsScreen;
