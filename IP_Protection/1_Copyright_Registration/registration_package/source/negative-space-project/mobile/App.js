/**
 * App.js
 * Main entry point for the Negative Space Mobile Application
 */

import React, { useState, useEffect } from 'react';
import { 
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  StatusBar,
  TouchableOpacity,
  ActivityIndicator,
  Alert
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

// Import screens
import HomeScreen from './src/screens/HomeScreen';
import CameraScreen from './src/screens/CameraScreen';
import VerificationScreen from './src/screens/VerificationScreen';
import BlockchainScreen from './src/screens/BlockchainScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import ARScreen from './src/screens/ARScreen';
import FieldVerificationScreen from './src/screens/FieldVerificationScreen';
        <Stack.Screen 
          name="FieldVerification" 
          options={{ title: 'Field Verification' }}
          component={FieldVerificationScreen}
        />

// Import API
import NegativeSpaceAPI from './api/NegativeSpaceAPI';

// Create the navigation stack
const Stack = createStackNavigator();

// Default server settings
const DEFAULT_SERVER_URL = '192.168.1.100:8080';

/**
 * Main application component
 */
const App = () => {
  const [api, setApi] = useState(null);
  const [isConnecting, setIsConnecting] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [offlineMode, setOfflineMode] = useState(false);

  // Initialize the API on app start
  useEffect(() => {
    initializeAPI();
  }, []);

  /**
   * Initialize the Negative Space API
   */
  const initializeAPI = async () => {
    try {
      setIsConnecting(true);
      
      // Create new API instance
      const newApi = new NegativeSpaceAPI(DEFAULT_SERVER_URL, {
        offlineMode: false,
        useWebsocket: true,
        useNativeBindings: true
      });
      
      // Try to connect
      const connected = await newApi.connect();
      
      setApi(newApi);
      setIsConnected(connected);
      
      if (!connected) {
        // If connection fails, prompt to use offline mode
        promptOfflineMode();
      }
    } catch (error) {
      console.error('API initialization error:', error);
      promptOfflineMode();
    } finally {
      setIsConnecting(false);
    }
  };

  /**
   * Prompt user to enable offline mode if connection fails
   */
  const promptOfflineMode = () => {
    Alert.alert(
      'Connection Failed',
      'Could not connect to the Negative Space server. Would you like to work in offline mode?',
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
   * Enable offline mode for the application
   */
  const enableOfflineMode = async () => {
    try {
      const offlineApi = new NegativeSpaceAPI('', {
        offlineMode: true,
        useWebsocket: false,
        useNativeBindings: true
      });
      
      setApi(offlineApi);
      setOfflineMode(true);
    } catch (error) {
      console.error('Failed to enable offline mode:', error);
      Alert.alert(
        'Offline Mode Error',
        'Could not enable offline mode. The application may not function correctly.'
      );
    }
  };

  // If still connecting, show loading indicator
  if (isConnecting) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text style={styles.loadingText}>Connecting to Negative Space server...</Text>
      </View>
    );
  }

  // Render the main application
  return (
    <NavigationContainer>
      <StatusBar barStyle="dark-content" />
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerStyle: {
            backgroundColor: '#4a90e2',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}
      >
        <Stack.Screen 
          name="AR" 
          options={{ title: 'AR Visualization' }}
          component={ARScreen}
        />
        <Stack.Screen 
          name="Home" 
          options={{ title: 'Negative Space' }}
        >
          {props => <HomeScreen {...props} api={api} offlineMode={offlineMode} />}
        </Stack.Screen>
        
        <Stack.Screen 
          name="Camera" 
          options={{ title: 'Capture Negative Space' }}
        >
          {props => <CameraScreen {...props} api={api} />}
        </Stack.Screen>
        
        <Stack.Screen 
          name="Verification" 
          options={{ title: 'Verify Signature' }}
        >
          {props => <VerificationScreen {...props} api={api} />}
        </Stack.Screen>
        
        <Stack.Screen 
          name="Blockchain" 
          options={{ title: 'Blockchain Details' }}
          component={BlockchainScreen}
        />
        
        <Stack.Screen 
          name="Settings" 
          options={{ title: 'Settings' }}
        >
          {props => (
            <SettingsScreen 
              {...props} 
              api={api} 
              offlineMode={offlineMode}
              setOfflineMode={setOfflineMode}
              reconnect={initializeAPI}
            />
          )}
        </Stack.Screen>
      </Stack.Navigator>
    </NavigationContainer>
  );
};

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  loadingText: {
    marginTop: 20,
    fontSize: 16,
    color: '#333',
  },
});

export default App;
