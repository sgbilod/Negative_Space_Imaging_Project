"""
Mobile AR Client for the Decentralized Notary Network

This module provides React Native components for interacting with the
Decentralized Time Notary Network from a mobile AR application.
"""

import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  ScrollView,
  SafeAreaView,
  StatusBar,
  Image,
} from 'react-native';
import { ViroARScene, ViroARSceneNavigator, ViroConstants } from 'react-viro';
import axios from 'axios';
import * as Location from 'expo-location';
import { Camera } from 'expo-camera';
import * as FileSystem from 'expo-file-system';
import * as Crypto from 'expo-crypto';
import MapView, { Marker } from 'react-native-maps';

// Configuration
const API_BASE_URL = 'https://notary-api.example.com';

// Utility functions
const hashDocument = async (fileUri) => {
  const fileContent = await FileSystem.readAsStringAsync(fileUri);
  return Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    fileContent
  );
};

// Main AR Scene component
const ProofOfViewARScene = ({ onCaptureSpatialSignature, nearbyLandmarks }) => {
  const [trackingStatus, setTrackingStatus] = useState('Initializing AR...');
  const [pointCloud, setPointCloud] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const recordingInterval = useRef(null);
  const pointsRef = useRef([]);

  // Handle AR tracking state changes
  const onTrackingUpdated = (state, reason) => {
    if (state === ViroConstants.TRACKING_NORMAL) {
      setTrackingStatus('AR Ready');
    } else if (state === ViroConstants.TRACKING_NONE) {
      setTrackingStatus(`Not Tracking: ${reason}`);
    }
  };

  // Handle point cloud updates
  const onARPointCloudUpdate = (pointCloud) => {
    if (isRecording) {
      // Add new points to the collection
      pointsRef.current = [...pointsRef.current, ...pointCloud.points];
    }
    // Update displayed point count
    setPointCloud(pointCloud.points);
  };

  // Start recording spatial data
  const startRecording = () => {
    pointsRef.current = [];
    setIsRecording(true);
    setRecordingTime(0);

    recordingInterval.current = setInterval(() => {
      setRecordingTime((prevTime) => {
        const newTime = prevTime + 1;
        
        // Auto-stop after 10 seconds
        if (newTime >= 10) {
          stopRecording();
        }
        
        return newTime;
      });
    }, 1000);
  };

  // Stop recording and process spatial data
  const stopRecording = () => {
    if (recordingInterval.current) {
      clearInterval(recordingInterval.current);
      recordingInterval.current = null;
    }

    setIsRecording(false);

    // Process and normalize the point cloud data
    if (pointsRef.current.length > 0) {
      const processedPoints = preprocessPointCloud(pointsRef.current);
      onCaptureSpatialSignature(processedPoints);
    } else {
      Alert.alert(
        'Insufficient Data', 
        'Not enough spatial points were captured. Please try again in a different area.'
      );
    }
  };

  // Preprocess the point cloud for signature generation
  const preprocessPointCloud = (points) => {
    // Filter out duplicate points
    const uniquePoints = points.filter((point, index, self) => 
      index === self.findIndex((p) => (
        p[0] === point[0] && p[1] === point[1] && p[2] === point[2]
      ))
    );

    // Take up to 100 points for processing
    return uniquePoints.slice(0, 100);
  };

  // Render AR markers for nearby landmarks
  const renderLandmarkMarkers = () => {
    if (!nearbyLandmarks || nearbyLandmarks.length === 0) return null;

    return nearbyLandmarks.map((landmark) => (
      <ViroARImageMarker
        key={landmark.landmark_id}
        position={[
          landmark.location.longitude, 
          0, 
          landmark.location.latitude
        ]}
        height={0.5}
        width={0.5}
        rotation={[-90, 0, 0]}
      >
        <ViroText
          text={landmark.name}
          scale={[0.5, 0.5, 0.5]}
          position={[0, 0.25, 0]}
          style={{ fontSize: 20, color: 'white' }}
        />
      </ViroARImageMarker>
    ));
  };

  return (
    <ViroARScene onTrackingUpdated={onTrackingUpdated} onARPointCloudUpdate={onARPointCloudUpdate}>
      {/* Status text */}
      <ViroText
        text={`${trackingStatus} | Points: ${pointCloud.length} | ${
          isRecording ? `Recording: ${recordingTime}s` : 'Ready'
        }`}
        position={[0, -0.5, -1]}
        style={{ fontSize: 16, color: 'white' }}
      />

      {/* Render landmark markers */}
      {renderLandmarkMarkers()}

      {/* Visual indicator for recording */}
      {isRecording && (
        <ViroText
          text="Recording Spatial Signature..."
          position={[0, 0, -1]}
          style={{ fontSize: 20, color: 'red' }}
          animation={{ name: 'pulse', run: true, loop: true }}
        />
      )}
    </ViroARScene>
  );
};

// Main application component
export const NotaryARApp = () => {
  const [isARMode, setIsARMode] = useState(false);
  const [userLocation, setUserLocation] = useState(null);
  const [nearbyLandmarks, setNearbyLandmarks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [spatialSignature, setSpatialSignature] = useState(null);
  const [selectedLandmark, setSelectedLandmark] = useState(null);
  const [nodeId, setNodeId] = useState(null);
  const [notarizations, setNotarizations] = useState([]);

  // Initialize location tracking
  useEffect(() => {
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setError('Permission to access location was denied');
        return;
      }

      try {
        setLoading(true);
        const location = await Location.getCurrentPositionAsync({});
        setUserLocation(location.coords);
        await fetchNearbyLandmarks(location.coords);
      } catch (err) {
        setError(`Error fetching location: ${err.message}`);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // Fetch nearby landmarks from the API
  const fetchNearbyLandmarks = async (coords) => {
    try {
      setLoading(true);
      const response = await axios.get(
        `${API_BASE_URL}/landmarks/near?latitude=${coords.latitude}&longitude=${coords.longitude}&radius_km=10`
      );
      setNearbyLandmarks(response.data.landmarks);
      setError(null);
    } catch (err) {
      setError(`Error fetching landmarks: ${err.message}`);
      setNearbyLandmarks([]);
    } finally {
      setLoading(false);
    }
  };

  // Handle spatial signature capture from AR
  const handleCaptureSpatialSignature = async (pointCloudData) => {
    try {
      setLoading(true);
      
      // In a real implementation, we would send the point cloud to the API
      // and receive back the generated signature
      // For Phase 1, we'll just use the raw points
      
      setSpatialSignature(pointCloudData);
      setIsARMode(false);
      setError(null);
      
      Alert.alert(
        'Signature Captured', 
        `Successfully captured spatial signature with ${pointCloudData.length} points.`
      );
    } catch (err) {
      setError(`Error processing spatial signature: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Register as a notary node
  const registerNode = async () => {
    try {
      setLoading(true);
      
      const response = await axios.post(`${API_BASE_URL}/nodes`, {
        owner_id: `user-${Date.now()}`, // In a real app, this would be a user ID
        owner_data: {
          device_info: `Mobile AR Client`,
          registration_time: new Date().toISOString()
        }
      });
      
      setNodeId(response.data.node_id);
      setError(null);
      
      Alert.alert(
        'Node Registered', 
        `You are now registered as notary node: ${response.data.node_id}`
      );
    } catch (err) {
      setError(`Error registering node: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Submit proof of view for a landmark
  const submitProofOfView = async () => {
    if (!nodeId) {
      Alert.alert('Error', 'Please register as a notary node first');
      return;
    }
    
    if (!selectedLandmark) {
      Alert.alert('Error', 'Please select a landmark');
      return;
    }
    
    if (!spatialSignature) {
      Alert.alert('Error', 'Please capture a spatial signature first');
      return;
    }
    
    try {
      setLoading(true);
      
      const response = await axios.post(`${API_BASE_URL}/nodes/proof-of-view`, {
        node_id: nodeId,
        landmark_id: selectedLandmark.landmark_id,
        proof_signature: spatialSignature
      });
      
      if (response.data.success) {
        Alert.alert(
          'Proof Submitted', 
          `Successfully submitted proof of view for ${selectedLandmark.name}. Match score: ${response.data.validation.match_score.toFixed(2)}`
        );
        setError(null);
      } else {
        setError(`Proof submission failed: ${response.data.reason}`);
        Alert.alert('Submission Failed', response.data.reason);
      }
    } catch (err) {
      setError(`Error submitting proof: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Notarize a document
  const notarizeDocument = async () => {
    if (!nodeId) {
      Alert.alert('Error', 'Please register as a notary node first');
      return;
    }
    
    try {
      setLoading(true);
      
      // In a real app, this would be a file picker or camera
      const documentHash = await Crypto.digestStringAsync(
        Crypto.CryptoDigestAlgorithm.SHA256,
        `Sample document content ${Date.now()}`
      );
      
      const response = await axios.post(`${API_BASE_URL}/notarize`, {
        document_hash: documentHash,
        metadata: {
          notarized_by: nodeId,
          device_type: 'Mobile AR Client',
          timestamp: new Date().toISOString()
        }
      });
      
      if (response.data.success) {
        const newNotarization = {
          id: response.data.notarization.notarization_id,
          documentHash: documentHash,
          timestamp: response.data.notarization.notarized_at,
          consensus: response.data.notarization.consensus_nodes
        };
        
        setNotarizations([newNotarization, ...notarizations]);
        
        Alert.alert(
          'Document Notarized', 
          `Successfully notarized document with ID: ${response.data.notarization.notarization_id}`
        );
        setError(null);
      } else {
        setError(`Notarization failed: ${response.data.reason}`);
        Alert.alert('Notarization Failed', response.data.reason);
      }
    } catch (err) {
      setError(`Error notarizing document: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Verify a notarization
  const verifyNotarization = async (notarizationId) => {
    try {
      setLoading(true);
      
      const response = await axios.post(`${API_BASE_URL}/verify`, {
        notarization_id: notarizationId
      });
      
      if (response.data.verified) {
        Alert.alert(
          'Verification Successful', 
          `The notarization ${notarizationId} is valid and verified on the blockchain.`
        );
      } else {
        Alert.alert(
          'Verification Failed', 
          `The notarization could not be verified: ${response.data.network_verification.reason}`
        );
      }
      
      setError(null);
    } catch (err) {
      setError(`Error verifying notarization: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Render the AR viewer
  if (isARMode) {
    return (
      <ViroARSceneNavigator
        initialScene={{
          scene: () => (
            <ProofOfViewARScene
              onCaptureSpatialSignature={handleCaptureSpatialSignature}
              nearbyLandmarks={nearbyLandmarks}
            />
          ),
        }}
        style={{ flex: 1 }}
      />
    );
  }

  // Render the main UI
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Decentralized Notary</Text>
        {nodeId && <Text style={styles.nodeId}>Node: {nodeId.substring(0, 8)}...</Text>}
      </View>
      
      {error && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}
      
      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#0066CC" />
          <Text style={styles.loadingText}>Processing...</Text>
        </View>
      )}
      
      {userLocation && (
        <View style={styles.mapContainer}>
          <MapView
            style={styles.map}
            initialRegion={{
              latitude: userLocation.latitude,
              longitude: userLocation.longitude,
              latitudeDelta: 0.0922,
              longitudeDelta: 0.0421,
            }}
          >
            {/* User location */}
            <Marker
              coordinate={{
                latitude: userLocation.latitude,
                longitude: userLocation.longitude,
              }}
              title="Your Location"
              pinColor="blue"
            />
            
            {/* Landmark markers */}
            {nearbyLandmarks.map((landmark) => (
              <Marker
                key={landmark.landmark_id}
                coordinate={{
                  latitude: landmark.location.latitude,
                  longitude: landmark.location.longitude,
                }}
                title={landmark.name}
                description={`${landmark.description} (${landmark.distance_km.toFixed(2)} km)`}
                onPress={() => setSelectedLandmark(landmark)}
              />
            ))}
          </MapView>
        </View>
      )}
      
      <View style={styles.infoContainer}>
        {selectedLandmark ? (
          <View style={styles.landmarkInfo}>
            <Text style={styles.landmarkName}>{selectedLandmark.name}</Text>
            <Text style={styles.landmarkDescription}>{selectedLandmark.description}</Text>
            <Text style={styles.landmarkDistance}>
              Distance: {selectedLandmark.distance_km.toFixed(2)} km
            </Text>
          </View>
        ) : (
          <Text style={styles.infoText}>
            Select a landmark on the map or use AR mode to capture a spatial signature
          </Text>
        )}
      </View>
      
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={styles.button}
          onPress={() => setIsARMode(true)}
          disabled={loading}
        >
          <Text style={styles.buttonText}>Start AR Mode</Text>
        </TouchableOpacity>
        
        {!nodeId && (
          <TouchableOpacity
            style={styles.button}
            onPress={registerNode}
            disabled={loading}
          >
            <Text style={styles.buttonText}>Register as Node</Text>
          </TouchableOpacity>
        )}
        
        {nodeId && selectedLandmark && spatialSignature && (
          <TouchableOpacity
            style={styles.button}
            onPress={submitProofOfView}
            disabled={loading}
          >
            <Text style={styles.buttonText}>Submit Proof of View</Text>
          </TouchableOpacity>
        )}
        
        {nodeId && (
          <TouchableOpacity
            style={styles.button}
            onPress={notarizeDocument}
            disabled={loading}
          >
            <Text style={styles.buttonText}>Notarize Document</Text>
          </TouchableOpacity>
        )}
      </View>
      
      {notarizations.length > 0 && (
        <View style={styles.notarizationsContainer}>
          <Text style={styles.sectionTitle}>Recent Notarizations</Text>
          <ScrollView style={styles.notarizationsList}>
            {notarizations.map((notarization) => (
              <TouchableOpacity
                key={notarization.id}
                style={styles.notarizationItem}
                onPress={() => verifyNotarization(notarization.id)}
              >
                <Text style={styles.notarizationId}>ID: {notarization.id.substring(0, 8)}...</Text>
                <Text style={styles.notarizationHash}>
                  Hash: {notarization.documentHash.substring(0, 10)}...
                </Text>
                <Text style={styles.notarizationTime}>
                  {new Date(notarization.timestamp).toLocaleString()}
                </Text>
                <Text style={styles.notarizationConsensus}>
                  Consensus: {notarization.consensus} nodes
                </Text>
                <Text style={styles.verifyText}>Tap to verify</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      )}
    </SafeAreaView>
  );
};

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  header: {
    padding: 16,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333333',
  },
  nodeId: {
    fontSize: 14,
    color: '#666666',
    marginTop: 4,
  },
  errorContainer: {
    backgroundColor: '#FFE6E6',
    padding: 12,
    margin: 16,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#FF9999',
  },
  errorText: {
    color: '#CC0000',
    fontSize: 14,
  },
  loadingContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
    zIndex: 1000,
  },
  loadingText: {
    marginTop: 8,
    fontSize: 16,
    color: '#0066CC',
  },
  mapContainer: {
    height: 200,
    margin: 16,
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  map: {
    flex: 1,
  },
  infoContainer: {
    margin: 16,
    padding: 16,
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  infoText: {
    fontSize: 14,
    color: '#666666',
    textAlign: 'center',
  },
  landmarkInfo: {
    alignItems: 'center',
  },
  landmarkName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333333',
    marginBottom: 4,
  },
  landmarkDescription: {
    fontSize: 14,
    color: '#666666',
    textAlign: 'center',
    marginBottom: 4,
  },
  landmarkDistance: {
    fontSize: 14,
    color: '#0066CC',
  },
  buttonContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    marginHorizontal: 16,
  },
  button: {
    backgroundColor: '#0066CC',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 8,
    margin: 8,
    minWidth: 150,
    alignItems: 'center',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  notarizationsContainer: {
    margin: 16,
    flex: 1,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333333',
    marginBottom: 8,
  },
  notarizationsList: {
    flex: 1,
  },
  notarizationItem: {
    backgroundColor: '#FFFFFF',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  notarizationId: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333333',
  },
  notarizationHash: {
    fontSize: 14,
    color: '#666666',
  },
  notarizationTime: {
    fontSize: 12,
    color: '#999999',
    marginTop: 4,
  },
  notarizationConsensus: {
    fontSize: 12,
    color: '#0066CC',
  },
  verifyText: {
    fontSize: 12,
    color: '#0066CC',
    textAlign: 'right',
    marginTop: 4,
    fontStyle: 'italic',
  },
});
