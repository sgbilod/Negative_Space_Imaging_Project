/**
 * FieldVerificationScreen.js
 * Screen for field verification of negative space signatures
 */

import React, { useState } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Alert
} from 'react-native';
import * as Location from 'expo-location';

const FieldVerificationScreen = ({ navigation, route }) => {
  const [location, setLocation] = useState(null);
  const [loadingLocation, setLoadingLocation] = useState(false);
  const [environment, setEnvironment] = useState('');
  const [notes, setNotes] = useState('');
  const [timestamp, setTimestamp] = useState(new Date().toISOString());

  // Get signature data from route params
  const signatureData = route.params?.signatureData || null;

  const getLocation = async () => {
    setLoadingLocation(true);
    try {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Location permission is required for field verification.');
        setLoadingLocation(false);
        return;
      }
      let loc = await Location.getCurrentPositionAsync({});
      setLocation(loc.coords);
    } catch (error) {
      Alert.alert('Error', 'Could not get location.');
    }
    setLoadingLocation(false);
  };

  const handleSubmit = () => {
    // Here you would send the verification record to the backend or save locally
    Alert.alert('Verification Recorded', 'Field verification data has been saved.');
    navigation.goBack();
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Field Verification</Text>
      <Text style={styles.subtitle}>Capture verification data in the field.</Text>
      <View style={styles.section}>
        <Text style={styles.label}>Timestamp:</Text>
        <Text style={styles.value}>{new Date(timestamp).toLocaleString()}</Text>
      </View>
      <View style={styles.section}>
        <Text style={styles.label}>Location:</Text>
        {location ? (
          <Text style={styles.value}>{`Lat: ${location.latitude}, Lon: ${location.longitude}`}</Text>
        ) : (
          <TouchableOpacity style={styles.button} onPress={getLocation} disabled={loadingLocation}>
            {loadingLocation ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>Get Location</Text>}
          </TouchableOpacity>
        )}
      </View>
      <View style={styles.section}>
        <Text style={styles.label}>Environmental Conditions:</Text>
        <TextInput
          style={styles.input}
          placeholder="e.g., sunny, indoors, low light"
          value={environment}
          onChangeText={setEnvironment}
        />
      </View>
      <View style={styles.section}>
        <Text style={styles.label}>Notes:</Text>
        <TextInput
          style={styles.input}
          placeholder="Additional notes"
          value={notes}
          onChangeText={setNotes}
        />
      </View>
      <TouchableOpacity style={styles.submitButton} onPress={handleSubmit}>
        <Text style={styles.submitButtonText}>Record Verification</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 16,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 16,
  },
  section: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    color: '#333',
    marginBottom: 4,
  },
  value: {
    fontSize: 14,
    color: '#4a90e2',
  },
  button: {
    backgroundColor: '#4a90e2',
    padding: 10,
    borderRadius: 6,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 14,
  },
  input: {
    backgroundColor: '#fff',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#ddd',
    padding: 10,
    fontSize: 14,
  },
  submitButton: {
    backgroundColor: '#4caf50',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 24,
  },
  submitButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default FieldVerificationScreen;
