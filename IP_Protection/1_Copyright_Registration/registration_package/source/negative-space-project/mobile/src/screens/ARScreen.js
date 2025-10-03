/**
 * ARScreen.js
 * Screen for AR visualization of negative space signatures
 */

import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import { ViroARSceneNavigator } from 'react-viro';
import ARScene from '../components/ARScene';

const ARScreen = ({ route }) => {
  // Get signature data from navigation params
  const signatureData = route.params?.signatureData || null;

  return (
    <SafeAreaView style={styles.container}>
      <ViroARSceneNavigator
        initialScene={{ scene: () => <ARScene signatureData={signatureData} /> }}
        style={{ flex: 1 }}
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
});

export default ARScreen;
