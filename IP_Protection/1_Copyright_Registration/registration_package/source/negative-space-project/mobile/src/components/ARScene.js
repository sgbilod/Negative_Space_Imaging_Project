// ARScene.js
// ViroReact-based AR scene for negative space visualization

import React, { useState } from 'react';
import { ViroARScene, ViroText, ViroPointCloud } from 'react-viro';
import { View, TouchableOpacity, Text, StyleSheet, Alert } from 'react-native';

const getColor = (i, total) => {
  // Simple color mapping: hue based on index
  const hue = Math.floor((i / total) * 360);
  return `hsl(${hue}, 80%, 60%)`;
};

const getAnalytics = (points) => {
  if (!points.length) return {};
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let sumX = 0, sumY = 0, sumZ = 0;
  points.forEach(p => {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    minZ = Math.min(minZ, p.z);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
    maxZ = Math.max(maxZ, p.z);
    sumX += p.x;
    sumY += p.y;
    sumZ += p.z;
  });
  const centroid = [sumX / points.length, sumY / points.length, sumZ / points.length];
  return {
    boundingBox: {
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ]
    },
    centroid
  };
};

// Native module integration scaffold
import NativeSignatureModule from '../../native/NativeSignatureModule';

const ARScene = ({ signatureData }) => {
  const [highlighted, setHighlighted] = useState(null);
  const [showOverlay, setShowOverlay] = useState(false);
  const [performanceMode, setPerformanceMode] = useState(false);
  const points = signatureData?.features?.map((f, i) => ({
    x: f[0],
    y: f[1],
    z: f[2],
    id: i,
    color: getColor(i, signatureData.features.length)
  })) || [];

  const analytics = getAnalytics(points);

  // Example: Use native module for fast centroid calculation if performanceMode
  const getNativeCentroid = () => {
    if (performanceMode && NativeSignatureModule?.getCentroid) {
      try {
        return NativeSignatureModule.getCentroid(points);
      } catch (e) {
        return analytics.centroid;
      }
    }
    return analytics.centroid;
  };

  const handleTap = (source) => {
    if (source.pointId !== undefined) {
      setHighlighted(source.pointId);
    }
  };

  const handleExport = () => {
    try {
      const exportData = JSON.stringify(points, null, 2);
      Alert.alert('Exported Point Cloud', exportData.length > 500 ? 'Exported JSON (truncated)' : exportData);
    } catch (e) {
      Alert.alert('Export Error', e.message);
    }
  };

  // Save AR snapshot (scaffold, real implementation would use camera API)
  const handleSaveSnapshot = () => {
    Alert.alert('Save Snapshot', 'AR snapshot saved to gallery (simulated).');
  };

  return (
    <View style={{ flex: 1 }}>
      <ViroARScene>
        <ViroText text="Negative Space Signature" position={[0, 0.1, -1]} style={{ fontSize: 20 }} />
        <ViroText text={`Points: ${points.length}`} position={[0, 0.2, -1]} style={{ fontSize: 14 }} />
        {points.length > 0 && (
          <ViroPointCloud 
            points={points}
            colors={points.map(p => p.color)}
            onClick={handleTap}
            highlightPoint={highlighted}
            highlightColor="#ff0000"
          />
        )}
        {highlighted !== null && (
          <ViroText 
            text={`Highlighted: ${highlighted}`}
            position={[0, 0.3, -1]}
            style={{ fontSize: 14, color: '#ff0000' }}
          />
        )}
      </ViroARScene>
      {/* Overlay UI */}
      <View style={styles.overlay} pointerEvents="box-none">
        <TouchableOpacity style={styles.overlayButton} onPress={() => setShowOverlay(!showOverlay)}>
          <Text style={styles.overlayButtonText}>{showOverlay ? 'Hide Info' : 'Show Info'}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.overlayButton} onPress={() => setPerformanceMode(!performanceMode)}>
          <Text style={styles.overlayButtonText}>{performanceMode ? 'Performance Mode: ON' : 'Performance Mode: OFF'}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.exportButton} onPress={handleSaveSnapshot}>
          <Text style={styles.exportButtonText}>Save AR Snapshot</Text>
        </TouchableOpacity>
        {showOverlay && (
          <View style={styles.infoBox}>
            <Text style={styles.infoTitle}>Point Cloud Analytics</Text>
            <Text style={styles.infoText}>Bounding Box: Min {JSON.stringify(analytics.boundingBox?.min)}, Max {JSON.stringify(analytics.boundingBox?.max)}</Text>
            <Text style={styles.infoText}>Centroid: {JSON.stringify(getNativeCentroid())}</Text>
            <TouchableOpacity style={styles.exportButton} onPress={handleExport}>
              <Text style={styles.exportButtonText}>Export Point Cloud</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 100,
    alignItems: 'flex-end',
    padding: 12
  },
  overlayButton: {
    backgroundColor: '#4a90e2',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6
  },
  overlayButtonText: {
    color: '#fff',
    fontWeight: 'bold'
  },
  infoBox: {
    backgroundColor: 'rgba(255,255,255,0.95)',
    marginTop: 8,
    padding: 12,
    borderRadius: 8,
    maxWidth: 300
  },
  infoTitle: {
    fontWeight: 'bold',
    fontSize: 16,
    marginBottom: 6
  },
  infoText: {
    fontSize: 13,
    marginBottom: 4
  },
  exportButton: {
    backgroundColor: '#4caf50',
    padding: 8,
    borderRadius: 6,
    marginTop: 8,
    alignItems: 'center'
  },
  exportButtonText: {
    color: '#fff',
    fontWeight: 'bold'
  }
});

export default ARScene;
