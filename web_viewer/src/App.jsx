/**
 * Negative Space Imaging Viewer Application
 *
 * Main React application component that integrates OHIF viewer with
 * quantum processing and negative space analysis capabilities.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { OHIFCornerstoneViewport } from '@ohif/ui';
import { useViewportGrid } from '@ohif/ui';
import { useImageLoad } from '@ohif/core';

import ViewerComponent from './components/ViewerComponent';
import QuantumProcessingOverlay from './components/QuantumProcessingOverlay';
import NegativeSpaceMeasurementTool from './components/NegativeSpaceMeasurementTool';
import apiService from './services/ApiService';

import './App.css';

const App = () => {
  // Application state
  const [currentImage, setCurrentImage] = useState(null);
  const [quantumResults, setQuantumResults] = useState(null);
  const [segmentationResults, setSegmentationResults] = useState(null);
  const [negativeSpaceMeasurements, setNegativeSpaceMeasurements] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState('');
  const [apiConnected, setApiConnected] = useState(false);

  // OHIF viewport management
  const viewportGridData = useViewportGrid();
  const [viewportGrid, viewportGridService] = viewportGridData || [{ viewports: [] }, { setViewportGrid: () => {} }];
  const imageLoadData = useImageLoad();
  const { imageLoadService } = imageLoadData || { imageLoadService: { loadImage: () => {}, cancelLoad: () => {} } };

  // Initialize API connection
  useEffect(() => {
    const checkApiConnection = async () => {
      try {
        const connected = await apiService.validateConnection();
        setApiConnected(connected);
        if (!connected) {
          console.warn('API connection failed - quantum features will be unavailable');
        }
      } catch (error) {
        console.error('API connection check failed:', error);
        setApiConnected(false);
      }
    };

    checkApiConnection();
  }, []);

  // Handle image loading
  const handleImageLoad = useCallback(async (imageFile) => {
    try {
      setIsProcessing(true);
      setProcessingStatus('Loading image...');

      // Load image into OHIF viewport
      await imageLoadService.loadImage(imageFile);
      setCurrentImage(imageFile);

      setProcessingStatus('Image loaded successfully');
    } catch (error) {
      console.error('Image loading failed:', error);
      setProcessingStatus('Failed to load image');
    } finally {
      setIsProcessing(false);
    }
  }, [imageLoadService]);

  // Handle quantum processing
  const handleQuantumProcessing = useCallback(async (options = {}) => {
    if (!currentImage || !apiConnected) {
      console.warn('Cannot process quantum analysis: no image or API not connected');
      return;
    }

    try {
      setIsProcessing(true);
      setProcessingStatus('Running quantum analysis...');

      const results = await apiService.processQuantumAnalysis(currentImage, options);
      setQuantumResults(results);

      setProcessingStatus('Quantum analysis complete');
    } catch (error) {
      console.error('Quantum processing failed:', error);
      setProcessingStatus('Quantum analysis failed');
    } finally {
      setIsProcessing(false);
    }
  }, [currentImage, apiConnected]);

  // Handle segmentation processing
  const handleSegmentationProcessing = useCallback(async (options = {}) => {
    if (!currentImage || !apiConnected) {
      console.warn('Cannot process segmentation: no image or API not connected');
      return;
    }

    try {
      setIsProcessing(true);
      setProcessingStatus('Running segmentation analysis...');

      const results = await apiService.processSegmentation(currentImage, options);
      setSegmentationResults(results);

      setProcessingStatus('Segmentation analysis complete');
    } catch (error) {
      console.error('Segmentation processing failed:', error);
      setProcessingStatus('Segmentation analysis failed');
    } finally {
      setIsProcessing(false);
    }
  }, [currentImage, apiConnected]);

  // Handle negative space measurement
  const handleNegativeSpaceMeasurement = useCallback(async (measurement) => {
    if (!currentImage || !apiConnected) {
      console.warn('Cannot analyze negative space: no image or API not connected');
      return;
    }

    try {
      setIsProcessing(true);
      setProcessingStatus('Analyzing negative space...');

      // Add measurement to local state
      const updatedMeasurements = [...negativeSpaceMeasurements, measurement];
      setNegativeSpaceMeasurements(updatedMeasurements);

      // Send to backend for analysis
      const results = await apiService.analyzeNegativeSpace(currentImage, measurement);

      // Update measurement with analysis results
      const measurementWithAnalysis = {
        ...measurement,
        analysis: results
      };

      setNegativeSpaceMeasurements(prev =>
        prev.map(m => m.id === measurement.id ? measurementWithAnalysis : m)
      );

      setProcessingStatus('Negative space analysis complete');
    } catch (error) {
      console.error('Negative space analysis failed:', error);
      setProcessingStatus('Negative space analysis failed');
    } finally {
      setIsProcessing(false);
    }
  }, [currentImage, apiConnected, negativeSpaceMeasurements]);

  // Handle batch processing
  const handleBatchProcessing = useCallback(async (imageFiles, options = {}) => {
    if (!apiConnected) {
      console.warn('Cannot process batch: API not connected');
      return;
    }

    try {
      setIsProcessing(true);
      setProcessingStatus('Processing batch...');

      const results = await apiService.batchProcess(imageFiles, options);

      setProcessingStatus('Batch processing complete');
      return results;
    } catch (error) {
      console.error('Batch processing failed:', error);
      setProcessingStatus('Batch processing failed');
      throw error;
    } finally {
      setIsProcessing(false);
    }
  }, [apiConnected]);

  // Handle configuration changes
  const handleConfigurationChange = useCallback(async (config) => {
    if (!apiConnected) {
      console.warn('Cannot update configuration: API not connected');
      return;
    }

    try {
      await apiService.configureProcessing(config);
      setProcessingStatus('Configuration updated');
    } catch (error) {
      console.error('Configuration update failed:', error);
      setProcessingStatus('Configuration update failed');
    }
  }, [apiConnected]);

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <h1>Negative Space Imaging Viewer</h1>
        <div className="status-indicators">
          <span className={`api-status ${apiConnected ? 'connected' : 'disconnected'}`}>
            API: {apiConnected ? 'Connected' : 'Disconnected'}
          </span>
          <span className={`processing-status ${isProcessing ? 'active' : 'idle'}`}>
            Status: {processingStatus || 'Ready'}
          </span>
        </div>
      </header>

      {/* Main Content */}
      <div className="app-content">
        {/* Control Panel */}
        <div className="control-panel">
          <div className="control-section">
            <h3>Image Operations</h3>
            <input
              type="file"
              accept=".dcm,.dicom,image/*"
              onChange={(e) => e.target.files[0] && handleImageLoad(e.target.files[0])}
              disabled={isProcessing}
            />
            <button
              onClick={() => handleQuantumProcessing()}
              disabled={!currentImage || !apiConnected || isProcessing}
            >
              Run Quantum Analysis
            </button>
            <button
              onClick={() => handleSegmentationProcessing()}
              disabled={!currentImage || !apiConnected || isProcessing}
            >
              Run Segmentation
            </button>
          </div>

          <div className="control-section">
            <h3>Analysis Results</h3>
            {quantumResults && (
              <div className="results-summary">
                <h4>Quantum Analysis</h4>
                <pre>{JSON.stringify(quantumResults, null, 2)}</pre>
              </div>
            )}
            {segmentationResults && (
              <div className="results-summary">
                <h4>Segmentation Results</h4>
                <pre>{JSON.stringify(segmentationResults, null, 2)}</pre>
              </div>
            )}
          </div>

          <div className="control-section">
            <h3>Negative Space Measurements</h3>
            <div className="measurements-list">
              {negativeSpaceMeasurements.map((measurement, index) => (
                <div key={measurement.id || index} className="measurement-item">
                  <h4>Measurement {index + 1}</h4>
                  <div>Area: {measurement.metrics?.area || 'N/A'}</div>
                  <div>Mean Intensity: {measurement.metrics?.meanIntensity?.toFixed(2) || 'N/A'}</div>
                  <div>NS Ratio: {measurement.metrics?.negativeSpaceRatio ?
                    (measurement.metrics.negativeSpaceRatio * 100).toFixed(1) + '%' : 'N/A'}</div>
                  {measurement.analysis && (
                    <div className="analysis-results">
                      <h5>Analysis Results</h5>
                      <pre>{JSON.stringify(measurement.analysis, null, 2)}</pre>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Viewer Area */}
        <div className="viewer-container">
          <ViewerComponent
            viewportGrid={viewportGrid}
            viewportGridService={viewportGridService}
            onMeasurement={handleNegativeSpaceMeasurement}
          />

          {/* Overlays */}
          {quantumResults && (
            <QuantumProcessingOverlay
              quantumResults={quantumResults}
              segmentationResults={segmentationResults}
            />
          )}

          {/* Custom Tools */}
          <NegativeSpaceMeasurementTool
            enabled={true}
            onMeasurement={handleNegativeSpaceMeasurement}
          />
        </div>
      </div>

      {/* Footer */}
      <footer className="app-footer">
        <p>Negative Space Imaging Project - Advanced Medical Image Analysis</p>
        <p>Powered by UNet++, Google Cirq, and OHIF Viewer</p>
      </footer>
    </div>
  );
};

export default App;
