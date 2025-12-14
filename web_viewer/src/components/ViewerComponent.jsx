/**
 * Negative Space Imaging OHIF Viewer Component
 *
 * This component provides a React wrapper around the OHIF Viewer for
 * visualizing DICOM data processed by the Negative Space Imaging system.
 * It includes custom extensions for quantum-enhanced image analysis.
 */

import React, { useEffect, useRef, useState } from 'react';
import PropTypes from 'prop-types';
import { OHIF } from '@ohif/core';
import { ViewportGrid, Toolbar, StudyBrowser } from '@ohif/ui';
import { useNavigate, useParams } from 'react-router-dom';
import axios from 'axios';

// Custom components for negative space analysis
import QuantumProcessingOverlay from './QuantumProcessingOverlay';
import NegativeSpaceMeasurementTool from './NegativeSpaceMeasurementTool';

const NegativeSpaceViewer = ({
  studyInstanceUID,
  seriesInstanceUID,
  config,
  onMeasurement,
  onAnalysisComplete
}) => {
  const viewerRef = useRef(null);
  const navigate = useNavigate();
  const { studyId, seriesId } = useParams();

  const [isLoading, setIsLoading] = useState(true);
  const [studyData, setStudyData] = useState(null);
  const [quantumAnalysis, setQuantumAnalysis] = useState(null);
  const [error, setError] = useState(null);

  // Initialize OHIF Viewer
  useEffect(() => {
    const initializeViewer = async () => {
      try {
        setIsLoading(true);

        // Initialize OHIF core
        const ohif = new OHIF();
        await ohif.init(config);

        // Load study data
        const studyUID = studyInstanceUID || studyId;
        if (studyUID) {
          await loadStudyData(studyUID);
        }

        // Set up event listeners
        setupEventListeners();

        setIsLoading(false);

      } catch (err) {
        console.error('Failed to initialize OHIF viewer:', err);
        setError(err.message);
        setIsLoading(false);
      }
    };

    initializeViewer();

    // Cleanup
    return () => {
      if (viewerRef.current) {
        // Clean up OHIF viewer
      }
    };
  }, [studyInstanceUID, seriesInstanceUID, studyId, seriesId, config]);

  // Load study data from backend
  const loadStudyData = async (studyUID) => {
    try {
      const response = await axios.get(`/api/studies/${studyUID}`);
      setStudyData(response.data);

      // Trigger quantum analysis if available
      if (response.data.hasQuantumAnalysis) {
        await performQuantumAnalysis(studyUID);
      }

    } catch (err) {
      console.error('Failed to load study data:', err);
      setError('Failed to load study data');
    }
  };

  // Perform quantum analysis on the study
  const performQuantumAnalysis = async (studyUID) => {
    try {
      const response = await axios.post(`/api/quantum/analyze/${studyUID}`, {
        analysisType: 'negative_space_detection',
        parameters: {
          sensitivity: 0.8,
          quantumLayers: 3
        }
      });

      setQuantumAnalysis(response.data);

      if (onAnalysisComplete) {
        onAnalysisComplete(response.data);
      }

    } catch (err) {
      console.error('Quantum analysis failed:', err);
      // Continue without quantum analysis
    }
  };

  // Set up OHIF event listeners
  const setupEventListeners = () => {
    // Listen for measurement events
    window.addEventListener('ohif:measurement:added', handleMeasurementEvent);
    window.addEventListener('ohif:measurement:removed', handleMeasurementEvent);
    window.addEventListener('ohif:measurement:modified', handleMeasurementEvent);

    // Listen for viewport events
    window.addEventListener('ohif:viewport:activated', handleViewportEvent);
  };

  // Handle measurement events
  const handleMeasurementEvent = (event) => {
    const { measurement, action } = event.detail;

    if (onMeasurement) {
      onMeasurement({
        ...measurement,
        action,
        quantumAnalysis: quantumAnalysis
      });
    }
  };

  // Handle viewport events
  const handleViewportEvent = (event) => {
    const { viewportId, studyInstanceUID, seriesInstanceUID } = event.detail;

    // Update URL if needed
    if (studyInstanceUID && seriesInstanceUID) {
      navigate(`/viewer/${studyInstanceUID}/${seriesInstanceUID}`, { replace: true });
    }
  };

  // Render loading state
  if (isLoading) {
    return (
      <div className="negative-space-viewer loading">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading Negative Space Imaging Viewer...</p>
        </div>
      </div>
    );
  }

  // Render error state
  if (error) {
    return (
      <div className="negative-space-viewer error">
        <div className="error-message">
          <h3>Error Loading Viewer</h3>
          <p>{error}</p>
          <button onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="negative-space-viewer" ref={viewerRef}>
      {/* OHIF Toolbar */}
      <div className="viewer-toolbar">
        <Toolbar />
      </div>

      {/* Main Viewer Layout */}
      <div className="viewer-layout">
        {/* Study Browser Sidebar */}
        <div className="viewer-sidebar">
          <StudyBrowser
            studies={studyData ? [studyData] : []}
            onStudySelect={(study) => loadStudyData(study.studyInstanceUID)}
          />
        </div>

        {/* Viewport Grid */}
        <div className="viewer-main">
          <ViewportGrid />

          {/* Quantum Processing Overlay */}
          {quantumAnalysis && (
            <QuantumProcessingOverlay
              analysis={quantumAnalysis}
              visible={true}
            />
          )}

          {/* Custom Measurement Tools */}
          <NegativeSpaceMeasurementTool
            enabled={true}
            onMeasurement={handleMeasurementEvent}
          />
        </div>
      </div>

      {/* Status Bar */}
      <div className="viewer-status">
        <div className="status-info">
          {studyData && (
            <span>
              Study: {studyData.studyInstanceUID} |
              Modality: {studyData.modalities} |
              Series: {studyData.series?.length || 0}
            </span>
          )}
          {quantumAnalysis && (
            <span className="quantum-status">
              Quantum Analysis: {quantumAnalysis.status}
            </span>
          )}
        </div>
      </div>

      <style jsx>{`
        .negative-space-viewer {
          height: 100vh;
          display: flex;
          flex-direction: column;
          background: #000;
          color: #fff;
        }

        .viewer-toolbar {
          height: 50px;
          background: #2a2a2a;
          border-bottom: 1px solid #444;
        }

        .viewer-layout {
          flex: 1;
          display: flex;
        }

        .viewer-sidebar {
          width: 300px;
          background: #1a1a1a;
          border-right: 1px solid #444;
          overflow-y: auto;
        }

        .viewer-main {
          flex: 1;
          position: relative;
        }

        .viewer-status {
          height: 30px;
          background: #2a2a2a;
          border-top: 1px solid #444;
          padding: 0 16px;
          display: flex;
          align-items: center;
          font-size: 12px;
        }

        .status-info {
          display: flex;
          gap: 16px;
        }

        .quantum-status {
          color: #00ff88;
        }

        .loading, .error {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 100vh;
        }

        .loading-spinner {
          text-align: center;
        }

        .spinner {
          width: 40px;
          height: 40px;
          border: 4px solid #444;
          border-top: 4px solid #00ff88;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 16px;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .error-message {
          text-align: center;
          color: #ff6b6b;
        }

        .error-message h3 {
          margin-bottom: 8px;
        }

        .error-message button {
          background: #00ff88;
          color: #000;
          border: none;
          padding: 8px 16px;
          border-radius: 4px;
          cursor: pointer;
          margin-top: 16px;
        }
      `}</style>
    </div>
  );
};

NegativeSpaceViewer.propTypes = {
  studyInstanceUID: PropTypes.string,
  seriesInstanceUID: PropTypes.string,
  config: PropTypes.object.isRequired,
  onMeasurement: PropTypes.func,
  onAnalysisComplete: PropTypes.func
};

export default NegativeSpaceViewer;
