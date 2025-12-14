/**
 * Quantum Processing Overlay Component
 *
 * Displays real-time quantum analysis results and processing metrics
 * overlaid on the medical imaging viewer.
 */

import React from 'react';
import PropTypes from 'prop-types';

const QuantumProcessingOverlay = ({ analysis, visible }) => {
  if (!visible || !analysis) {
    return null;
  }

  const {
    processingTime,
    confidenceScore,
    quantumLayers,
    detectedFeatures,
    status
  } = analysis;

  return (
    <div className="quantum-overlay">
      <div className="overlay-header">
        <h4>Quantum Analysis</h4>
        <span className={`status ${status.toLowerCase()}`}>
          {status}
        </span>
      </div>

      <div className="overlay-content">
        <div className="metric">
          <label>Processing Time:</label>
          <span>{processingTime?.toFixed(3)}s</span>
        </div>

        <div className="metric">
          <label>Confidence:</label>
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{ width: `${confidenceScore * 100}%` }}
            />
            <span className="confidence-text">
              {(confidenceScore * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        <div className="metric">
          <label>Quantum Layers:</label>
          <span>{quantumLayers}</span>
        </div>

        {detectedFeatures && detectedFeatures.length > 0 && (
          <div className="features">
            <label>Detected Features:</label>
            <ul>
              {detectedFeatures.map((feature, index) => (
                <li key={index}>
                  {feature.name}: {feature.confidence.toFixed(2)}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <style jsx>{`
        .quantum-overlay {
          position: absolute;
          top: 16px;
          right: 16px;
          background: rgba(0, 0, 0, 0.8);
          border: 1px solid #00ff88;
          border-radius: 8px;
          padding: 16px;
          min-width: 250px;
          font-family: monospace;
          font-size: 12px;
          z-index: 1000;
        }

        .overlay-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
          border-bottom: 1px solid #444;
          padding-bottom: 8px;
        }

        .overlay-header h4 {
          margin: 0;
          color: #00ff88;
          font-size: 14px;
        }

        .status {
          padding: 2px 8px;
          border-radius: 4px;
          font-size: 10px;
          font-weight: bold;
          text-transform: uppercase;
        }

        .status.completed { background: #00ff88; color: #000; }
        .status.processing { background: #ffa500; color: #000; }
        .status.failed { background: #ff6b6b; color: #fff; }

        .overlay-content {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .metric {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .metric label {
          color: #ccc;
          flex: 1;
        }

        .metric span {
          color: #fff;
          font-weight: bold;
        }

        .confidence-bar {
          position: relative;
          width: 80px;
          height: 12px;
          background: #333;
          border-radius: 6px;
          overflow: hidden;
        }

        .confidence-fill {
          height: 100%;
          background: linear-gradient(90deg, #ff6b6b, #ffa500, #00ff88);
          transition: width 0.3s ease;
        }

        .confidence-text {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          text-align: center;
          line-height: 12px;
          font-size: 10px;
          font-weight: bold;
          color: #000;
        }

        .features {
          margin-top: 8px;
        }

        .features label {
          display: block;
          color: #ccc;
          margin-bottom: 4px;
        }

        .features ul {
          margin: 0;
          padding-left: 16px;
          list-style-type: none;
        }

        .features li {
          color: #fff;
          font-size: 11px;
          margin-bottom: 2px;
        }
      `}</style>
    </div>
  );
};

QuantumProcessingOverlay.propTypes = {
  analysis: PropTypes.shape({
    processingTime: PropTypes.number,
    confidenceScore: PropTypes.number,
    quantumLayers: PropTypes.number,
    detectedFeatures: PropTypes.arrayOf(
      PropTypes.shape({
        name: PropTypes.string,
        confidence: PropTypes.number
      })
    ),
    status: PropTypes.string
  }),
  visible: PropTypes.bool
};

export default QuantumProcessingOverlay;
