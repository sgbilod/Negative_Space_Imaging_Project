/**
 * Negative Space Measurement Tool
 *
 * Custom measurement tool for analyzing negative space patterns in medical images.
 * Integrates with OHIF's measurement system to provide specialized analysis
 * for detecting anomalies in "negative space" regions.
 */

import React, { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import { cornerstone, cornerstoneTools } from '@ohif/core';

const NegativeSpaceMeasurementTool = ({ enabled, onMeasurement }) => {
  const toolRef = useRef(null);

  useEffect(() => {
    if (enabled) {
      initializeTool();
    } else {
      destroyTool();
    }

    return () => {
      destroyTool();
    };
  }, [enabled]);

  const initializeTool = () => {
    try {
      // Register custom tool with Cornerstone
      const NegativeSpaceTool = cornerstoneTools.NegativeSpaceTool || createNegativeSpaceTool();

      // Add tool to Cornerstone
      cornerstoneTools.addTool(NegativeSpaceTool);

      // Set tool as active
      cornerstoneTools.setToolActive('NegativeSpace', { mouseButtonMask: 1 });

      toolRef.current = NegativeSpaceTool;

      console.log('Negative Space Measurement Tool initialized');

    } catch (error) {
      console.error('Failed to initialize Negative Space tool:', error);
    }
  };

  const destroyTool = () => {
    if (toolRef.current) {
      try {
        cornerstoneTools.removeTool('NegativeSpace');
        toolRef.current = null;
        console.log('Negative Space Measurement Tool destroyed');
      } catch (error) {
        console.error('Failed to destroy Negative Space tool:', error);
      }
    }
  };

  const createNegativeSpaceTool = () => {
    // Create custom measurement tool for negative space analysis
    const NegativeSpaceTool = function() {
      const toolInterface = {
        name: 'NegativeSpace',
        supportedInteractionTypes: ['Mouse'],
        configuration: {
          drawHandles: true,
          drawHandlesOnHover: true,
          hideHandlesIfMoving: false,
          renderDashed: false
        }
      };

      // Tool activation handler
      toolInterface.toolActivate = (element, mouseButtonMask) => {
        // Set up event listeners for measurements
        element.addEventListener('cornerstoneToolsMeasurementAdded', handleMeasurementAdded);
        element.addEventListener('cornerstoneToolsMeasurementModified', handleMeasurementModified);
        element.addEventListener('cornerstoneToolsMeasurementRemoved', handleMeasurementRemoved);
      };

      // Tool deactivation handler
      toolInterface.toolDeactivate = (element, mouseButtonMask) => {
        // Clean up event listeners
        element.removeEventListener('cornerstoneToolsMeasurementAdded', handleMeasurementAdded);
        element.removeEventListener('cornerstoneToolsMeasurementModified', handleMeasurementModified);
        element.removeEventListener('cornerstoneToolsMeasurementRemoved', handleMeasurementRemoved);
      };

      // Mouse event handlers
      toolInterface.mouseDownCallback = (evt) => {
        const eventData = evt.detail;
        const element = eventData.element;
        const startCoords = eventData.startCoords;

        // Start negative space measurement
        const measurementData = {
          toolType: 'NegativeSpace',
          toolName: 'Negative Space Analysis',
          visible: true,
          active: true,
          color: '#00ff88',
          handles: {
            start: {
              x: startCoords.x,
              y: startCoords.y,
              highlight: true,
              active: true
            },
            end: {
              x: startCoords.x,
              y: startCoords.y,
              highlight: false,
              active: false
            }
          }
        };

        // Add measurement to Cornerstone
        cornerstoneTools.addToolState(element, 'NegativeSpace', measurementData);

        return true; // Event handled
      };

      toolInterface.mouseMoveCallback = (evt) => {
        const eventData = evt.detail;
        const element = eventData.element;
        const currentCoords = eventData.currentCoords;

        // Update measurement handles
        const toolState = cornerstoneTools.getToolState(element, 'NegativeSpace');
        if (toolState && toolState.data && toolState.data.length > 0) {
          const measurement = toolState.data[toolState.data.length - 1];
          measurement.handles.end.x = currentCoords.x;
          measurement.handles.end.y = currentCoords.y;

          // Trigger measurement update
          cornerstone.updateImage(element);
        }

        return true;
      };

      toolInterface.mouseUpCallback = (evt) => {
        const eventData = evt.detail;
        const element = eventData.element;

        // Finalize measurement
        const toolState = cornerstoneTools.getToolState(element, 'NegativeSpace');
        if (toolState && toolState.data && toolState.data.length > 0) {
          const measurement = toolState.data[toolState.data.length - 1];

          // Calculate negative space metrics
          const metrics = calculateNegativeSpaceMetrics(measurement, element);

          // Add metrics to measurement
          measurement.metrics = metrics;
          measurement.area = metrics.area;
          measurement.meanIntensity = metrics.meanIntensity;
          measurement.stdIntensity = metrics.stdIntensity;

          // Trigger measurement complete event
          const measurementEvent = new CustomEvent('negativeSpaceMeasurementComplete', {
            detail: {
              measurement: measurement,
              metrics: metrics
            }
          });
          element.dispatchEvent(measurementEvent);
        }

        return true;
      };

      // Rendering function
      toolInterface.renderToolData = (evt) => {
        const eventData = evt.detail;
        const element = eventData.element;
        const context = eventData.canvasContext;

        const toolState = cornerstoneTools.getToolState(element, 'NegativeSpace');
        if (!toolState || !toolState.data) return;

        // Render each measurement
        toolState.data.forEach((measurement) => {
          renderNegativeSpaceMeasurement(context, measurement);
        });
      };

      return toolInterface;
    };

    return NegativeSpaceTool;
  };

  // Event handlers for measurements
  const handleMeasurementAdded = (evt) => {
    const measurement = evt.detail.measurement;
    if (onMeasurement) {
      onMeasurement({
        ...measurement,
        action: 'added'
      });
    }
  };

  const handleMeasurementModified = (evt) => {
    const measurement = evt.detail.measurement;
    if (onMeasurement) {
      onMeasurement({
        ...measurement,
        action: 'modified'
      });
    }
  };

  const handleMeasurementRemoved = (evt) => {
    const measurement = evt.detail.measurement;
    if (onMeasurement) {
      onMeasurement({
        ...measurement,
        action: 'removed'
      });
    }
  };

  // Utility functions
  const calculateNegativeSpaceMetrics = (measurement, element) => {
    try {
      const image = cornerstone.getImage(element);
      const pixelData = image.getPixelData();

      const start = measurement.handles.start;
      const end = measurement.handles.end;

      // Calculate bounding box
      const minX = Math.min(start.x, end.x);
      const maxX = Math.max(start.x, end.x);
      const minY = Math.min(start.y, end.y);
      const maxY = Math.max(start.y, end.y);

      // Extract pixel values in the region
      const pixels = [];
      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const pixelIndex = y * image.width + x;
          if (pixelIndex < pixelData.length) {
            pixels.push(pixelData[pixelIndex]);
          }
        }
      }

      // Calculate metrics
      const area = pixels.length;
      const meanIntensity = pixels.reduce((sum, val) => sum + val, 0) / area;
      const variance = pixels.reduce((sum, val) => sum + Math.pow(val - meanIntensity, 2), 0) / area;
      const stdIntensity = Math.sqrt(variance);

      // Negative space specific metrics
      const lowIntensityPixels = pixels.filter(p => p < meanIntensity * 0.5).length;
      const negativeSpaceRatio = lowIntensityPixels / area;

      return {
        area,
        meanIntensity,
        stdIntensity,
        negativeSpaceRatio,
        lowIntensityPixels,
        boundingBox: { minX, maxX, minY, maxY }
      };

    } catch (error) {
      console.error('Failed to calculate negative space metrics:', error);
      return {
        area: 0,
        meanIntensity: 0,
        stdIntensity: 0,
        negativeSpaceRatio: 0,
        lowIntensityPixels: 0,
        boundingBox: { minX: 0, maxX: 0, minY: 0, maxY: 0 }
      };
    }
  };

  const renderNegativeSpaceMeasurement = (context, measurement) => {
    const start = measurement.handles.start;
    const end = measurement.handles.end;

    // Set drawing style
    context.strokeStyle = measurement.color || '#00ff88';
    context.lineWidth = 2;
    context.fillStyle = 'rgba(0, 255, 136, 0.1)';

    // Draw measurement rectangle
    const width = Math.abs(end.x - start.x);
    const height = Math.abs(end.y - start.y);

    context.fillRect(start.x, start.y, width, height);
    context.strokeRect(start.x, start.y, width, height);

    // Draw handles
    context.fillStyle = measurement.color || '#00ff88';
    context.beginPath();
    context.arc(start.x, start.y, 4, 0, 2 * Math.PI);
    context.fill();

    context.beginPath();
    context.arc(end.x, end.y, 4, 0, 2 * Math.PI);
    context.fill();

    // Draw measurement text
    if (measurement.metrics) {
      const textX = end.x + 10;
      const textY = end.y - 10;

      context.fillStyle = 'rgba(0, 0, 0, 0.7)';
      context.fillRect(textX - 2, textY - 12, 150, 40);

      context.fillStyle = '#fff';
      context.font = '12px monospace';
      context.fillText(`Area: ${measurement.metrics.area}`, textX, textY);
      context.fillText(`Mean: ${measurement.metrics.meanIntensity.toFixed(2)}`, textX, textY + 14);
      context.fillText(`NS Ratio: ${(measurement.metrics.negativeSpaceRatio * 100).toFixed(1)}%`, textX, textY + 28);
    }
  };

  return null; // This component doesn't render anything visible
};

NegativeSpaceMeasurementTool.propTypes = {
  enabled: PropTypes.bool,
  onMeasurement: PropTypes.func
};

export default NegativeSpaceMeasurementTool;
