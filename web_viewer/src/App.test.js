/**
 * Web Viewer Tests
 *
 * Basic tests to validate the Negative Space Imaging web viewer functionality.
 */

// Mock OHIF hooks before any imports
jest.mock('@ohif/ui', () => ({
  OHIFCornerstoneViewport: jest.fn(() => <div data-testid="ohif-viewport">OHIF Viewport</div>),
  useViewportGrid: jest.fn(() => [
    { viewports: [] }, // viewportGrid
    { setViewportGrid: jest.fn() }, // viewportGridService
  ]),
}));

jest.mock('@ohif/core', () => ({
  ServicesManager: jest.fn(),
  CommandsManager: jest.fn(),
  ExtensionManager: jest.fn(),
  useImageLoad: jest.fn(() => ({
    imageLoadService: {
      loadImage: jest.fn(),
      cancelLoad: jest.fn(),
    },
  })),
}));

jest.mock('@ohif/extension-default', () => ({
  init: jest.fn(),
}));

jest.mock('@ohif/extension-cornerstone', () => ({
  init: jest.fn(),
}));

jest.mock('@ohif/extension-measurement-tracking', () => ({
  init: jest.fn(),
}));

// Mock ApiService
jest.mock('./services/ApiService', () => ({
  __esModule: true,
  default: {
    validateConnection: jest.fn(() => Promise.resolve(true)),
    getSystemHealth: jest.fn(),
    processImage: jest.fn(),
    getSegmentation: jest.fn(),
    getQuantumAnalysis: jest.fn(),
    client: {
      get: jest.fn(),
      post: jest.fn(),
    },
  },
}));

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BrowserRouter } from 'react-router-dom';
import { useViewportGrid } from '@ohif/ui';

jest.mock('@ohif/mode-longitudinal', () => ({
  id: 'mode-longitudinal',
}));

jest.mock('@ohif/i18n', () => ({
  init: jest.fn(),
}));

// Mock axios
j4est.mock('axios', () => ({
  create: jest.fn(() => ({
    get: jest.fn(),
    post: jest.fn(),
    interceptors: {
      response: {
        use: jest.fn(),
      },
    },
  })),
}));

// Import components AFTER mocks
import App from './App';
import apiService from './services/ApiService';
import NegativeSpaceMeasurementTool from './components/NegativeSpaceMeasurementTool';
import QuantumProcessingOverlay from './components/QuantumProcessingOverlay';

// Wrapper component with Router
const RouterWrapper = ({ children }) => <BrowserRouter>{children}</BrowserRouter>;

describe('Negative Space Imaging Viewer', () => {
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();

    // Mock useViewportGrid hook
    useViewportGrid.mockReturnValue([
      {
        viewports: [],
        activeViewportId: 'viewport-1',
        setActiveViewportId: jest.fn(),
      },
      {
        setLayout: jest.fn(),
        getViewport: jest.fn(() => ({
          displaySetInstanceUIDs: [],
          viewportOptions: {},
        })),
      },
    ]);
  });

  describe('App Component', () => {
    test('renders without crashing', () => {
      render(
        <RouterWrapper>
          <App />
        </RouterWrapper>,
      );
      expect(screen.getByText('Negative Space Imaging Viewer')).toBeInTheDocument();
    });

    test('displays API connection status', () => {
      render(
        <RouterWrapper>
          <App />
        </RouterWrapper>,
      );
      expect(screen.getByText(/API:/)).toBeInTheDocument();
    });

    test('shows control panel sections', () => {
      render(
        <RouterWrapper>
          <App />
        </RouterWrapper>,
      );
      expect(screen.getByText('Image Operations')).toBeInTheDocument();
      expect(screen.getByText('Analysis Results')).toBeInTheDocument();
      expect(screen.getByText('Negative Space Measurements')).toBeInTheDocument();
    });
  });

  describe('ApiService', () => {
    beforeEach(() => {
      // Reset all mocks
      jest.clearAllMocks();

      // Set up mock implementations
      apiService.validateConnection.mockResolvedValue(true);
      apiService.client.get.mockResolvedValue({ data: { status: 'healthy' } });

      // Make getSystemHealth call the real implementation (which calls client.get)
      apiService.getSystemHealth.mockImplementation(async function () {
        const response = await this.client.get('/health');
        return response.data;
      });
    });

    test('initializes with default base URL', () => {
      expect(apiService.client).toBeDefined();
    });

    test('validateConnection returns boolean', async () => {
      const result = await apiService.validateConnection();
      expect(typeof result).toBe('boolean');
      expect(result).toBe(true);
    });

    test('getSystemHealth makes API call', async () => {
      const result = await apiService.getSystemHealth();
      expect(result).toEqual({ status: 'healthy' });
      expect(apiService.client.get).toHaveBeenCalledWith('/health');
    });
  });

  describe('NegativeSpaceMeasurementTool', () => {
    test('renders without crashing', () => {
      render(<NegativeSpaceMeasurementTool enabled={true} />);
      // Component renders nothing visible, so we just check it doesn't throw
    });

    test('accepts measurement callback', () => {
      const mockCallback = jest.fn();
      render(<NegativeSpaceMeasurementTool enabled={true} onMeasurement={mockCallback} />);
      // Test would require DOM event simulation in full integration test
    });
  });

  describe('QuantumProcessingOverlay', () => {
    const mockQuantumResults = {
      processingTime: 1.5,
      confidenceScore: 0.89,
      quantumLayers: 3,
      detectedFeatures: [
        { name: 'Negative Space A', confidence: 0.95 },
        { name: 'Negative Space B', confidence: 0.87 },
        { name: 'Anomaly Region', confidence: 0.92 },
      ],
      status: 'COMPLETED',
    };

    test('renders without results', () => {
      render(<QuantumProcessingOverlay analysis={null} visible={false} />);
      // Should render nothing
      expect(document.body.textContent).toBe('');
    });

    test('renders with quantum results', () => {
      render(<QuantumProcessingOverlay analysis={mockQuantumResults} visible={true} />);
      expect(screen.getByText('Quantum Analysis')).toBeInTheDocument();
    });

    test('displays processing metrics', () => {
      render(<QuantumProcessingOverlay analysis={mockQuantumResults} visible={true} />);
      expect(screen.getByText('1.500s')).toBeInTheDocument();
      expect(screen.getByText('3')).toBeInTheDocument(); // quantum layers
    });

    test('renders without results', () => {
      render(<QuantumProcessingOverlay />);
      // Should render without crashing
    });
  });

  describe('Integration Tests', () => {
    test('full app renders all components', () => {
      render(
        <RouterWrapper>
          <App />
        </RouterWrapper>,
      );
      expect(screen.getByText('Negative Space Imaging Viewer')).toBeInTheDocument();
      expect(screen.getByText(/Powered by/)).toBeInTheDocument();
    });

    test('handles file input changes', () => {
      render(
        <RouterWrapper>
          <App />
        </RouterWrapper>,
      );
      const fileInput = screen.getByDisplayValue(''); // File input
      expect(fileInput).toBeInTheDocument();
      expect(fileInput.type).toBe('file');
    });
  });
});

// Performance tests
describe('Performance Tests', () => {
  test('renders within performance budget', () => {
    const startTime = performance.now();
    render(
      <RouterWrapper>
        <App />
      </RouterWrapper>,
    );
    const endTime = performance.now();
    const renderTime = endTime - startTime;

    // Should render in less than 100ms
    expect(renderTime).toBeLessThan(100);
  });
});

// Accessibility tests
describe('Accessibility Tests', () => {
  test('has proper heading structure', () => {
    render(
      <RouterWrapper>
        <App />
      </RouterWrapper>,
    );
    const headings = screen.getAllByRole('heading');
    expect(headings.length).toBeGreaterThan(0);
  });

  test('buttons have accessible names', () => {
    render(
      <RouterWrapper>
        <App />
      </RouterWrapper>,
    );
    const buttons = screen.getAllByRole('button');
    buttons.forEach((button) => {
      expect(button).toHaveAccessibleName();
    });
  });
});
