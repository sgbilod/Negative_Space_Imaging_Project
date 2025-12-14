/**
 * OHIF Configuration for Negative Space Imaging Project
 *
 * This configuration file sets up the OHIF Viewer for visualizing DICOM data
 * processed by the Negative Space Imaging system. It includes custom extensions
 * for advanced medical imaging capabilities.
 */

const config = {
  // Application Configuration
  app: {
    title: 'Negative Space Imaging Viewer',
    version: '1.0.0',
    description: 'Advanced DICOM viewer with quantum-enhanced image processing',
  },

  // Router Configuration
  router: {
    base: '/',
    history: 'browser',
  },

  // Data Source Configuration
  dataSources: [
    {
      friendlyName: 'Negative Space DICOM Server',
      namespace: '@ohif/extension-default.dataSourcesModule.dicomweb',
      sourceName: 'dicomweb',
      configuration: {
        name: 'dicomweb',
        wadoUriRoot: 'http://localhost:8000/dicom-web',
        qidoRoot: 'http://localhost:8000/dicom-web',
        wadoRoot: 'http://localhost:8000/dicom-web',
        qidoSupportsIncludeField: false,
        imageRendering: 'wadors',
        thumbnailRendering: 'wadors',
        enableStudyLazyLoad: true,
        supportsFuzzyMatching: false,
        supportsWildcard: true,
        staticWado: true,
        singlepart: 'bulkdata,video,pdf',
        bulkDataURI: {
          enabled: true,
          root: 'http://localhost:8000/dicom-web/bulkdata',
        },
      },
    },
  ],

  // Default Data Source
  defaultDataSourceName: 'dicomweb',

  // UI Configuration
  ui: {
    theme: 'default',
    studyListFunctionsEnabled: true,
    displaySetNavigationLoopOverSeries: false,
    displaySetNavigationMultipleViewports: true,
    displaySetNavigationSkipUnnecessaryLayouts: false,
    autoPositionMeasurementsTextCallOuts: 'TRLB',
    useMiddleClickToOpenContextMenu: true,
  },

  // Hotkeys Configuration
  hotkeys: [
    {
      commandName: 'setToolActive',
      commandOptions: { toolName: 'Zoom' },
      label: 'Zoom',
      keys: ['z'],
    },
    {
      commandName: 'setToolActive',
      commandOptions: { toolName: 'Wwwc' },
      label: 'Window/Level',
      keys: ['w'],
    },
    {
      commandName: 'setToolActive',
      commandOptions: { toolName: 'Pan' },
      label: 'Pan',
      keys: ['p'],
    },
    {
      commandName: 'setToolActive',
      commandOptions: { toolName: 'Length' },
      label: 'Length Measurement',
      keys: ['l'],
    },
    {
      commandName: 'setToolActive',
      commandOptions: { toolName: 'Angle' },
      label: 'Angle Measurement',
      keys: ['a'],
    },
    {
      commandName: 'rotateViewportCW',
      label: 'Rotate Viewport Clockwise',
      keys: ['r'],
    },
    {
      commandName: 'flipViewportHorizontal',
      label: 'Flip Viewport Horizontally',
      keys: ['h'],
    },
    {
      commandName: 'flipViewportVertical',
      label: 'Flip Viewport Vertically',
      keys: ['v'],
    },
    {
      commandName: 'invertViewport',
      label: 'Invert Viewport',
      keys: ['i'],
    },
  ],

  // Extensions Configuration
  extensions: [
    '@ohif/extension-default',
    '@ohif/extension-dicom-web',
    '@ohif/extension-dicom-microscopy',
    '@ohif/extension-dicom-pdf',
  ],

  // Modes Configuration
  modes: [
    {
      id: 'viewer',
      routeName: 'viewer',
      displayName: 'Basic Viewer',
      viewports: [
        {
          namespace: '@ohif/extension-default.viewportModule.cornerstone',
        },
      ],
      hangingProtocols: [
        '@ohif/extension-default.hangingProtocolModule.petLayout',
        '@ohif/extension-default.hangingProtocolModule.mnGrid',
        '@ohif/extension-default.hangingProtocolModule.dynamicVolume',
      ],
      toolbar: [
        {
          id: 'MeasurementTools',
          type: 'ohif.splitButton',
          props: {
            groupId: 'MeasurementTools',
            primary: '@ohif/extension-default.buttonGroup.tool',
            secondary: '@ohif/extension-default.buttonGroup.measurement',
            items: [
              '@ohif/extension-default.action.measurement.showAll',
              '@ohif/extension-default.action.measurement.hideAll',
              '@ohif/extension-default.action.measurement.deleteAll',
            ],
          },
        },
        {
          id: 'ZoomTools',
          type: 'ohif.splitButton',
          props: {
            groupId: 'ZoomTools',
            primary: '@ohif/extension-default.buttonGroup.tool',
            secondary: '@ohif/extension-default.buttonGroup.zoom',
            items: [
              '@ohif/extension-default.action.zoom.zoomIn',
              '@ohif/extension-default.action.zoom.zoomOut',
              '@ohif/extension-default.action.zoom.fitToWindow',
              '@ohif/extension-default.action.zoom.reset',
            ],
          },
        },
        {
          id: 'WindowLevelTools',
          type: 'ohif.splitButton',
          props: {
            groupId: 'WindowLevelTools',
            primary: '@ohif/extension-default.buttonGroup.tool',
            secondary: '@ohif/extension-default.buttonGroup.windowLevel',
            items: [
              '@ohif/extension-default.action.windowLevel.presetSoftTissue',
              '@ohif/extension-default.action.windowLevel.presetLung',
              '@ohif/extension-default.action.windowLevel.presetLiver',
              '@ohif/extension-default.action.windowLevel.presetBone',
              '@ohif/extension-default.action.windowLevel.reset',
            ],
          },
        },
      ],
    },
  ],

  // Hanging Protocols
  hangingProtocols: [
    {
      id: 'default',
      name: 'Default',
      protocolMatchingRules: [],
      displaySetSelectors: [],
      defaultViewport: {
        viewportOptions: {
          toolGroupId: 'default',
          allowUntrackedOperations: false,
        },
        displayArea: {
          imageArea: [0, 1, 1, 1],
          imageCanvas: [0, 1, 1, 1],
        },
      },
    },
  ],

  // Customization
  customization: {
    // Custom measurement tools for negative space analysis
    measurementTools: [
      {
        id: 'negativeSpaceMeasurement',
        name: 'Negative Space Analysis',
        toolClass: 'NegativeSpaceMeasurementTool',
        configuration: {
          enabled: true,
          showAnalysis: true,
        },
      },
    ],

    // Custom viewport overlays
    viewportOverlays: [
      {
        id: 'quantumProcessingOverlay',
        component: 'QuantumProcessingOverlay',
        configuration: {
          showProcessingTime: true,
          showConfidenceScore: true,
        },
      },
    ],
  },

  // Performance Configuration
  performance: {
    preloadAdjacentSeries: true,
    useSharedArrayBuffer: true,
    decodeConfig: {
      convertFloatPixelDataToInt: false,
      use16BitDataType: true,
    },
  },

  // Error Handling
  errorHandler: {
    onError: (error, errorInfo) => {
      console.error('OHIF Viewer Error:', error, errorInfo);
      // Send error to monitoring service
      if (window.gtag) {
        window.gtag('event', 'exception', {
          description: error.toString(),
          fatal: false,
        });
      }
    },
  },
};

export default config;
