/**
 * Negative Space Imaging Viewer - Main Entry Point
 *
 * Initializes the React application with OHIF integration and quantum processing capabilities.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';

// OHIF Core imports
import { ServicesManager, CommandsManager, ExtensionManager } from '@ohif/core';
import { init as initExtensions } from '@ohif/extension-default';
import { init as initCornerstone } from '@ohif/extension-cornerstone';

// Custom components
import App from './App';

// Styles
import './App.css';

// Initialize OHIF services
const servicesManager = new ServicesManager();
const commandsManager = new CommandsManager();
const extensionManager = new ExtensionManager({
  servicesManager,
  commandsManager,
  extensionConfig: {},
  dataSourceConfig: {},
});

// Initialize cornerstone extension
initCornerstone({
  servicesManager,
  commandsManager,
  extensionManager,
  configuration: {
    cornerstone: {
      // Cornerstone configuration
    },
  },
});

// Initialize default extensions
initExtensions({
  servicesManager,
  commandsManager,
  extensionManager,
  configuration: {
    default: {
      // Default extension configuration
    },
  },
});

// Application configuration
const appConfig = {
  routerBasename: '/',
  showStudyList: false,
  extensions: [],
  modes: [],
  customizationService: {},
  hotkeys: {},
};

// Create root element
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render application
root.render(
  <React.StrictMode>
    <BrowserRouter basename={appConfig.routerBasename}>
      <App
        config={appConfig}
        servicesManager={servicesManager}
        commandsManager={commandsManager}
        extensionManager={extensionManager}
      />
    </BrowserRouter>
  </React.StrictMode>,
);

// Performance monitoring
if (process.env.NODE_ENV === 'development') {
  // Enable React DevTools
  if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
    window.__REACT_DEVTOOLS_GLOBAL_HOOK__.inject = () => {};
  }

  // Log initialization
  console.log('Negative Space Imaging Viewer initialized in development mode');
  console.log('OHIF Services:', servicesManager.services);
  console.log('Available Extensions:', extensionManager.registeredExtensions);
}

// Hot Module Replacement for development
if (module.hot) {
  module.hot.accept('./App', () => {
    const NextApp = require('./App').default;
    root.render(
      <React.StrictMode>
        <BrowserRouter basename={appConfig.routerBasename}>
          <NextApp
            config={appConfig}
            servicesManager={servicesManager}
            commandsManager={commandsManager}
            extensionManager={extensionManager}
          />
        </BrowserRouter>
      </React.StrictMode>,
    );
  });
}
