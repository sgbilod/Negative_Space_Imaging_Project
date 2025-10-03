# Negative Space Mobile Application

This directory contains the mobile application framework for the Negative Space Imaging Project. The mobile app allows field verification of negative space signatures using a smartphone camera.

## Features

- Real-time negative space capture using the device camera
- Local signature extraction and verification
- Offline verification capabilities
- Blockchain integration for online verification
- Augmented reality visualization of negative spaces
- Multi-signature support

## Project Structure

- `api/` - API bridge between Python backend and mobile frontend
- `src/` - React Native application source code
- `assets/` - Images, icons, and other static assets
- `docs/` - Mobile application documentation

## Getting Started

1. Install React Native development environment
2. Run `npm install` to install dependencies
3. Run `npm start` to start the development server
4. Use a mobile device or emulator to test the application

## Development Workflow

1. Use the API bridge to communicate with the Python backend
2. Develop UI components in the `src/components` directory
3. Add screens in the `src/screens` directory
4. Test on both Android and iOS platforms

## API Bridge

The API bridge provides a communication layer between the Python backend and the React Native frontend. It includes:

- WebSocket interface for real-time data exchange
- REST API for basic operations
- Native module bindings for performance-critical operations
