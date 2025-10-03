# Deployment Instructions

## Backend
1. Install dependencies:
   pip install -r requirements.txt
2. Run migrations and setup database (if needed)
3. Start backend server:
   python src/main.py

## Mobile App
1. Install dependencies:
   npm install
2. Build for Android:
   npx react-native run-android
3. Build for iOS:
   npx react-native run-ios
4. For production builds, use:
   npx react-native build-android --variant=release
   npx react-native build-ios --configuration Release

## Blockchain
1. Deploy smart contracts using Truffle/Hardhat
2. Update backend with contract addresses

## CI/CD Pipeline (sample: GitHub Actions)

# .github/workflows/deploy.yml
name: Deploy Project
on:
  push:
    branches: [ main ]
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      - name: Install Python dependencies
        run: pip install -r requirements.txt
      - name: Run Python tests
        run: pytest
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install Node dependencies
        run: npm install
      - name: Run JS tests
        run: npm test
      - name: Build Mobile App
        run: |
          npx react-native build-android --variant=release
          npx react-native build-ios --configuration Release
      # Add deployment steps as needed
