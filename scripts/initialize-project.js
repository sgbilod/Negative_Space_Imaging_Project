#!/usr/bin/env node

/**
 * Project initialization script for Negative Space Imaging System
 * 
 * This script sets up the development environment and installs all dependencies.
 */

const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const readline = require('readline');

// Configuration
const config = {
  // Project info
  projectName: 'Negative Space Imaging System',
  repositoryUrl: 'https://github.com/yourusername/negative-space-imaging.git',
  
  // Directories to create
  directories: [
    'logs',
    'uploads',
    'uploads/images',
    'uploads/thumbnails',
    'src/middleware',
    'src/schemas',
    'src/types',
    'src/utils/imaging',
    'tests/integration',
    'tests/e2e'
  ],
  
  // Required system dependencies
  systemDependencies: {
    node: '>=18.0.0',
    npm: '>=9.0.0',
    postgres: '>=15.0'
  }
};

// ANSI color codes for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

// Helper functions
function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

function logHeader(message) {
  console.log('\n');
  log(`=== ${message} ===`, `${colors.bright}${colors.cyan}`);
}

function logSuccess(message) {
  log(`✓ ${message}`, colors.green);
}

function logError(message) {
  log(`✗ ${message}`, colors.red);
}

function logWarning(message) {
  log(`⚠ ${message}`, colors.yellow);
}

function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: options.silent ? 'ignore' : 'inherit',
      shell: true,
      ...options
    });
    
    child.on('close', (code) => {
      if (code !== 0 && !options.ignoreError) {
        reject(new Error(`Command failed with exit code ${code}`));
      } else {
        resolve();
      }
    });
  });
}

function checkSystemDependency(name, versionCommand, extractVersion, minVersion) {
  try {
    const output = execSync(versionCommand, { encoding: 'utf8' });
    const version = extractVersion(output);
    
    if (!version) {
      logWarning(`Could not determine ${name} version. Required: ${minVersion}`);
      return false;
    }
    
    const installed = version.trim();
    logSuccess(`${name} version ${installed} is installed`);
    return true;
  } catch (error) {
    logError(`${name} is not installed or not in PATH`);
    return false;
  }
}

function createDirectoryIfNotExists(dir) {
  const fullPath = path.resolve(dir);
  if (!fs.existsSync(fullPath)) {
    fs.mkdirSync(fullPath, { recursive: true });
    logSuccess(`Created directory: ${dir}`);
  } else {
    log(`Directory already exists: ${dir}`, colors.dim);
  }
}

function copyEnvFile() {
  const sourceFile = path.resolve('.env.example');
  const targetFile = path.resolve('.env');
  
  if (!fs.existsSync(targetFile)) {
    fs.copyFileSync(sourceFile, targetFile);
    logSuccess('Created .env file from .env.example');
  } else {
    log('.env file already exists', colors.dim);
  }
}

// Main execution
async function main() {
  try {
    logHeader(`Initializing ${config.projectName}`);
    
    // Check system dependencies
    logHeader('Checking System Dependencies');
    
    const nodeInstalled = checkSystemDependency(
      'Node.js',
      'node --version',
      (output) => output.replace('v', ''),
      config.systemDependencies.node
    );
    
    const npmInstalled = checkSystemDependency(
      'npm',
      'npm --version',
      (output) => output,
      config.systemDependencies.npm
    );
    
    const postgresInstalled = checkSystemDependency(
      'PostgreSQL',
      'psql --version',
      (output) => {
        const match = output.match(/\d+\.\d+/);
        return match ? match[0] : null;
      },
      config.systemDependencies.postgres
    );
    
    if (!nodeInstalled || !npmInstalled) {
      logError('Required dependencies are missing. Please install them and try again.');
      process.exit(1);
    }
    
    if (!postgresInstalled) {
      logWarning('PostgreSQL is not installed. You will need it to run the application.');
      log('Installation guide: https://www.postgresql.org/download/');
    }
    
    // Create required directories
    logHeader('Creating Directory Structure');
    
    for (const dir of config.directories) {
      createDirectoryIfNotExists(dir);
    }
    
    // Setup environment variables
    logHeader('Setting up Environment');
    copyEnvFile();
    
    // Install dependencies
    logHeader('Installing Dependencies');
    log('This may take a few minutes...');
    
    await runCommand('npm', ['install']);
    logSuccess('Installed npm dependencies');
    
    // Setup git (if not already initialized)
    logHeader('Setting up Git Repository');
    
    if (!fs.existsSync('.git')) {
      await runCommand('git', ['init']);
      await runCommand('git', ['add', '.']);
      await runCommand('git', ['commit', '-m', 'Initial commit']);
      logSuccess('Initialized git repository');
      
      const addRemote = await promptYesNo('Do you want to add a remote repository?');
      if (addRemote) {
        const remote = await prompt('Enter the remote repository URL:', config.repositoryUrl);
        await runCommand('git', ['remote', 'add', 'origin', remote]);
        logSuccess(`Added remote repository: ${remote}`);
      }
    } else {
      log('Git repository already initialized', colors.dim);
    }
    
    // Setup database
    logHeader('Setting up Database');
    
    const setupDb = await promptYesNo('Do you want to setup the PostgreSQL database now?');
    if (setupDb) {
      log('Creating database...');
      try {
        // Extract database name from DATABASE_URL in .env
        const envContent = fs.readFileSync('.env', 'utf8');
        const dbUrlMatch = envContent.match(/DATABASE_URL=.*\/([^:]+)$/m);
        const dbName = dbUrlMatch ? dbUrlMatch[1] : 'negative_space';
        
        await runCommand('createdb', [dbName], { ignoreError: true });
        logSuccess(`Created database: ${dbName}`);
      } catch (error) {
        logWarning('Could not create database automatically. You may need to create it manually.');
      }
    }
    
    // Final steps
    logHeader('Finalizing Setup');
    logSuccess(`${config.projectName} has been initialized successfully!`);
    
    log('\nNext steps:', colors.bright);
    log('1. Review and update the .env file with your configuration');
    log('2. Start the development server: npm run dev');
    log('3. Run tests to verify setup: npm test');
    log('\nHappy coding! 🚀');
    
  } catch (error) {
    logError(`Initialization failed: ${error.message}`);
    process.exit(1);
  }
}

// Prompt utilities
function createInterface() {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });
}

function prompt(question, defaultValue) {
  const rl = createInterface();
  
  return new Promise((resolve) => {
    rl.question(`${question} ${defaultValue ? `(${defaultValue}) ` : ''}`, (answer) => {
      rl.close();
      resolve(answer || defaultValue);
    });
  });
}

function promptYesNo(question) {
  const rl = createInterface();
  
  return new Promise((resolve) => {
    rl.question(`${question} (y/n) `, (answer) => {
      rl.close();
      resolve(answer.toLowerCase() === 'y');
    });
  });
}

// Run the script
main().catch((error) => {
  console.error('Unhandled error:', error);
  process.exit(1);
});
