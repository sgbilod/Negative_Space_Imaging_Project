/**
 * Python Bridge Module
 *
 * Handles communication between Express API and Python analysis engine
 * via subprocess execution.
 *
 * @module utils/pythonBridge
 */

import { spawn } from 'child_process';
import path from 'path';
import logger from './logger';

/**
 * Analysis configuration
 */
interface AnalysisConfig {
  timeout?: number;
  retries?: number;
  verbose?: boolean;
}

/**
 * Analysis result from Python
 */
interface AnalysisResult {
  negative_space_percentage: number;
  contours_count: number;
  regions: object[];
  edges_detected: boolean;
  processing_time: number;
  [key: string]: any;
}

const DEFAULT_CONFIG: AnalysisConfig = {
  timeout: 300000, // 5 minutes
  retries: 1,
  verbose: false,
};

const PYTHON_SCRIPT = path.join(
  process.cwd(),
  'src',
  'python',
  'negative_space',
  '__init__.py'
);

/**
 * Call Python analysis engine
 * Spawns subprocess to run Python analysis on image
 */
export async function callPythonAnalysis(
  imagePath: string,
  config: AnalysisConfig = {}
): Promise<AnalysisResult> {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  let attempt = 0;

  while (attempt <= (finalConfig.retries || 0)) {
    try {
      attempt++;
      logger.debug(
        `Analysis attempt ${attempt} for: ${imagePath}`
      );

      const result = await executeAnalysis(imagePath, finalConfig);
      return result;
    } catch (error) {
      if (attempt <= (finalConfig.retries || 0)) {
        logger.warn(
          `Analysis failed on attempt ${attempt}, retrying...`,
          error
        );
        await delay(1000);
        continue;
      }

      logger.error(`Analysis failed after ${attempt} attempts`, error);
      throw error;
    }
  }

  throw new Error('Analysis failed');
}

/**
 * Execute Python analysis script
 */
function executeAnalysis(
  imagePath: string,
  config: AnalysisConfig
): Promise<AnalysisResult> {
  return new Promise((resolve, reject) => {
    let stdout = '';
    let stderr = '';
    let timedOut = false;

    // Spawn Python process
    const pythonProcess = spawn('python', [
      '-m',
      'negative_space.core.analyzer',
      '--image',
      imagePath,
      config.verbose ? '--verbose' : '',
    ].filter(Boolean));

    // Set timeout
    const timeout = setTimeout(() => {
      timedOut = true;
      pythonProcess.kill('SIGTERM');
      logger.error('Python process timeout');
      reject(new Error('Analysis timeout'));
    }, config.timeout || DEFAULT_CONFIG.timeout);

    // Handle stdout
    pythonProcess.stdout?.on('data', (data) => {
      stdout += data.toString();
    });

    // Handle stderr
    pythonProcess.stderr?.on('data', (data) => {
      stderr += data.toString();
      logger.debug(`Python stderr: ${data}`);
    });

    // Handle process exit
    pythonProcess.on('close', (code) => {
      clearTimeout(timeout);

      if (timedOut) {
        return;
      }

      if (code !== 0) {
        logger.error(`Python process exited with code ${code}`);
        return reject(
          new Error(`Python analysis failed: ${stderr || 'Unknown error'}`)
        );
      }

      try {
        // Parse JSON output from Python
        const result = JSON.parse(stdout) as AnalysisResult;
        logger.debug(`Analysis completed successfully`);
        resolve(result);
      } catch (parseError) {
        logger.error('Failed to parse Python output', parseError);
        reject(new Error('Invalid Python output'));
      }
    });

    // Handle process errors
    pythonProcess.on('error', (error) => {
      clearTimeout(timeout);
      logger.error('Failed to spawn Python process', error);
      reject(new Error(`Process error: ${error.message}`));
    });
  });
}

/**
 * Spawn Python process with custom script
 */
export function spawnPythonProcess(
  scriptPath: string,
  args: string[] = [],
  config: AnalysisConfig = {}
): Promise<string> {
  return new Promise((resolve, reject) => {
    const finalConfig = { ...DEFAULT_CONFIG, ...config };
    let stdout = '';
    let stderr = '';

    const pythonProcess = spawn('python', [scriptPath, ...args]);

    const timeout = setTimeout(() => {
      pythonProcess.kill('SIGTERM');
      reject(new Error('Process timeout'));
    }, finalConfig.timeout);

    pythonProcess.stdout?.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr?.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      clearTimeout(timeout);

      if (code !== 0) {
        reject(new Error(`Process failed: ${stderr}`));
      } else {
        resolve(stdout);
      }
    });

    pythonProcess.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });
  });
}

/**
 * Batch analysis for multiple images
 */
export async function batchAnalysis(
  imagePaths: string[],
  config: AnalysisConfig = {}
): Promise<Map<string, AnalysisResult>> {
  const results = new Map<string, AnalysisResult>();

  for (const imagePath of imagePaths) {
    try {
      const result = await callPythonAnalysis(imagePath, config);
      results.set(imagePath, result);
    } catch (error) {
      logger.error(`Batch analysis failed for ${imagePath}`, error);
    }
  }

  return results;
}

/**
 * Check if Python is available
 */
export async function checkPythonAvailability(): Promise<boolean> {
  try {
    return await new Promise((resolve) => {
      const pythonProcess = spawn('python', ['--version']);
      let available = false;

      pythonProcess.stdout?.on('data', () => {
        available = true;
      });

      pythonProcess.stderr?.on('data', () => {
        available = true;
      });

      pythonProcess.on('close', () => {
        resolve(available);
      });

      pythonProcess.on('error', () => {
        resolve(false);
      });

      setTimeout(() => {
        pythonProcess.kill();
        resolve(available);
      }, 5000);
    });
  } catch (error) {
    logger.error('Python availability check failed', error);
    return false;
  }
}

/**
 * Get analysis metadata
 */
export async function getAnalysisMetadata(): Promise<object> {
  try {
    return await new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', [
        '-m',
        'negative_space.core.analyzer',
        '--metadata',
      ]);

      let stdout = '';

      pythonProcess.stdout?.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            resolve(JSON.parse(stdout));
          } catch (error) {
            reject(error);
          }
        } else {
          reject(new Error('Failed to get metadata'));
        }
      });

      pythonProcess.on('error', reject);
    });
  } catch (error) {
    logger.error('Failed to get analysis metadata', error);
    throw error;
  }
}

/**
 * Utility: delay function
 */
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default {
  callPythonAnalysis,
  spawnPythonProcess,
  batchAnalysis,
  checkPythonAvailability,
  getAnalysisMetadata,
};
