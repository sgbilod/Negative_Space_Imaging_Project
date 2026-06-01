/**
 * GENERATE ENVIRONMENT EXAMPLE
 *
 * Generates .env.example from actual environment
 * Features:
 *   - Creates example file with masked secrets
 *   - Includes documentation for each variable
 *   - Groups by category
 *   - Provides default values
 *   - Safe to commit to repository
 */

'use strict';

const fs = require('fs');
const path = require('path');

/**
 * Environment variable documentation
 */
const envDocumentation = {
  // Application
  NODE_ENV: {
    category: 'Application',
    description: 'Runtime environment (development, staging, production)',
    example: 'development',
    required: false,
    default: 'development',
  },
  PORT: {
    category: 'Application',
    description: 'HTTP server port',
    example: '3000',
    required: false,
    default: '3000',
  },
  HOST: {
    category: 'Application',
    description: 'HTTP server host/listen address',
    example: 'localhost',
    required: false,
    default: 'localhost',
  },

  // Database
  DATABASE_URL: {
    category: 'Database',
    description: 'PostgreSQL connection string (postgres://user:pass@host:port/dbname)',
    example: 'postgres://postgres:postgres@localhost:5432/negative_space',
    required: true,
    default: null,
  },
  DB_SSL: {
    category: 'Database',
    description: 'Enable SSL for database connections (true/false)',
    example: 'false',
    required: false,
    default: 'false',
  },
  DB_LOGGING: {
    category: 'Database',
    description: 'Enable SQL query logging (true/false)',
    example: 'false',
    required: false,
    default: 'false',
  },
  DB_SYNC: {
    category: 'Database',
    description: 'Auto-sync database schema on startup (true/false)',
    example: 'false',
    required: false,
    default: 'false',
  },
  DB_POOL_MIN: {
    category: 'Database',
    description: 'Minimum database connection pool size',
    example: '2',
    required: false,
    default: '2',
  },
  DB_POOL_MAX: {
    category: 'Database',
    description: 'Maximum database connection pool size',
    example: '10',
    required: false,
    default: '10',
  },

  // Redis
  REDIS_URL: {
    category: 'Redis',
    description: 'Redis connection URL (redis://[user:pass@]host:port) - Optional',
    example: 'redis://localhost:6379',
    required: false,
    default: null,
  },

  // JWT
  JWT_SECRET: {
    category: 'JWT Authentication',
    description: 'Secret key for signing JWT tokens (minimum 32 characters)',
    example: 'your-secret-key-minimum-32-characters-long',
    required: true,
    default: null,
  },
  JWT_EXPIRES_IN: {
    category: 'JWT Authentication',
    description: 'JWT token expiration time',
    example: '24h',
    required: false,
    default: '24h',
  },
  JWT_REFRESH_SECRET: {
    category: 'JWT Authentication',
    description: 'Secret key for refresh tokens (minimum 32 characters)',
    example: 'your-refresh-secret-minimum-32-characters-long',
    required: false,
    default: null,
  },
  JWT_REFRESH_EXPIRES_IN: {
    category: 'JWT Authentication',
    description: 'Refresh token expiration time',
    example: '7d',
    required: false,
    default: '7d',
  },

  // Security
  BCRYPT_SALT_ROUNDS: {
    category: 'Security',
    description: 'Bcrypt salt rounds for password hashing',
    example: '12',
    required: false,
    default: '12',
  },
  COOKIE_SECRET: {
    category: 'Security',
    description: 'Secret key for signing session cookies (minimum 32 characters)',
    example: 'your-cookie-secret-minimum-32-characters-long',
    required: false,
    default: null,
  },
  CSRF_ENABLED: {
    category: 'Security',
    description: 'Enable CSRF protection (true/false)',
    example: 'true',
    required: false,
    default: 'true',
  },
  CORS_ALLOWED_ORIGINS: {
    category: 'Security',
    description: 'Allowed CORS origins (comma-separated)',
    example: 'http://localhost:3000,https://example.com',
    required: false,
    default: 'http://localhost:3000',
  },
  SESSION_TIMEOUT: {
    category: 'Security',
    description: 'Session timeout in milliseconds',
    example: '3600000',
    required: false,
    default: '3600000',
  },

  // Logging
  LOG_LEVEL: {
    category: 'Logging',
    description: 'Log level (error, warn, info, debug, trace)',
    example: 'info',
    required: false,
    default: 'info',
  },
  LOG_DIR: {
    category: 'Logging',
    description: 'Directory for log files',
    example: 'logs',
    required: false,
    default: 'logs',
  },
  LOG_FILENAME: {
    category: 'Logging',
    description: 'Log filename',
    example: 'app.log',
    required: false,
    default: 'app.log',
  },
  LOG_MAX_SIZE: {
    category: 'Logging',
    description: 'Maximum log file size (e.g., 10m, 100k)',
    example: '10m',
    required: false,
    default: '10m',
  },
  LOG_MAX_FILES: {
    category: 'Logging',
    description: 'Number of log files to retain',
    example: '7',
    required: false,
    default: '7',
  },

  // File Storage
  UPLOAD_DIR: {
    category: 'File Storage',
    description: 'Directory for file uploads',
    example: 'uploads',
    required: false,
    default: 'uploads',
  },
  MAX_FILE_SIZE: {
    category: 'File Storage',
    description: 'Maximum file upload size in bytes (10485760 = 10MB)',
    example: '10485760',
    required: false,
    default: '10485760',
  },
  ALLOWED_FILE_TYPES: {
    category: 'File Storage',
    description: 'Allowed MIME types (comma-separated)',
    example: 'image/jpeg,image/png,image/tiff,application/dicom',
    required: false,
    default: 'image/jpeg,image/png,image/tiff,application/dicom',
  },

  // Imaging
  DEFAULT_ALGORITHM: {
    category: 'Imaging',
    description: 'Default imaging algorithm to use',
    example: 'advanced',
    required: false,
    default: 'advanced',
  },
  MAX_IMAGE_DIMENSION: {
    category: 'Imaging',
    description: 'Maximum image dimension in pixels',
    example: '4096',
    required: false,
    default: '4096',
  },
  PROCESSING_TIMEOUT: {
    category: 'Imaging',
    description: 'Image processing timeout in milliseconds',
    example: '30000',
    required: false,
    default: '30000',
  },
  ENABLE_GPU: {
    category: 'Imaging',
    description: 'Enable GPU acceleration (true/false)',
    example: 'false',
    required: false,
    default: 'false',
  },
  CACHE_TTL: {
    category: 'Imaging',
    description: 'Cache time-to-live in seconds',
    example: '86400',
    required: false,
    default: '86400',
  },

  // HIPAA Compliance
  HIPAA_ENABLED: {
    category: 'HIPAA Compliance',
    description: 'Enable HIPAA compliance mode (true/false)',
    example: 'false',
    required: false,
    default: 'false',
  },
  DATA_ENCRYPTION: {
    category: 'HIPAA Compliance',
    description: 'Enable data encryption at rest (true/false)',
    example: 'false',
    required: false,
    default: 'false',
  },
  AUDIT_LOG_ENABLED: {
    category: 'HIPAA Compliance',
    description: 'Enable audit logging (true/false)',
    example: 'false',
    required: false,
    default: 'false',
  },
  AUTO_LOGOUT: {
    category: 'HIPAA Compliance',
    description: 'Auto-logout timeout in milliseconds (900000 = 15 minutes)',
    example: '900000',
    required: false,
    default: '900000',
  },

  // AWS
  AWS_ACCESS_KEY_ID: {
    category: 'AWS (Optional)',
    description: 'AWS access key ID (only if using AWS S3)',
    example: null,
    required: false,
    default: null,
  },
  AWS_SECRET_ACCESS_KEY: {
    category: 'AWS (Optional)',
    description: 'AWS secret access key (only if using AWS S3)',
    example: null,
    required: false,
    default: null,
  },
  AWS_REGION: {
    category: 'AWS (Optional)',
    description: 'AWS region (only if using AWS)',
    example: 'us-east-1',
    required: false,
    default: 'us-east-1',
  },
  AWS_S3_BUCKET: {
    category: 'AWS (Optional)',
    description: 'AWS S3 bucket name (only if using AWS S3)',
    example: null,
    required: false,
    default: null,
  },

  // API Keys
  API_KEY: {
    category: 'API Keys',
    description: 'API key for external services (if needed)',
    example: null,
    required: false,
    default: null,
  },
  API_SECRET: {
    category: 'API Keys',
    description: 'API secret for external services (if needed)',
    example: null,
    required: false,
    default: null,
  },
};

/**
 * Generate .env.example file
 */
class EnvExampleGenerator {
  /**
   * Generate example content
   */
  static generateContent() {
    let content = '';

    // Header
    content += this.generateHeader();

    // Group by category
    const categories = this.getCategories();

    for (const category of categories) {
      content += this.generateCategory(category);
    }

    // Footer
    content += this.generateFooter();

    return content;
  }

  /**
   * Get unique categories
   */
  static getCategories() {
    const categories = new Set();
    Object.values(envDocumentation).forEach((doc) => {
      categories.add(doc.category);
    });
    return Array.from(categories).sort();
  }

  /**
   * Generate header section
   */
  static generateHeader() {
    return `# ENVIRONMENT CONFIGURATION
# Copy this file to .env and fill in actual values
# DO NOT commit the actual .env file to version control
#
# For required variables, uncomment and set the value
# For optional variables, leave commented or set as needed

`;
  }

  /**
   * Generate category section
   */
  static generateCategory(category) {
    let content = `\n# ============================================\n`;
    content += `# ${category.toUpperCase()}\n`;
    content += `# ============================================\n\n`;

    // Get all variables in this category
    const variables = Object.entries(envDocumentation)
      .filter(([, doc]) => doc.category === category)
      .sort(([a], [b]) => a.localeCompare(b));

    for (const [varName, doc] of variables) {
      content += this.generateVariable(varName, doc);
    }

    return content;
  }

  /**
   * Generate variable documentation
   */
  static generateVariable(name, doc) {
    let content = '';

    // Comment with description
    content += `# ${doc.description}\n`;

    // Add requirement indicator
    if (doc.required) {
      content += `# ⚠️  REQUIRED\n`;
    }

    // Add default if available
    if (doc.default !== null) {
      content += `# Default: ${doc.default}\n`;
    }

    // Add example if available
    if (doc.example !== null) {
      content += `# Example: ${doc.example}\n`;
    }

    // Determine if should be commented
    const shouldComment = doc.default !== null || !doc.required;
    const prefix = shouldComment ? '# ' : '';

    // Generate variable line
    if (doc.example !== null) {
      content += `${prefix}${name}=${doc.example}\n`;
    } else if (doc.default !== null) {
      content += `${prefix}${name}=${doc.default}\n`;
    } else {
      content += `${prefix}${name}=\n`;
    }

    content += '\n';
    return content;
  }

  /**
   * Generate footer section
   */
  static generateFooter() {
    return `
# ============================================
# SECURITY NOTES
# ============================================
#
# 1. Keep .env file secure and never commit to version control
# 2. All secrets must be at least 32 characters long
# 3. Use strong, unique values for all credentials
# 4. Rotate secrets regularly in production
# 5. Use different values for dev/staging/production
# 6. Consider using a secret management service (Vault, AWS Secrets Manager, etc.)
#
# ============================================
# QUICK START
# ============================================
#
# 1. Copy this file: cp .env.example .env
# 2. Edit .env with your local values
# 3. Run verification: npm run verify:environment
# 4. Start development: npm run dev
#
`;
  }

  /**
   * Generate and write to file
   */
  static generate(outputPath = '.env.example') {
    const content = this.generateContent();

    try {
      fs.writeFileSync(outputPath, content, 'utf-8');
      return {
        success: true,
        file: outputPath,
        size: content.length,
        message: `Generated ${outputPath} (${content.length} bytes)`,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        message: `Failed to generate ${outputPath}: ${error.message}`,
      };
    }
  }

  /**
   * Generate masked config from current environment
   */
  static generateFromEnvironment(outputPath = '.env.example') {
    let content = '';

    // Header
    content += this.generateHeader();

    // Group by category
    const categories = this.getCategories();

    for (const category of categories) {
      content += `\n# ============================================\n`;
      content += `# ${category.toUpperCase()}\n`;
      content += `# ============================================\n\n`;

      const variables = Object.entries(envDocumentation)
        .filter(([, doc]) => doc.category === category)
        .sort(([a], [b]) => a.localeCompare(b));

      for (const [varName, doc] of variables) {
        const envValue = process.env[varName];

        content += `# ${doc.description}\n`;

        if (doc.required) {
          content += `# ⚠️  REQUIRED\n`;
        }

        // Mask sensitive values
        const isSensitive =
          varName.includes('SECRET') ||
          varName.includes('PASSWORD') ||
          varName.includes('KEY') ||
          varName.includes('TOKEN') ||
          varName.includes('CREDENTIAL');

        let displayValue = envValue;
        if (envValue && isSensitive) {
          displayValue = '*'.repeat(Math.min(envValue.length, 20));
        }

        content += `# Current: ${displayValue || '(not set)'}\n`;
        content += `# Example: ${doc.example || '(no example)'}\n`;

        if (envValue) {
          const prefix = envValue ? '' : '# ';
          content += `${prefix}${varName}=${displayValue || envValue}\n`;
        } else {
          content += `# ${varName}=${doc.example || ''}\n`;
        }

        content += '\n';
      }
    }

    // Footer
    content += this.generateFooter();

    try {
      fs.writeFileSync(outputPath, content, 'utf-8');
      return {
        success: true,
        file: outputPath,
        size: content.length,
        message: `Generated ${outputPath} from current environment (${content.length} bytes)`,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        message: `Failed to generate ${outputPath}: ${error.message}`,
      };
    }
  }

  /**
   * Validate existing .env.example
   */
  static validate(envExamplePath = '.env.example') {
    try {
      if (!fs.existsSync(envExamplePath)) {
        return {
          valid: false,
          message: `${envExamplePath} not found`,
        };
      }

      const content = fs.readFileSync(envExamplePath, 'utf-8');
      const lines = content.split('\n');

      const variables = new Set();
      let commentedRequired = 0;

      for (const line of lines) {
        const trimmed = line.trim();

        if (trimmed && !trimmed.startsWith('#')) {
          const [varName] = trimmed.split('=');
          if (varName) {
            variables.add(varName);

            const doc = envDocumentation[varName];
            if (doc && doc.required && line.startsWith('# ')) {
              commentedRequired++;
            }
          }
        }
      }

      const warnings = [];
      if (commentedRequired > 0) {
        warnings.push(`${commentedRequired} required variables are commented out`);
      }

      return {
        valid: true,
        variableCount: variables.size,
        warnings,
        message: `Valid .env.example with ${variables.size} variables`,
      };
    } catch (error) {
      return {
        valid: false,
        error: error.message,
        message: `Failed to validate ${envExamplePath}: ${error.message}`,
      };
    }
  }
}

// CLI Usage
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0] || 'generate';
  const outputPath = args[1] || '.env.example';

  console.log(`\n📋 Environment Example Generator\n`);

  let result;

  if (command === 'generate') {
    console.log(`Generating ${outputPath}...`);
    result = EnvExampleGenerator.generate(outputPath);
  } else if (command === 'from-env') {
    console.log(`Generating ${outputPath} from current environment...`);
    result = EnvExampleGenerator.generateFromEnvironment(outputPath);
  } else if (command === 'validate') {
    console.log(`Validating ${outputPath}...`);
    result = EnvExampleGenerator.validate(outputPath);
  } else {
    console.error(`Unknown command: ${command}`);
    console.error(`\nUsage:`);
    console.error(`  npm run generate:env [generate|from-env|validate] [outputPath]`);
    console.error(`\nExamples:`);
    console.error(`  npm run generate:env generate .env.example`);
    console.error(`  npm run generate:env from-env .env.example`);
    console.error(`  npm run generate:env validate .env.example`);
    process.exit(1);
  }

  console.log(`\n✅ ${result.message}\n`);

  if (result.error) {
    console.error(`❌ Error: ${result.error}`);
    process.exit(1);
  }

  if (result.warnings && result.warnings.length > 0) {
    console.warn(`⚠️  Warnings:`);
    result.warnings.forEach((w) => console.warn(`   - ${w}`));
  }

  process.exit(result.success !== false ? 0 : 1);
}

module.exports = {
  EnvExampleGenerator,
  envDocumentation,
};
