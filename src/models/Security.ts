import { Model, DataTypes, Optional } from 'sequelize';
import { sequelize } from '../database/connection';
import { logger } from '../utils/logger';
import crypto from 'crypto';

/**
 * Audit log attributes interface
 */
export interface AuditLogAttributes {
  id: string;
  userId: string | null;
  action: string;
  resource: string;
  resourceId: string | null;
  details: Record<string, unknown>;
  ipAddress: string;
  userAgent: string;
  outcome: 'SUCCESS' | 'FAILURE';
  timestamp: Date;
}

/**
 * Audit log creation attributes
 */
export interface AuditLogCreationAttributes extends Optional<AuditLogAttributes, 'id' | 'timestamp'> {}

/**
 * HIPAA-compliant audit logging model
 */
export class AuditLog extends Model<AuditLogAttributes, AuditLogCreationAttributes> implements AuditLogAttributes {
  public id!: string;
  public userId!: string | null;
  public action!: string;
  public resource!: string;
  public resourceId!: string | null;
  public details!: Record<string, unknown>;
  public ipAddress!: string;
  public userAgent!: string;
  public outcome!: 'SUCCESS' | 'FAILURE';
  public timestamp!: Date;
}

/**
 * Initialize Audit Log model
 */
AuditLog.init(
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    userId: {
      type: DataTypes.UUID,
      allowNull: true,
      references: {
        model: 'users',
        key: 'id',
      },
      onDelete: 'SET NULL',
    },
    action: {
      type: DataTypes.STRING(50),
      allowNull: false,
      comment: 'Type of action performed (VIEW, EDIT, CREATE, DELETE, LOGIN, etc.)',
    },
    resource: {
      type: DataTypes.STRING(100),
      allowNull: false,
      comment: 'The resource being accessed (PATIENT_RECORD, IMAGE, ANALYSIS, etc.)',
    },
    resourceId: {
      type: DataTypes.STRING,
      allowNull: true,
      comment: 'Identifier of the specific resource being accessed',
    },
    details: {
      type: DataTypes.JSONB,
      allowNull: false,
      defaultValue: {},
      comment: 'Additional contextual details about the action',
    },
    ipAddress: {
      type: DataTypes.STRING(45),
      allowNull: false,
      comment: 'IP address of the user performing the action',
    },
    userAgent: {
      type: DataTypes.STRING,
      allowNull: false,
      comment: 'User agent of the client used',
    },
    outcome: {
      type: DataTypes.ENUM('SUCCESS', 'FAILURE'),
      allowNull: false,
      comment: 'Whether the action succeeded or failed',
    },
    timestamp: {
      type: DataTypes.DATE,
      allowNull: false,
      defaultValue: DataTypes.NOW,
      comment: 'When the action occurred',
    },
  },
  {
    sequelize,
    modelName: 'AuditLog',
    tableName: 'audit_logs',
    timestamps: false,
    indexes: [
      {
        name: 'audit_logs_user_id_idx',
        fields: ['userId'],
      },
      {
        name: 'audit_logs_resource_id_idx',
        fields: ['resourceId'],
      },
      {
        name: 'audit_logs_timestamp_idx',
        fields: ['timestamp'],
      },
    ],
  }
);

/**
 * Security-related utility functions
 */
export class Security {
  /**
   * Create HIPAA-compliant audit log entry
   */
  static async createAuditLog(data: AuditLogCreationAttributes): Promise<AuditLog> {
    try {
      const log = await AuditLog.create(data);
      logger.debug('Audit log created', { id: log.id, action: log.action, resource: log.resource });
      return log;
    } catch (error) {
      logger.error('Failed to create audit log', { error, data });
      // Don't throw - audit logging should never break application flow
      return null as unknown as AuditLog;
    }
  }

  /**
   * Generate a secure random token
   */
  static generateSecureToken(length = 32): string {
    return crypto.randomBytes(length).toString('hex');
  }

  /**
   * Hash sensitive data
   */
  static hashData(data: string): string {
    return crypto.createHash('sha256').update(data).digest('hex');
  }
  
  /**
   * Encrypt sensitive data
   */
  static encryptData(text: string, key: string): string {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-cbc', Buffer.from(key, 'hex'), iv);
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    return `${iv.toString('hex')}:${encrypted}`;
  }
  
  /**
   * Decrypt sensitive data
   */
  static decryptData(text: string, key: string): string {
    const [ivHex, encryptedText] = text.split(':');
    const iv = Buffer.from(ivHex, 'hex');
    const decipher = crypto.createDecipheriv('aes-256-cbc', Buffer.from(key, 'hex'), iv);
    let decrypted = decipher.update(encryptedText, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
  }
}
