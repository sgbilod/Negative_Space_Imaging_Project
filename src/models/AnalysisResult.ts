/**
 * Analysis Result Model
 *
 * Stores the results of negative space analysis on images.
 * Contains quantitative metrics and raw analysis data.
 *
 * @module models/AnalysisResult
 */

import { DataTypes, Model, Sequelize, ForeignKey, JSON } from 'sequelize';
import { Image } from './Image';

export interface AnalysisData {
  [key: string]: unknown;
  total_negative_space?: number;
  average_confidence?: number;
  processing_time_ms?: number;
  contours_count?: number;
}

export class AnalysisResult extends Model {
  public id!: number;
  public image_id!: ForeignKey<Image['id']>;
  public negative_space_percentage!: number;
  public regions_count!: number;
  public processing_time_ms!: number;
  public raw_data!: AnalysisData;
  public created_at!: Date;
  public updated_at!: Date;

  /**
   * Serialize analysis result for API response
   */
  public serialize(): object {
    return {
      id: this.id,
      image_id: this.image_id,
      negative_space_percentage: this.negative_space_percentage,
      regions_count: this.regions_count,
      processing_time_ms: this.processing_time_ms,
      raw_data: this.raw_data,
      created_at: this.created_at,
    };
  }

  /**
   * Get summary statistics
   */
  public getSummary(): object {
    return {
      negative_space: this.negative_space_percentage,
      regions: this.regions_count,
      processing_time: `${this.processing_time_ms}ms`,
    };
  }

  /**
   * Check if result quality is acceptable
   */
  public isQualityAcceptable(minConfidence: number = 0.5): boolean {
    if (!this.raw_data.average_confidence) {
      return false;
    }
    return this.raw_data.average_confidence >= minConfidence;
  }
}

/**
 * Initialize AnalysisResult model
 */
export function initAnalysisResultModel(sequelize: Sequelize): typeof AnalysisResult {
  AnalysisResult.init(
    {
      id: {
        type: DataTypes.INTEGER,
        autoIncrement: true,
        primaryKey: true,
      },
      image_id: {
        type: DataTypes.INTEGER,
        allowNull: false,
        references: {
          model: 'images',
          key: 'id',
        },
        onDelete: 'CASCADE',
      },
      negative_space_percentage: {
        type: DataTypes.FLOAT,
        allowNull: false,
        validate: {
          min: 0,
          max: 100,
        },
      },
      regions_count: {
        type: DataTypes.INTEGER,
        allowNull: false,
        validate: {
          min: 0,
        },
      },
      processing_time_ms: {
        type: DataTypes.FLOAT,
        allowNull: false,
      },
      raw_data: {
        type: DataTypes.JSON,
        allowNull: true,
        defaultValue: {},
      },
      created_at: {
        type: DataTypes.DATE,
        defaultValue: DataTypes.NOW,
      },
      updated_at: {
        type: DataTypes.DATE,
        defaultValue: DataTypes.NOW,
      },
    },
    {
      sequelize,
      tableName: 'analysis_results',
      timestamps: true,
      underscored: true,
    }
  );

  return AnalysisResult;
}
