-- Migration 003: Create analysis_results table
-- Description: Create the analysis_results table for storing image analysis output
-- Created: 2025-11-08

CREATE TABLE IF NOT EXISTS analysis_results (
  id SERIAL PRIMARY KEY,
  image_id INTEGER NOT NULL,
  negative_space_percentage NUMERIC(5, 2) NOT NULL,
  positive_space_percentage NUMERIC(5, 2) NOT NULL,
  regions_count INTEGER NOT NULL,
  processing_time_ms INTEGER NOT NULL,
  confidence_score NUMERIC(3, 2) NOT NULL,
  status VARCHAR(50) NOT NULL DEFAULT 'completed',
  raw_data JSONB,
  visualization_data JSONB,
  error_message TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_analysis_results_image_id FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
  CONSTRAINT check_negative_space_percentage CHECK (negative_space_percentage >= 0 AND negative_space_percentage <= 100),
  CONSTRAINT check_positive_space_percentage CHECK (positive_space_percentage >= 0 AND positive_space_percentage <= 100),
  CONSTRAINT check_confidence_score CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_analysis_results_image_id ON analysis_results(image_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_results_status ON analysis_results(status);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_analysis_results_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER analysis_results_updated_at_trigger
BEFORE UPDATE ON analysis_results
FOR EACH ROW
EXECUTE FUNCTION update_analysis_results_updated_at();

-- Add comments
COMMENT ON TABLE analysis_results IS 'Stores results from image analysis by the Python engine';
COMMENT ON COLUMN analysis_results.image_id IS 'Foreign key reference to images table';
COMMENT ON COLUMN analysis_results.negative_space_percentage IS 'Percentage of image identified as negative space (0-100)';
COMMENT ON COLUMN analysis_results.regions_count IS 'Number of identified regions in the image';
COMMENT ON COLUMN analysis_results.raw_data IS 'JSON data with detailed analysis information';
COMMENT ON COLUMN analysis_results.visualization_data IS 'JSON data for frontend visualization';
COMMENT ON COLUMN analysis_results.status IS 'Status of the analysis: pending, processing, completed, failed';
