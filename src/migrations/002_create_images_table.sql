-- Migration 002: Create images table
-- Description: Create the images table for storing uploaded image metadata
-- Created: 2025-11-08

CREATE TABLE IF NOT EXISTS images (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  filename VARCHAR(255) NOT NULL,
  original_filename VARCHAR(255) NOT NULL,
  file_size INTEGER NOT NULL,
  storage_path VARCHAR(500) NOT NULL,
  mime_type VARCHAR(100),
  width INTEGER,
  height INTEGER,
  uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_images_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_images_user_id ON images(user_id);
CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_images_uploaded_at ON images(uploaded_at DESC);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_images_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER images_updated_at_trigger
BEFORE UPDATE ON images
FOR EACH ROW
EXECUTE FUNCTION update_images_updated_at();

-- Add comments
COMMENT ON TABLE images IS 'Stores metadata for uploaded images';
COMMENT ON COLUMN images.user_id IS 'Foreign key reference to users table';
COMMENT ON COLUMN images.storage_path IS 'File system path where image is stored';
COMMENT ON COLUMN images.mime_type IS 'MIME type of the image (e.g., image/jpeg)';
