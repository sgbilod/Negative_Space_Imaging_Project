/**
 * Upload Page Component
 * Handles file upload with drag-and-drop, preview, and progress tracking
 * @author Negative Space Imaging Project
 * @version 1.0.0
 */

import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Container,
  Paper,
  Typography,
  Alert,
  LinearProgress,
  Card,
  CardContent,
  CardMedia,
  Grid,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  Close as CloseIcon,
  CheckCircle as CheckCircleIcon,
  Image as ImageIcon,
  GetApp as GetAppIcon,
} from '@mui/icons-material';
import { useImageUpload } from '../hooks/useImageUpload';
import { useNotification } from '../hooks/useNotification';
import { useAuth } from '../hooks/useAuth';

/**
 * Uploaded file interface
 */
interface UploadedFile {
  id: string;
  file: File;
  preview: string;
  progress: number;
  status: 'pending' | 'uploading' | 'completed' | 'failed';
  error?: string;
}

/**
 * Upload Page Component
 * Features:
 * - Drag-and-drop file upload
 * - File input button
 * - Multiple file upload
 * - File preview with thumbnail
 * - Upload progress bar
 * - File validation
 * - Success message with link to results
 * - Error handling and retry
 * - Responsive design
 */
const UploadPage: React.FC = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { uploadImage, uploading } = useImageUpload();
  const { showNotification } = useNotification();
  const { getAccessToken } = useAuth();

  // State management
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [successDialogOpen, setSuccessDialogOpen] = useState(false);
  const [successAnalysisId, setSuccessAnalysisId] = useState<string | null>(null);

  // Constants
  const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff'];
  const MAX_FILES = 5;

  /**
   * Validate file before upload
   */
  const validateFile = (file: File): string | null => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return 'Only JPEG, PNG, WebP, and TIFF images are allowed';
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File size must be less than ${MAX_FILE_SIZE / (1024 * 1024)}MB`;
    }
    return null;
  };

  /**
   * Handle drag events
   */
  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  /**
   * Handle file drop
   */
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles) {
      handleFiles(Array.from(droppedFiles));
    }
  };

  /**
   * Handle file input change
   */
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(Array.from(e.target.files));
    }
  };

  /**
   * Process files
   */
  const handleFiles = async (files: File[]) => {
    // Check file count limit
    if (uploadedFiles.length + files.length > MAX_FILES) {
      showNotification(
        `Maximum ${MAX_FILES} files allowed. You already have ${uploadedFiles.length} file(s).`,
        'warning',
      );
      return;
    }

    // Validate and add files
    const newFiles: UploadedFile[] = [];

    for (const file of files) {
      const error = validateFile(file);
      if (error) {
        showNotification(`${file.name}: ${error}`, 'error');
        continue;
      }

      // Create preview
      const preview = URL.createObjectURL(file);

      newFiles.push({
        id: `${Date.now()}-${Math.random()}`,
        file,
        preview,
        progress: 0,
        status: 'pending',
      });
    }

    setUploadedFiles((prev) => [...prev, ...newFiles]);
    showNotification(`${newFiles.length} file(s) ready to upload`, 'success');
  };

  /**
   * Handle file upload
   */
  const handleUpload = async (fileId: string) => {
    const fileItem = uploadedFiles.find((f) => f.id === fileId);
    if (!fileItem) return;

    try {
      // Update status to uploading
      setUploadedFiles((prev) =>
        prev.map((f) => (f.id === fileId ? { ...f, status: 'uploading', progress: 0 } : f)),
      );

      // Get access token and upload
      const accessToken = getAccessToken();
      if (!accessToken) {
        throw new Error('Not authenticated');
      }

      // Upload file
      const analysisId = await uploadImage(fileItem.file, accessToken);

      // Update status to completed
      setUploadedFiles((prev) =>
        prev.map((f) => (f.id === fileId ? { ...f, status: 'completed', progress: 100 } : f)),
      );

      showNotification('Upload successful!', 'success');
      setSuccessAnalysisId(analysisId);
      setSuccessDialogOpen(true);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';

      // Update status to failed
      setUploadedFiles((prev) =>
        prev.map((f) => (f.id === fileId ? { ...f, status: 'failed', error: errorMessage } : f)),
      );

      showNotification(`Upload failed: ${errorMessage}`, 'error');
    }
  };

  /**
   * Handle upload all
   */
  const handleUploadAll = async () => {
    const pendingFiles = uploadedFiles.filter((f) => f.status === 'pending');
    for (const file of pendingFiles) {
      await handleUpload(file.id);
    }
  };

  /**
   * Remove file from list
   */
  const handleRemoveFile = (fileId: string) => {
    setUploadedFiles((prev) => {
      const file = prev.find((f) => f.id === fileId);
      if (file) {
        URL.revokeObjectURL(file.preview);
      }
      return prev.filter((f) => f.id !== fileId);
    });
  };

  /**
   * Handle clear all
   */
  const handleClearAll = () => {
    uploadedFiles.forEach((f) => URL.revokeObjectURL(f.preview));
    setUploadedFiles([]);
  };

  /**
   * Format file size
   */
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  const pendingCount = uploadedFiles.filter((f) => f.status === 'pending').length;
  const completedCount = uploadedFiles.filter((f) => f.status === 'completed').length;

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ marginBottom: 4 }}>
        <Typography
          variant="h3"
          component="h1"
          sx={{
            fontWeight: 700,
            marginBottom: 1,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Upload Images
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Upload images for analysis. Supported formats: JPEG, PNG, WebP, TIFF
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Upload Area */}
        <Grid item xs={12} md={8}>
          {/* Drop Zone */}
          <Paper
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            sx={{
              p: 4,
              textAlign: 'center',
              border: '2px dashed',
              borderColor: dragActive ? '#667eea' : '#ddd',
              borderRadius: 2,
              backgroundColor: dragActive ? 'rgba(102, 126, 234, 0.05)' : 'transparent',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              marginBottom: 3,
              '&:hover': {
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.05)',
              },
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={ALLOWED_TYPES.join(',')}
              onChange={handleFileInputChange}
              aria-label="Upload image files"
              style={{ display: 'none' }}
            />

            <CloudUploadIcon
              sx={{
                fontSize: 64,
                color: '#667eea',
                marginBottom: 2,
              }}
            />

            <Typography variant="h5" sx={{ fontWeight: 600, marginBottom: 1 }}>
              Drag and drop your images here
            </Typography>

            <Typography variant="body2" color="textSecondary" sx={{ marginBottom: 2 }}>
              or
            </Typography>

            <Button
              variant="contained"
              onClick={() => fileInputRef.current?.click()}
              sx={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                textTransform: 'none',
                fontWeight: 600,
              }}
            >
              Select Files
            </Button>

            <Typography variant="caption" display="block" color="textSecondary" sx={{ mt: 2 }}>
              Max file size: {MAX_FILE_SIZE / (1024 * 1024)}MB • Max files: {MAX_FILES}
            </Typography>
          </Paper>

          {/* Uploaded Files List */}
          {uploadedFiles.length > 0 && (
            <Card>
              <CardContent>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    mb: 2,
                  }}
                >
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Files ({uploadedFiles.length})
                  </Typography>
                  <Button size="small" color="error" onClick={handleClearAll}>
                    Clear All
                  </Button>
                </Box>

                <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                  <Chip
                    label={`Pending: ${pendingCount}`}
                    variant="outlined"
                    color={pendingCount > 0 ? 'warning' : 'default'}
                  />
                  <Chip
                    label={`Completed: ${completedCount}`}
                    variant="outlined"
                    color={completedCount > 0 ? 'success' : 'default'}
                  />
                </Box>

                {uploadedFiles.map((fileItem) => (
                  <Box
                    key={fileItem.id}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 2,
                      padding: 1.5,
                      borderBottom: '1px solid #eee',
                      '&:last-child': { borderBottom: 'none' },
                    }}
                  >
                    {/* File Preview */}
                    <Box
                      sx={{
                        width: 60,
                        height: 60,
                        borderRadius: 1,
                        backgroundColor: '#f5f5f5',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        overflow: 'hidden',
                        flexShrink: 0,
                      }}
                    >
                      <img
                        src={fileItem.preview}
                        alt={fileItem.file.name}
                        style={{
                          width: '100%',
                          height: '100%',
                          objectFit: 'cover',
                        }}
                      />
                    </Box>

                    {/* File Info */}
                    <Box flex={1} minWidth={0}>
                      <Typography
                        variant="body2"
                        sx={{
                          fontWeight: 600,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {fileItem.file.name}
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        {formatFileSize(fileItem.file.size)}
                      </Typography>

                      {/* Progress Bar */}
                      {fileItem.status === 'uploading' && (
                        <Box sx={{ mt: 1 }}>
                          <LinearProgress variant="determinate" value={fileItem.progress} />
                          <Typography variant="caption" color="textSecondary">
                            {fileItem.progress}%
                          </Typography>
                        </Box>
                      )}

                      {/* Error Message */}
                      {fileItem.status === 'failed' && (
                        <Typography
                          variant="caption"
                          sx={{
                            color: '#f44336',
                            display: 'block',
                            mt: 0.5,
                          }}
                        >
                          {fileItem.error}
                        </Typography>
                      )}
                    </Box>

                    {/* Status and Actions */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexShrink: 0 }}>
                      {fileItem.status === 'completed' && (
                        <CheckCircleIcon sx={{ color: '#4caf50' }} />
                      )}

                      {fileItem.status === 'pending' && (
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => handleUpload(fileItem.id)}
                          disabled={uploading}
                          sx={{
                            borderColor: '#667eea',
                            color: '#667eea',
                          }}
                        >
                          Upload
                        </Button>
                      )}

                      {fileItem.status === 'failed' && (
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => handleUpload(fileItem.id)}
                          disabled={uploading}
                          sx={{
                            borderColor: '#f44336',
                            color: '#f44336',
                          }}
                        >
                          Retry
                        </Button>
                      )}

                      <IconButton
                        size="small"
                        onClick={() => handleRemoveFile(fileItem.id)}
                        disabled={fileItem.status === 'uploading'}
                      >
                        <CloseIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                    </Box>
                  </Box>
                ))}
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Upload Tips Sidebar */}
        <Grid item xs={12} md={4}>
          <Card sx={{ position: 'sticky', top: 20 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, marginBottom: 2 }}>
                Upload Tips
              </Typography>

              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, marginBottom: 0.5 }}>
                    💡 Best Practices
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Use high-resolution images for better analysis results.
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, marginBottom: 0.5 }}>
                    🎯 Recommended Format
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    PNG or TIFF for maximum quality and precision.
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, marginBottom: 0.5 }}>
                    ⚡ Processing Time
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Analysis typically completes in 30-60 seconds per image.
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, marginBottom: 0.5 }}>
                    📊 What We Analyze
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Negative space areas, geometric patterns, contrast distribution.
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Upload All Button */}
      {pendingCount > 0 && (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            gap: 2,
            mt: 4,
          }}
        >
          <Button
            variant="contained"
            size="large"
            onClick={handleUploadAll}
            disabled={uploading || pendingCount === 0}
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              textTransform: 'none',
              fontWeight: 600,
              px: 4,
            }}
          >
            Upload All ({pendingCount})
          </Button>
        </Box>
      )}

      {/* Success Dialog */}
      <Dialog open={successDialogOpen} onClose={() => setSuccessDialogOpen(false)}>
        <DialogTitle sx={{ textAlign: 'center', fontWeight: 600 }}>
          ✅ Upload Successful!
        </DialogTitle>
        <DialogContent>
          <Typography sx={{ my: 2 }}>
            Your image has been uploaded and analysis is starting. You'll receive a notification
            when it's complete.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSuccessDialogOpen(false)} sx={{ color: '#667eea' }}>
            Continue Uploading
          </Button>
          <Button
            variant="contained"
            onClick={() => {
              if (successAnalysisId) {
                navigate(`/analysis/${successAnalysisId}`);
              }
              setSuccessDialogOpen(false);
            }}
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            }}
          >
            View Results
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default UploadPage;
