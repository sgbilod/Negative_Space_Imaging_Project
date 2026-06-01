import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Grid,
  TextField,
  Typography,
  Paper,
  Alert,
} from '@mui/material';
import { CloudUpload as UploadIcon } from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const Input = styled('input')({
  display: 'none',
});

interface ProcessingResult {
  jobId: string;
  status: string;
  progress: number;
  result?: string;
}

const ImageProcessing: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState<ProcessingResult | null>(null);
  const [error, setError] = useState<string>('');
  const [processingParams, setProcessingParams] = useState({
    contrast: 0,
    brightness: 0,
    sharpness: 0,
  });

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setError('');
    }
  };

  const handleParamChange = (param: keyof typeof processingParams) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setProcessingParams({
      ...processingParams,
      [param]: Number(event.target.value),
    });
  };

  const handleProcess = async () => {
    if (!selectedFile) {
      setError('Please select an image file');
      return;
    }

    setProcessing(true);
    setError('');

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('params', JSON.stringify(processingParams));

    try {
      const response = await fetch('http://localhost:8000/images/process', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Processing failed');
      }

      const data = await response.json();
      setResult(data);

      // Start polling for job status
      pollJobStatus(data.jobId);
    } catch (err) {
      setError('Failed to process image. Please try again.');
    } finally {
      setProcessing(false);
    }
  };

  const pollJobStatus = async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(
          `http://localhost:8000/jobs/${jobId}`,
          {
            headers: {
              Authorization: `Bearer ${localStorage.getItem('token')}`,
            },
          }
        );

        if (!response.ok) {
          throw new Error('Failed to fetch job status');
        }

        const data = await response.json();
        setResult(data);

        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Error polling job status:', err);
        clearInterval(interval);
      }
    }, 2000);
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Image Processing
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upload Image
              </Typography>
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  p: 3,
                }}
              >
                <label htmlFor="image-upload">
                  <Input
                    id="image-upload"
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                  />
                  <Button
                    variant="contained"
                    component="span"
                    startIcon={<UploadIcon />}
                  >
                    Select Image
                  </Button>
                </label>
                {selectedFile && (
                  <Typography sx={{ mt: 2 }}>
                    Selected: {selectedFile.name}
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Processing Parameters
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  label="Contrast"
                  type="number"
                  value={processingParams.contrast}
                  onChange={handleParamChange('contrast')}
                  InputProps={{ inputProps: { min: -100, max: 100 } }}
                />
                <TextField
                  label="Brightness"
                  type="number"
                  value={processingParams.brightness}
                  onChange={handleParamChange('brightness')}
                  InputProps={{ inputProps: { min: -100, max: 100 } }}
                />
                <TextField
                  label="Sharpness"
                  type="number"
                  value={processingParams.sharpness}
                  onChange={handleParamChange('sharpness')}
                  InputProps={{ inputProps: { min: -100, max: 100 } }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleProcess}
              disabled={!selectedFile || processing}
              sx={{ minWidth: 200 }}
            >
              {processing ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                'Process Image'
              )}
            </Button>
          </Box>
        </Grid>

        {result && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Processing Status
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography>Status: {result.status}</Typography>
                <Typography>Progress: {result.progress}%</Typography>
              </Box>
              {result.status === 'completed' && result.result && (
                <Box
                  component="img"
                  sx={{
                    maxWidth: '100%',
                    height: 'auto',
                  }}
                  src={`data:image/jpeg;base64,${result.result}`}
                  alt="Processed"
                />
              )}
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default ImageProcessing;
