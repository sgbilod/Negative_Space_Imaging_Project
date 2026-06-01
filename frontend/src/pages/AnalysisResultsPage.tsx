/**
 * Analysis Results Page Component
 * Displays analysis results with image viewer, statistics, and export options
 * @author Negative Space Imaging Project
 * @version 1.0.0
 */

import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Container,
  Card,
  CardContent,
  Grid,
  Typography,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  Download as DownloadIcon,
  Share as ShareIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  RotateRight as RotateRightIcon,
  Compare as CompareIcon,
  MoreVert as MoreVertIcon,
} from '@mui/icons-material';
import { useNotification } from '../hooks/useNotification';

/**
 * Analysis data interface
 */
interface AnalysisData {
  id: string;
  fileName: string;
  originalImage: string;
  analyzedImage: string;
  analysisDate: string;
  status: 'completed' | 'processing' | 'failed';
  statistics: {
    totalAreas: number;
    averageConfidence: number;
    dominantColor: string;
    contrastRatio: number;
    processingTime: number;
  };
  regions: Array<{
    id: string;
    area: number;
    confidence: number;
    location: string;
    description: string;
  }>;
}

/**
 * Analysis Results Page Component
 * Features:
 * - Original and analyzed image display with zoom
 * - Statistics panel
 * - Detailed regions list
 * - Export options (PNG, CSV, JSON)
 * - Share functionality
 * - Responsive design
 */
const AnalysisResultsPage: React.FC = () => {
  const { imageId } = useParams<{ imageId: string }>();
  const navigate = useNavigate();
  const { showNotification } = useNotification();

  // State management
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [compareMode, setCompareMode] = useState(false);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [shareUrl, setShareUrl] = useState('');
  const [exportMenuAnchor, setExportMenuAnchor] = useState<null | HTMLElement>(null);

  /**
   * Fetch analysis data
   */
  useEffect(() => {
    const fetchAnalysisData = async () => {
      try {
        setLoading(true);
        setError('');

        // Mock data for development
        // In production: fetch from API using imageId
        setAnalysis({
          id: imageId || '1',
          fileName: 'landscape_001.jpg',
          originalImage: 'https://via.placeholder.com/800x600?text=Original+Image',
          analyzedImage: 'https://via.placeholder.com/800x600?text=Analyzed+Image',
          analysisDate: new Date().toISOString(),
          status: 'completed',
          statistics: {
            totalAreas: 12,
            averageConfidence: 0.94,
            dominantColor: '#2A2A2A',
            contrastRatio: 8.5,
            processingTime: 45,
          },
          regions: [
            {
              id: '1',
              area: 2400,
              confidence: 0.98,
              location: 'Top-Left',
              description: 'Large negative space area',
            },
            {
              id: '2',
              area: 1800,
              confidence: 0.92,
              location: 'Bottom-Right',
              description: 'Medium shadow region',
            },
            {
              id: '3',
              area: 1200,
              confidence: 0.89,
              location: 'Center',
              description: 'Central void space',
            },
          ],
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load analysis';
        setError(message);
        showNotification(message, 'error');
      } finally {
        setLoading(false);
      }
    };

    if (imageId) {
      fetchAnalysisData();
    }
  }, [imageId, showNotification]);

  /**
   * Handle zoom in
   */
  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev + 0.2, 3));
  };

  /**
   * Handle zoom out
   */
  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev - 0.2, 0.5));
  };

  /**
   * Handle rotation
   */
  const handleRotate = () => {
    setRotation((prev) => (prev + 90) % 360);
  };

  /**
   * Handle export
   */
  const handleExport = (format: 'png' | 'csv' | 'json') => {
    if (!analysis) return;

    setExportMenuAnchor(null);

    try {
      let content = '';
      let filename = '';

      switch (format) {
        case 'png':
          // In production: use canvas or server-side export
          showNotification('PNG export started', 'success');
          break;

        case 'csv':
          content = 'Region ID,Area,Confidence,Location,Description\n';
          analysis.regions.forEach((region) => {
            content += `${region.id},${region.area},${region.confidence},${region.location},"${region.description}"\n`;
          });
          filename = `analysis-${analysis.id}-results.csv`;
          downloadFile(content, filename, 'text/csv');
          showNotification('CSV exported successfully', 'success');
          break;

        case 'json':
          content = JSON.stringify(analysis, null, 2);
          filename = `analysis-${analysis.id}-results.json`;
          downloadFile(content, filename, 'application/json');
          showNotification('JSON exported successfully', 'success');
          break;
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Export failed';
      showNotification(message, 'error');
    }
  };

  /**
   * Download file helper
   */
  const downloadFile = (content: string, filename: string, type: string) => {
    const element = document.createElement('a');
    element.setAttribute('href', `data:${type};charset=utf-8,${encodeURIComponent(content)}`);
    element.setAttribute('download', filename);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  /**
   * Handle share
   */
  const handleShare = () => {
    const url = `${window.location.origin}/analysis/${analysis?.id}`;
    setShareUrl(url);
    setShareDialogOpen(true);
  };

  /**
   * Copy share URL
   */
  const handleCopyShareUrl = () => {
    navigator.clipboard.writeText(shareUrl);
    showNotification('Share URL copied to clipboard', 'success');
  };

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '80vh',
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (error || !analysis) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error || 'Analysis not found'}
        </Alert>
        <Button variant="outlined" onClick={() => navigate('/dashboard')}>
          Back to Dashboard
        </Button>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Box>
          <Typography
            variant="h3"
            component="h1"
            sx={{
              fontWeight: 700,
              marginBottom: 0.5,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Analysis Results
          </Typography>
          <Typography variant="body1" color="textSecondary">
            {analysis.fileName}
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<ShareIcon />}
            onClick={handleShare}
            sx={{
              borderColor: '#667eea',
              color: '#667eea',
              textTransform: 'none',
            }}
          >
            Share
          </Button>

          <Button
            variant="contained"
            endIcon={<MoreVertIcon />}
            onClick={(e) => setExportMenuAnchor(e.currentTarget)}
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              textTransform: 'none',
            }}
          >
            Export
          </Button>

          <Menu
            anchorEl={exportMenuAnchor}
            open={Boolean(exportMenuAnchor)}
            onClose={() => setExportMenuAnchor(null)}
          >
            <MenuItem onClick={() => handleExport('png')}>
              <DownloadIcon sx={{ mr: 1 }} /> PNG Image
            </MenuItem>
            <MenuItem onClick={() => handleExport('csv')}>
              <DownloadIcon sx={{ mr: 1 }} /> CSV Data
            </MenuItem>
            <MenuItem onClick={() => handleExport('json')}>
              <DownloadIcon sx={{ mr: 1 }} /> JSON Report
            </MenuItem>
          </Menu>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Image Viewer */}
        <Grid item xs={12} lg={8}>
          <Card sx={{ borderRadius: 2 }}>
            <CardContent>
              {/* Toolbar */}
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  mb: 2,
                  pb: 2,
                  borderBottom: '1px solid #eee',
                }}
              >
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  Image Viewer
                </Typography>

                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<ZoomInIcon />}
                    onClick={handleZoomIn}
                    sx={{
                      borderColor: '#667eea',
                      color: '#667eea',
                    }}
                  >
                    +
                  </Button>

                  <Typography variant="body2" sx={{ px: 1, display: 'flex', alignItems: 'center' }}>
                    {Math.round(zoom * 100)}%
                  </Typography>

                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<ZoomOutIcon />}
                    onClick={handleZoomOut}
                    sx={{
                      borderColor: '#667eea',
                      color: '#667eea',
                    }}
                  >
                    -
                  </Button>

                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<RotateRightIcon />}
                    onClick={handleRotate}
                    sx={{
                      borderColor: '#667eea',
                      color: '#667eea',
                    }}
                  >
                    Rotate
                  </Button>

                  <Button
                    size="small"
                    variant={compareMode ? 'contained' : 'outlined'}
                    startIcon={<CompareIcon />}
                    onClick={() => setCompareMode(!compareMode)}
                    sx={
                      compareMode
                        ? {
                            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                          }
                        : {
                            borderColor: '#667eea',
                            color: '#667eea',
                          }
                    }
                  >
                    Compare
                  </Button>
                </Box>
              </Box>

              {/* Images */}
              <Box
                sx={{
                  display: 'flex',
                  gap: 2,
                  overflow: 'auto',
                  backgroundColor: '#f5f5f5',
                  borderRadius: 1,
                  p: 2,
                  minHeight: 400,
                }}
              >
                {/* Original Image */}
                <Box
                  sx={{
                    flex: compareMode ? 1 : '0 0 100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Box
                    component="img"
                    src={analysis.originalImage}
                    alt="Original"
                    sx={{
                      maxWidth: '100%',
                      maxHeight: '100%',
                      transform: `scale(${zoom}) rotate(${rotation}deg)`,
                      transition: 'transform 0.3s ease',
                      borderRadius: 1,
                    }}
                  />
                </Box>

                {/* Analyzed Image */}
                {compareMode && (
                  <Box
                    sx={{
                      flex: 1,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <Box
                      component="img"
                      src={analysis.analyzedImage}
                      alt="Analyzed"
                      sx={{
                        maxWidth: '100%',
                        maxHeight: '100%',
                        transform: `scale(${zoom}) rotate(${rotation}deg)`,
                        transition: 'transform 0.3s ease',
                        borderRadius: 1,
                      }}
                    />
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Statistics */}
        <Grid item xs={12} lg={4}>
          <Card sx={{ borderRadius: 2, mb: 3 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Statistics
              </Typography>

              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <StatItem
                  label="Total Areas Found"
                  value={analysis.statistics.totalAreas}
                  unit=""
                />
                <StatItem
                  label="Average Confidence"
                  value={analysis.statistics.averageConfidence}
                  unit="%"
                  format={(v: number) => `${(v * 100).toFixed(0)}`}
                />
                <StatItem
                  label="Contrast Ratio"
                  value={analysis.statistics.contrastRatio}
                  unit=":1"
                />
                <StatItem
                  label="Processing Time"
                  value={analysis.statistics.processingTime}
                  unit="s"
                />
                <Box>
                  <Typography variant="caption" color="textSecondary">
                    Dominant Color
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                    <Box
                      sx={{
                        width: 32,
                        height: 32,
                        backgroundColor: analysis.statistics.dominantColor,
                        borderRadius: 1,
                        border: '1px solid #ddd',
                      }}
                    />
                    <Typography variant="body2">{analysis.statistics.dominantColor}</Typography>
                  </Box>
                </Box>
              </Box>
            </CardContent>
          </Card>

          {/* Status Chip */}
          <Chip
            label={`Status: ${analysis.status.charAt(0).toUpperCase() + analysis.status.slice(1)}`}
            color={analysis.status === 'completed' ? 'success' : 'warning'}
            variant="outlined"
            sx={{ width: '100%', py: 3, fontSize: '1rem' }}
          />
        </Grid>
      </Grid>

      {/* Regions Table */}
      <Card sx={{ mt: 3, borderRadius: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Detected Regions ({analysis.regions.length})
          </Typography>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                  <TableCell sx={{ fontWeight: 600 }}>ID</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Area (px²)</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Confidence</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Location</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Description</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {analysis.regions.map((region) => (
                  <TableRow key={region.id} hover>
                    <TableCell>{region.id}</TableCell>
                    <TableCell>{region.area.toLocaleString()}</TableCell>
                    <TableCell>
                      <Chip
                        label={`${(region.confidence * 100).toFixed(0)}%`}
                        color={region.confidence > 0.9 ? 'success' : 'warning'}
                        size="small"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>{region.location}</TableCell>
                    <TableCell>{region.description}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Share Dialog */}
      <Dialog
        open={shareDialogOpen}
        onClose={() => setShareDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Share Analysis Results</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            value={shareUrl}
            variant="outlined"
            InputProps={{ readOnly: true }}
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShareDialogOpen(false)}>Close</Button>
          <Button
            variant="contained"
            onClick={handleCopyShareUrl}
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            }}
          >
            Copy to Clipboard
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

/**
 * Statistic Item Component
 */
interface StatItemProps {
  label: string;
  value: number;
  unit: string;
  format?: (v: number) => string;
}

const StatItem: React.FC<StatItemProps> = ({ label, value, unit, format }) => (
  <Box>
    <Typography variant="caption" color="textSecondary">
      {label}
    </Typography>
    <Typography variant="h5" sx={{ fontWeight: 700, color: '#667eea' }}>
      {format ? format(value) : value}
      <Typography component="span" variant="body2" color="textSecondary" sx={{ ml: 0.5 }}>
        {unit}
      </Typography>
    </Typography>
  </Box>
);

export default AnalysisResultsPage;
