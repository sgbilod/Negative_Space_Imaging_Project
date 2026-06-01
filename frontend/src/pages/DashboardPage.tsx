/**
 * Dashboard Page Component
 * Main dashboard showing user welcome, stats, recent analyses, and quick actions
 * @author Negative Space Imaging Project
 * @version 1.0.0
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Grid,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Menu,
  MenuItem,
  Avatar,
  Divider,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  History as HistoryIcon,
  BarChart as BarChartIcon,
  Settings as SettingsIcon,
  Logout as LogoutIcon,
  MoreVert as MoreVertIcon,
  ImageSearch as ImageSearchIcon,
  CheckCircle as CheckCircleIcon,
  PendingActions as PendingActionsIcon,
} from '@mui/icons-material';
import { useAuth } from '../hooks/useAuth';
import { useNotification } from '../hooks/useNotification';

/**
 * Analysis result interface
 */
interface Analysis {
  id: string;
  fileName: string;
  uploadedAt: string;
  status: 'completed' | 'processing' | 'failed';
  areasFound: number;
  confidence: number;
}

/**
 * Dashboard stats interface
 */
interface DashboardStats {
  totalImages: number;
  completedAnalyses: number;
  processingAnalyses: number;
  totalAreasFound: number;
}

/**
 * Dashboard Page Component
 * Features:
 * - Welcome message with user name
 * - Quick stats display
 * - Recent analyses table
 * - Quick action buttons
 * - User menu dropdown
 * - Logout functionality
 */
const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const { showNotification } = useNotification();

  // State management
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  /**
   * Fetch dashboard data
   */
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        setError('');

        // In production, replace with actual API calls
        // const [statsResponse, analysesResponse] = await Promise.all([
        //   fetch(`${API_URL}/dashboard/stats`, {
        //     headers: { Authorization: `Bearer ${getAccessToken()}` }
        //   }),
        //   fetch(`${API_URL}/dashboard/recent-analyses`, {
        //     headers: { Authorization: `Bearer ${getAccessToken()}` }
        //   })
        // ]);

        // Mock data for development
        setStats({
          totalImages: 24,
          completedAnalyses: 18,
          processingAnalyses: 2,
          totalAreasFound: 156,
        });

        setAnalyses([
          {
            id: '1',
            fileName: 'landscape_001.jpg',
            uploadedAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
            status: 'completed',
            areasFound: 12,
            confidence: 0.95,
          },
          {
            id: '2',
            fileName: 'urban_scene.png',
            uploadedAt: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
            status: 'completed',
            areasFound: 8,
            confidence: 0.87,
          },
          {
            id: '3',
            fileName: 'abstract_art.jpg',
            uploadedAt: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
            status: 'processing',
            areasFound: 0,
            confidence: 0,
          },
          {
            id: '4',
            fileName: 'portrait_002.jpg',
            uploadedAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
            status: 'completed',
            areasFound: 15,
            confidence: 0.92,
          },
          {
            id: '5',
            fileName: 'nature_photo.jpg',
            uploadedAt: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
            status: 'completed',
            areasFound: 11,
            confidence: 0.89,
          },
        ]);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load dashboard data';
        setError(message);
        showNotification(message, 'error');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [showNotification]);

  /**
   * Format date to readable string
   */
  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(hours / 24);

    if (hours < 1) {
      return 'Just now';
    } else if (hours < 24) {
      return `${hours}h ago`;
    } else if (days < 7) {
      return `${days}d ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  /**
   * Get status chip color
   */
  const getStatusColor = (status: string): 'success' | 'warning' | 'error' => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default' as any;
    }
  };

  /**
   * Handle user menu open
   */
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  /**
   * Handle user menu close
   */
  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  /**
   * Handle logout
   */
  const handleLogout = async () => {
    try {
      await logout();
      showNotification('Logged out successfully', 'success');
      navigate('/login');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Logout failed';
      showNotification(message, 'error');
    }
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

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header Section */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 4,
          flexWrap: 'wrap',
          gap: 2,
        }}
      >
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
            Welcome back, {user?.firstName}! 👋
          </Typography>
          <Typography variant="body1" color="textSecondary">
            Here's what's happening with your imaging analysis
          </Typography>
        </Box>

        {/* User Menu */}
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="contained"
            startIcon={<CloudUploadIcon />}
            onClick={() => navigate('/upload')}
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              textTransform: 'none',
              fontWeight: 600,
            }}
          >
            Upload Image
          </Button>

          <Button
            onClick={handleMenuOpen}
            sx={{
              borderRadius: '50%',
              minWidth: 'auto',
              width: 48,
              height: 48,
              p: 0,
            }}
          >
            <Avatar
              sx={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                cursor: 'pointer',
              }}
            >
              {user?.firstName?.[0]}
              {user?.lastName?.[0]}
            </Avatar>
          </Button>

          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
          >
            <MenuItem disabled>
              <Typography variant="caption" color="textSecondary">
                {user?.email}
              </Typography>
            </MenuItem>
            <Divider />
            <MenuItem
              onClick={() => {
                navigate('/settings');
                handleMenuClose();
              }}
            >
              <SettingsIcon sx={{ mr: 1 }} /> Settings
            </MenuItem>
            <MenuItem onClick={handleLogout}>
              <LogoutIcon sx={{ mr: 1 }} /> Logout
            </MenuItem>
          </Menu>
        </Box>
      </Box>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ marginBottom: 3 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Stats Grid */}
      <Grid container spacing={2} sx={{ marginBottom: 4 }}>
        {stats &&
          [
            {
              label: 'Total Images',
              value: stats.totalImages,
              icon: <ImageSearchIcon />,
              color: '#667eea',
            },
            {
              label: 'Completed',
              value: stats.completedAnalyses,
              icon: <CheckCircleIcon />,
              color: '#4caf50',
            },
            {
              label: 'Processing',
              value: stats.processingAnalyses,
              icon: <PendingActionsIcon />,
              color: '#ff9800',
            },
            {
              label: 'Areas Found',
              value: stats.totalAreasFound,
              icon: <BarChartIcon />,
              color: '#f44336',
            },
          ].map((stat, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card
                sx={{
                  background:
                    'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
                  border: '1px solid #667eea',
                  borderRadius: 2,
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0 12px 24px rgba(102, 126, 234, 0.2)',
                  },
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Box
                      sx={{
                        p: 1.5,
                        borderRadius: '50%',
                        background: stat.color,
                        color: 'white',
                        display: 'flex',
                      }}
                    >
                      {stat.icon}
                    </Box>
                    <Box flex={1}>
                      <Typography color="textSecondary" variant="body2">
                        {stat.label}
                      </Typography>
                      <Typography variant="h4" sx={{ fontWeight: 700 }}>
                        {stat.value}
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
      </Grid>

      {/* Recent Analyses Section */}
      <Card sx={{ borderRadius: 2, boxShadow: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <HistoryIcon sx={{ color: '#667eea' }} />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Recent Analyses
            </Typography>
          </Box>

          {analyses.length === 0 ? (
            <Box
              sx={{
                textAlign: 'center',
                py: 4,
              }}
            >
              <ImageSearchIcon
                sx={{
                  fontSize: 48,
                  color: '#ccc',
                  marginBottom: 2,
                }}
              />
              <Typography color="textSecondary">
                No analyses yet. Start by uploading an image!
              </Typography>
              <Button
                variant="contained"
                startIcon={<CloudUploadIcon />}
                onClick={() => navigate('/upload')}
                sx={{
                  mt: 2,
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  textTransform: 'none',
                }}
              >
                Upload First Image
              </Button>
            </Box>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                    <TableCell sx={{ fontWeight: 600 }}>File Name</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Status</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 600 }}>
                      Areas Found
                    </TableCell>
                    <TableCell align="right" sx={{ fontWeight: 600 }}>
                      Confidence
                    </TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Uploaded</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 600 }}>
                      Action
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {analyses.map((analysis) => (
                    <TableRow
                      key={analysis.id}
                      sx={{
                        '&:hover': { backgroundColor: '#f9f9f9' },
                        cursor: 'pointer',
                      }}
                    >
                      <TableCell>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {analysis.fileName}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={analysis.status.charAt(0).toUpperCase() + analysis.status.slice(1)}
                          color={getStatusColor(analysis.status)}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2">{analysis.areasFound}</Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography
                          variant="body2"
                          sx={{
                            color: analysis.confidence > 0.85 ? '#4caf50' : '#ff9800',
                            fontWeight: 600,
                          }}
                        >
                          {analysis.confidence > 0
                            ? `${(analysis.confidence * 100).toFixed(0)}%`
                            : '-'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption" color="textSecondary">
                          {formatDate(analysis.uploadedAt)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => navigate(`/analysis/${analysis.id}`)}
                          sx={{
                            borderColor: '#667eea',
                            color: '#667eea',
                            '&:hover': {
                              backgroundColor: '#667eea',
                              color: 'white',
                            },
                          }}
                        >
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
        <CardActions sx={{ justifyContent: 'space-between', p: 2, borderTop: '1px solid #eee' }}>
          <Button size="small" color="primary">
            View All
          </Button>
          <Button
            size="small"
            variant="outlined"
            onClick={() => navigate('/upload')}
            sx={{
              borderColor: '#667eea',
              color: '#667eea',
              textTransform: 'none',
            }}
          >
            New Analysis
          </Button>
        </CardActions>
      </Card>
    </Container>
  );
};

export default DashboardPage;
