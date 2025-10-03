import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  CheckCircle as SuccessIcon,
} from '@mui/icons-material';
import { SelectChangeEvent } from '@mui/material/Select';

interface SecurityEvent {
  id: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  description: string;
  source: string;
}

const severityIcons = {
  low: <InfoIcon color="info" />,
  medium: <WarningIcon sx={{ color: 'orange' }} />,
  high: <ErrorIcon color="error" />,
  critical: <ErrorIcon sx={{ color: 'darkred' }} />,
};

const severityColors = {
  low: 'info',
  medium: 'warning',
  high: 'error',
  critical: 'error',
};

const SecurityMonitor: React.FC = () => {
  const [events, setEvents] = useState<SecurityEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filters, setFilters] = useState({
    severity: 'all',
    category: 'all',
  });

  const fetchSecurityEvents = async () => {
    try {
      const queryParams = new URLSearchParams();
      if (filters.severity !== 'all') queryParams.append('severity', filters.severity);
      if (filters.category !== 'all') queryParams.append('category', filters.category);

      const response = await fetch(
        `http://localhost:8000/security-events?${queryParams}`,
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`,
          },
        }
      );

      if (!response.ok) {
        throw new Error('Failed to fetch security events');
      }

      const data = await response.json();
      setEvents(data.events);
    } catch (err) {
      setError('Failed to load security events');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSecurityEvents();
    // Set up real-time updates
    const ws = new WebSocket('ws://localhost:8000/ws/security-events');

    ws.onmessage = (event) => {
      const newEvent = JSON.parse(event.data);
      setEvents((prevEvents) => [newEvent, ...prevEvents]);
    };

    return () => {
      ws.close();
    };
  }, [filters]);

  const handleFilterChange = (
    event: SelectChangeEvent,
    filterType: 'severity' | 'category'
  ) => {
    setFilters((prev) => ({
      ...prev,
      [filterType]: event.target.value,
    }));
  };

  const getEventStats = () => {
    return {
      total: events.length,
      critical: events.filter((e) => e.severity === 'critical').length,
      high: events.filter((e) => e.severity === 'high').length,
      medium: events.filter((e) => e.severity === 'medium').length,
      low: events.filter((e) => e.severity === 'low').length,
    };
  };

  const stats = getEventStats();

  if (loading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="200px"
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Security Monitor
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={2.4}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Events
              </Typography>
              <Typography variant="h4">{stats.total}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={2.4}>
          <Card sx={{ bgcolor: 'error.dark' }}>
            <CardContent>
              <Typography color="white" gutterBottom>
                Critical
              </Typography>
              <Typography variant="h4" color="white">
                {stats.critical}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={2.4}>
          <Card sx={{ bgcolor: 'error.light' }}>
            <CardContent>
              <Typography color="white" gutterBottom>
                High
              </Typography>
              <Typography variant="h4" color="white">
                {stats.high}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={2.4}>
          <Card sx={{ bgcolor: 'warning.main' }}>
            <CardContent>
              <Typography color="white" gutterBottom>
                Medium
              </Typography>
              <Typography variant="h4" color="white">
                {stats.medium}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={2.4}>
          <Card sx={{ bgcolor: 'info.main' }}>
            <CardContent>
              <Typography color="white" gutterBottom>
                Low
              </Typography>
              <Typography variant="h4" color="white">
                {stats.low}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ mb: 3 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>Severity</InputLabel>
              <Select
                value={filters.severity}
                label="Severity"
                onChange={(e) => handleFilterChange(e, 'severity')}
              >
                <MenuItem value="all">All Severities</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="low">Low</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>Category</InputLabel>
              <Select
                value={filters.category}
                label="Category"
                onChange={(e) => handleFilterChange(e, 'category')}
              >
                <MenuItem value="all">All Categories</MenuItem>
                <MenuItem value="authentication">Authentication</MenuItem>
                <MenuItem value="authorization">Authorization</MenuItem>
                <MenuItem value="system">System</MenuItem>
                <MenuItem value="data">Data</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Severity</TableCell>
              <TableCell>Timestamp</TableCell>
              <TableCell>Category</TableCell>
              <TableCell>Description</TableCell>
              <TableCell>Source</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {events.map((event) => (
              <TableRow key={event.id}>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {severityIcons[event.severity]}
                    <Chip
                      label={event.severity.toUpperCase()}
                      color={severityColors[event.severity] as any}
                      size="small"
                    />
                  </Box>
                </TableCell>
                <TableCell>
                  {new Date(event.timestamp).toLocaleString()}
                </TableCell>
                <TableCell>{event.category}</TableCell>
                <TableCell>{event.description}</TableCell>
                <TableCell>{event.source}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default SecurityMonitor;
