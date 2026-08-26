import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { createTheme } from '@mui/material/styles';
import Login from '../pages/Login';
import Dashboard from '../pages/Dashboard';
import ImageProcessing from '../pages/ImageProcessing';
import SecurityMonitor from '../pages/SecurityMonitor';
import { AuthProvider } from '../contexts/AuthContext';

const theme = createTheme();

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <BrowserRouter>
    <ThemeProvider theme={theme}>
      <AuthProvider>{children}</AuthProvider>
    </ThemeProvider>
  </BrowserRouter>
);

describe('Login Page', () => {
  test('renders login form', () => {
    render(<Login />, { wrapper });
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  test('handles login submission', async () => {
    render(<Login />, { wrapper });

    fireEvent.change(screen.getByLabelText(/username/i), {
      target: { value: 'testuser' },
    });
    fireEvent.change(screen.getByLabelText(/password/i), {
      target: { value: 'testpass' },
    });

    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(window.location.pathname).toBe('/');
    });
  });
});

describe('Dashboard', () => {
  test('renders dashboard components', async () => {
    render(<Dashboard />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText(/active jobs/i)).toBeInTheDocument();
      expect(screen.getByText(/system load/i)).toBeInTheDocument();
    });
  });

  test('displays metrics', async () => {
    render(<Dashboard />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText(/system load history/i)).toBeInTheDocument();
    });
  });
});

describe('Image Processing', () => {
  test('renders image upload form', () => {
    render(<ImageProcessing />, { wrapper });

    expect(screen.getByText(/select image/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/contrast/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/brightness/i)).toBeInTheDocument();
  });

  test('handles image processing', async () => {
    render(<ImageProcessing />, { wrapper });

    const file = new File(['dummy image'], 'test.jpg', { type: 'image/jpeg' });
    const input = screen.getByLabelText(/select image/i);

    Object.defineProperty(input, 'files', {
      value: [file],
    });

    fireEvent.change(input);

    fireEvent.click(screen.getByRole('button', { name: /process image/i }));

    await waitFor(() => {
      expect(screen.getByText(/processing status/i)).toBeInTheDocument();
    });
  });
});

describe('Security Monitor', () => {
  test('renders security events table', async () => {
    render(<SecurityMonitor />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText(/security monitor/i)).toBeInTheDocument();
      expect(screen.getByText(/severity/i)).toBeInTheDocument();
    });
  });

  test('filters security events', async () => {
    render(<SecurityMonitor />, { wrapper });

    const severitySelect = screen.getByLabelText(/severity/i);
    fireEvent.mouseDown(severitySelect);

    const criticalOption = screen.getByText(/critical/i);
    fireEvent.click(criticalOption);

    await waitFor(() => {
      expect(screen.getByText(/total events/i)).toBeInTheDocument();
    });
  });
});
