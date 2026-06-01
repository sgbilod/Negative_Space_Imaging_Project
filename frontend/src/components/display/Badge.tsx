import React from 'react';
import {
  Chip,
  ChipProps as MuiChipProps,
  Box,
  Avatar,
  Typography,
  useTheme,
  alpha,
} from '@mui/material';
import ErrorIcon from '@mui/icons-material/Error';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import InfoIcon from '@mui/icons-material/Info';
import WarningIcon from '@mui/icons-material/Warning';

export type BadgeVariant = 'default' | 'success' | 'error' | 'warning' | 'info';

interface BadgeProps extends Omit<MuiChipProps, 'color' | 'variant' | 'icon'> {
  variant?: BadgeVariant;
  icon?: React.ReactElement | null;
  count?: number;
  max?: number;
  pulse?: boolean;
}

interface StatusBadgeProps extends BadgeProps {
  status: 'active' | 'inactive' | 'pending' | 'error' | 'warning';
  label?: string;
}

/**
 * Badge Component
 *
 * Styled badge/chip component with:
 * - Semantic variants (success, error, warning, info)
 * - Icon support
 * - Count display with max
 * - Animated pulse effect
 */
const Badge: React.FC<BadgeProps> = ({
  variant = 'default',
  icon,
  count,
  max = 99,
  pulse = false,
  children,
  ...rest
}) => {
  const theme = useTheme();

  const variantStyles: Record<BadgeVariant, { bg: string; text: string; border: string }> = {
    default: {
      bg: theme.palette.action.hover,
      text: theme.palette.text.primary,
      border: theme.palette.divider,
    },
    success: {
      bg: alpha(theme.palette.success.main, 0.1),
      text: theme.palette.success.main,
      border: theme.palette.success.light,
    },
    error: {
      bg: alpha(theme.palette.error.main, 0.1),
      text: theme.palette.error.main,
      border: theme.palette.error.light,
    },
    warning: {
      bg: alpha(theme.palette.warning.main, 0.1),
      text: theme.palette.warning.main,
      border: theme.palette.warning.light,
    },
    info: {
      bg: alpha(theme.palette.info.main, 0.1),
      text: theme.palette.info.main,
      border: theme.palette.info.light,
    },
  };

  const styles = variantStyles[variant];

  const displayCount = count !== undefined && count > max ? `${max}+` : count;

  const badgeContent = count !== undefined ? `${displayCount}` : children;

  return (
    <Chip
      {...rest}
      icon={
        icon ? (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              '& svg': {
                width: 18,
                height: 18,
              },
            }}
          >
            {icon}
          </Box>
        ) : undefined
      }
      label={badgeContent}
      sx={{
        backgroundColor: styles.bg,
        color: styles.text,
        border: `1px solid ${styles.border}`,
        fontWeight: 500,
        fontSize: '0.85rem',
        animation: pulse ? `pulse 2s ease-in-out infinite` : undefined,
        '@keyframes pulse': {
          '0%, 100%': {
            opacity: 1,
            boxShadow: `0 0 0 0 ${alpha(theme.palette.primary.main, 0.7)}`,
          },
          '50%': {
            opacity: 0.8,
            boxShadow: `0 0 0 8px ${alpha(theme.palette.primary.main, 0)}`,
          },
        },
      }}
    />
  );
};

/**
 * StatusBadge Component
 *
 * Status-specific badge with:
 * - Predefined status variants
 * - Automatic icon selection
 * - Semantic color coding
 */
const StatusBadge: React.FC<StatusBadgeProps> = ({ status, label, ...rest }) => {
  const statusConfig: Record<
    string,
    { variant: BadgeVariant; icon: React.ReactElement | null; defaultLabel: string }
  > = {
    active: {
      variant: 'success',
      icon: <CheckCircleIcon />,
      defaultLabel: 'Active',
    },
    inactive: {
      variant: 'default',
      icon: <InfoIcon />,
      defaultLabel: 'Inactive',
    },
    pending: {
      variant: 'warning',
      icon: <WarningIcon />,
      defaultLabel: 'Pending',
    },
    error: {
      variant: 'error',
      icon: <ErrorIcon />,
      defaultLabel: 'Error',
    },
    warning: {
      variant: 'warning',
      icon: <WarningIcon />,
      defaultLabel: 'Warning',
    },
  };

  const config = statusConfig[status];

  return (
    <Badge
      variant={config.variant}
      icon={config.icon}
      label={label || config.defaultLabel}
      {...rest}
    />
  );
};

/**
 * CountBadge Component
 *
 * Badge specifically for displaying counts with:
 * - Max value capping (shows 99+)
 * - Consistent sizing
 */
interface CountBadgeProps extends Omit<BadgeProps, 'children'> {
  count: number;
}

const CountBadge: React.FC<CountBadgeProps> = ({ count, max = 99, ...rest }) => {
  const theme = useTheme();
  const displayCount = count > max ? `${max}+` : count;

  return (
    <Box
      sx={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        minWidth: 24,
        height: 24,
        px: 0.75,
        borderRadius: '12px',
        backgroundColor: theme.palette.error.main,
        color: theme.palette.error.contrastText,
        fontSize: '0.75rem',
        fontWeight: 600,
      }}
    >
      <Typography variant="caption" sx={{ fontWeight: 600 }}>
        {displayCount}
      </Typography>
    </Box>
  );
};

export { Badge, StatusBadge, CountBadge };
export default Badge;
