import React from 'react';
import {
  Card as MuiCard,
  CardProps as MuiCardProps,
  CardHeader,
  CardContent,
  CardActions,
  CardMedia,
  Typography,
  Box,
  Divider,
  useTheme,
  alpha,
} from '@mui/material';

interface CardProps extends MuiCardProps {
  title?: string;
  subtitle?: string;
  image?: string;
  imageHeight?: number;
  actions?: React.ReactNode;
  footer?: React.ReactNode;
  dense?: boolean;
  elevation?: number;
  interactive?: boolean;
  onClick?: () => void;
}

/**
 * Card Component
 *
 * Enhanced MUI Card wrapper with:
 * - Standardized header/content/footer structure
 * - Image support
 * - Action buttons
 * - Interactive hover effects
 * - Consistent styling
 */
const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  image,
  imageHeight = 200,
  actions,
  footer,
  dense = false,
  children,
  interactive = false,
  onClick,
  elevation,
  ...rest
}) => {
  const theme = useTheme();
  const [hovered, setHovered] = React.useState(false);

  return (
    <MuiCard
      {...rest}
      elevation={hovered && interactive ? (elevation || 4) + 4 : elevation || 1}
      onMouseEnter={() => interactive && setHovered(true)}
      onMouseLeave={() => interactive && setHovered(false)}
      onClick={onClick}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        transition: theme.transitions.create(['box-shadow', 'transform'], {
          easing: theme.transitions.easing.easeInOut,
          duration: theme.transitions.duration.shorter,
        }),
        cursor: interactive ? 'pointer' : 'default',
        transform: hovered && interactive ? 'translateY(-4px)' : 'translateY(0)',
        backgroundColor: theme.palette.background.paper,
        ...rest.sx,
      }}
    >
      {/* Image Section */}
      {image && (
        <CardMedia
          component="img"
          height={imageHeight}
          image={image}
          alt={title || 'Card image'}
          sx={{
            objectFit: 'cover',
          }}
        />
      )}

      {/* Header Section */}
      {(title || subtitle) && (
        <CardHeader
          title={title}
          subheader={subtitle}
          titleTypographyProps={{
            variant: dense ? 'subtitle1' : 'h6',
            fontWeight: 600,
          }}
          subheaderTypographyProps={{
            variant: 'caption',
            color: 'text.secondary',
          }}
          sx={{
            pb: dense ? 1 : 2,
            pt: dense ? 1 : 2,
          }}
        />
      )}

      {/* Content Section */}
      <CardContent
        sx={{
          flex: 1,
          pb: dense ? 1 : 2,
          pt: (title || subtitle) && !dense ? 0 : 2,
          '&:last-child': {
            pb: dense ? 1 : 2,
          },
        }}
      >
        {typeof children === 'string' ? (
          <Typography variant="body2" color="text.secondary">
            {children}
          </Typography>
        ) : (
          children
        )}
      </CardContent>

      {/* Actions Section */}
      {actions && (
        <>
          <Divider />
          <CardActions
            sx={{
              justifyContent: 'flex-end',
              pt: dense ? 1 : 2,
              pb: dense ? 1 : 2,
            }}
          >
            {actions}
          </CardActions>
        </>
      )}

      {/* Footer Section */}
      {footer && (
        <>
          <Divider />
          <Box
            sx={{
              px: 2,
              py: dense ? 1 : 1.5,
              backgroundColor: alpha(theme.palette.primary.main, 0.02),
              fontSize: '0.75rem',
              color: 'text.secondary',
            }}
          >
            {footer}
          </Box>
        </>
      )}
    </MuiCard>
  );
};

export default Card;
