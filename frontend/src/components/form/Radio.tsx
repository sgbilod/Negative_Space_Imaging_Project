import React from 'react';
import {
  Radio as MuiRadio,
  RadioProps as MuiRadioProps,
  RadioGroup as MuiRadioGroup,
  FormControlLabel,
  FormControl,
  FormHelperText,
  Typography,
  Box,
  Stack,
} from '@mui/material';
import { FieldError } from 'react-hook-form';

interface RadioProps extends Omit<MuiRadioProps, 'error'> {
  label?: string;
  error?: FieldError;
  helperText?: string;
}

interface RadioGroupProps {
  legend?: string;
  options: Array<{ label: string; value: string | number; description?: string }>;
  value: string | number;
  onChange: (value: string | number) => void;
  error?: FieldError;
  helperText?: string;
  disabled?: boolean;
  row?: boolean;
}

/**
 * Radio Component
 *
 * Single radio button wrapper with:
 * - Label support
 * - React Hook Form integration
 * - Error display
 */
const Radio: React.FC<RadioProps> = ({ label, error, helperText, ...rest }) => {
  const hasError = !!error;
  const errorMessage = error?.message || helperText || '';

  return (
    <Box>
      {label ? (
        <FormControlLabel control={<MuiRadio {...rest} />} label={label} />
      ) : (
        <MuiRadio {...rest} />
      )}
      {errorMessage && <FormHelperText error={hasError}>{errorMessage}</FormHelperText>}
    </Box>
  );
};

/**
 * RadioGroup Component
 *
 * Multiple radio buttons wrapper with:
 * - Grouped radio management
 * - Legend support
 * - Description support for each option
 * - React Hook Form integration
 * - Horizontal/vertical layout
 */
const RadioGroup: React.FC<RadioGroupProps> = ({
  legend,
  options,
  value,
  onChange,
  error,
  helperText,
  disabled = false,
  row = false,
}) => {
  const hasError = !!error;
  const errorMessage = error?.message || helperText || '';

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onChange(event.target.value);
  };

  return (
    <FormControl error={hasError} fullWidth component="fieldset" disabled={disabled}>
      {legend && (
        <Typography component="legend" variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
          {legend}
        </Typography>
      )}

      <MuiRadioGroup
        row={row}
        value={value}
        onChange={handleChange}
        sx={{
          display: 'flex',
          flexDirection: row ? 'row' : 'column',
          gap: row ? 2 : 0,
        }}
      >
        {options.map((option) => (
          <Stack key={`radio-${option.value}`} direction="row" alignItems="flex-start" spacing={1}>
            <FormControlLabel
              control={<MuiRadio />}
              value={option.value}
              label={option.label}
              sx={{ m: 0 }}
            />
            {option.description && (
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{
                  mt: 0.5,
                  ml: option.label ? 4 : 0,
                }}
              >
                {option.description}
              </Typography>
            )}
          </Stack>
        ))}
      </MuiRadioGroup>

      {errorMessage && <FormHelperText>{errorMessage}</FormHelperText>}
    </FormControl>
  );
};

export { Radio, RadioGroup };
export default Radio;
