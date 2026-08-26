import React from 'react';
import {
  Checkbox as MuiCheckbox,
  CheckboxProps as MuiCheckboxProps,
  FormControlLabel,
  FormControl,
  FormHelperText,
  FormGroup,
  Box,
  Typography,
} from '@mui/material';
import { FieldError } from 'react-hook-form';

interface CheckboxProps extends Omit<MuiCheckboxProps, 'error'> {
  label?: string;
  error?: FieldError;
  helperText?: string;
  formControl?: boolean;
}

interface CheckboxGroupProps {
  legend?: string;
  options: Array<{ label: string; value: string | number }>;
  selected: Array<string | number>;
  onChange: (selected: Array<string | number>) => void;
  error?: FieldError;
  helperText?: string;
  disabled?: boolean;
}

/**
 * Checkbox Component
 *
 * Single checkbox wrapper with:
 * - Label support
 * - React Hook Form integration
 * - Error display
 */
const Checkbox: React.FC<CheckboxProps> = ({
  label,
  error,
  helperText,
  formControl = false,
  ...rest
}) => {
  const hasError = !!error;
  const errorMessage = error?.message || helperText || '';

  if (formControl) {
    return (
      <FormControl error={hasError} fullWidth>
        <FormControlLabel
          control={<MuiCheckbox {...rest} />}
          label={label}
          slotProps={{
            typography: {
              variant: 'body2',
            },
          }}
        />
        {errorMessage && <FormHelperText>{errorMessage}</FormHelperText>}
      </FormControl>
    );
  }

  return (
    <Box>
      {label ? (
        <FormControlLabel control={<MuiCheckbox {...rest} />} label={label} />
      ) : (
        <MuiCheckbox {...rest} />
      )}
      {errorMessage && <FormHelperText error={hasError}>{errorMessage}</FormHelperText>}
    </Box>
  );
};

/**
 * CheckboxGroup Component
 *
 * Multiple checkboxes wrapper with:
 * - Grouped checkbox management
 * - Legend support
 * - React Hook Form integration
 * - Error display
 */
const CheckboxGroup: React.FC<CheckboxGroupProps> = ({
  legend,
  options,
  selected,
  onChange,
  error,
  helperText,
  disabled = false,
}) => {
  const hasError = !!error;
  const errorMessage = error?.message || helperText || '';

  const handleChange = (value: string | number) => {
    const newSelected = selected.includes(value)
      ? selected.filter((item) => item !== value)
      : [...selected, value];

    onChange(newSelected);
  };

  return (
    <FormControl error={hasError} fullWidth component="fieldset">
      {legend && (
        <Typography component="legend" variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
          {legend}
        </Typography>
      )}

      <FormGroup>
        {options.map((option) => (
          <FormControlLabel
            key={`checkbox-${option.value}`}
            control={
              <MuiCheckbox
                checked={selected.includes(option.value)}
                onChange={() => handleChange(option.value)}
                disabled={disabled}
              />
            }
            label={option.label}
          />
        ))}
      </FormGroup>

      {errorMessage && <FormHelperText>{errorMessage}</FormHelperText>}
    </FormControl>
  );
};

export { Checkbox, CheckboxGroup };
export default Checkbox;
