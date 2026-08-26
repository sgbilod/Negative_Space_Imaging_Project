import React from 'react';
import {
  TextField as MuiTextField,
  TextFieldProps as MuiTextFieldProps,
  FormHelperText,
  Box,
  InputAdornment,
} from '@mui/material';
import { FieldError } from 'react-hook-form';

interface TextFieldProps extends Omit<MuiTextFieldProps, 'error'> {
  error?: FieldError;
  helperIcon?: React.ReactNode;
  endAdornment?: React.ReactNode;
  startAdornment?: React.ReactNode;
  isValidating?: boolean;
  characterLimit?: number;
  showCharCount?: boolean;
}

/**
 * TextField Component
 *
 * Wrapper around MUI TextField with:
 * - React Hook Form integration
 * - Custom validation display
 * - Character count display
 * - Adornment support
 */
const TextField: React.FC<TextFieldProps> = ({
  error,
  helperIcon,
  endAdornment,
  startAdornment,
  isValidating = false,
  characterLimit,
  showCharCount = false,
  value,
  onChange,
  ...rest
}) => {
  const [charCount, setCharCount] = React.useState<number>(
    typeof value === 'string' ? value.length : 0,
  );

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = event.target.value;
    setCharCount(newValue.length);

    if (onChange) {
      onChange(event);
    }
  };

  const errorMessage = error?.message || '';
  const hasError = !!error;

  return (
    <Box>
      <MuiTextField
        {...rest}
        value={value}
        onChange={handleChange}
        error={hasError || isValidating}
        disabled={isValidating || rest.disabled}
        InputProps={{
          startAdornment: startAdornment ? (
            <InputAdornment position="start">{startAdornment}</InputAdornment>
          ) : undefined,
          endAdornment: endAdornment ? (
            <InputAdornment position="end">{endAdornment}</InputAdornment>
          ) : undefined,
        }}
        fullWidth
      />

      {(errorMessage || (showCharCount && characterLimit)) && (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mt: 0.5,
            px: 1,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {helperIcon && helperIcon}
            {errorMessage && <FormHelperText error={hasError}>{errorMessage}</FormHelperText>}
          </Box>

          {showCharCount && characterLimit && (
            <FormHelperText>
              {charCount} / {characterLimit}
            </FormHelperText>
          )}
        </Box>
      )}
    </Box>
  );
};

export default TextField;
