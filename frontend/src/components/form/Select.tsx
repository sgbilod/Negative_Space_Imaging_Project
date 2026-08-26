import React from 'react';
import {
  Select as MuiSelect,
  SelectProps as MuiSelectProps,
  MenuItem,
  FormControl,
  InputLabel,
  FormHelperText,
  CircularProgress,
  Box,
} from '@mui/material';
import { FieldError } from 'react-hook-form';

export interface SelectOption {
  label: string;
  value: string | number;
  disabled?: boolean;
  group?: string;
}

interface SelectProps extends Omit<MuiSelectProps, 'error'> {
  label?: string;
  error?: FieldError;
  options?: SelectOption[];
  isLoading?: boolean;
  isValidating?: boolean;
  emptyLabel?: string;
  groupedOptions?: Record<string, SelectOption[]>;
  helperText?: string;
}

/**
 * Select Component
 *
 * Wrapper around MUI Select with:
 * - React Hook Form integration
 * - Option grouping support
 * - Loading state
 * - Custom validation display
 * - Enhanced accessibility
 */
const Select: React.FC<SelectProps> = ({
  label,
  error,
  options = [],
  isLoading = false,
  isValidating = false,
  emptyLabel = 'Select an option',
  groupedOptions,
  helperText,
  value,
  onChange,
  ...rest
}) => {
  const hasError = !!error;
  const errorMessage = error?.message || '';
  const id = rest.id || `select-${label}`;

  return (
    <FormControl fullWidth error={hasError || isValidating} disabled={isLoading || rest.disabled}>
      {label && <InputLabel id={`${id}-label`}>{label}</InputLabel>}

      <MuiSelect
        {...rest}
        labelId={`${id}-label`}
        id={id}
        value={value}
        onChange={onChange}
        label={label}
        endAdornment={isValidating ? <CircularProgress color="inherit" size={20} /> : undefined}
      >
        {/* Empty option */}
        {!rest.multiple && <MenuItem value="">{emptyLabel}</MenuItem>}

        {/* Grouped options */}
        {groupedOptions &&
          Object.entries(groupedOptions).map(([group, items]) => [
            <MenuItem key={`group-${group}`} disabled divider>
              <strong>{group}</strong>
            </MenuItem>,
            ...items.map((option) => (
              <MenuItem
                key={`${group}-${option.value}`}
                value={option.value}
                disabled={option.disabled}
              >
                {option.label}
              </MenuItem>
            )),
          ])}

        {/* Flat options */}
        {!groupedOptions &&
          options.map((option) => (
            <MenuItem
              key={`option-${option.value}`}
              value={option.value}
              disabled={option.disabled || isLoading}
            >
              {option.label}
            </MenuItem>
          ))}
      </MuiSelect>

      {(errorMessage || helperText) && (
        <FormHelperText>{errorMessage || helperText}</FormHelperText>
      )}
    </FormControl>
  );
};

export default Select;
