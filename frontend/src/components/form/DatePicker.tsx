import React from 'react';
import {
  TextField,
  TextFieldProps,
  FormControl,
  FormHelperText,
  Box,
  InputLabel,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import { FieldError } from 'react-hook-form';

interface DatePickerProps extends Omit<TextFieldProps, 'error' | 'type'> {
  error?: FieldError;
  helperText?: string;
  minDate?: string;
  maxDate?: string;
  disableWeekends?: boolean;
  disabledDates?: string[];
  label?: string;
}

interface DateRangePickerProps {
  startDate: string;
  endDate: string;
  onStartDateChange: (date: string) => void;
  onEndDateChange: (date: string) => void;
  error?: FieldError;
  helperText?: string;
  minDate?: string;
  maxDate?: string;
  disabled?: boolean;
  label?: string;
}

/**
 * DatePicker Component
 *
 * Date input wrapper with:
 * - React Hook Form integration
 * - Date range constraints
 * - Weekend/custom date disabling
 * - Mobile-responsive
 */
const DatePicker: React.FC<DatePickerProps> = ({
  error,
  helperText,
  minDate,
  maxDate,
  disableWeekends = false,
  disabledDates = [],
  label = 'Select Date',
  value,
  onChange,
  ...rest
}) => {
  const hasError = !!error;
  const errorMessage = error?.message || helperText || '';

  const isDisabledDate = (dateString: string): boolean => {
    if (disabledDates.includes(dateString)) return true;

    if (disableWeekends) {
      const date = new Date(dateString);
      const dayOfWeek = date.getDay();
      if (dayOfWeek === 0 || dayOfWeek === 6) return true;
    }

    return false;
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedDate = event.target.value;

    // Validate date constraints
    if (minDate && selectedDate < minDate) return;
    if (maxDate && selectedDate > maxDate) return;
    if (isDisabledDate(selectedDate)) return;

    if (onChange) {
      onChange(event);
    }
  };

  return (
    <Box>
      <TextField
        {...rest}
        label={label}
        type="date"
        value={value}
        onChange={handleChange}
        error={hasError}
        disabled={rest.disabled}
        inputProps={{
          min: minDate,
          max: maxDate,
        }}
        InputLabelProps={{
          shrink: true,
        }}
        fullWidth
      />
      {errorMessage && <FormHelperText error={hasError}>{errorMessage}</FormHelperText>}
    </Box>
  );
};

/**
 * DateRangePicker Component
 *
 * Date range input with:
 * - Start and end date inputs
 * - Date constraint validation
 * - React Hook Form integration
 * - End date must be after start date
 */
const DateRangePicker: React.FC<DateRangePickerProps> = ({
  startDate,
  endDate,
  onStartDateChange,
  onEndDateChange,
  error,
  helperText,
  minDate,
  maxDate,
  disabled = false,
  label = 'Select Date Range',
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const hasError = !!error;
  const errorMessage = error?.message || helperText || '';

  const handleStartDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newStartDate = event.target.value;

    // Ensure start date is not after end date
    if (endDate && newStartDate > endDate) return;

    onStartDateChange(newStartDate);
  };

  const handleEndDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newEndDate = event.target.value;

    // Ensure end date is not before start date
    if (startDate && newEndDate < startDate) return;

    onEndDateChange(newEndDate);
  };

  return (
    <FormControl fullWidth error={hasError} disabled={disabled}>
      <InputLabel shrink>{label}</InputLabel>

      <Box
        sx={{
          display: 'flex',
          gap: 2,
          flexDirection: isMobile ? 'column' : 'row',
          mb: errorMessage ? 0 : 1,
        }}
      >
        <TextField
          label="From"
          type="date"
          value={startDate}
          onChange={handleStartDateChange}
          error={hasError}
          disabled={disabled}
          inputProps={{
            min: minDate,
            max: endDate || maxDate,
          }}
          InputLabelProps={{
            shrink: true,
          }}
          sx={{ flex: 1 }}
        />

        <TextField
          label="To"
          type="date"
          value={endDate}
          onChange={handleEndDateChange}
          error={hasError}
          disabled={disabled}
          inputProps={{
            min: startDate || minDate,
            max: maxDate,
          }}
          InputLabelProps={{
            shrink: true,
          }}
          sx={{ flex: 1 }}
        />
      </Box>

      {errorMessage && <FormHelperText>{errorMessage}</FormHelperText>}
    </FormControl>
  );
};

export { DatePicker, DateRangePicker };
export default DatePicker;
