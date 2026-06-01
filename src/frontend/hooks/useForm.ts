import { useState, useCallback } from 'react';
import { ZodSchema, ZodError } from 'zod';

export interface FormState<T> {
  values: T;
  errors: Partial<Record<keyof T, string>>;
  touched: Partial<Record<keyof T, boolean>>;
  isSubmitting: boolean;
}

export interface UseFormReturn<T> {
  state: FormState<T>;
  handleChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => void;
  handleSubmit: (onSubmit: (values: T) => Promise<void> | void) => (e: React.FormEvent) => Promise<void>;
  reset: () => void;
  setFieldValue: (field: keyof T, value: unknown) => void;
  setFieldError: (field: keyof T, error: string) => void;
  setFieldTouched: (field: keyof T, touched: boolean) => void;
}

export const useForm = <T extends Record<string, unknown>>(
  initialValues: T,
  validationSchema?: ZodSchema
): UseFormReturn<T> => {
  const [state, setState] = useState<FormState<T>>({
    values: initialValues,
    errors: {},
    touched: {},
    isSubmitting: false,
  });

  const validateForm = useCallback(
    (values: T): Partial<Record<keyof T, string>> => {
      if (!validationSchema) return {};

      try {
        validationSchema.parse(values);
        return {};
      } catch (err) {
        if (err instanceof ZodError) {
          const errors: Partial<Record<keyof T, string>> = {};
          err.errors.forEach((error) => {
            const path = error.path[0] as keyof T;
            errors[path] = error.message;
          });
          return errors;
        }
        return {};
      }
    },
    [validationSchema]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
      const { name, value, type } = e.target;
      const inputValue = type === 'checkbox' ? (e.target as HTMLInputElement).checked : value;

      setState((prev) => ({
        ...prev,
        values: {
          ...prev.values,
          [name]: inputValue,
        },
        touched: {
          ...prev.touched,
          [name]: true,
        },
      }));
    },
    []
  );

  const handleSubmit = useCallback(
    (onSubmit: (values: T) => Promise<void> | void) =>
      async (e: React.FormEvent) => {
        e.preventDefault();
        const errors = validateForm(state.values);

        setState((prev) => ({
          ...prev,
          errors,
          touched: Object.keys(prev.values).reduce(
            (acc, key) => {
              acc[key as keyof T] = true;
              return acc;
            },
            {} as Record<keyof T, boolean>
          ),
        }));

        if (Object.keys(errors).length === 0) {
          setState((prev) => ({ ...prev, isSubmitting: true }));
          try {
            await onSubmit(state.values);
          } catch (err) {
            console.error('Form submission error:', err);
          } finally {
            setState((prev) => ({ ...prev, isSubmitting: false }));
          }
        }
      },
    [state.values, validateForm]
  );

  const reset = useCallback(() => {
    setState({
      values: initialValues,
      errors: {},
      touched: {},
      isSubmitting: false,
    });
  }, [initialValues]);

  const setFieldValue = useCallback((field: keyof T, value: unknown) => {
    setState((prev) => ({
      ...prev,
      values: {
        ...prev.values,
        [field]: value,
      },
    }));
  }, []);

  const setFieldError = useCallback((field: keyof T, error: string) => {
    setState((prev) => ({
      ...prev,
      errors: {
        ...prev.errors,
        [field]: error,
      },
    }));
  }, []);

  const setFieldTouched = useCallback((field: keyof T, touched: boolean) => {
    setState((prev) => ({
      ...prev,
      touched: {
        ...prev.touched,
        [field]: touched,
      },
    }));
  }, []);

  return {
    state,
    handleChange,
    handleSubmit,
    reset,
    setFieldValue,
    setFieldError,
    setFieldTouched,
  };
};
