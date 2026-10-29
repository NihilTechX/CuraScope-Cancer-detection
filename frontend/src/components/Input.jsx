import React, { forwardRef, useState } from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, CheckCircle2 } from 'lucide-react';
import { generateId } from '../utils/accessibility';
import { springs, useReducedMotion } from '../utils/motion';

/**
 * Input Component
 * Text and number inputs with clinical styling and fluid interactions
 */
const Input = forwardRef((
  {
    label,
    type = 'text',
    value,
    onChange,
    placeholder,
    error,
    helperText,
    success,
    required = false,
    disabled = false,
    fullWidth = false,
    id,
    className = '',
    ...props
  },
  ref
) => {
  const [isFocused, setIsFocused] = useState(false);
  const prefersReducedMotion = useReducedMotion();
  
  const inputId = id || generateId('input');
  const errorId = error ? `${inputId}-error` : undefined;
  const helperId = helperText ? `${inputId}-helper` : undefined;

  const hasValue = value !== '' && value !== null && value !== undefined;

  return (
    <div className={fullWidth ? 'w-full' : ''}>
      {label && (
        <label
          htmlFor={inputId}
          className="block font-medium mb-1.5"
          style={{
            fontSize: 'var(--text-xs)',
            color: error ? 'var(--risk-high)' : 'var(--slate-700)',
            fontWeight: 'var(--font-semibold)',
          }}
        >
          {label}
          {required && (
            <span
              className="ml-1"
              style={{ color: 'var(--risk-high)' }}
              aria-label="required"
            >
              *
            </span>
          )}
        </label>
      )}

      <div className="relative">
        <input
          ref={ref}
          id={inputId}
          type={type}
          value={value}
          onChange={onChange}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          disabled={disabled}
          required={required}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={error ? errorId : helperId}
          className={`
            w-full px-3 py-2 rounded-lg
            bg-white
            ${type === 'number' ? 'tabular-nums' : ''}
            ${className}
          `}
          style={{
            fontSize: 'var(--text-sm)',
            color: 'var(--slate-900)',
            border: `1px solid ${
              error
                ? 'var(--risk-high)'
                : isFocused
                ? 'var(--clinical-blue-500)'
                : hasValue
                ? 'var(--slate-400)'
                : 'var(--slate-300)'
            }`,
            outline: 'none',
            transition: `border-color var(--duration-quick) var(--ease-out), box-shadow var(--duration-quick) var(--ease-out)`,
            boxShadow: isFocused
              ? error
                ? '0 0 0 3px rgba(198, 40, 40, 0.1)'
                : '0 0 0 3px rgba(46, 107, 168, 0.1)'
              : 'none',
            opacity: disabled ? 0.5 : 1,
            cursor: disabled ? 'not-allowed' : 'text',
            backgroundColor: disabled ? 'var(--slate-50)' : 'white',
          }}
          {...props}
        />

        {/* Status icons */}
        {(error || success) && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={prefersReducedMotion ? { duration: 0.01 } : springs.quick}
            className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none"
          >
            {error && (
              <AlertCircle
                className="w-4 h-4"
                style={{ color: 'var(--risk-high)' }}
              />
            )}
            {success && !error && (
              <CheckCircle2
                className="w-4 h-4"
                style={{ color: 'var(--risk-low)' }}
              />
            )}
          </motion.div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <motion.p
          id={errorId}
          role="alert"
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={prefersReducedMotion ? { duration: 0.01 } : springs.quick}
          className="mt-1.5 flex items-start gap-1"
          style={{
            fontSize: 'var(--text-xs)',
            color: 'var(--risk-high)',
          }}
        >
          <span>{error}</span>
        </motion.p>
      )}

      {/* Helper text */}
      {helperText && !error && (
        <p
          id={helperId}
          className="mt-1.5"
          style={{
            fontSize: 'var(--text-xs)',
            color: 'var(--slate-500)',
          }}
        >
          {helperText}
        </p>
      )}
    </div>
  );
});

Input.displayName = 'Input';

export default Input;
