/**
 * CuraScope Accessibility Utilities
 * Following better_interface principles
 */

/**
 * Generate unique ID for ARIA relationships
 * @param {string} prefix - ID prefix
 * @returns {string} Unique ID
 */
export function generateId(prefix = 'id') {
  return `${prefix}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Announce message to screen readers
 * @param {string} message - Message to announce
 * @param {string} priority - 'polite' or 'assertive'
 */
export function announceToScreenReader(message, priority = 'polite') {
  const announcement = document.createElement('div');
  announcement.setAttribute('role', 'status');
  announcement.setAttribute('aria-live', priority);
  announcement.setAttribute('aria-atomic', 'true');
  announcement.className = 'sr-only';
  announcement.textContent = message;

  document.body.appendChild(announcement);

  // Remove after announcement
  setTimeout(() => {
    document.body.removeChild(announcement);
  }, 1000);
}

/**
 * Focus trap for modals and dialogs
 * @param {HTMLElement} element - Container element
 * @returns {function} Cleanup function
 */
export function trapFocus(element) {
  const focusableElements = element.querySelectorAll(
    'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
  );

  const firstFocusable = focusableElements[0];
  const lastFocusable = focusableElements[focusableElements.length - 1];

  function handleTabKey(e) {
    if (e.key !== 'Tab') return;

    if (e.shiftKey) {
      // Shift + Tab
      if (document.activeElement === firstFocusable) {
        e.preventDefault();
        lastFocusable.focus();
      }
    } else {
      // Tab
      if (document.activeElement === lastFocusable) {
        e.preventDefault();
        firstFocusable.focus();
      }
    }
  }

  element.addEventListener('keydown', handleTabKey);

  // Focus first element
  firstFocusable?.focus();

  // Return cleanup function
  return () => {
    element.removeEventListener('keydown', handleTabKey);
  };
}

/**
 * Check if element is visible to screen readers
 * @param {HTMLElement} element
 * @returns {boolean}
 */
export function isVisibleToScreenReader(element) {
  return (
    element.offsetWidth > 0 &&
    element.offsetHeight > 0 &&
    window.getComputedStyle(element).visibility !== 'hidden' &&
    !element.hasAttribute('aria-hidden')
  );
}

/**
 * Get accessible label for element
 * @param {HTMLElement} element
 * @returns {string}
 */
export function getAccessibleLabel(element) {
  // Check aria-label
  if (element.hasAttribute('aria-label')) {
    return element.getAttribute('aria-label');
  }

  // Check aria-labelledby
  if (element.hasAttribute('aria-labelledby')) {
    const labelId = element.getAttribute('aria-labelledby');
    const labelElement = document.getElementById(labelId);
    return labelElement?.textContent || '';
  }

  // Check associated label
  if (element.id) {
    const label = document.querySelector(`label[for="${element.id}"]`);
    return label?.textContent || '';
  }

  // Fallback to text content
  return element.textContent || '';
}

/**
 * Keyboard navigation helper
 * @param {KeyboardEvent} event
 * @param {object} handlers - Key handlers { Enter, Escape, ArrowUp, etc. }
 */
export function handleKeyboardNavigation(event, handlers) {
  const handler = handlers[event.key];
  if (handler) {
    event.preventDefault();
    handler(event);
  }
}

/**
 * Check color contrast ratio (WCAG AA compliance)
 * @param {string} foreground - Hex color
 * @param {string} background - Hex color
 * @returns {number} Contrast ratio
 */
export function getContrastRatio(foreground, background) {
  const getLuminance = (hex) => {
    const rgb = parseInt(hex.slice(1), 16);
    const r = ((rgb >> 16) & 0xff) / 255;
    const g = ((rgb >> 8) & 0xff) / 255;
    const b = (rgb & 0xff) / 255;

    const [rs, gs, bs] = [r, g, b].map((c) =>
      c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
    );

    return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
  };

  const l1 = getLuminance(foreground);
  const l2 = getLuminance(background);

  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);

  return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Check if contrast meets WCAG AA standards
 * @param {string} foreground - Hex color
 * @param {string} background - Hex color
 * @param {boolean} isLargeText - Text is 18pt+ or 14pt+ bold
 * @returns {boolean}
 */
export function meetsContrastRequirements(foreground, background, isLargeText = false) {
  const ratio = getContrastRatio(foreground, background);
  const requiredRatio = isLargeText ? 3 : 4.5;
  return ratio >= requiredRatio;
}

/**
 * Screen reader only CSS class helper
 */
export const srOnlyClass = 'absolute w-px h-px p-0 -m-px overflow-hidden whitespace-nowrap border-0';

/**
 * Skip link component helper
 * @param {string} targetId - ID of main content
 * @returns {object} Props for skip link
 */
export function getSkipLinkProps(targetId = 'main-content') {
  return {
    href: `#${targetId}`,
    className: 'sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-clinical-blue-700 focus:text-white focus:rounded-md',
    onClick: (e) => {
      e.preventDefault();
      const target = document.getElementById(targetId);
      if (target) {
        target.focus();
        target.scrollIntoView({ behavior: 'smooth' });
      }
    },
  };
}

/**
 * Touch target size validator
 * @param {HTMLElement} element
 * @returns {boolean} true if meets 44x44px minimum
 */
export function meetsTouchTargetSize(element) {
  const rect = element.getBoundingClientRect();
  return rect.width >= 44 && rect.height >= 44;
}

/**
 * ARIA live region helper
 * @param {string} message - Message to announce
 * @param {string} politeness - 'polite', 'assertive', or 'off'
 */
export function createLiveRegion(message, politeness = 'polite') {
  let liveRegion = document.getElementById('aria-live-region');
  
  if (!liveRegion) {
    liveRegion = document.createElement('div');
    liveRegion.id = 'aria-live-region';
    liveRegion.setAttribute('role', 'status');
    liveRegion.setAttribute('aria-live', politeness);
    liveRegion.setAttribute('aria-atomic', 'true');
    liveRegion.className = srOnlyClass;
    document.body.appendChild(liveRegion);
  }

  liveRegion.textContent = message;
}
