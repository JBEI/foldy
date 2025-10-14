import { useEffect } from 'react';

type KeyHandler = (event: KeyboardEvent) => void;

const isInputElement = (element: EventTarget | null): boolean => {
  if (!element || !(element instanceof HTMLElement)) {
    return false;
  }

  // Check for common input elements
  if (element instanceof HTMLInputElement ||
      element instanceof HTMLTextAreaElement ||
      element instanceof HTMLSelectElement) {
    return true;
  }

  // Check for contenteditable elements
  if (element.hasAttribute('contenteditable')) {
    return true;
  }

  return false;
};

export const useKeyboardIntercept = (key: string, handler: KeyHandler) => {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Don't intercept if target is an input element
      if (isInputElement(event.target)) {
        return;
      }

      if (event.key === key) {
        handler(event);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [key, handler]);
};
