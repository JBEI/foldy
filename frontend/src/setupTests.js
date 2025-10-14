// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// Add TextEncoder/TextDecoder for MSW
global.TextEncoder = require('util').TextEncoder;
global.TextDecoder = require('util').TextDecoder;

// Mock environment variables
window.ENV = {
  VITE_BACKEND_URL: 'http://localhost:5000'
};

// Mock vite's import.meta.env
process.env.VITE_BACKEND_URL = 'http://localhost:5000';

// Proper mock for axiosInstance
jest.mock('./services/axiosInstance.tsx', () => {
  return {
    __esModule: true,
    default: {
      get: jest.fn(() => Promise.resolve({ data: {} })),
      post: jest.fn(() => Promise.resolve({ data: {} })),
      put: jest.fn(() => Promise.resolve({ data: {} })),
      delete: jest.fn(() => Promise.resolve({ data: {} })),
      interceptors: {
        request: { use: jest.fn(), eject: jest.fn() },
        response: { use: jest.fn(), eject: jest.fn() }
      }
    },
  };
});

// Mock for axios
jest.mock('axios', () => ({
  create: jest.fn(() => ({
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
    interceptors: {
      request: { use: jest.fn(), eject: jest.fn() },
      response: { use: jest.fn(), eject: jest.fn() }
    }
  })),
  get: jest.fn(),
  post: jest.fn(),
  put: jest.fn(),
  delete: jest.fn(),
}));

// Global fetch mock setup
global.fetch = jest.fn();

// Mock matchMedia
window.matchMedia = window.matchMedia || function() {
    return {
        matches: false,
        addListener: function() {},
        removeListener: function() {},
        addEventListener: function() {},
        removeEventListener: function() {},
        dispatchEvent: function() {},
    };
};

// Mock ResizeObserver
class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
}

window.ResizeObserver = window.ResizeObserver || ResizeObserverMock;

// Mock getComputedStyle
window.getComputedStyle = window.getComputedStyle || function() {
    return {
        getPropertyValue: function() {}
    };
};

// Suppress specific warnings
const originalError = console.error;
const originalWarn = console.warn;

beforeAll(() => {
  console.error = (...args) => {
    if (
        /Warning.*not wrapped in act/.test(args[0]) ||
        /Warning.*Received .* for a non-boolean attribute/.test(args[0]) ||
        (typeof args[0] === 'string' &&
         (args[0].includes('Warning: ReactDOM.render') ||
          args[0].includes('Warning: React.createElement')))
    ) {
        return;
    }
    originalError.call(console, ...args);
  };

  console.warn = (...args) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('Warning: validateDOMNesting')
    ) {
      return;
    }
    originalWarn.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
  console.warn = originalWarn;
});
