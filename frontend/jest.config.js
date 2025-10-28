module.exports = {
  // Use the jsdom environment for testing React components
  testEnvironment: 'jsdom',

  // Look for test files in these patterns
  testMatch: [
    '<rootDir>/src/**/__tests__/**/*.{js,jsx,ts,tsx}',
    '<rootDir>/src/**/*.{spec,test}.{js,jsx,ts,tsx}',
  ],

  // Files to ignore when testing
  testPathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/public/',
  ],

  // Transform files before testing
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': 'babel-jest',
  },

  // Handle non-JavaScript files
  moduleNameMapper: {
    '\\.(css|less|scss|sass)$': '<rootDir>/src/test-utils/styleMock.js',
    '\\.(jpg|jpeg|png|gif|webp|svg)$': '<rootDir>/src/test-utils/fileMock.js',
  },

  // Setup file to run before tests
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.js'],

  // Default coverage collection
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/index.{js,jsx,ts,tsx}',
    '!src/serviceWorker.{js,jsx,ts,tsx}',
    '!src/setupTests.{js,jsx,ts,tsx}',
  ],

  // Minimum coverage threshold
  coverageThreshold: {
    global: {
      statements: 10,
      branches: 10,
      functions: 10,
      lines: 10,
    },
  },
};
