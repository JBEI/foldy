import React, { ReactElement } from 'react';
import { render, RenderOptions, RenderResult } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
// Instead of importing msw directly, we'll use a different approach
// for axios mocking that's more compatible with the current setup
const mockServer = {
  use: jest.fn(),
  listen: jest.fn(),
  close: jest.fn(),
  resetHandlers: jest.fn(),
};

// Mock API handlers are now implemented through jest.mock() in each test file

// Custom render function that includes providers
const AllProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <BrowserRouter>
      {children}
    </BrowserRouter>
  );
};

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  route?: string;
}

// Custom render function that wraps the UI in necessary providers
function customRender(
  ui: ReactElement,
  options?: CustomRenderOptions
): RenderResult {
  const { route = '/', ...renderOptions } = options || {};

  // Set the URL before rendering
  window.history.pushState({}, 'Test page', route);

  return render(ui, { wrapper: AllProviders, ...renderOptions });
}

// Re-export everything from testing-library
export * from '@testing-library/react';

// Override the render method
export { customRender as render, userEvent };

// Helper for mocking axios responses
export const mockAxiosResponse = (data: any) => {
  return Promise.resolve({
    data,
    status: 200,
    statusText: 'OK',
    headers: {},
    config: {},
  });
};

// Helper for mocking axios error responses
export const mockAxiosError = (status = 500, message = 'Error') => {
  const error = new Error(message) as any;
  error.response = {
    status,
    data: { message },
  };
  return Promise.reject(error);
};
