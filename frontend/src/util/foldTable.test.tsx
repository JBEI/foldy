import React from 'react';
import { render, screen } from '@testing-library/react';
import { makeFoldTable } from './foldTable';
import { BrowserRouter } from 'react-router-dom';

// Mock the Link component in case the custom render doesn't handle it
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  Link: ({ to, children }: { to: string; children: React.ReactNode }) => (
    <a href={to} data-testid="mock-link">
      {children}
    </a>
  ),
}));

describe('foldTable', () => {
  it('should create a table element', () => {
    // Create a basic fold object with required properties
    const mockFolds = [{
      id: 1,
      name: 'Test Fold 1',
      user_id: 1,
      sequence: 'ACDEFGHIK',
      created_at: '2023-01-01T12:00:00Z',
      jobs: [{ type: 'models', state: 'finished' }],
      tags: [], // Required for the spread operator
      tagstring: '',
      is_public: false
    }];

    // Just verify the function returns a table element
    const table = makeFoldTable(mockFolds);
    expect(table.type).toBe('div');
  });
});
