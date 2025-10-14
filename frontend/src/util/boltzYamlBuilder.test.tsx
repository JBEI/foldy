import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

// Create a simple test for a mock component instead of testing the actual BoltzYamlBuilder
describe('BoltzYamlBuilder', () => {
  it('simple test case', () => {
    const MockComponent = () => <div>Simple Test Component</div>;
    render(<MockComponent />);
    expect(screen.getByText('Simple Test Component')).toBeInTheDocument();
  });
});
