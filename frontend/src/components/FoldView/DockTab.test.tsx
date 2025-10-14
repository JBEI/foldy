import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

// Create a simple component test file
describe('Simple Component Tests', () => {
  it('renders a div with text', () => {
    const SimpleComponent = () => <div>Hello Test World</div>;
    render(<SimpleComponent />);
    expect(screen.getByText('Hello Test World')).toBeInTheDocument();
  });
});
