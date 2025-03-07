import { describe, expect, it } from '@jest/globals';
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import BoltzYamlBuilder from './boltzYamlBuilder';
import '@testing-library/jest-dom';

describe('BoltzYamlBuilder', () => {
    const sampleYaml = `
version: 1
sequences:
  - protein:
      id: [A]
      sequence: MVTPE
`;

    it('renders without crashing', () => {
        render(<BoltzYamlBuilder />);
        expect(screen.getByText('YAML Version')).toBeInTheDocument();
        expect(screen.getByText('Boltz YAML Editor')).toBeInTheDocument();
    });

    it('loads initial YAML correctly', () => {
        render(<BoltzYamlBuilder initialYaml={sampleYaml} />);

        // Check if the sequence field is populated
        const sequenceInput = screen.getByDisplayValue('MVTPE');
        expect(sequenceInput).toBeInTheDocument();

        // Check if the chain ID is populated
        const chainIdInput = screen.getByDisplayValue('A');
        expect(chainIdInput).toBeInTheDocument();
    });

    it('calls onSave with updated YAML when form is submitted', async () => {
        const mockOnSave = jest.fn();
        render(<BoltzYamlBuilder onSave={mockOnSave} />);

        // Find and click the "Show YAML Editor" button
        fireEvent.click(screen.getByText('Show YAML Editor'));

        // Find the YAML textarea and update it
        const textarea = screen.getByRole('textbox');
        fireEvent.change(textarea, { target: { value: sampleYaml } });

        // Find and click the save button
        fireEvent.click(screen.getByText('Save YAML'));

        // Check if onSave was called with the correct YAML
        expect(mockOnSave).toHaveBeenCalled();
        const savedYaml = mockOnSave.mock.calls[0][0];
        expect(savedYaml).toContain('version: 1');
        expect(savedYaml).toContain('sequence: MVTPE');
    });
});