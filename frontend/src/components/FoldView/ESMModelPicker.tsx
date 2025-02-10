import React from 'react';

// Add this constant for model options
const ESM_MODELS = [
    { value: 'esmc_600m', label: 'ESM-C (600M parameters, academic only)' },
    { value: 'esmc_300m', label: 'ESM-C (300M parameters, academic or commercial)' },
    { value: 'esm3-open', label: 'ESM-3 (works with structures, academic only)' },
    { value: 'esm2_t33_650M_UR50D', label: 'ESM-2 (650M parameters)' },
    { value: 'esm2_t36_3B_UR50D', label: 'ESM-2 (3B parameters)' },
    { value: 'esm2_t48_15B_UR50D', label: 'ESM-2 (15B parameters)' },
    { value: 'esm1v_t33_650M_UR90S_1', label: 'ESM-1v-1 (650M parameters)' },
    { value: 'esm1v_t33_650M_UR90S_2', label: 'ESM-1v-2 (650M parameters)' },
    { value: 'esm1v_t33_650M_UR90S_3', label: 'ESM-1v-3 (650M parameters)' },
    { value: 'esm1v_t33_650M_UR90S_4', label: 'ESM-1v-4 (650M parameters)' },
    { value: 'esm1v_t33_650M_UR90S_5', label: 'ESM-1v-5 (650M parameters)' },
    { value: 'esm1v', label: 'ESM-1v' },
];

interface ESMModelPickerProps {
    value: string;
    onChange: (value: string) => void;
    label?: string;
}

export const ESMModelPicker: React.FC<ESMModelPickerProps> = ({
    value,
    onChange,
    label = "ESM Model"
}) => {
    return (
        <div style={{ flex: 1, minWidth: '200px' }}>
            <label className="uk-form-label">{label}</label>
            <select
                className="uk-select"
                value={value}
                onChange={(e) => onChange(e.target.value)}
            >
                {ESM_MODELS.map(model => (
                    <option key={model.value} value={model.value}>
                        {model.label}
                    </option>
                ))}
            </select>
        </div>
    );
};
