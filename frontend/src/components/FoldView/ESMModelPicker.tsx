import React from 'react';
import { Select, Typography } from 'antd';

const { Text } = Typography;

// Add this constant for model options
const ESM_MODELS = [
    { value: 'esmc_600m', label: 'ESM-C (600M parameters, academic only)' },
    { value: 'esmc_300m', label: 'ESM-C (300M parameters, academic or commercial)' },
    { value: 'esm3-open', label: 'ESM-3 (works with structures, academic only)' },
    { value: 'esm2_t6_8M_UR50D', label: 'ESM-2 (8M parameters)' },
    { value: 'esm2_t12_35M_UR50D', label: 'ESM-2 (35M parameters)' },
    { value: 'esm2_t30_150M_UR50D', label: 'ESM-2 (150M parameters)' },
    { value: 'esm2_t33_650M_UR50D', label: 'ESM-2 (650M parameters)' },
    { value: 'esm2_t36_3B_UR50D', label: 'ESM-2 (3B parameters)' },
    { value: 'esm2_t48_15B_UR50D', label: 'ESM-2 (15B parameters)' },
    { value: 'esm1v_t33_650M_UR90S_ensemble', label: 'ESM-1v (ensemble) (650M parameters)' },
    { value: 'esm1v_t33_650M_UR90S_1', label: 'ESM-1v-1 (650M parameters)' },
    { value: 'esm1v_t33_650M_UR90S_2', label: 'ESM-1v-2 (650M parameters)' },
    { value: 'esm1v_t33_650M_UR90S_3', label: 'ESM-1v-3 (650M parameters)' },
    { value: 'esm1v_t33_650M_UR90S_4', label: 'ESM-1v-4 (650M parameters)' },
    { value: 'esm1v_t33_650M_UR90S_5', label: 'ESM-1v-5 (650M parameters)' },
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
            <Text strong style={{ marginBottom: '8px', display: 'block' }}>
                {label}
            </Text>
            <Select
                value={value}
                onChange={onChange}
                style={{ width: '100%' }}
                options={ESM_MODELS}
            />
        </div>
    );
};
