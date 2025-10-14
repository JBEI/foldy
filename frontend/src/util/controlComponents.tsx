import React, { CSSProperties } from 'react';
import { Input, InputNumber, Checkbox, Select, Upload, Typography } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';

const { Text } = Typography;

interface NumberInputControlProps {
    label: string;
    value: number;
    onChange: (value: number) => void;
    min?: number;
    max?: number;
    step?: number;
    style?: CSSProperties;
    inputWidth?: string;
}

export const NumberInputControl: React.FC<NumberInputControlProps> = ({
    label,
    value,
    onChange,
    min,
    max,
    step = 1,
    style = {},
    inputWidth = '100px'
}) => {
    const containerStyle: CSSProperties = {
        marginBottom: '16px',
        ...style
    };

    return (
        <div style={containerStyle}>
            <Text strong style={{ marginBottom: '8px', display: 'block' }}>
                {label}:
            </Text>
            <InputNumber
                value={value}
                onChange={(val) => onChange(val || 0)}
                style={{ width: inputWidth }}
                min={min}
                max={max}
                step={step}
            />
        </div>
    );
};

interface CheckboxControlProps {
    label: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
    style?: CSSProperties;
}

export const CheckboxControl: React.FC<CheckboxControlProps> = ({
    label,
    checked,
    onChange,
    style = {}
}) => {
    const containerStyle: CSSProperties = {
        marginBottom: '16px',
        ...style
    };

    return (
        <div style={containerStyle}>
            <Checkbox
                checked={checked}
                onChange={(e) => onChange(e.target.checked)}
            >
                {label}
            </Checkbox>
        </div>
    );
};

interface TextInputControlProps {
    label: string;
    value: string;
    onChange: (value: string) => void;
    placeholder?: string;
    style?: CSSProperties;
    inputStyle?: CSSProperties;
}

export const TextInputControl: React.FC<TextInputControlProps> = ({
    label,
    value,
    onChange,
    placeholder,
    style = {},
    inputStyle = {}
}) => {
    const containerStyle: CSSProperties = {
        marginBottom: '16px',
        ...style
    };

    return (
        <div style={containerStyle}>
            <Text strong style={{ marginBottom: '8px', display: 'block' }}>
                {label}
            </Text>
            <Input
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                style={inputStyle}
            />
        </div>
    );
};

interface TextAreaControlProps {
    label: string;
    value: string;
    onChange: (value: string) => void;
    placeholder?: string;
    rows?: number;
    style?: CSSProperties;
    inputStyle?: CSSProperties;
    disabled?: boolean;
}

export const TextAreaControl: React.FC<TextAreaControlProps> = ({
    label,
    value,
    onChange,
    placeholder,
    rows = 5,
    style = {},
    inputStyle = {},
    disabled = false
}) => {
    const containerStyle: CSSProperties = {
        marginBottom: '16px',
        ...style
    };

    return (
        <div style={containerStyle}>
            <Text strong style={{ marginBottom: '8px', display: 'block' }}>
                {label}
            </Text>
            <Input.TextArea
                rows={rows}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                style={inputStyle}
                disabled={disabled}
            />
        </div>
    );
};

interface SelectControlProps {
    label: string;
    value: string;
    onChange: (value: string) => void;
    options: { value: string; label: string }[];
    style?: CSSProperties;
    selectStyle?: CSSProperties;
}

export const SelectControl: React.FC<SelectControlProps> = ({
    label,
    value,
    onChange,
    options,
    style = {},
    selectStyle = {}
}) => {
    const containerStyle: CSSProperties = {
        marginBottom: '16px',
        ...style
    };

    return (
        <div style={containerStyle}>
            <Text strong style={{ marginBottom: '8px', display: 'block' }}>
                {label}
            </Text>
            <Select
                value={value}
                onChange={onChange}
                style={{ width: '100%', ...selectStyle }}
                options={options}
            />
        </div>
    );
};

interface FileUploadControlProps {
    label: string;
    onChange: (file: File | null) => void;
    accept?: string;
    selectedFile?: File | null;
    style?: CSSProperties;
}

export const FileUploadControl: React.FC<FileUploadControlProps> = ({
    label,
    onChange,
    accept,
    selectedFile,
    style = {}
}) => {
    const containerStyle: CSSProperties = {
        marginBottom: '16px',
        ...style
    };

    const handleChange = (info: any) => {
        const { file } = info;
        if (file?.originFileObj) {
            onChange(file.originFileObj);
        } else {
            onChange(null);
        }
    };

    return (
        <div style={containerStyle}>
            <Text strong style={{ marginBottom: '8px', display: 'block' }}>
                {label}
            </Text>
            <Upload
                maxCount={1}
                accept={accept}
                onChange={handleChange}
                beforeUpload={() => false} // Prevent auto upload
                fileList={selectedFile ? [{
                    uid: '-1',
                    name: selectedFile.name,
                    status: 'done',
                    originFileObj: selectedFile
                } as UploadFile] : []}
            >
                <button style={{ border: 'none', background: 'none', padding: 0 }}>
                    <UploadOutlined /> Choose File
                </button>
            </Upload>
        </div>
    );
};

interface MultiSelectControlProps {
    label: string;
    options: { key: string; label: string }[];
    selectedValues: string[];
    onChange: (selectedValues: string[]) => void;
    size?: number;
    style?: CSSProperties;
}

export const MultiSelectControl: React.FC<MultiSelectControlProps> = ({
    label,
    options,
    selectedValues,
    onChange,
    size,
    style = {}
}) => {
    const containerStyle: CSSProperties = {
        marginBottom: '16px',
        ...style
    };

    const selectOptions = options.map(option => ({
        value: option.key,
        label: option.label
    }));

    return (
        <div style={containerStyle}>
            <Text strong style={{ marginBottom: '8px', display: 'block' }}>
                {label}
            </Text>
            <Select
                mode="multiple"
                value={selectedValues}
                onChange={onChange}
                options={selectOptions}
                style={{ width: '100%' }}
                placeholder="Select items"
            />
            <Text type="secondary" style={{ fontSize: '12px', marginTop: '4px', display: 'block' }}>
                Selected {selectedValues.length} item(s)
            </Text>
        </div>
    );
};
