import React, { ReactNode, CSSProperties } from 'react';
import { Spin } from 'antd';
import './tabComponents.css';

interface TabContainerProps {
    children: ReactNode;
    style?: CSSProperties;
}

export const TabContainer: React.FC<TabContainerProps> = ({ children, style = {} }) => {
    const defaultStyle: CSSProperties = {
        padding: '16px',
        backgroundColor: 'transparent',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        ...style
    };

    return <div className="tab-container" style={defaultStyle}>{children}</div>;
};

interface SectionCardProps {
    children: ReactNode;
    style?: CSSProperties;
    className?: string;
}

export const SectionCard: React.FC<SectionCardProps> = ({ children, style = {}, className = '' }) => {
    const defaultStyle: CSSProperties = {
        padding: '20px',
        backgroundColor: '#ffffff',
        borderRadius: '6px',
        border: '1px solid #d9d9d9',
        ...style
    };

    return <section style={defaultStyle} className={`section-card ${className}`}>{children}</section>;
};

interface CollapsibleSectionProps {
    title: string;
    isOpen: boolean;
    onToggle: () => void;
    children: ReactNode;
    style?: CSSProperties;
    headerStyle?: CSSProperties;
    contentStyle?: CSSProperties;
}

export const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
    title,
    isOpen,
    onToggle,
    children,
    style = {},
    headerStyle = {},
    contentStyle = {}
}) => {
    const defaultHeaderStyle: CSSProperties = {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "16px 20px",
        backgroundColor: "#ffffff",
        border: "1px solid #d9d9d9",
        borderRadius: "6px",
        cursor: "pointer",
        fontWeight: "bold",
        marginBottom: isOpen ? "8px" : "0",
        userSelect: 'none',
        transition: 'background-color 0.2s ease',
        ...headerStyle
    };

    const defaultContentStyle: CSSProperties = {
        padding: '20px',
        backgroundColor: '#ffffff',
        borderRadius: '6px',
        border: '1px solid #d9d9d9',
        transition: 'opacity 0.2s ease, transform 0.2s ease',
        opacity: isOpen ? 1 : 0,
        transform: isOpen ? 'translateY(0)' : 'translateY(-10px)',
        ...contentStyle
    };

    return (
        <div style={style}>
            <div
                className="collapsible-header"
                style={defaultHeaderStyle}
                onClick={onToggle}
            >
                <span>{title}</span>
                <span style={{ fontSize: '14px' }}>{isOpen ? "▲" : "▼"}</span>
            </div>
            {isOpen && (
                <div style={defaultContentStyle}>
                    {children}
                </div>
            )}
        </div>
    );
};

interface FormRowProps {
    children: ReactNode;
    style?: CSSProperties;
    gap?: string;
}

export const FormRow: React.FC<FormRowProps> = ({ children, style = {}, gap = '20px' }) => {
    const defaultStyle: CSSProperties = {
        display: 'flex',
        gap,
        flexWrap: 'wrap',
        alignItems: 'flex-start',
        ...style
    };

    return <div className="form-row" style={defaultStyle}>{children}</div>;
};

interface FormFieldProps {
    children: ReactNode;
    style?: CSSProperties;
    flex?: string;
    minWidth?: string;
}

export const FormField: React.FC<FormFieldProps> = ({
    children,
    style = {},
    flex = '1',
    minWidth = '200px'
}) => {
    const defaultStyle: CSSProperties = {
        flex,
        minWidth,
        ...style
    };

    return <div className="form-field" style={defaultStyle}>{children}</div>;
};

interface TableSectionProps {
    title: string;
    children: ReactNode;
    style?: CSSProperties;
    scrollable?: boolean;
    extra?: ReactNode;
}

export const TableSection: React.FC<TableSectionProps> = ({
    title,
    children,
    style = {},
    scrollable = true,
    extra
}) => {
    const sectionStyle: CSSProperties = {
        padding: '20px',
        backgroundColor: '#ffffff',
        borderRadius: '6px',
        border: '1px solid #d9d9d9',
        ...(scrollable && { overflowX: 'auto' }),
        ...style
    };

    const headerStyle: CSSProperties = {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '15px'
    };

    return (
        <section style={sectionStyle}>
            {extra ? (
                <div style={headerStyle}>
                    <h3 style={{ margin: 0, overflowWrap: 'anywhere' }}>{title}</h3>
                    {extra}
                </div>
            ) : (
                <h3 style={{ overflowWrap: 'anywhere' }}>{title}</h3>
            )}
            {children}
        </section>
    );
};

interface DescriptionSectionProps {
    title: string;
    children: ReactNode;
    style?: CSSProperties;
}

export const DescriptionSection: React.FC<DescriptionSectionProps> = ({
    title,
    children,
    style = {}
}) => {
    return (
        <SectionCard style={style}>
            <h3 style={{ marginBottom: '10px' }}>{title}</h3>
            <div>{children}</div>
        </SectionCard>
    );
};

interface ButtonGroupProps {
    children: ReactNode;
    style?: CSSProperties;
    gap?: string;
}

export const ButtonGroup: React.FC<ButtonGroupProps> = ({
    children,
    style = {},
    gap = '10px'
}) => {
    const defaultStyle: CSSProperties = {
        marginTop: '10px',
        display: 'flex',
        gap,
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'flex-start',
        ...style
    };

    return <div className="button-group" style={defaultStyle}>{children}</div>;
};

interface LoadingSpinnerProps {
    message?: string;
    size?: number;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
    message = "Loading...",
    size = 4
}) => {
    const spinSize = size >= 4 ? 'large' : size >= 2 ? 'default' : 'small';

    return (
        <div className="loading-spinner">
            <div style={{ textAlign: 'center', padding: '20px 0' }}>
                <Spin size={spinSize} />
                {message && <p style={{ marginTop: '10px' }}>{message}</p>}
            </div>
        </div>
    );
};

interface ErrorMessageProps {
    message: string;
    details?: string;
    onRetry?: () => void;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({
    message,
    details,
    onRetry
}) => {
    return (
        <div className="error-message">
            <p><strong>Error:</strong> {message}</p>
            {details && <p>{details}</p>}
            {onRetry && (
                <button
                    className="uk-button uk-button-default uk-button-small"
                    onClick={onRetry}
                    style={{ marginTop: '10px' }}
                >
                    Retry
                </button>
            )}
        </div>
    );
};

interface SuccessMessageProps {
    message: string;
}

export const SuccessMessage: React.FC<SuccessMessageProps> = ({ message }) => {
    return (
        <div className="success-message">
            <p>{message}</p>
        </div>
    );
};

interface EmptyStateProps {
    title: string;
    message: string;
    actionButton?: {
        text: string;
        onClick: () => void;
    };
}

export const EmptyState: React.FC<EmptyStateProps> = ({
    title,
    message,
    actionButton
}) => {
    return (
        <div style={{
            textAlign: 'center',
            padding: '40px 20px',
            color: '#666'
        }}>
            <h4 style={{ marginBottom: '10px', color: '#999' }}>{title}</h4>
            <p style={{ marginBottom: '20px' }}>{message}</p>
            {actionButton && (
                <button
                    className="uk-button uk-button-primary"
                    onClick={actionButton.onClick}
                >
                    {actionButton.text}
                </button>
            )}
        </div>
    );
};

interface ResponsiveTableProps {
    children: ReactNode;
    style?: CSSProperties;
}

export const ResponsiveTable: React.FC<ResponsiveTableProps> = ({ children, style = {} }) => {
    const containerStyle: CSSProperties = {
        overflowX: 'auto',
        maxWidth: '100%',
        WebkitOverflowScrolling: 'touch',
        ...style
    };

    return (
        <div className="responsive-table-container" style={containerStyle}>
            <table className="uk-table uk-table-striped">
                {children}
            </table>
        </div>
    );
};
