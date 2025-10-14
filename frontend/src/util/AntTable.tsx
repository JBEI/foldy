import React from 'react';
import { Table, TableProps, Button, Tooltip, Space } from 'antd';
import { ColumnsType, ExpandableConfig } from 'antd/es/table';
import './AntTable.css';

export interface ActionButtonProps {
    icon: React.ReactNode;
    onClick: () => void;
    tooltip: string;
    disabled?: boolean;
    danger?: boolean;
    type?: 'default' | 'primary' | 'dashed' | 'link' | 'text';
}

export const ActionButton: React.FC<ActionButtonProps> = ({
    icon,
    onClick,
    tooltip,
    disabled = false,
    danger = false,
    type = 'text'
}) => {
    return (
        <Tooltip title={tooltip}>
            <Button
                type={type}
                icon={icon}
                onClick={onClick}
                disabled={disabled}
                danger={danger}
                size="small"
                style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minWidth: '32px',
                    height: '32px'
                }}
            />
        </Tooltip>
    );
};

export interface AntTableColumn<T = any> {
    key: string;
    title: string;
    dataIndex?: string;
    render?: (value: any, record: T, index: number) => React.ReactNode;
    width?: number | string;
    align?: 'left' | 'center' | 'right';
    sortable?: boolean;
    sorter?: boolean | ((a: T, b: T) => number);
    fixed?: 'left' | 'right';
    ellipsis?: boolean;
    filterDropdown?: React.ReactNode;
    onFilter?: (value: string | number | boolean, record: T) => boolean;
    filters?: Array<{ text: string; value: string | number | boolean }>;
}

export interface AntTableProps<T = any> extends Omit<TableProps<T>, 'columns' | 'dataSource' | 'expandable'> {
    columns: AntTableColumn<T>[];
    dataSource: T[];
    loading?: boolean;
    pagination?: TableProps<T>['pagination'];
    rowKey?: string | ((record: T) => string);
    onRow?: (record: T, index?: number) => React.HTMLAttributes<HTMLTableRowElement>;
    scroll?: { x?: number | string; y?: number | string };
    size?: 'small' | 'middle' | 'large';
    bordered?: boolean;
    showHeader?: boolean;
    sticky?: boolean;
    summary?: (data: readonly T[]) => React.ReactNode;
    expandableContent?: (record: T) => React.ReactNode;
}

// Default expandable content renderer - shows object properties as key-value pairs
export const defaultExpandableContent = <T extends Record<string, any>>(record: T): React.ReactNode => {
    const entries = Object.entries(record).filter(([key, value]) =>
        value !== null && value !== undefined && value !== ''
    );

    const detailColumns = [
        {
            title: 'Property',
            dataIndex: 'key',
            key: 'key',
            width: 200,
            render: (key: string) => <strong>{key}</strong>,
        },
        {
            title: 'Value',
            dataIndex: 'value',
            key: 'value',
            render: (value: any) => {
                if (typeof value === 'object') {
                    return <pre style={{ margin: 0, fontSize: '12px' }}>{JSON.stringify(value, null, 2)}</pre>;
                }
                if (typeof value === 'boolean') {
                    return value ? 'true' : 'false';
                }
                return String(value);
            },
        },
    ];

    const detailData = entries.map(([key, value]) => ({
        key,
        value,
    }));

    return (
        <Table
            columns={detailColumns}
            dataSource={detailData}
            pagination={false}
            size="small"
            bordered
            rowKey="key"
            style={{ margin: '16px 0' }}
        />
    );
};

export const AntTable = <T extends Record<string, any>>({
    columns,
    dataSource,
    loading = false,
    pagination = false,
    rowKey = 'id',
    onRow,
    scroll = { x: 'max-content' },
    size = 'middle',
    bordered = true,
    showHeader = true,
    sticky = false,
    summary,
    expandableContent,
    ...rest
}: AntTableProps<T>) => {
    const antColumns: ColumnsType<T> = columns.map((col) => ({
        key: col.key,
        title: col.title,
        dataIndex: col.dataIndex || col.key,
        render: col.render,
        width: col.width,
        align: col.align,
        sorter: col.sortable ? (col.sorter || true) : false,
        fixed: col.fixed,
        ellipsis: col.ellipsis,
        filterDropdown: col.filterDropdown,
        onFilter: col.onFilter,
        filters: col.filters,
    }));

    // Configure expandable functionality if expandableContent is provided
    const expandableConfig: ExpandableConfig<T> | undefined = expandableContent ? {
        expandedRowRender: expandableContent || defaultExpandableContent,
        rowExpandable: () => true,
    } : undefined;

    return (
        <Table<T>
            columns={antColumns}
            dataSource={dataSource}
            loading={loading}
            pagination={pagination}
            rowKey={rowKey}
            onRow={onRow}
            scroll={scroll}
            size="small"
            bordered={bordered}
            showHeader={showHeader}
            sticky={sticky}
            summary={summary}
            expandable={expandableConfig}
            className={`foldy-table ${rest.className || ''}`}
            style={rest.style}
            {...rest}
        />
    );
};

// Helper function to create action buttons in a Space component
export const createActionButtons = (buttons: ActionButtonProps[]) => {
    return (
        <Space size="small">
            {buttons.map((button, index) => (
                <ActionButton key={index} {...button} />
            ))}
        </Space>
    );
};

export default AntTable;
