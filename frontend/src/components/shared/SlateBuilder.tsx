import React, { useState } from 'react';
import { Modal, Table, Button, Typography, Input, Space } from 'antd';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;

interface SlateBuilderProps {
    open: boolean;
    onClose: () => void;
    onConfirm: (selectedSeqIds: string[]) => void;
    seqIds: string[];
    title?: string;
}

interface MutantRow {
    key: string;
    seqId: string;
}

const SlateBuilder: React.FC<SlateBuilderProps> = ({
    open,
    onClose,
    onConfirm,
    seqIds,
    title = "Build Slate"
}) => {
    const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>(seqIds);
    const [extraSeqIds, setExtraSeqIds] = useState<string>('');

    const data: MutantRow[] = seqIds.map(seqId => ({
        key: seqId,
        seqId: seqId
    }));

    const columns: ColumnsType<MutantRow> = [
        {
            title: 'Sequence ID',
            dataIndex: 'seqId',
            key: 'seqId',
            sorter: (a, b) => a.seqId.localeCompare(b.seqId),
        },
    ];

    const rowSelection = {
        selectedRowKeys,
        onChange: (newSelectedRowKeys: React.Key[]) => {
            setSelectedRowKeys(newSelectedRowKeys);
        },
        onSelectAll: (selected: boolean, selectedRows: MutantRow[], changeRows: MutantRow[]) => {
            if (selected) {
                setSelectedRowKeys(seqIds);
            } else {
                setSelectedRowKeys([]);
            }
        },
    };

    const handleConfirm = () => {
        // Combine selected sequence IDs with extra sequence IDs
        const selectedSeqs = selectedRowKeys as string[];
        const extraSeqs = extraSeqIds
            .split(',')
            .map(id => id.trim())
            .filter(id => id.length > 0);

        // Deduplicate the combined list
        const allSeqs = [...new Set([...selectedSeqs, ...extraSeqs])];

        onConfirm(allSeqs);
        onClose();
    };

    const handleCancel = () => {
        setSelectedRowKeys(seqIds); // Reset to initial state
        setExtraSeqIds(''); // Reset extra sequence IDs
        onClose();
    };

    return (
        <Modal
            title={title}
            open={open}
            onCancel={handleCancel}
            width={600}
            footer={[
                <Button key="cancel" onClick={handleCancel}>
                    Cancel
                </Button>,
                <Button key="confirm" type="primary" onClick={handleConfirm}>
                    Add to Slate ({(() => {
                        const selectedSeqs = selectedRowKeys as string[];
                        const extraSeqs = extraSeqIds.split(',').map(id => id.trim()).filter(id => id.length > 0);
                        return new Set([...selectedSeqs, ...extraSeqs]).size;
                    })()} mutants)
                </Button>,
            ]}
        >
            <div style={{ marginBottom: '16px' }}>
                <Text>
                    Finalize the slate for this round. Select the mutants you want to add to your slate. All mutants are selected by default.
                </Text>
            </div>

            <Table
                rowSelection={rowSelection}
                columns={columns}
                dataSource={data}
                pagination={false}
                scroll={{ y: 300 }}
                size="small"
            />

            <div style={{ marginTop: '16px', marginBottom: '16px' }}>
                <Text strong>Additional Sequence IDs (optional):</Text>
                <Text type="secondary" style={{ display: 'block', marginBottom: '8px' }}>
                    Enter additional sequence IDs to include in the slate, separated by commas
                </Text>
                <Input.TextArea
                    value={extraSeqIds}
                    onChange={(e) => setExtraSeqIds(e.target.value)}
                    placeholder="e.g., custom_mutant1, custom_mutant2, special_variant"
                    rows={2}
                />
            </div>

            <div style={{ textAlign: 'center' }}>
                <Text type="secondary">
                    {selectedRowKeys.length} selected from table
                    {extraSeqIds.trim() && ` + ${extraSeqIds.split(',').filter(id => id.trim()).length} additional`}
                    {' = '}
                    {(() => {
                        const selectedSeqs = selectedRowKeys as string[];
                        const extraSeqs = extraSeqIds.split(',').map(id => id.trim()).filter(id => id.length > 0);
                        return new Set([...selectedSeqs, ...extraSeqs]).size;
                    })()} total mutants
                </Text>
            </div>
        </Modal>
    );
};

export default SlateBuilder;
