import React from 'react';
import { Modal, Typography, Space, Tag } from 'antd';
import { Embedding } from '../../types/types';

const { Text } = Typography;

interface EmbeddingParametersModalProps {
    open: boolean;
    onClose: () => void;
    embedding: Embedding | null;
}

export const EmbeddingParametersModal: React.FC<EmbeddingParametersModalProps> = ({
    open,
    onClose,
    embedding
}) => {
    if (!embedding) return null;

    const parseSequenceIds = (seqIds: string | null) => {
        if (!seqIds) return [];
        return seqIds.split(',').map(id => id.trim()).filter(id => id);
    };

    const parseLayers = (layers: string | null) => {
        if (!layers) return [];
        return layers.split(',').map(layer => layer.trim()).filter(layer => layer);
    };

    return (
        <Modal
            title={`Embedding Run Parameters: ${embedding.name}`}
            open={open}
            onCancel={onClose}
            footer={null}
            width={600}
        >
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
                <div>
                    <Text strong>Batch Name:</Text>
                    <div style={{ marginTop: '4px' }}>
                        <Tag color="blue">{embedding.name}</Tag>
                    </div>
                </div>

                <div>
                    <Text strong>Model:</Text>
                    <div style={{ marginTop: '4px' }}>
                        <Tag color="green">{embedding.embedding_model}</Tag>
                    </div>
                </div>

                <div>
                    <Text strong>Extra Sequence IDs:</Text>
                    <div style={{ marginTop: '4px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                        {parseSequenceIds(embedding.extra_seq_ids).length > 0 ? (
                            parseSequenceIds(embedding.extra_seq_ids).map((seqId, index) => (
                                <Tag key={index} color="cyan">{seqId}</Tag>
                            ))
                        ) : (
                            <Text type="secondary">None</Text>
                        )}
                    </div>
                </div>

                <div>
                    <Text strong>DMS Starting Sequence IDs:</Text>
                    <div style={{ marginTop: '4px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                        {parseSequenceIds(embedding.dms_starting_seq_ids).length > 0 ? (
                            parseSequenceIds(embedding.dms_starting_seq_ids).map((seqId, index) => (
                                <Tag key={index} color="orange">{seqId}</Tag>
                            ))
                        ) : (
                            <Text type="secondary">None</Text>
                        )}
                    </div>
                </div>

                <div>
                    <Text strong>Extra Layers:</Text>
                    <div style={{ marginTop: '4px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                        {parseLayers(embedding.extra_layers).length > 0 ? (
                            parseLayers(embedding.extra_layers).map((layer, index) => (
                                <Tag key={index} color="purple">{layer}</Tag>
                            ))
                        ) : (
                            <Text type="secondary">Default layers</Text>
                        )}
                    </div>
                </div>

                <div>
                    <Text strong>Fold ID:</Text>
                    <div style={{ marginTop: '4px' }}>
                        <Text type="secondary">{embedding.fold_id}</Text>
                    </div>
                </div>

                <div>
                    <Text strong>Run ID:</Text>
                    <div style={{ marginTop: '4px' }}>
                        <Text type="secondary">{embedding.id}</Text>
                    </div>
                </div>
            </Space>
        </Modal>
    );
};
