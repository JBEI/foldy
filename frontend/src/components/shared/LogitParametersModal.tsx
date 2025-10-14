import React from 'react';
import { Modal, Typography, Space, Tag } from 'antd';
import { Naturalness } from '../../types/types';

const { Text } = Typography;

interface NaturalnessParametersModalProps {
    open: boolean;
    onClose: () => void;
    logit: Naturalness | null;
}

export const LogitParametersModal: React.FC<NaturalnessParametersModalProps> = ({
    open,
    onClose,
    logit
}) => {
    if (!logit) return null;

    return (
        <Modal
            title={`Naturalness Run Parameters: ${logit.name}`}
            open={open}
            onCancel={onClose}
            footer={null}
            width={500}
        >
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
                <div>
                    <Text strong>Run Name:</Text>
                    <div style={{ marginTop: '4px' }}>
                        <Tag color="blue">{logit.name}</Tag>
                    </div>
                </div>

                <div>
                    <Text strong>Model:</Text>
                    <div style={{ marginTop: '4px' }}>
                        <Tag color="green">{logit.logit_model}</Tag>
                    </div>
                </div>

                <div>
                    <Text strong>Use Structure:</Text>
                    <div style={{ marginTop: '4px' }}>
                        <Tag color={logit.use_structure ? "success" : "default"}>
                            {logit.use_structure ? "Yes" : "No"}
                        </Tag>
                    </div>
                </div>

                <div>
                    <Text strong>Depth Two Logits:</Text>
                    <div style={{ marginTop: '4px' }}>
                        <Tag color={logit.get_depth_two_logits ? "success" : "default"}>
                            {logit.get_depth_two_logits ? "Yes" : "No"}
                        </Tag>
                    </div>
                </div>

                <div>
                    <Text strong>Fold ID:</Text>
                    <div style={{ marginTop: '4px' }}>
                        <Text type="secondary">{logit.fold_id}</Text>
                    </div>
                </div>

                <div>
                    <Text strong>Run ID:</Text>
                    <div style={{ marginTop: '4px' }}>
                        <Text type="secondary">{logit.id}</Text>
                    </div>
                </div>
            </Space>
        </Modal>
    );
};
