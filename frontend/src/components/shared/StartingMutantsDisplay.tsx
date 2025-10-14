import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Space } from 'antd';
import { EditOutlined } from '@ant-design/icons';

interface StartingMutantsDisplayProps {
    campaignId: number;
    roundNumber: number;
    inputTemplates?: string;
    style?: React.CSSProperties;
}

const StartingMutantsDisplay: React.FC<StartingMutantsDisplayProps> = ({
    campaignId,
    roundNumber,
    inputTemplates,
    style
}) => {
    const navigate = useNavigate();

    const handleEditClick = () => {
        navigate(`/campaigns/${campaignId}/${roundNumber}/startingSequences`);
    };

    const mutantsList = inputTemplates
        ? inputTemplates.split(',').map(t => t.trim()).join(', ')
        : 'None selected';

    return (
        <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            gap: '8px',
            ...style
        }}>
            <span>
                <strong>Starting mutants:</strong> {mutantsList}
            </span>
            <Button
                type="link"
                icon={<EditOutlined />}
                onClick={handleEditClick}
                size="small"
                style={{ padding: '0 4px', minWidth: 'auto' }}
            >
                Edit
            </Button>
        </div>
    );
};

export default StartingMutantsDisplay;
