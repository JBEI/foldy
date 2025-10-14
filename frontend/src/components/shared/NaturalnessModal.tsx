import React, { useState } from 'react';
import { Modal, Alert, Button as AntButton, Typography } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { startLogits } from '../../api/embedApi';
import { notify } from '../../services/NotificationService';
import { ESMModelPicker } from '../FoldView/ESMModelPicker';
import { FormRow, FormField } from '../../util/tabComponents';
import { TextInputControl, CheckboxControl } from '../../util/controlComponents';
import { Logit } from '../../types/types';

const { Text, Paragraph, Title } = Typography;

interface NaturalnessModalProps {
    open: boolean;
    onClose: () => void;
    foldIds: number[];
    title?: string;
    templateNaturalnessRun?: Logit;
}

export const NaturalnessModal: React.FC<NaturalnessModalProps> = ({
    open,
    onClose,
    foldIds,
    title = "New Naturalness Run",
    templateNaturalnessRun
}) => {
    const [runName, setRunName] = useState<string>(templateNaturalnessRun?.name || '');
    const [logitModel, setLogitModel] = useState<string>(templateNaturalnessRun?.logit_model || 'esmc_600m');
    const [useStructure, setUseStructure] = useState<boolean>(templateNaturalnessRun?.use_structure || false);
    const [getDepthTwoLogits, setGetDepthTwoLogits] = useState<boolean>(templateNaturalnessRun?.get_depth_two_logits || false);
    const [showHelpModal, setShowHelpModal] = useState<boolean>(false);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const handleStartLogit = async () => {
        if (!runName.trim()) {
            notify.error('Run name is required.');
            return;
        }

        setIsLoading(true);

        try {
            const promises = foldIds.map(foldId =>
                startLogits(foldId, runName, logitModel, useStructure, getDepthTwoLogits)
            );

            await Promise.all(promises);

            notify.success(`Started naturalness runs for ${foldIds.length} fold(s)`);

            // Reset form
            setRunName('');
            setLogitModel('esmc_600m');
            setUseStructure(false);
            setGetDepthTwoLogits(false);

            onClose();
        } catch (error) {
            notify.error(`Failed to start naturalness runs: ${error}`);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <>
            <Modal
                title={title}
                open={open}
                onCancel={onClose}
                footer={[
                    <AntButton key="cancel" onClick={onClose}>
                        Cancel
                    </AntButton>,
                    <AntButton
                        key="start"
                        type="primary"
                        onClick={handleStartLogit}
                        disabled={!runName.trim()}
                        loading={isLoading}
                    >
                        Start Naturalness Run{foldIds.length > 1 ? 's' : ''}
                    </AntButton>
                ]}
                width={600}
            >
                {/* Help Alert */}
                <Alert
                    message="What is Naturalness?"
                    description={
                        <div>
                            <Paragraph>
                                Naturalness uses protein language models to score how "natural" each possible amino acid mutation looks.
                                Higher scores indicate mutations that are more likely to maintain protein function.
                            </Paragraph>
                            <AntButton
                                type="link"
                                icon={<QuestionCircleOutlined />}
                                onClick={() => setShowHelpModal(true)}
                                style={{ padding: 0 }}
                            >
                                View detailed naturalness guide
                            </AntButton>
                        </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: '20px' }}
                />

                <div style={{ marginBottom: '16px' }}>
                    <Text strong>Target Folds:</Text> {foldIds.length} fold{foldIds.length > 1 ? 's' : ''}
                </div>

                <FormRow>
                    <FormField>
                        <TextInputControl
                            label="Name"
                            value={runName}
                            onChange={setRunName}
                            placeholder="Enter run name"
                        />
                    </FormField>

                    <FormField>
                        <ESMModelPicker
                            value={logitModel}
                            onChange={setLogitModel}
                        />
                    </FormField>
                </FormRow>

                <FormRow>
                    <FormField>
                        <CheckboxControl
                            label="Use Structure (experimental)"
                            checked={useStructure}
                            onChange={setUseStructure}
                        />
                    </FormField>

                    <FormField>
                        <CheckboxControl
                            label="Get Depth Two Logits (experimental)"
                            checked={getDepthTwoLogits}
                            onChange={setGetDepthTwoLogits}
                        />
                    </FormField>
                </FormRow>
            </Modal>

            {/* Detailed Help Modal */}
            <Modal
                title="Naturalness Analysis Guide"
                open={showHelpModal}
                onCancel={() => setShowHelpModal(false)}
                footer={[
                    <AntButton key="close" onClick={() => setShowHelpModal(false)}>
                        Close
                    </AntButton>
                ]}
                width={700}
            >
                <div>
                    <Title level={4}>What is Naturalness?</Title>
                    <Paragraph>
                        Naturalness analysis uses protein language models (PLMs) to evaluate how "natural" or likely
                        each possible amino acid substitution appears based on evolutionary patterns learned from
                        millions of protein sequences.
                    </Paragraph>

                    <Title level={4}>How to Use</Title>
                    <ul>
                        <li><Text strong>Model Selection:</Text> Choose from different PLMs (ESM-C models recommended)</li>
                        <li><Text strong>Structure Integration:</Text> Optionally include 3D structure information</li>
                        <li><Text strong>Depth Two Logits:</Text> Advanced option for pair mutation analysis</li>
                    </ul>

                    <Title level={4}>Interpreting Results</Title>
                    <Paragraph>
                        The heatmap shows naturalness scores for each position-residue combination:
                    </Paragraph>
                    <ul>
                        <li><Text strong>Higher scores:</Text> More "natural" mutations, likely to preserve function</li>
                        <li><Text strong>Lower scores:</Text> Less natural mutations, may disrupt protein</li>
                        <li><Text strong>Wild-type masking:</Text> Option to hide original residues for clearer visualization</li>
                    </ul>

                    <Alert
                        message="Estimated Cost"
                        description="~$1 per naturalness run"
                        type="success"
                        showIcon
                        style={{ marginTop: '16px' }}
                    />

                    <Paragraph style={{ marginTop: '16px' }}>
                        Results can be downloaded as CSV files containing naturalness scores for all single mutations.
                    </Paragraph>
                </div>
            </Modal>
        </>
    );
};
