import React, { useState, useMemo, useEffect } from 'react';
import { Modal, Form, Input, Select, Upload, Button as AntButton, Card, Divider, InputNumber, Alert, Typography, Row, Col } from 'antd';
import { UploadOutlined, PlayCircleOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import { FaRedo, FaRocket } from 'react-icons/fa';
import { FileInfo, FewShot, CampaignRound } from '../../types/types';
import { runFewShot } from '../../api/fewShotApi';
import { notify } from '../../services/NotificationService';

const { Text, Paragraph, Title } = Typography;

const FEW_SHOT_PRESETS = {
    'folde_default_mlp': {
        mode: 'TorchMLPFewShotModel',
        params: `{
    "pretrain": true,
    "pretrain_epochs": 50,
    "pretrain_val_frequency": 1,
    "ensemble_size": 5,
    "hidden_dims": [100, 50],
    "dropout": 0.2,
    "learning_rate": 0.0003,
    "weight_decay": 0.00001,
    "train_epochs": 200,
    "train_patience": 40,
    "val_frequency": 1,
    "do_validation_with_pair_fraction": 0.2,
    "decision_mode": "constantliar",
    "lie_noise_stddev_multiplier": 0.25
}`
    },
    'evolvepro': {
        mode: 'RandomForestFewShotModel',
        params: `{
    "n_estimators": 100,
    "criterion": "friedman_mse",
    "max_depth": null,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 1.0,
    "max_leaf_nodes": null,
    "min_impurity_decrease": 0.0,
    "bootstrap": true,
    "oob_score": false,
    "n_jobs": null,
    "random_state": 1,
    "verbose": 0,
    "warm_start": false,
    "ccp_alpha": 0.0,
    "max_samples": null
}`
    },
    'custom': {
        mode: null,
        params: ''
    }
};

interface FewShotModalProps {
    open: boolean;
    onClose: (createdFewShot?: FewShot) => void;
    foldId: number;
    files: FileInfo[] | null;
    evolutions: FewShot[] | null;
    campaignRounds?: CampaignRound[] | null;
    title?: string;
    templateFewShot: Partial<FewShot>;
    defaultActivityFileSource?: 'upload' | 'evolution' | 'campaign';
    fewShotCampaignRoundIdForActivityFile?: number;
    fewShotRunIdForActivityFile?: number;
}

export const FewShotModal: React.FC<FewShotModalProps> = ({
    open,
    onClose,
    foldId,
    files,
    evolutions,
    campaignRounds,
    title = "New FewShot Run",
    templateFewShot,
    defaultActivityFileSource = 'upload',
    fewShotCampaignRoundIdForActivityFile,
    fewShotRunIdForActivityFile
}) => {
    const [fewShotName, setFewShotName] = useState<string>('');
    const [activityFile, setActivityFile] = useState<File | null>(null);
    const [activityFileSource, setActivityFileSource] = useState<'upload' | 'evolution' | 'campaign'>('upload');
    const [selectedFewShotForActivity, setSelectedFewShotForActivity] = useState<number | null>(null);
    const [selectedCampaignRoundForActivity, setSelectedCampaignRoundForActivity] = useState<number | null>(null);
    const [mode, setMode] = useState<string>('TorchMLPFewShotModel');
    const [numMutants, setNumMutants] = useState<number>(24);
    const [selectedEmbeddingPaths, setSelectedEmbeddingPaths] = useState<string[]>([]);
    const [selectedNaturalnessPaths, setSelectedNaturalnessPaths] = useState<string[]>([]);
    const [finetuningModelCheckpoint, setFinetuningModelCheckpoint] = useState<string>('facebook/esm2_t6_8M_UR50D');
    const [fewShotParams, setFewShotParams] = useState<string>('');
    const [selectedPreset, setSelectedPreset] = useState<string>('folde_default_mlp');
    const [showHelpModal, setShowHelpModal] = useState<boolean>(false);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    // Sort evolutions (FewShot) by date_created (newest first)
    const sortedEvolutions = useMemo(() => {
        if (!evolutions) return [];
        return [...evolutions].sort((a, b) => {
            if (!a.date_created && !b.date_created) return 0;
            if (!a.date_created) return 1;
            if (!b.date_created) return -1;
            return new Date(b.date_created).getTime() - new Date(a.date_created).getTime();
        });
    }, [evolutions]);

    // Sort campaign rounds by round_number (newest first)
    const sortedCampaignRounds = useMemo(() => {
        if (!campaignRounds) return [];
        return [...campaignRounds].sort((a, b) => b.round_number - a.round_number);
    }, [campaignRounds]);

    const availableEmbeddingFiles = files?.filter(file =>
        file.key.includes('embed')
    ) || [];
    const availableNaturalnessFiles = files?.filter(file =>
        file.key.includes('naturalness') && file.key.endsWith('.csv')
    ) || [];

    // Stable template reference - only changes when content actually changes
    const templateString = useMemo(() =>
        JSON.stringify(templateFewShot), [templateFewShot]
    );

    // Template reactivity - apply template values when modal opens or template changes
    useEffect(() => {
        if (!open) return; // Only initialize when modal is open

        // Always apply template values (could be empty object for "new" mode)
        setFewShotName(templateFewShot.name || '');
        setMode(templateFewShot.mode || 'TorchMLPFewShotModel');
        setNumMutants(templateFewShot.num_mutants || 24);
        setSelectedEmbeddingPaths(templateFewShot.embedding_files ? templateFewShot.embedding_files.split(',') : []);
        setSelectedNaturalnessPaths(templateFewShot.naturalness_files ? templateFewShot.naturalness_files.split(',') : []);
        setFinetuningModelCheckpoint(templateFewShot.finetuning_model_checkpoint || 'facebook/esm2_t6_8M_UR50D');
        setFewShotParams(templateFewShot.few_shot_params || '');

        // Smart preset logic - if template has params, show custom; otherwise use default
        if (templateFewShot.few_shot_params) {
            setSelectedPreset('custom');
        } else {
            onPresetChange('folde_default_mlp'); // Apply default preset + trigger logic
        }

        // Activity source settings
        setActivityFileSource(defaultActivityFileSource);
        setSelectedFewShotForActivity(fewShotRunIdForActivityFile || null);
        setSelectedCampaignRoundForActivity(fewShotCampaignRoundIdForActivityFile || null);
    }, [open, templateString, defaultActivityFileSource, fewShotRunIdForActivityFile, fewShotCampaignRoundIdForActivityFile]);

    const isValidJson = (jsonString: string): boolean => {
        if (!jsonString.trim()) return true;
        try {
            JSON.parse(jsonString);
            return true;
        } catch {
            return false;
        }
    };

    const jsonValidationStatus = useMemo(() => {
        if (fewShotParams.trim() === '') return '';
        return isValidJson(fewShotParams) ? 'success' : 'error';
    }, [fewShotParams]);

    const onPresetChange = (preset: string) => {
        setSelectedPreset(preset);
        if (preset !== 'custom') {
            const presetConfig = FEW_SHOT_PRESETS[preset as keyof typeof FEW_SHOT_PRESETS];
            setFewShotParams(presetConfig.params);
            if (presetConfig.mode) {
                setMode(presetConfig.mode);
            }
        }
    };

    const handleEvolve = async () => {
        const hasActivitySource = activityFileSource === 'upload' ? activityFile :
            activityFileSource === 'evolution' ? selectedFewShotForActivity :
                selectedCampaignRoundForActivity;

        if (!hasActivitySource || (selectedEmbeddingPaths.length === 0) || (selectedNaturalnessPaths.length === 0)) {
            notify.warning('Please fill in all required fields');
            return;
        }

        setIsLoading(true);

        try {
            notify.info('Starting FewShot...');
            const foldEvolution = await runFewShot(
                fewShotName,
                foldId,
                activityFileSource === 'upload' ? activityFile : null,
                activityFileSource === 'evolution' ? selectedFewShotForActivity : null,
                activityFileSource === 'campaign' ? selectedCampaignRoundForActivity : null,
                mode,
                numMutants,
                selectedEmbeddingPaths,
                selectedNaturalnessPaths,
                mode === 'finetuning' ? finetuningModelCheckpoint : undefined,
                fewShotParams
            );
            notify.success(`FewShot process started with id ${foldEvolution.id} and name ${foldEvolution.name}`);

            // Reset form and close modal with created FewShot
            handleClose(foldEvolution);
        } catch (error) {
            notify.error(`Failed to start FewShot process: ${error}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleClose = (createdFewShot?: FewShot) => {
        setIsLoading(false);
        onClose(createdFewShot);
    };

    const handleCancel = () => {
        handleClose();
    };

    return (
        <>
            <Modal
                title={title}
                open={open}
                onCancel={handleCancel}
                width={800}
                footer={[
                    <AntButton key="cancel" onClick={handleCancel}>
                        Cancel
                    </AntButton>,
                    <AntButton
                        key="submit"
                        type="primary"
                        icon={<PlayCircleOutlined />}
                        onClick={handleEvolve}
                        loading={isLoading}
                        disabled={
                            fewShotName === '' ||
                            (activityFileSource === 'upload' && !activityFile) ||
                            (activityFileSource === 'evolution' && !selectedFewShotForActivity) ||
                            (activityFileSource === 'campaign' && !selectedCampaignRoundForActivity) ||
                            ((mode === 'randomforest' || mode === 'mlp') && selectedEmbeddingPaths.length === 0) ||
                            (mode === 'finetuning' && !finetuningModelCheckpoint)
                        }
                    >
                        Start FewShot
                    </AntButton>
                ]}
            >
                {/* Help Alert */}
                <Alert
                    message="What is a FewShot Run?"
                    description={
                        <div>
                            <Paragraph>
                                In a FewShot run, a machine learning model is trained on your protein activity measurements, and that model is used to predict the activity of many other possible mutations. Then a slate of mutants is recommended for screening in the next round.

                                This tool facilitates low-N directed protein optimization,
                                with as little as 16 screened mutants per round.
                            </Paragraph>
                            <AntButton
                                type="link"
                                icon={<QuestionCircleOutlined />}
                                onClick={() => setShowHelpModal(true)}
                                style={{ padding: 0 }}
                            >
                                View detailed setup instructions
                            </AntButton>
                        </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: '20px' }}
                />

                <Form layout="vertical">
                    <Row gutter={16}>
                        <Col span={12}>
                            <Form.Item
                                label="FewShot Name"
                                required
                                help="Give your FewShot run a descriptive name"
                            >
                                <Input
                                    value={fewShotName}
                                    onChange={(e) => setFewShotName(e.target.value)}
                                    placeholder="e.g., round1_high_activity"
                                />
                            </Form.Item>
                        </Col>
                        <Col span={12}>
                            <Form.Item
                                label="Slate Size"
                                required
                                help="How many top mutants to recommend"
                            >
                                <InputNumber
                                    value={numMutants}
                                    onChange={(value) => setNumMutants(value || 24)}
                                    min={1}
                                    style={{ width: '100%' }}
                                />
                            </Form.Item>
                        </Col>
                    </Row>

                    <Divider>File Selection</Divider>

                    <Form.Item
                        label="Activity File Source"
                        required
                        help="Choose the source of your activity data"
                    >
                        <Row gutter={16}>
                            <Col span={8}>
                                <Card
                                    size="small"
                                    style={{
                                        cursor: 'pointer',
                                        border: activityFileSource === 'upload' ? '2px solid #1890ff' : '1px solid #d9d9d9'
                                    }}
                                    onClick={() => {
                                        setActivityFileSource('upload');
                                        setSelectedFewShotForActivity(null);
                                        setSelectedCampaignRoundForActivity(null);
                                    }}
                                >
                                    <div style={{ textAlign: 'center' }}>
                                        <UploadOutlined style={{ fontSize: '24px', marginBottom: '8px' }} />
                                        <div><strong>Upload New File</strong></div>
                                        <div style={{ fontSize: '12px', color: '#666' }}>
                                            Upload an Excel file with activity data
                                        </div>
                                    </div>
                                </Card>
                            </Col>
                            <Col span={8}>
                                <Card
                                    size="small"
                                    style={{
                                        cursor: 'pointer',
                                        border: activityFileSource === 'evolution' ? '2px solid #1890ff' : '1px solid #d9d9d9'
                                    }}
                                    onClick={() => {
                                        setActivityFileSource('evolution');
                                        setActivityFile(null);
                                        setSelectedCampaignRoundForActivity(null);
                                    }}
                                >
                                    <div style={{ textAlign: 'center' }}>
                                        <FaRedo style={{ fontSize: '24px', marginBottom: '8px' }} />
                                        <div><strong>Use Existing FewShot</strong></div>
                                        <div style={{ fontSize: '12px', color: '#666' }}>
                                            Reuse activity data from a previous FewShot run
                                        </div>
                                    </div>
                                </Card>
                            </Col>
                            <Col span={8}>
                                <Card
                                    size="small"
                                    style={{
                                        cursor: 'pointer',
                                        border: activityFileSource === 'campaign' ? '2px solid #1890ff' : '1px solid #d9d9d9'
                                    }}
                                    onClick={() => {
                                        setActivityFileSource('campaign');
                                        setActivityFile(null);
                                        setSelectedFewShotForActivity(null);
                                    }}
                                >
                                    <div style={{ textAlign: 'center' }}>
                                        <FaRocket style={{ fontSize: '24px', marginBottom: '8px' }} />
                                        <div><strong>Campaign Round</strong></div>
                                        <div style={{ fontSize: '12px', color: '#666' }}>
                                            Use activity data from a campaign round
                                        </div>
                                    </div>
                                </Card>
                            </Col>
                        </Row>
                    </Form.Item>

                    {activityFileSource === 'upload' && (
                        <Form.Item
                            label="Upload Activity File"
                            required
                            help="Excel file with seq_id and activity columns"
                        >
                            <Upload
                                beforeUpload={(file) => {
                                    setActivityFile(file);
                                    return false;
                                }}
                                accept=".xlsx,.xls"
                                maxCount={1}
                                fileList={activityFile ? [{
                                    uid: '1',
                                    name: activityFile.name,
                                    status: 'done'
                                }] : []}
                                onRemove={() => setActivityFile(null)}
                            >
                                <AntButton icon={<UploadOutlined />}>
                                    Select Activity File (.xlsx/.xls)
                                </AntButton>
                            </Upload>
                        </Form.Item>
                    )}

                    {activityFileSource === 'evolution' && (
                        <Form.Item
                            label="Select FewShot for Activity Data"
                            required
                            help="Choose a finished FewShot run to reuse its activity data"
                        >
                            <Select
                                value={selectedFewShotForActivity}
                                onChange={setSelectedFewShotForActivity}
                                style={{ width: '100%' }}
                                placeholder="Select a FewShot run"
                            >
                                {sortedEvolutions.map(slateBuild => (
                                    <Select.Option key={slateBuild.id} value={slateBuild.id}>
                                        {slateBuild.name}
                                    </Select.Option>
                                ))}
                            </Select>
                            {sortedEvolutions.length === 0 && (
                                <Alert
                                    message="No finished FewShot runs available"
                                    description="You need at least one completed FewShot run to use this option."
                                    type="info"
                                    showIcon
                                    style={{ marginTop: '8px' }}
                                />
                            )}
                        </Form.Item>
                    )}

                    {activityFileSource === 'campaign' && (
                        <Form.Item
                            label="Select Campaign Round for Activity Data"
                            required
                            help="Choose a campaign round with uploaded activity data"
                        >
                            <Select
                                value={selectedCampaignRoundForActivity}
                                onChange={setSelectedCampaignRoundForActivity}
                                style={{ width: '100%' }}
                                placeholder="Select a campaign round"
                            >
                                {sortedCampaignRounds.filter(round => round.result_activity_fpath).map(round => (
                                    <Select.Option key={round.id} value={round.id}>
                                        Round {round.round_number} - {new Date(round.date_started).toLocaleDateString()}
                                    </Select.Option>
                                ))}
                            </Select>
                            {sortedCampaignRounds.filter(round => round.result_activity_fpath).length === 0 && (
                                <Alert
                                    message="No campaign rounds with activity data available"
                                    description="You need at least one campaign round with uploaded activity data to use this option."
                                    type="warning"
                                    showIcon
                                    style={{ marginTop: '8px' }}
                                />
                            )}
                        </Form.Item>
                    )}

                    <Form.Item
                        label="Multi-Mutant Embedding Files"
                        required
                        help="Select embedding files generated in the Embed tab. This defines the pool of mutants that will be evaluated."
                    >
                        <Select
                            mode="multiple"
                            value={selectedEmbeddingPaths}
                            onChange={setSelectedEmbeddingPaths}
                            style={{ width: '100%' }}
                            placeholder="Select embedding files"
                        >
                            {availableEmbeddingFiles.map(file => (
                                <Select.Option key={file.key} value={file.key}>
                                    {file.key.split('/').pop() || file.key}
                                </Select.Option>
                            ))}
                        </Select>
                    </Form.Item>

                    <Form.Item
                        label="Single Mutant Naturalness Files"
                        help="Select naturalness files. We recommend using ESM-C 600M."
                    >
                        <Select
                            mode="multiple"
                            value={selectedNaturalnessPaths}
                            onChange={setSelectedNaturalnessPaths}
                            style={{ width: '100%' }}
                            placeholder="Select naturalness files"
                        >
                            {availableNaturalnessFiles.map(file => (
                                <Select.Option key={file.key} value={file.key}>
                                    {file.key.split('/').pop() || file.key}
                                </Select.Option>
                            ))}
                        </Select>
                    </Form.Item>

                    <Divider>Model Parameters</Divider>

                    <Form.Item
                        label="Presets"
                        help="Choose a preset configuration or select 'Custom' to define your own"
                    >
                        <Select
                            value={selectedPreset}
                            onChange={(preset) => onPresetChange(preset)}
                            style={{ width: '100%' }}
                        >
                            <Select.Option value="folde_default_mlp">FolDE Default MLP</Select.Option>
                            <Select.Option value="evolvepro">EvolvePro</Select.Option>
                            <Select.Option value="custom">Custom</Select.Option>
                        </Select>
                    </Form.Item>

                    <Form.Item
                        label="Model Choice"
                        required
                        help="ML algorithm to use for predictions"
                    >
                        <Select
                            value={mode}
                            onChange={(newMode) => {
                                setMode(newMode);
                                if (selectedPreset !== 'custom') {
                                    setSelectedPreset('custom');
                                }
                            }}
                            style={{ width: '100%' }}
                        >
                            <Select.Option value="TorchMLPFewShotModel">MLP Few Shot Model (Recommended)</Select.Option>
                            <Select.Option value="RandomForestFewShotModel">Random Forest Few Shot Model</Select.Option>
                            <Select.Option value="randomforest">(Legacy) Random Forest</Select.Option>
                            <Select.Option value="mlp">(Legacy) Multi-Layer Perceptron</Select.Option>
                            <Select.Option value="finetuning">(Legacy) Finetuning</Select.Option>
                        </Select>
                    </Form.Item>

                    {mode === 'finetuning' && (
                        <Form.Item
                            label="Model Checkpoint"
                            help="Pre-trained model to fine-tune (legacy mode only)"
                        >
                            <Select
                                value={finetuningModelCheckpoint}
                                onChange={setFinetuningModelCheckpoint}
                                style={{ width: '100%' }}
                            >
                                <Select.Option value="facebook/esm2_t6_8M_UR50D">ESM2 (8M params)</Select.Option>
                                <Select.Option value="facebook/esm2_t33_650M_UR50D">ESM2 (650M params)</Select.Option>
                                <Select.Option value="facebook/esm2_t48_15B_UR50D">ESM2 (15B params)</Select.Option>
                            </Select>
                        </Form.Item>
                    )}

                    <Form.Item
                        label="Few Shot Parameters (JSON format)"
                        help="Model-specific parameters in JSON format. Border color indicates validity."
                        validateStatus={jsonValidationStatus}
                    >
                        <Input.TextArea
                            value={fewShotParams}
                            onChange={(e) => {
                                const value = e.target.value;
                                setFewShotParams(value);
                                if (selectedPreset !== 'custom') {
                                    const presetConfig = FEW_SHOT_PRESETS[selectedPreset as keyof typeof FEW_SHOT_PRESETS];
                                    if (value !== presetConfig.params) {
                                        setSelectedPreset('custom');
                                    }
                                }
                            }}
                            placeholder='{"key": "value"}'
                            rows={4}
                            style={{
                                fontFamily: 'monospace',
                                borderColor: jsonValidationStatus === 'success' ? '#52c41a' :
                                    jsonValidationStatus === 'error' ? '#ff4d4f' : undefined,
                                borderWidth: jsonValidationStatus ? '2px' : undefined
                            }}
                        />
                    </Form.Item>
                </Form>
            </Modal>

            {/* Detailed Help Modal */}
            <Modal
                title="FewShot Run Setup Guide"
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
                    <Title level={4}>Required Inputs</Title>
                    <Paragraph>
                        Each FewShot run requires:
                    </Paragraph>
                    <ul>
                        <li>
                            <Text strong>Activity Excel File:</Text> A file with columns 'seq_id' and 'activity' containing
                            your measured mutant activities
                        </li>
                        <li>
                            <Text strong>Embedding Files:</Text> Embeddings generated in the Embed tab, containing
                            embeddings for both measured mutants and all mutants you wish to screen
                        </li>
                        <li>
                            <Text strong>Naturalness Files:</Text> Naturalness scores for single mutants of the protein. We recommend using ESM-C 600M
                        </li>
                    </ul>

                    <Title level={4}>Example Activity File Format</Title>
                    <div style={{ textAlign: 'center', margin: '20px 0' }}>
                        <img
                            style={{ width: "300px", border: '1px solid #d9d9d9', borderRadius: '4px' }}
                            src={`/evolve_activity_excel_example.png`}
                            alt="Example activity file format showing seq_id and activity columns"
                        />
                    </div>

                    <Title level={4}>Mode Selection Guide</Title>
                    <ul>
                        <li><Text strong>MLP Few Shot Model:</Text> Recommended for most use cases</li>
                        <li><Text strong>Random Forest Few Shot Model:</Text> Alternative ML approach</li>
                        <li><Text strong>Legacy modes:</Text> Older implementations, use new modes when possible</li>
                    </ul>

                    <Title level={4}>Parameters</Title>
                    <Paragraph>
                        <Text strong>Few Shot Parameters:</Text> JSON configuration for the ML model.
                        Use presets for common configurations or customize with your own JSON.
                    </Paragraph>

                    <Alert
                        message="Estimated Cost"
                        description="~$0.05 per FewShot round"
                        type="success"
                        showIcon
                        style={{ marginTop: '16px' }}
                    />

                    <Paragraph style={{ marginTop: '16px' }}>
                        Once complete, you can download the predicted activities for all mutants from the Files tab.
                    </Paragraph>
                </div>
            </Modal>
        </>
    );
};

export default FewShotModal;
