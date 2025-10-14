import React, { useState, useEffect } from 'react';
import {
    Card,
    Typography,
    Button,
    Alert,
    Row,
    Col,
    Spin,
    Table,
    Upload,
    Form,
    Tooltip,
    Space
} from 'antd';
import {
    CheckCircleOutlined,
    ClockCircleOutlined,
    ExclamationCircleOutlined,
    PlusOutlined,
    SelectOutlined,
    UploadOutlined
} from '@ant-design/icons';
import { FaRocket } from 'react-icons/fa';
import Plot from 'react-plotly.js';
import type { Campaign, CampaignRound, Fold, Naturalness } from '../../types/types';
import { updateCampaignRound, uploadCampaignRoundActivityFile } from '../../api/campaignApi';
import { notify } from '../../services/NotificationService';
import NaturalnessResults from '../shared/NaturalnessResults';
import { NaturalnessModal } from '../shared/NaturalnessModal';
import MutantSlateCard from '../shared/MutantSlateCard';
import { isOutputComplete } from '../../util/statusHelpers';

const { Title, Text, Paragraph } = Typography;

interface ZeroShotCampaignRoundViewProps {
    campaign: Campaign;
    currentRound: CampaignRound;
    fold: Fold;
    naturalnessCsvData: string | null;
    activityData: Array<{ seq_id: string, activity: number }> | null;
    onRefresh: () => void;
    onRefreshRound?: () => void;
    buildSlate: (seqIds: string[]) => void;
}

const ZeroShotCampaignRoundView: React.FC<ZeroShotCampaignRoundViewProps> = ({
    campaign,
    currentRound,
    fold,
    naturalnessCsvData,
    activityData,
    onRefresh,
    onRefreshRound,
    buildSlate
}) => {
    const [showNaturalnessModal, setShowNaturalnessModal] = useState(false);
    const [uploadingActivity, setUploadingActivity] = useState(false);
    const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

    useEffect(() => {
        const handleResize = () => {
            setIsMobile(window.innerWidth < 768);
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const matchingNaturalnessRuns = fold.naturalness_runs?.filter(run =>
        run.logit_model === campaign.naturalness_model
    ) || [];

    const handleSelectNaturalnessRun = async (naturalnessRunId: number) => {
        try {
            await updateCampaignRound(campaign.id!, currentRound.id, { naturalness_run_id: naturalnessRunId });
            onRefresh();
            notify.success('Naturalness run selected successfully');
        } catch (error) {
            notify.error('Failed to select naturalness run');
            console.error('Error selecting naturalness run:', error);
        }
    };

    const handleActivityFileUpload = async (file: File) => {
        if (!campaign?.id) return false;

        setUploadingActivity(true);
        try {
            await uploadCampaignRoundActivityFile(campaign.id, currentRound.round_number, file);
            notify.success('Activity file uploaded successfully');
            onRefresh();
        } catch (error: any) {
            notify.error(error.response?.data?.message || 'Failed to upload activity file');
        } finally {
            setUploadingActivity(false);
        }
        return false;
    };

    const handleStartNaturalnessRun = () => {
        setShowNaturalnessModal(true);
    };

    const handleNaturalnessModalClose = () => {
        setShowNaturalnessModal(false);
        onRefresh();
    };


    // If a naturalness run is already selected, show the results
    if ((currentRound.naturalness_run && !currentRound.naturalness_run_id) || (!currentRound.naturalness_run && currentRound.naturalness_run_id)) {
        console.error('Naturalness run not found for round which is strange because naturalness_run_id is set', currentRound);
    }
    if (currentRound.naturalness_run_id && currentRound.naturalness_run) {
        return (
            <div style={{
                display: 'grid',
                gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr',
                gap: '20px',
                alignItems: 'start'
            }}>
                <Card
                    title={
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <CheckCircleOutlined style={{ color: '#52c41a' }} />
                            <span>Naturalness Results: {currentRound.naturalness_run.name}</span>
                        </div>
                    }
                >
                    {naturalnessCsvData ? (
                        <NaturalnessResults
                            naturalnessCsvData={naturalnessCsvData}
                            yamlConfig={fold?.yaml_config || null}
                            setSelectedSubsequence={() => { }} // TODO: implement if needed
                            runName={currentRound.naturalness_run.name}
                            onBuildSlate={buildSlate}
                            disableSlateBuilder={!!currentRound.slate_seq_ids}
                            disableRowSelection={true}
                        />
                    ) : (
                        <div style={{ textAlign: 'center', padding: '20px' }}>
                            <Spin size="large" />
                            <div style={{ marginTop: '16px' }}>
                                <Text type="secondary">Loading naturalness data...</Text>
                            </div>
                        </div>
                    )}
                </Card>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <MutantSlateCard
                        currentRound={currentRound}
                        activityData={activityData}
                        onActivityFileUpload={handleActivityFileUpload}
                        uploadingActivity={uploadingActivity}
                        showActivityPlot={true}
                        foldId={campaign.fold_id.toString()}
                    />
                </div>
            </div>
        );
    }

    // No naturalness run selected yet
    if (matchingNaturalnessRuns.length === 0) {
        return (
            <>
                <Card>
                    <div style={{ marginBottom: '24px' }}>
                        <Title level={4}>
                            <ExclamationCircleOutlined style={{ color: '#faad14', marginRight: '8px' }} />
                            Missing Naturalness Run
                        </Title>
                        <Paragraph>
                            We're not seeing any naturalness runs that used the proper protein language model
                            (<Text code>{campaign.naturalness_model}</Text>).
                            Would you like to start a run?
                        </Paragraph>
                    </div>

                    <Space>
                        <Button
                            type="primary"
                            icon={<PlusOutlined />}
                            onClick={handleStartNaturalnessRun}
                        >
                            Start Naturalness Run
                        </Button>
                    </Space>
                </Card>

                <NaturalnessModal
                    open={showNaturalnessModal}
                    onClose={handleNaturalnessModalClose}
                    foldIds={[campaign.fold_id]}
                    title="Start Naturalness Run for Campaign"
                    templateNaturalnessRun={{
                        name: `${campaign.name}_round_${currentRound.round_number}_naturalness`,
                        logit_model: campaign.naturalness_model || 'esm2_t33_650M_UR50D',
                        use_structure: false,
                        get_depth_two_logits: false
                    }}
                />
            </>
        );
    }

    // Check if any runs are still running
    const incompleteRun = matchingNaturalnessRuns.find(run => !run.output_fpath);
    if (incompleteRun) {
        return (
            <>
                <Card>
                    <div style={{ textAlign: 'center', marginBottom: '24px' }}>
                        <ClockCircleOutlined style={{ fontSize: '48px', color: '#faad14', marginBottom: '16px' }} />
                        <Title level={4}>Waiting for Naturalness Run</Title>
                        <Paragraph>
                            Waiting on <Text strong>{incompleteRun.name}</Text> naturalness run to complete.
                        </Paragraph>
                        <Alert
                            message="If this takes more than 24 hours, it has probably failed, and you should restart it."
                            type="warning"
                            showIcon
                            style={{ marginBottom: '16px' }}
                        />
                    </div>

                    <Row justify="center">
                        <Col>
                            <Space>
                                <Button onClick={onRefresh}>
                                    Refresh Status
                                </Button>
                                <Button
                                    onClick={() => {
                                        // Get the invokation matching the naturalness run
                                        const jobId = incompleteRun.invokation_id;
                                        if (jobId) {
                                            // Navigate to the logs tab with the specific job ID
                                            window.open(`/fold/${campaign.fold_id}/logs#logs_${jobId}`, '_blank');
                                        }
                                    }}
                                    disabled={!incompleteRun.id}
                                >
                                    View Logs
                                </Button>
                                <Button
                                    type="primary"
                                    icon={<PlusOutlined />}
                                    onClick={handleStartNaturalnessRun}
                                >
                                    Start New Run
                                </Button>
                            </Space>
                        </Col>
                    </Row>
                </Card>

                <NaturalnessModal
                    open={showNaturalnessModal}
                    onClose={handleNaturalnessModalClose}
                    foldIds={[campaign.fold_id]}
                    title="Start Naturalness Run for Campaign"
                    templateNaturalnessRun={{
                        name: `${campaign.name}_round_${currentRound.round_number}_naturalness`,
                        logit_model: campaign.naturalness_model || 'esm2_t33_650M_UR50D',
                        use_structure: false,
                        get_depth_two_logits: false
                    }}
                />
            </>
        );
    }

    // Show selection table for completed runs
    const columns = [
        {
            title: 'Name',
            dataIndex: 'name',
            key: 'name',
            ellipsis: true,
            width: '30%',
        },
        {
            title: 'Model',
            dataIndex: 'logit_model',
            key: 'logit_model',
            ellipsis: true,
            render: (model: string) => (
                <Tooltip title={model}>
                    <span>{model}</span>
                </Tooltip>
            ),
        },
        {
            title: 'Status',
            key: 'status',
            width: '80px',
            render: (record: Naturalness) => (
                isOutputComplete(record.output_fpath) ? (
                    <span style={{ color: '#52c41a' }}>✓ Complete</span>
                ) : (
                    <span style={{ color: '#faad14' }}>⏳ Running</span>
                )
            ),
        },
        {
            title: 'Action',
            key: 'action',
            width: '100px',
            render: (record: Naturalness) => (
                isOutputComplete(record.output_fpath) ? (
                    <Button
                        type="primary"
                        icon={<SelectOutlined />}
                        onClick={() => handleSelectNaturalnessRun(record.id!)}
                        size="small"
                    >
                        Select
                    </Button>
                ) : (
                    <Button disabled size="small">Not Ready</Button>
                )
            ),
        },
    ];

    return (
        <>
            <Card>
                <div style={{ marginBottom: '24px' }}>
                    <CheckCircleOutlined style={{ fontSize: '48px', color: '#52c41a', marginBottom: '16px' }} />
                    <Title level={4}>Select Naturalness Run</Title>
                    <Paragraph>
                        Choose from {matchingNaturalnessRuns.length} available naturalness run(s)
                        using <Text code>{campaign.naturalness_model}</Text>.
                    </Paragraph>
                </div>

                <div style={{ overflowX: 'auto' }}>
                    <Table
                        dataSource={matchingNaturalnessRuns}
                        columns={columns}
                        rowKey="id"
                        pagination={false}
                        size="small"
                        scroll={{ x: true }}
                    />
                </div>
            </Card>

            <NaturalnessModal
                open={showNaturalnessModal}
                onClose={handleNaturalnessModalClose}
                foldIds={[campaign.fold_id]}
                title="Start Naturalness Run for Campaign"
                templateNaturalnessRun={{
                    name: `${campaign.name}_round_${currentRound.round_number}_naturalness`,
                    logit_model: campaign.naturalness_model || 'esm2_t33_650M_UR50D',
                    use_structure: false,
                    get_depth_two_logits: false
                }}
            />
        </>
    );
};

export default ZeroShotCampaignRoundView;
