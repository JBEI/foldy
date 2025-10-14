import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Card,
    Typography,
    Button,
    Modal,
    Spin,
} from 'antd';
import {
    ExperimentOutlined,
    ExclamationCircleOutlined
} from '@ant-design/icons';
import type { Campaign, CampaignRound, Fold } from '../../types/types';
import { getFold } from '../../api/foldApi';
import { getCampaignRoundActivityData, updateCampaignRound } from '../../api/campaignApi';
import { getFile } from '../../api/fileApi';
import { notify } from '../../services/NotificationService';
import SlateBuilder from '../shared/SlateBuilder';
import ZeroShotCampaignRoundView from './ZeroShotCampaignRoundView';
import FewShotCampaignRoundView from './FewShotCampaignRoundView';

const { Title, Text } = Typography;

interface CampaignRoundComponentProps {
    campaign: Campaign;
    currentRound: CampaignRound;
    subpage?: string;
    onRefresh: () => void;
    onRefreshRound?: () => void;
}

const CampaignRoundComponent: React.FC<CampaignRoundComponentProps> = ({
    campaign,
    currentRound,
    subpage,
    onRefresh,
    onRefreshRound
}) => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(true);
    const [fold, setFold] = useState<Fold | null>(null);
    const [naturalnessCsvData, setNaturalnessCsvData] = useState<string | null>(null);
    const [showSlateBuilder, setShowSlateBuilder] = useState(false);
    const [slateBuilderSeqIds, setSlateBuilderSeqIds] = useState<string[]>([]);
    const [activityData, setActivityData] = useState<Array<{ seq_id: string, activity: number }> | null>(null);

    // Load naturalness CSV data when needed
    const loadNaturalnessCsvData = useCallback(async (naturalness: any) => {
        if (!naturalness.output_fpath) {
            notify.error('Naturalness run output file not found');
            return;
        }

        try {
            const fileBlob = await getFile(campaign.fold_id, naturalness.output_fpath);
            const reader = new FileReader();
            reader.onload = (e) => {
                const fileString = e.target?.result as string;
                setNaturalnessCsvData(fileString);
            };
            reader.readAsText(fileBlob);
        } catch (error) {
            notify.error('Failed to load naturalness data');
            console.error('Error loading naturalness CSV:', error);
        }
    }, [campaign.fold_id]);

    // Load fold data
    const loadFoldData = useCallback(async () => {
        setLoading(true);
        try {
            const foldData = await getFold(campaign.fold_id);
            setFold(foldData);
        } catch (error) {
            notify.error('Failed to load fold data');
            console.error('Error loading fold:', error);
        } finally {
            setLoading(false);
        }
    }, [campaign.fold_id]);

    // Load activity data
    const loadActivityData = useCallback(async () => {
        if (!campaign?.id || !currentRound.result_activity_fpath) return;

        try {
            const response = await getCampaignRoundActivityData(campaign.id, currentRound.round_number);
            setActivityData(response.data);
        } catch (error: any) {
            console.error('Failed to load activity data:', error);
            notify.error('Failed to load activity data');
        }
    }, [campaign?.id, currentRound.result_activity_fpath, currentRound.round_number]);

    // Effects
    useEffect(() => {
        loadFoldData();
    }, [loadFoldData]);

    useEffect(() => {
        // Load CSV data if a naturalness run is selected
        if (currentRound.naturalness_run && currentRound.naturalness_run.output_fpath) {
            loadNaturalnessCsvData(currentRound.naturalness_run);
        }
    }, [currentRound.naturalness_run_id, currentRound.naturalness_run, loadNaturalnessCsvData]);

    useEffect(() => {
        if (currentRound.result_activity_fpath && campaign?.id) {
            loadActivityData();
        }
    }, [currentRound.result_activity_fpath, campaign?.id, currentRound.round_number, loadActivityData]);

    const handleMeasurementChoice = async (hasMeasurements: boolean) => {
        const mode = hasMeasurements ? 'few-shot' : 'zero-shot';

        try {
            await updateCampaignRound(campaign.id!, currentRound.id, { mode });
            onRefresh();
        } catch (error) {
            notify.error('Failed to update campaign round mode');
            console.error('Error updating round mode:', error);
        }
    };

    const buildSlate = (seqIds: string[]) => {
        setSlateBuilderSeqIds(seqIds);
        setShowSlateBuilder(true);
    };

    const handleSlateConfirm = async (selectedSeqIds: string[]) => {
        Modal.confirm({
            title: 'Lock in Slate',
            icon: <ExclamationCircleOutlined />,
            width: 600,
            content: (
                <div>
                    <p>Are you sure you want to lock in these <strong>{selectedSeqIds.length} mutants</strong> as your slate?</p>
                    <p><strong>You cannot change the slate after it has been locked in.</strong> If you need to change the slate, you will need to restart the round.</p>
                    <p><em>Note that you can test whatever mutants you want, you will not be limited to the slate.</em></p>
                </div>
            ),
            okText: 'Lock in Slate',
            okType: 'primary',
            cancelText: 'Cancel',
            onOk: async () => {
                try {
                    const slateSeqIdsString = selectedSeqIds.join(',');
                    await updateCampaignRound(campaign.id!, currentRound.id, { slate_seq_ids: slateSeqIdsString });
                    onRefresh();
                    notify.success(`Added ${selectedSeqIds.length} mutants to slate`);
                } catch (error) {
                    notify.error('Failed to update slate');
                    console.error('Error updating slate:', error);
                }
            }
        });
    };


    // Auto-redirect to appropriate subpage for few-shot rounds (only when missing required state)
    useEffect(() => {
        if (currentRound.mode === 'few-shot' && !loading) {
            // If no input templates, redirect to startingSequences
            if (!currentRound.input_templates && subpage !== 'startingSequences') {
                navigate(`/campaigns/${campaign.id}/${currentRound.round_number}/startingSequences`, { replace: true });
                return;
            }

            // If has templates but no few-shot run, redirect to buildFewShotRun
            if (currentRound.input_templates && !currentRound.few_shot_run && subpage !== 'buildFewShotRun') {
                navigate(`/campaigns/${campaign.id}/${currentRound.round_number}/buildFewShotRun`, { replace: true });
                return;
            }

            // DO NOT redirect users away from subpages if they have the required state
            // Users should be able to manually navigate to subpages even when state exists
        }
    }, [currentRound, loading, subpage, navigate, campaign.id]);

    if (loading) {
        return (
            <div style={{ textAlign: 'center', padding: '48px' }}>
                <Spin size="large" />
                <div style={{ marginTop: '16px' }}>
                    <Text type="secondary">Loading workflow...</Text>
                </div>
            </div>
        );
    }

    if (!fold) {
        return <div>Error: Could not load fold data</div>;
    }

    // Determine workflow step based on round and mode
    const renderWorkflowContent = () => {
        // First round with no mode set - automatically set to zero-shot
        if (!currentRound.mode && currentRound.round_number === 1) {
            handleMeasurementChoice(false); // false = zero-shot
            return null;
        }

        // Round > 1 with no mode - default to few-shot
        if (!currentRound.mode && currentRound.round_number > 1) {
            handleMeasurementChoice(true);
            return null;
        }

        // Route to appropriate subcomponent based on mode
        if (currentRound.mode === 'zero-shot') {
            return (
                <ZeroShotCampaignRoundView
                    campaign={campaign}
                    currentRound={currentRound}
                    fold={fold}
                    naturalnessCsvData={naturalnessCsvData}
                    activityData={activityData}
                    onRefresh={onRefresh}
                    onRefreshRound={onRefreshRound}
                    buildSlate={buildSlate}
                />
            );
        } else if (currentRound.mode === 'few-shot') {
            return (
                <FewShotCampaignRoundView
                    campaign={campaign}
                    currentRound={currentRound}
                    fold={fold}
                    activityData={activityData}
                    subpage={subpage}
                    onRefresh={onRefresh}
                    onRefreshRound={onRefreshRound}
                    buildSlate={buildSlate}
                />
            );
        }

        return <div>Unknown workflow mode</div>;
    };


    return (
        <div style={{ marginBottom: '48px' }}>
            {/* Header with round info */}
            <div style={{ marginBottom: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap', marginBottom: '8px' }}>
                    <ExperimentOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
                    <Title level={2} style={{ margin: 0, wordBreak: 'break-word' }}>
                        Round {currentRound.round_number}
                        {currentRound.slate_seq_ids && (
                            <Text type="secondary" style={{ fontSize: '16px', fontWeight: 'normal', marginLeft: '12px' }}>
                                - {currentRound.slate_seq_ids.split(',').length} mutants in slate
                            </Text>
                        )}
                    </Title>
                </div>
                <Text type="secondary">
                    Started on {new Date(currentRound.date_started).toLocaleString()}
                </Text>

                {/* Few Shot Steps Navigation */}
                {currentRound.mode === 'few-shot' && (
                    <div style={{
                        marginTop: '16px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        flexWrap: 'wrap'
                    }}>
                        <span
                            style={{
                                color: (subpage === 'startingSequences') ? '#262626' : '#8c8c8c',
                                fontWeight: (subpage === 'startingSequences') ? 600 : 400,
                                cursor: 'pointer',
                                fontSize: '14px'
                            }}
                            onClick={() => navigate(`/campaigns/${campaign.id}/${currentRound.round_number}/startingSequences`)}
                        >
                            Choose Starting Mutants
                        </span>

                        <span style={{ color: '#d9d9d9' }}>→</span>

                        <span
                            style={{
                                color: (subpage === 'buildFewShotRun') ? '#262626' : '#8c8c8c',
                                fontWeight: (subpage === 'buildFewShotRun') ? 600 : 400,
                                cursor: 'pointer',
                                fontSize: '14px'
                            }}
                            onClick={() => navigate(`/campaigns/${campaign.id}/${currentRound.round_number}/buildFewShotRun`)}
                        >
                            Start FewShot Run
                        </span>

                        <span style={{ color: '#d9d9d9' }}>→</span>

                        <span
                            style={{
                                color: (!subpage && currentRound.few_shot_run) ? '#262626' : '#8c8c8c',
                                fontWeight: (!subpage && currentRound.few_shot_run) ? 600 : 400,
                                cursor: currentRound.few_shot_run ? 'pointer' : 'default',
                                fontSize: '14px'
                            }}
                            onClick={() => {
                                if (currentRound.few_shot_run) {
                                    navigate(`/campaigns/${campaign.id}/${currentRound.round_number}`, { replace: true });
                                }
                            }}
                        >
                            Results
                        </span>
                    </div>
                )}
            </div>

            {renderWorkflowContent()}

            <SlateBuilder
                open={showSlateBuilder}
                onClose={() => setShowSlateBuilder(false)}
                onConfirm={handleSlateConfirm}
                seqIds={slateBuilderSeqIds}
                title="Build Slate for Campaign"
            />
        </div>
    );
};

export default CampaignRoundComponent;
