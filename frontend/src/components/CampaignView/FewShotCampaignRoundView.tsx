import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Card,
    Typography,
    Button,
    Spin,
    Space,
    Table,
    Tooltip
} from 'antd';
import {
    CheckCircleOutlined,
    FileTextOutlined,
    DeleteOutlined,
    PlusOutlined
} from '@ant-design/icons';
import type { Campaign, CampaignRound, Fold, FewShot } from '../../types/types';
import { updateCampaignRound, uploadCampaignRoundActivityFile, getCampaignRoundActivityData } from '../../api/campaignApi';
import { getFewShotDebugInfo, getFewShotPredictedSlate, SlateData, deleteFewShot } from '../../api/fewShotApi';
import { notify } from '../../services/NotificationService';
import FewShotMutantTable from '../shared/FewShotMutantTable';
import FewShotDebugPlots from '../shared/FewShotDebugPlots';
import MutantSlateCard from '../shared/MutantSlateCard';
import InputTemplateSelection from './InputTemplateSelection';
import FewShotRunBuilder from './FewShotRunBuilder';
import StartingMutantsDisplay from '../shared/StartingMutantsDisplay';
import { getStatusDisplay, getFewShotStatus } from '../../util/statusHelpers';
import { Selection } from '../FoldView/StructurePane';

const { Title, Text } = Typography;

interface FewShotCampaignRoundViewProps {
    campaign: Campaign;
    currentRound: CampaignRound;
    fold: Fold;
    activityData: Array<{ seq_id: string, activity: number }> | null;
    subpage?: string;
    onRefresh: () => void;
    onRefreshRound?: () => void;
    buildSlate?: (seqIds: string[]) => void;
}


interface FewShotResultsContentProps {
    fewShotRun: FewShot;
    fold: Fold;
    campaign: Campaign;
    setSelectedSubsequence: (selection: Selection | null) => void;
    buildSlate?: (seqIds: string[]) => void;
    disableSlateBuilder?: boolean;
}

const FewShotResultsContent: React.FC<FewShotResultsContentProps> = ({
    fewShotRun,
    fold,
    campaign,
    setSelectedSubsequence,
    buildSlate,
    disableSlateBuilder
}) => {
    const [slateData, setSlateData] = useState<SlateData[] | null>(null);
    const [debugData, setDebugData] = useState<any>(null);
    const [sortOptions, setSortOptions] = useState<{ [key: string]: string[] } | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadFewShotData = async () => {
            setLoading(true);
            try {
                // Load slate data from new API
                if (!fewShotRun.id) {
                    throw new Error('FewShot run has no ID');
                }
                const slateResponse = await getFewShotPredictedSlate(fewShotRun.id, {
                    selectedOnly: true
                });
                setSlateData(slateResponse.data);

                // Load debug data using helper function
                const { debugData, sortOptions } = await getFewShotDebugInfo(campaign.fold_id, fewShotRun);
                setDebugData(debugData);
                setSortOptions(sortOptions);
            } catch (error) {
                console.error('Error loading FewShot data:', error);
                notify.error(`Failed to load FewShot results: ${error}`);
            } finally {
                setLoading(false);
            }
        };

        loadFewShotData();
    }, [fewShotRun.id, campaign.fold_id]);

    if (loading) {
        return (
            <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                <Spin size="large" />
                <div style={{ marginTop: '16px' }}>
                    <Text type="secondary">Loading FewShot results...</Text>
                </div>
            </div>
        );
    }

    return (
        <div>
            {slateData && (
                <div style={{ marginBottom: '24px' }}>
                    <FewShotMutantTable
                        yamlConfig={fold?.yaml_config || null}
                        slateData={slateData}
                        setSelectedSubsequence={setSelectedSubsequence}
                        sortOptions={sortOptions}
                        onBuildSlate={buildSlate}
                        disableSlateBuilder={disableSlateBuilder}
                    />
                </div>
            )}

            {debugData && (
                <div>
                    <h3>Training Metrics</h3>
                    <FewShotDebugPlots debugData={debugData} />
                </div>
            )}

            {!slateData && !debugData && (
                <div style={{ textAlign: 'center', padding: '20px' }}>
                    <Text type="secondary">No results data available</Text>
                </div>
            )}
        </div>
    );
};

const FewShotCampaignRoundView: React.FC<FewShotCampaignRoundViewProps> = ({
    campaign,
    currentRound,
    fold,
    activityData,
    subpage,
    onRefresh,
    buildSlate
}) => {
    const navigate = useNavigate();
    // Stable empty template reference for "new" FewShot mode
    const emptyTemplate = useMemo(() => ({}), []);

    // All hooks must be declared at the top level
    const [uploadingActivity, setUploadingActivity] = useState(false);
    const [selectedSeqIds, setSelectedSeqIds] = useState<string[]>([]);
    const [savingTemplates, setSavingTemplates] = useState(false);
    const [allPriorRoundsActivity, setAllPriorRoundsActivity] = useState<Array<{ seq_id: string, activity: number, round_number: number }>>([]);
    const [loadingActivityData, setLoadingActivityData] = useState(false);
    // Embedding state moved to FewShotRunBuilder component
    const [priorRoundActivityData, setPriorRoundActivityData] = useState<Array<{ seq_id: string, activity: number }> | null>(null);
    const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
    const [nameFilter, setNameFilter] = useState<string | null>(null);


    // All useEffect hooks must come before any conditional returns
    useEffect(() => {
        const handleResize = () => {
            setIsMobile(window.innerWidth < 768);
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    useEffect(() => {
        const fetchPriorRoundActivityData = async () => {
            if (currentRound.round_number <= 1) {
                setPriorRoundActivityData(null);
                return;
            }

            try {
                // Find the previous round
                const priorRound = campaign.rounds?.find(round =>
                    round.round_number === currentRound.round_number - 1
                );

                if (!priorRound || !priorRound.result_activity_fpath) {
                    console.warn('No prior round with activity data found');
                    setPriorRoundActivityData(null);
                    return;
                }

                const response = await getCampaignRoundActivityData(campaign.id!, priorRound.round_number);
                console.log('Prior round activity data:', response.data);
                setPriorRoundActivityData(response.data);
            } catch (error) {
                console.error('Failed to fetch prior round activity data:', error);
                setPriorRoundActivityData(null);
            }
        };

        fetchPriorRoundActivityData();
    }, [campaign.id, campaign.rounds, currentRound.round_number]);

    // Auto-embedding selection useEffect moved to FewShotRunBuilder component

    // Load activity data from all prior rounds for template selection
    useEffect(() => {
        const loadAllPriorRoundsActivity = async () => {
            if (!campaign.rounds || campaign.rounds.length <= 1) return;

            setLoadingActivityData(true);
            try {
                const allActivity: Array<{ seq_id: string, activity: number, round_number: number }> = [];

                // Get activity data from all prior rounds
                const priorRounds = campaign.rounds.filter(round =>
                    round.round_number < currentRound.round_number && round.result_activity_fpath
                );

                for (const round of priorRounds) {
                    try {
                        const activityResponse = await getCampaignRoundActivityData(campaign.id!, round.round_number);
                        activityResponse.data.forEach(item => {
                            allActivity.push({
                                seq_id: item.seq_id,
                                activity: item.activity,
                                round_number: round.round_number
                            });
                        });
                    } catch (error) {
                        console.warn(`Failed to load activity data for round ${round.round_number}:`, error);
                    }
                }

                setAllPriorRoundsActivity(allActivity);
            } catch (error) {
                console.error('Error loading prior rounds activity data:', error);
                notify.error('Failed to load activity data from prior rounds');
            } finally {
                setLoadingActivityData(false);
            }
        };

        loadAllPriorRoundsActivity();
    }, [campaign.id, campaign.rounds, currentRound.round_number]);

    // Get all FewShot runs for this fold, sorted by date (newest first) and filtered by name
    const availableFewShots = useMemo(() => {
        if (!fold.few_shots) return [];

        let filteredShots = [...fold.few_shots];

        // Apply name filter
        if (nameFilter === 'selected') {
            filteredShots = filteredShots.filter(shot => shot.id === currentRound.few_shot_run?.id);
        } else if (nameFilter === 'round') {
            const roundPrefix = `${campaign.name}_R${currentRound.round_number}`;
            filteredShots = filteredShots.filter(shot => shot.name.startsWith(roundPrefix));
        }
        // 'all' filter shows everything, so no additional filtering needed

        return filteredShots.sort((a, b) => {
            if (!a.date_created && !b.date_created) return 0;
            if (!a.date_created) return 1;
            if (!b.date_created) return -1;
            return new Date(b.date_created).getTime() - new Date(a.date_created).getTime();
        });
    }, [fold.few_shots, nameFilter, currentRound.few_shot_run?.id, campaign.name, currentRound.round_number]);

    // Initialize selected seq IDs from current round's input templates
    useEffect(() => {
        setSelectedSeqIds(currentRound.input_templates?.split(',') || []);
    }, [currentRound.input_templates]);



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



    const handleSaveTemplateSelection = async () => {
        setSavingTemplates(true);
        try {
            const templatesCsv = selectedSeqIds.join(',');

            await updateCampaignRound(campaign.id!, currentRound.id, {
                input_templates: templatesCsv
            });

            onRefresh();
            notify.success(`Selected ${selectedSeqIds.length} mutants to pass into the next round`);
            // Navigate to buildFewShotRun after saving templates
            navigate(`/campaigns/${campaign.id}/${currentRound.round_number}/buildFewShotRun`);
        } catch (error) {
            notify.error('Failed to save template selection');
            console.error('Error saving templates:', error);
        } finally {
            setSavingTemplates(false);
        }
    };


    // Render based on subpage
    if (subpage === 'startingSequences') {

        return (
            <InputTemplateSelection
                allPriorRoundsActivity={allPriorRoundsActivity}
                loadingActivityData={loadingActivityData}
                savingTemplates={savingTemplates}
                selectedSeqIds={selectedSeqIds}
                setSelectedSeqIds={!currentRound.slate_seq_ids ? setSelectedSeqIds : undefined}
                onSaveTemplateSelection={handleSaveTemplateSelection}
                isEditable={!currentRound.slate_seq_ids}
            />
        );
    }

    if (subpage === 'buildFewShotRun') {
        return (
            <FewShotRunBuilder
                campaign={campaign}
                currentRound={currentRound}
                fold={fold}
                onRefresh={onRefresh}
                emptyTemplate={emptyTemplate}
                priorRoundActivityData={priorRoundActivityData}
            />
        );
    }


    // Helper functions for FewShot management
    const handleSelectFewShot = async (fewShotId: number) => {
        try {
            await updateCampaignRound(campaign.id!, currentRound.id, {
                few_shot_run_id: fewShotId
            });
            notify.success('FewShot run selected successfully');
            onRefresh();
        } catch (error: any) {
            notify.error(error.response?.data?.message || 'Failed to select FewShot run');
        }
    };

    const handleDeleteFewShot = async (fewShotId: number) => {
        try {
            await deleteFewShot(fewShotId);
            notify.success('FewShot run deleted successfully');
            // If this was the selected FewShot, clear the selection
            if (currentRound.few_shot_run?.id === fewShotId) {
                await updateCampaignRound(campaign.id!, currentRound.id, {
                    few_shot_run_id: null
                });
            }
            onRefresh();
        } catch (error: any) {
            notify.error(error.response?.data?.message || 'Failed to delete FewShot run');
        }
    };

    const openFewShotLogs = (fewShot: FewShot) => {
        if (fewShot.invokation_id) {
            window.open(`/fold/${campaign.fold_id}/logs#logs_${fewShot.invokation_id}`, '_blank');
        }
    };


    // Default view - show results if no subpage and few_shot_run exists
    if (!currentRound.few_shot_run) {
        return (
            <div>No matching workflow state</div>
        );
    }
    const fewShotStatus = getFewShotStatus(currentRound.few_shot_run, fold.jobs || null);
    const statusDisplay = getStatusDisplay(fewShotStatus);
    const isComplete = statusDisplay.text === 'Complete';
    console.log('currentRound.few_shot_run', currentRound.few_shot_run);


    return (
        <div>
            {/* Comprehensive FewShot Management Card */}
            <Card style={{ marginBottom: '20px' }}>
                <div style={{ marginBottom: '24px' }}>
                    <Title level={4}>FewShot Runs</Title>

                    <StartingMutantsDisplay
                        campaignId={campaign.id!}
                        roundNumber={currentRound.round_number}
                        inputTemplates={currentRound.input_templates || undefined}
                        style={{ marginBottom: '20px' }}
                    />

                    {/* FewShot runs table */}
                    <div style={{ overflowX: 'auto' }}>
                        <Table
                            dataSource={availableFewShots}
                            rowKey="id"
                            size="small"
                            pagination={false}
                            scroll={{ x: true }}
                            onChange={(_, filters) => {
                                if (filters.name && filters.name.length > 0) {
                                    setNameFilter(filters.name[0] as string);
                                } else {
                                    setNameFilter(null);
                                }
                            }}
                            columns={[
                                {
                                    title: 'Name',
                                    dataIndex: 'name',
                                    key: 'name',
                                    ellipsis: true,
                                    filters: [
                                        { text: 'Just selected few shot run', value: 'selected' },
                                        { text: 'Few shot runs for this round', value: 'round' },
                                        { text: 'All few shot runs', value: 'all' }
                                    ],
                                    filteredValue: nameFilter ? [nameFilter] : null,
                                    onFilter: () => true, // Filtering is handled in useMemo
                                    render: (name: string, record: FewShot) => (
                                        <span style={{
                                            fontWeight: record.id === currentRound.few_shot_run?.id ? 600 : 'normal'
                                        }}>
                                            {name}
                                        </span>
                                    )
                                },
                                {
                                    title: 'Date',
                                    key: 'date',
                                    width: 100,
                                    render: (_, fewShot: FewShot) => {
                                        if (!fewShot.date_created) return "N/A";
                                        try {
                                            const date = new Date(fewShot.date_created);
                                            return new Intl.DateTimeFormat('en-US', {
                                                dateStyle: "short",
                                                timeZone: "America/Los_Angeles"
                                            }).format(date);
                                        } catch {
                                            return "Error";
                                        }
                                    }
                                },
                                {
                                    title: 'Status',
                                    key: 'status',
                                    width: 120,
                                    render: (_, fewShot: FewShot) => {
                                        const status = getFewShotStatus(fewShot, fold.jobs || null);
                                        const display = getStatusDisplay(status);
                                        return (
                                            <span style={{ color: display.color }}>
                                                {display.icon} {display.text}
                                            </span>
                                        );
                                    }
                                },
                                {
                                    title: 'Actions',
                                    key: 'actions',
                                    width: 280,
                                    render: (_, fewShot: FewShot) => {
                                        const isCurrentlySelected = fewShot.id === currentRound.few_shot_run?.id;
                                        const fewShotStatusForAction = getFewShotStatus(fewShot, fold.jobs || null);
                                        const statusDisplayForAction = getStatusDisplay(fewShotStatusForAction);
                                        const isCompleteRun = statusDisplayForAction.text === 'Complete';

                                        return (
                                            <Space>
                                                {isCurrentlySelected ? (
                                                    <Button
                                                        size="small"
                                                        disabled
                                                        style={{
                                                            backgroundColor: '#f5f5f5',
                                                            borderColor: '#d9d9d9',
                                                            color: '#8c8c8c'
                                                        }}
                                                    >
                                                        {currentRound.slate_seq_ids ? 'Locked' : 'Selected'}
                                                    </Button>
                                                ) : isCompleteRun && !currentRound.slate_seq_ids ? (
                                                    <Button
                                                        type="primary"
                                                        size="small"
                                                        icon={<CheckCircleOutlined />}
                                                        onClick={() => handleSelectFewShot(fewShot.id!)}
                                                    >
                                                        Select
                                                    </Button>
                                                ) : isCompleteRun && currentRound.slate_seq_ids ? (
                                                    <Button
                                                        size="small"
                                                        disabled
                                                        style={{
                                                            backgroundColor: '#f5f5f5',
                                                            borderColor: '#d9d9d9',
                                                            color: '#8c8c8c'
                                                        }}
                                                    >
                                                        Locked
                                                    </Button>
                                                ) : null}
                                                <Tooltip title="View logs">
                                                    <Button
                                                        type="text"
                                                        size="small"
                                                        icon={<FileTextOutlined />}
                                                        onClick={() => openFewShotLogs(fewShot)}
                                                    />
                                                </Tooltip>
                                                <Tooltip title="Delete">
                                                    <Button
                                                        type="text"
                                                        size="small"
                                                        danger
                                                        icon={<DeleteOutlined />}
                                                        onClick={() => {
                                                            if (window.confirm(`Are you sure you want to delete FewShot run "${fewShot.name}"?`)) {
                                                                handleDeleteFewShot(fewShot.id!);
                                                            }
                                                        }}
                                                        disabled={isCurrentlySelected}
                                                    />
                                                </Tooltip>
                                            </Space>
                                        );
                                    }
                                }
                            ]}
                        />
                    </div>

                    {/* Start New Few Shot Run button */}
                    <div style={{ marginTop: '16px', textAlign: 'center' }}>
                        <Button
                            type="primary"
                            icon={<PlusOutlined />}
                            onClick={() => navigate(`/campaigns/${campaign.id}/${currentRound.round_number}/buildFewShotRun`)}
                        >
                            Start New Few Shot Run
                        </Button>
                    </div>
                </div>
            </Card>

            {/* Existing two-column grid layout */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr',
                gap: '20px',
                alignItems: 'start'
            }}>
                <Card
                    title={
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ color: statusDisplay.color }}>{statusDisplay.icon}</span>
                            <span>FewShot Results: {currentRound.few_shot_run.name}</span>
                        </div>
                    }
                >
                    {isComplete ? (
                        <FewShotResultsContent
                            fewShotRun={currentRound.few_shot_run}
                            fold={fold}
                            campaign={campaign}
                            setSelectedSubsequence={() => { }} // TODO: Implement if needed
                            buildSlate={buildSlate}
                            disableSlateBuilder={!!currentRound.slate_seq_ids}
                        />
                    ) : (
                        <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                            <div style={{ fontSize: '48px', marginBottom: '16px' }}>
                                {statusDisplay.icon}
                            </div>
                            <Title level={4} style={{ color: statusDisplay.color }}>
                                {statusDisplay.text}
                            </Title>
                            <Text type="secondary">
                                FewShot run is {statusDisplay.text.toLowerCase()}. Please wait for it to complete.
                            </Text>
                            <div style={{ marginTop: '16px', display: 'flex', gap: '8px', justifyContent: 'center' }}>
                                <Button
                                    type="default"
                                    size="small"
                                    onClick={onRefresh}
                                >
                                    Refresh
                                </Button>
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
                        hasNextRound={campaign.rounds?.some(round => round.round_number === currentRound.round_number + 1) || false}
                    />
                </div>
            </div>
        </div>
    );
};

export default FewShotCampaignRoundView;
