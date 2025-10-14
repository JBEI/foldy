import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import StartingMutantsDisplay from '../shared/StartingMutantsDisplay';
import {
    Card,
    Typography,
    Button,
    Alert,
    Empty,
    Table,
    Select,
    Tooltip
} from 'antd';
import {
    PlusOutlined
} from '@ant-design/icons';
import type { Campaign, CampaignRound, Fold } from '../../types/types';
import { updateCampaignRound } from '../../api/campaignApi';
import { notify } from '../../services/NotificationService';
import { EmbeddingModal } from '../shared/EmbeddingModal';
import { FewShotModal } from '../shared/FewShotModal';
import { getEmbeddingStatus, getNaturalnessStatus, getStatusDisplay } from '../../util/statusHelpers';

const { Title, Text, Paragraph } = Typography;

interface EmbeddingSelectionRow {
    key: string;
    requirement: string;
    availableEmbeddings: any[];
    selectedEmbeddingId?: number;
    newEmbeddingTemplate: any;
}

interface EmbeddingSelectionTableProps {
    title: string;
    description: string;
    rows: EmbeddingSelectionRow[];
    selectedEmbeddings: Record<string, number>;
    onEmbeddingChange: (rowKey: string, embeddingId: number | 'new', row?: EmbeddingSelectionRow) => void;
    maybeGetRequirementsErrorMessage?: () => string | null;
}

interface FewShotRunBuilderProps {
    campaign: Campaign;
    currentRound: CampaignRound;
    fold: Fold;
    onRefresh: () => void;
    emptyTemplate: any;
    priorRoundActivityData: Array<{ seq_id: string, activity: number }> | null;
}

const FewShotRunBuilder: React.FC<FewShotRunBuilderProps> = ({
    campaign,
    currentRound,
    fold,
    onRefresh,
    emptyTemplate,
    priorRoundActivityData
}) => {
    // State management (moved from parent component)
    const [selectedEmbeddings, setSelectedEmbeddings] = useState<Record<string, number>>({});
    const [selectedNaturalnessRun, setSelectedNaturalnessRun] = useState<number | null>(null);
    const [showEmbeddingModal, setShowEmbeddingModal] = useState(false);
    const [embeddingModalTemplate, setEmbeddingModalTemplate] = useState<any>(null);
    const [showFewShotModal, setShowFewShotModal] = useState(false);
    const [fewShotTemplate, setFewShotTemplate] = useState<any>(null);

    // Variables and functions used throughout the component
    const allMatchingEmbeddings = fold?.embeddings?.filter(embedding =>
        embedding.embedding_model === campaign.embedding_model &&
        ((!embedding.domain_boundaries && !campaign.domain_boundaries) || embedding.domain_boundaries === campaign.domain_boundaries)
    ) || [];

    const inputTemplates = currentRound.input_templates?.split(',').map(t => t.trim()) || [];
    const selectedTemplates = ['WT', ...inputTemplates.filter(t => t !== 'WT')];

    // Auto-select first available embedding for few-shot step 2
    useEffect(() => {
        if (!currentRound.input_templates || !fold?.embeddings) return;

        if (allMatchingEmbeddings.length === 0) return;

        const autoSelections: Record<string, number> = {};

        // Auto-select for naturalness warmstart
        const naturalnessWarmstartEmbeddings = allMatchingEmbeddings.filter(embedding =>
            embedding.dms_starting_seq_ids?.includes('WT')
        );
        if (naturalnessWarmstartEmbeddings.length > 0 && !selectedEmbeddings['naturalness-warmstart']) {
            autoSelections['naturalness-warmstart'] = naturalnessWarmstartEmbeddings[0].id;
        }

        // Auto-select for template embeddings
        selectedTemplates.forEach(template => {
            const key = `template-${template}`;
            const templateEmbeddings = allMatchingEmbeddings.filter(embedding =>
                embedding.dms_starting_seq_ids?.includes(template)
            );
            if (templateEmbeddings.length > 0 && !selectedEmbeddings[key]) {
                autoSelections[key] = templateEmbeddings[0].id;
            }
        });

        if (Object.keys(autoSelections).length > 0) {
            setSelectedEmbeddings(prev => ({
                ...prev,
                ...autoSelections
            }));
        }
    }, [fold?.embeddings, campaign.embedding_model, currentRound.input_templates, selectedEmbeddings, allMatchingEmbeddings, selectedTemplates]);

    // Step 2: Run Selection & Results - Auto-select naturalness run
    const matchingNaturalnessRuns = fold?.naturalness_runs?.filter(run =>
        run.logit_model === campaign.naturalness_model
    ) || [];

    const handleEmbeddingModalClose = async () => {
        setShowEmbeddingModal(false);
        setEmbeddingModalTemplate(null);

        try {
            // Force refresh by calling onRefresh which should update fold data in parent
            onRefresh();
        } catch (error) {
            console.error('Error refreshing fold data:', error);
            notify.error('Failed to refresh fold data');
        }
    };

    const handleEmbeddingChange = (rowKey: string, embeddingId: number | 'new', row?: EmbeddingSelectionRow) => {
        if (embeddingId === 'new' && row) {
            setEmbeddingModalTemplate(row.newEmbeddingTemplate);
            setShowEmbeddingModal(true);
        } else if (typeof embeddingId === 'number') {
            setSelectedEmbeddings(prev => ({
                ...prev,
                [rowKey]: embeddingId
            }));
        }
    };

    // Create template FewShot object with pre-populated values
    const createTemplateFewShot = () => {
        // Get selected naturalness run
        const selectedNaturalnessRunData = matchingNaturalnessRuns.find(run =>
            run.id === selectedNaturalnessRun
        );

        // Get selected embedding paths (deduplicated)
        const embeddingPaths: string[] = [];
        const uniqueEmbeddingIds = Array.from(new Set(Object.values(selectedEmbeddings)));
        uniqueEmbeddingIds.forEach(embeddingId => {
            const embedding = allMatchingEmbeddings.find(e => e.id === embeddingId);
            if (embedding?.output_fpath) {
                embeddingPaths.push(embedding.output_fpath);
            }
        });

        console.log(`FewShot template has the following naturalness paths: ${selectedNaturalnessRunData?.output_fpath} for ${selectedNaturalnessRun}`);

        return {
            name: `${campaign.name}_R${currentRound.round_number}_fewshot`,
            mode: 'TorchMLPFewShotModel',
            num_mutants: 24,
            embedding_files: embeddingPaths.length > 0 ? embeddingPaths.join(',') : undefined,
            naturalness_files: selectedNaturalnessRunData?.output_fpath ? selectedNaturalnessRunData.output_fpath : undefined,
            finetuning_model_checkpoint: 'facebook/esm2_t6_8M_UR50D',
        };
    };

    // Function to check if all requirements are satisfied
    const areAllRequirementsSatisfied = () => {
        // Check if naturalness run is selected and complete
        if (!selectedNaturalnessRun) return false;
        const selectedNaturalnessRunData = matchingNaturalnessRuns.find(run => run.id === selectedNaturalnessRun);
        if (!selectedNaturalnessRunData) return false;
        const naturalnessStatus = getNaturalnessStatus(selectedNaturalnessRunData, fold.jobs || null);
        const naturalnessStatusDisplay = getStatusDisplay(naturalnessStatus);
        if (naturalnessStatusDisplay.text !== 'Complete') return false;

        // Check if all embeddings are selected and complete
        const requiredEmbeddingKeys = [
            'naturalness-warmstart',
            ...selectedTemplates.map(template => `template-${template}`),
            'activity-measurements'
        ];

        for (const key of requiredEmbeddingKeys) {
            const selectedEmbeddingId = selectedEmbeddings[key];
            if (!selectedEmbeddingId) return false;

            const selectedEmbedding = allMatchingEmbeddings.find(e => e.id === selectedEmbeddingId);
            if (!selectedEmbedding) return false;

            const embeddingStatus = getEmbeddingStatus(selectedEmbedding, fold.jobs || null);
            const embeddingStatusDisplay = getStatusDisplay(embeddingStatus);
            if (embeddingStatusDisplay.text !== 'Complete') return false;
        }

        return true;
    };

    const handleLaunchFewShotModal = () => {
        const template = createTemplateFewShot();
        setFewShotTemplate(template);
        console.log('Launching few shot modal with template ', template);
        setShowFewShotModal(true);
    };

    const handleFewShotCreated = async (createdFewShot: any) => {
        try {
            // Update the campaign round with the created FewShot ID
            await updateCampaignRound(campaign.id!, currentRound.id, {
                few_shot_run_id: createdFewShot.id
            });

            // Refresh to get the updated campaign round
            onRefresh();
            notify.success(`FewShot run "${createdFewShot.name}" started successfully`);
        } catch (error) {
            notify.error('Failed to update campaign round with FewShot run');
            console.error('Error updating campaign round:', error);
            // Still refresh to see if it worked
            onRefresh();
        }
    };

    // EmbeddingSelectionTable component
    const EmbeddingSelectionTable: React.FC<EmbeddingSelectionTableProps> = ({
        title,
        description,
        rows,
        selectedEmbeddings,
        onEmbeddingChange,
        maybeGetRequirementsErrorMessage
    }) => {
        const getStatusForRow = (row: EmbeddingSelectionRow) => {
            const selectedEmbeddingId = selectedEmbeddings[row.key];

            if (!selectedEmbeddingId) {
                return { color: '#ff4d4f', text: 'Unsatisfied', icon: '✗' };
            }

            const selectedEmbedding = row.availableEmbeddings.find(e => e.id === selectedEmbeddingId);
            if (!selectedEmbedding) {
                return { color: '#ff4d4f', text: 'Unsatisfied', icon: '✗' };
            }

            const embeddingStatus = getEmbeddingStatus(selectedEmbedding, fold.jobs || null);
            const statusDisplay = getStatusDisplay(embeddingStatus);

            // If the embedding is complete, the requirement is satisfied
            if (statusDisplay.text === 'Complete') {
                return { color: statusDisplay.color, text: 'Satisfied', icon: statusDisplay.icon };
            }

            // Otherwise show the actual status (Running, Failed, etc.)
            return statusDisplay;
        };

        const columns = [
            {
                title: 'Requirement',
                dataIndex: 'requirement',
                key: 'requirement',
                width: '35%',
                ellipsis: true
            },
            {
                title: 'Embedding',
                key: 'embedding',
                width: '45%',
                render: (record: EmbeddingSelectionRow) => {
                    const truncateText = (text: string, maxLength: number = 25) => {
                        return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
                    };

                    const options = [
                        ...record.availableEmbeddings.map(embedding => {
                            const embeddingStatus = getEmbeddingStatus(embedding, fold.jobs || null);
                            const statusDisplay = getStatusDisplay(embeddingStatus);
                            const truncatedName = truncateText(embedding.name);
                            return {
                                value: embedding.id,
                                label: `${truncatedName} | ${statusDisplay.icon} ${statusDisplay.text}`,
                                title: `${embedding.name} | ${statusDisplay.icon} ${statusDisplay.text}` // Full name for hover tooltip
                            };
                        }),
                        {
                            value: 'new',
                            label: '+ Start New Embedding',
                            title: 'Start a new embedding run'
                        }
                    ];

                    const selectedOption = options.find(opt => opt.value === selectedEmbeddings[record.key]);
                    const hoverText = selectedOption ? selectedOption.title || selectedOption.label : '';

                    let startingValue: number | null = selectedEmbeddings[record.key];
                    // Check if starting value exists in options
                    if (startingValue && !options.find(opt => opt.value === startingValue)) {
                        // If not found, show error and return null
                        notify.error(`Embedding ${startingValue} is not a valid option. Setting to null.`);
                        startingValue = null;
                    }

                    return (
                        <div style={{ overflow: 'hidden' }}>
                            <Tooltip title={hoverText} placement="topLeft">
                                <Select
                                    style={{ width: '100%' }}
                                    placeholder="Select embedding..."
                                    value={startingValue}
                                    onChange={(value) => onEmbeddingChange(record.key, value, record)}
                                    options={options}
                                    showSearch
                                    optionFilterProp="label"
                                    dropdownStyle={{
                                        maxWidth: '600px'
                                    }}
                                />
                            </Tooltip>
                        </div>
                    );
                }
            },
            {
                title: 'Status',
                key: 'status',
                width: '20%',
                ellipsis: true,
                render: (record: EmbeddingSelectionRow) => {
                    const status = getStatusForRow(record);
                    return <span style={{ color: status.color }}>{status.icon} {status.text}</span>;
                }
            }
        ];

        return (
            <Card size="small" title={title}>
                {maybeGetRequirementsErrorMessage && maybeGetRequirementsErrorMessage() && (
                    <Alert
                        message="Requirements Error"
                        description={maybeGetRequirementsErrorMessage()}
                        type="error"
                        style={{ marginBottom: '16px' }}
                    />
                )}
                <Paragraph>{description}</Paragraph>
                <div style={{ overflowX: 'auto' }}>
                    <Table
                        dataSource={rows}
                        columns={columns}
                        rowKey="key"
                        pagination={false}
                        size="small"
                        scroll={{ x: true }}
                    />
                </div>
            </Card>
        );
    };

    return (
        <div>
            <Card style={{ marginBottom: '20px' }}>
                <div style={{ marginBottom: '24px' }}>
                    <Title level={4}>Prepare Few-Shot Data</Title>
                    <Paragraph>
                        Configure the naturalness and embedding runs needed for few-shot predictions.
                    </Paragraph>
                </div>

                <StartingMutantsDisplay
                    campaignId={campaign.id!}
                    roundNumber={currentRound.round_number}
                    inputTemplates={currentRound.input_templates || undefined}
                    style={{ marginBottom: '24px' }}
                />
            </Card>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <Card size="small" title="A) Naturalness Run">
                    <Paragraph>Select a naturalness run to use for few-shot predictions.</Paragraph>
                    {matchingNaturalnessRuns.length > 0 ? (
                        <div style={{ overflowX: 'auto' }}>
                            <Table
                                dataSource={[{
                                    key: 'naturalness-selection',
                                    requirement: 'Naturalness predictions',
                                    availableRuns: matchingNaturalnessRuns
                                }]}
                                columns={[
                                    {
                                        title: 'Requirement',
                                        dataIndex: 'requirement',
                                        key: 'requirement',
                                        width: '35%',
                                        ellipsis: true
                                    },
                                    {
                                        title: 'Naturalness Run',
                                        key: 'naturalness',
                                        width: '45%',
                                        render: () => {
                                            const truncateText = (text: string, maxLength: number = 25) => {
                                                return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
                                            };

                                            const options = matchingNaturalnessRuns.map(run => {
                                                const naturalnessStatus = getNaturalnessStatus(run, fold.jobs || null);
                                                const statusDisplay = getStatusDisplay(naturalnessStatus);
                                                const truncatedName = truncateText(run.name);
                                                return {
                                                    value: run.id,
                                                    label: `${truncatedName} | ${statusDisplay.icon} ${statusDisplay.text}`,
                                                    title: `${run.name} | ${statusDisplay.icon} ${statusDisplay.text}`
                                                };
                                            });

                                            const selectedOption = options.find(opt => opt.value === selectedNaturalnessRun);
                                            const hoverText = selectedOption ? selectedOption.title || selectedOption.label : '';

                                            return (
                                                <div style={{ overflow: 'hidden' }}>
                                                    <Tooltip title={hoverText} placement="topLeft">
                                                        <Select
                                                            style={{ width: '100%' }}
                                                            placeholder="Select naturalness run..."
                                                            value={selectedNaturalnessRun}
                                                            onChange={(value) => setSelectedNaturalnessRun(value)}
                                                            options={options}
                                                            showSearch
                                                            optionFilterProp="label"
                                                            dropdownStyle={{
                                                                maxWidth: '600px'
                                                            }}
                                                        />
                                                    </Tooltip>
                                                </div>
                                            );
                                        }
                                    },
                                    {
                                        title: 'Status',
                                        key: 'status',
                                        width: '20%',
                                        ellipsis: true,
                                        render: () => {
                                            if (!selectedNaturalnessRun) {
                                                return <span style={{ color: '#ff4d4f' }}>❌ Unsatisfied</span>;
                                            }

                                            const selectedRun = matchingNaturalnessRuns.find(run => run.id === selectedNaturalnessRun);
                                            if (!selectedRun) {
                                                return <span style={{ color: '#ff4d4f' }}>❌ Unsatisfied</span>;
                                            }

                                            const naturalnessStatus = getNaturalnessStatus(selectedRun, fold.jobs || null);
                                            const statusDisplay = getStatusDisplay(naturalnessStatus);

                                            // If complete, show satisfied
                                            if (statusDisplay.text === 'Complete') {
                                                return <span style={{ color: statusDisplay.color }}>{statusDisplay.icon} Satisfied</span>;
                                            }

                                            // Otherwise show actual status
                                            return <span style={{ color: statusDisplay.color }}>{statusDisplay.icon} {statusDisplay.text}</span>;
                                        }
                                    }
                                ]}
                                rowKey="key"
                                pagination={false}
                                size="small"
                                scroll={{ x: true }}
                            />
                        </div>
                    ) : (
                        <Empty description="No matching naturalness runs" />
                    )}
                </Card>
                <EmbeddingSelectionTable
                    title="B) Embeddings for Naturalness Warm Start"
                    description="All single mutant variants of the WT sequence must be embedded for naturalness warm-start."
                    rows={[{
                        key: 'naturalness-warmstart',
                        requirement: 'Single mutant embeddings',
                        availableEmbeddings: allMatchingEmbeddings.filter(embedding =>
                            embedding.dms_starting_seq_ids?.split(',').some(id => id.trim() === 'WT')
                        ),
                        newEmbeddingTemplate: {
                            name: `${campaign.name}_naturalness_warmstart`,
                            embedding_model: campaign.embedding_model,
                            dms_starting_seq_ids: 'WT',
                            extra_seq_ids: '',
                            domain_boundaries: campaign.domain_boundaries
                        }
                    }]}
                    selectedEmbeddings={selectedEmbeddings}
                    onEmbeddingChange={handleEmbeddingChange}
                />
                {/* </Col>

                <Col sm={24} lg={12}> */}
                <EmbeddingSelectionTable
                    title="C) Embeddings for New Proteins"
                    description={`All candidate sequences, which are the single mutant variants of all ${selectedTemplates.length} templates, must be embedded in order to be evaluated.`}
                    rows={selectedTemplates.map(template => ({
                        key: `template-${template}`,
                        requirement: `Embeddings for ${template}`,
                        availableEmbeddings: allMatchingEmbeddings.filter(embedding =>
                            embedding.dms_starting_seq_ids?.split(',').some(id => id.trim() === template)
                        ),
                        newEmbeddingTemplate: {
                            name: `${campaign.name}_template_${template}`,
                            embedding_model: campaign.embedding_model,
                            dms_starting_seq_ids: template,
                            extra_seq_ids: '',
                            domain_boundaries: campaign.domain_boundaries
                        }
                    }))}
                    selectedEmbeddings={selectedEmbeddings}
                    onEmbeddingChange={handleEmbeddingChange}
                />
                {/* </Col>

                <Col sm={24} lg={12}> */}
                <EmbeddingSelectionTable
                    title="D) Embeddings for Activity Measurements"
                    description={`All measurements from prior round (Round ${currentRound.round_number - 1}) must be embedded in order to train the model.`}
                    rows={[{
                        key: 'activity-measurements',
                        requirement: 'Embeddings for Measured Mutants',
                        availableEmbeddings: (() => {
                            if (!priorRoundActivityData) return allMatchingEmbeddings;

                            // Get deduplicated sequence IDs from prior round activity data
                            const priorRoundSeqIds = Array.from(new Set(priorRoundActivityData.map(item => item.seq_id)));
                            const priorRoundSeqIdsSet = new Set(priorRoundSeqIds);

                            // Filter embeddings that have exactly these sequence IDs in extra_seq_ids
                            return allMatchingEmbeddings.filter(embedding => {
                                if (!embedding.extra_seq_ids) return false;

                                const embeddingSeqIds = new Set(embedding.extra_seq_ids.split(',').map(id => id.trim()).filter(id => id));

                                // Check if the sets are exactly equal (same elements, same size)
                                return embeddingSeqIds.size === priorRoundSeqIdsSet.size &&
                                    Array.from(priorRoundSeqIdsSet).every(id => embeddingSeqIds.has(id));
                            });
                        })(),
                        newEmbeddingTemplate: {
                            name: `${campaign.name}_R${currentRound.round_number - 1}_activity_measurements`,
                            embedding_model: campaign.embedding_model,
                            dms_starting_seq_ids: '',
                            extra_seq_ids: priorRoundActivityData ?
                                Array.from(new Set(priorRoundActivityData.map(item => item.seq_id))).join(',') :
                                '',
                            domain_boundaries: campaign.domain_boundaries
                        }
                    }]}
                    selectedEmbeddings={selectedEmbeddings}
                    onEmbeddingChange={handleEmbeddingChange}
                />
            </div>

            {/* FewShot Run Launcher Section */}
            <Card style={{ marginTop: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                    <span><strong>Run Few-Shot Prediction</strong></span>
                </div>

                <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                    <Title level={4}>
                        {areAllRequirementsSatisfied() ? 'Ready to Run FewShot Prediction' : 'Complete Requirements to Run FewShot Prediction'}
                    </Title>
                    <Text type="secondary" style={{ display: 'block', marginBottom: '32px' }}>
                        {areAllRequirementsSatisfied()
                            ? 'All requirements have been configured. Start the FewShot activity prediction to generate mutant recommendations.'
                            : 'Please complete all embedding selections and ensure the naturalness run is finished before starting FewShot prediction.'
                        }
                    </Text>

                    <Button
                        type="primary"
                        size="large"
                        icon={<PlusOutlined />}
                        onClick={handleLaunchFewShotModal}
                        disabled={!areAllRequirementsSatisfied()}
                        style={{
                            ...(areAllRequirementsSatisfied() && { backgroundColor: '#1890ff' }),
                            fontSize: '16px',
                            height: '48px',
                            padding: '0 32px'
                        }}
                    >
                        Run FewShot Activity Prediction and Slate Builder
                    </Button>
                </div>
            </Card>


            <EmbeddingModal
                key={embeddingModalTemplate ? JSON.stringify(embeddingModalTemplate) : 'defaultEmbeddingModal'}
                open={showEmbeddingModal}
                onClose={handleEmbeddingModalClose}
                foldIds={[campaign.fold_id]}
                title="Start Embedding Run for Few-Shot"
                templateEmbedding={embeddingModalTemplate}
            />

            <FewShotModal
                open={showFewShotModal}
                onClose={(createdFewShot?: any) => {
                    setShowFewShotModal(false);
                    setFewShotTemplate(null);

                    // If a FewShot was created, update the campaign round with its ID
                    console.log('Closing the few shot modal with createdFewShot ', createdFewShot);
                    if (createdFewShot) {
                        handleFewShotCreated(createdFewShot);
                    } else {
                        onRefresh(); // Fallback to full refresh
                    }
                }}
                foldId={campaign.fold_id}
                files={null} // Files not available in current Fold type
                evolutions={fold.few_shots || null}
                campaignRounds={campaign.rounds || null}
                title={`FewShot Run - ${campaign.name} Round ${currentRound.round_number}`}
                templateFewShot={fewShotTemplate || emptyTemplate}
                defaultActivityFileSource="campaign"
                fewShotCampaignRoundIdForActivityFile={(() => {
                    // Find the previous round with activity data
                    const priorRound = campaign.rounds?.find(round =>
                        round.round_number === currentRound.round_number - 1 && round.result_activity_fpath
                    );
                    return priorRound?.id;
                })()}
            />
        </div>
    );
};

export default FewShotRunBuilder;
