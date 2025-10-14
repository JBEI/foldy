import React, { useState } from 'react';
import {
    Card,
    Typography,
    Button,
    Alert,
    Table,
    Upload,
    Form
} from 'antd';
import {
    UploadOutlined,
    ExperimentOutlined,
    DownloadOutlined
} from '@ant-design/icons';
import { FaRocket } from 'react-icons/fa';
import Plot from 'react-plotly.js';
import { DnaBuildModal } from './DnaBuildModal';
import type { CampaignRound } from '../../types/types';

const { Title, Text } = Typography;

interface MutantSlateCardProps {
    currentRound: CampaignRound;
    activityData: Array<{ seq_id: string, activity: number }> | null;
    onActivityFileUpload: (file: File) => Promise<boolean>;
    uploadingActivity: boolean;
    showActivityPlot?: boolean;
    foldId?: string;
    hasNextRound?: boolean;
}

const MutantSlateCard: React.FC<MutantSlateCardProps> = ({
    currentRound,
    activityData,
    onActivityFileUpload,
    uploadingActivity,
    showActivityPlot = true,
    foldId,
    hasNextRound = false
}) => {
    const [dnaBuildModalOpen, setDnaBuildModalOpen] = useState(false);
    const slateSeqIds = currentRound.slate_seq_ids ? currentRound.slate_seq_ids.split(',').map(id => id.trim()) : [];
    const hasSlate = slateSeqIds.length > 0;
    const hasActivityFile = !!currentRound.result_activity_fpath;

    // Create activity data lookup
    const activityLookup = activityData ? activityData.reduce((acc, item) => {
        const baseName = item.seq_id;
        if (!acc[baseName]) {
            acc[baseName] = [];
        }
        acc[baseName].push(item.activity);
        return acc;
    }, {} as Record<string, number[]>) : {};

    // Get mean activity for a sequence
    const getMeanActivity = (seqId: string) => {
        const activities = activityLookup[seqId];
        if (!activities || activities.length === 0) return null;
        return activities.reduce((sum, activity) => sum + activity, 0) / activities.length;
    };

    // Parse sequence ID into alleles (mutations)
    const parseSequenceId = (seqId: string): Set<string> => {
        // Handle WT as special case (no mutations)
        if (seqId.toUpperCase() === 'WT') {
            return new Set();
        }

        // Split by underscores to get individual alleles
        return new Set(seqId.split('_'));
    };

    // Find parent sequence for a given sequence
    const findParent = (seqId: string, inputTemplates: string[]): string => {

        // Return unknown for homology-based sequences
        if (seqId.startsWith('HOM_')) {
            return 'unknown';
        }

        const seqAlleles = parseSequenceId(seqId);

        // For each input template, check if it's exactly one mutation away
        for (const template of inputTemplates) {
            const templateAlleles = parseSequenceId(template);

            // Calculate number of mutations that must be added to the template to get to
            // seq, or added to seq to get to template.
            const extraAllelesInSeq = Array.from(seqAlleles).filter(allele => !templateAlleles.has(allele));
            const extraAllelesInTemplate = Array.from(templateAlleles).filter(allele => !seqAlleles.has(allele));

            // If exactly one mutation must be added to seq to get to template,
            // then this is the parent.
            if (extraAllelesInSeq.length === 1 && extraAllelesInTemplate.length === 0) {
                return template;
            }

            // If exactly one mutation must be added to template to get to seq,
            // that's a weird case but allowed.
            if (extraAllelesInSeq.length === 0 && extraAllelesInTemplate.length === 1) {
                return template;
            }

            // If exactly one residue must be changed to get from one to the other, that
            // is allowed too.
            if (extraAllelesInSeq.length === 1 && extraAllelesInTemplate.length === 1) {
                const seqDiffLocus = extraAllelesInSeq[0].slice(1, -1);
                const templateDiffLocus = extraAllelesInTemplate[0].slice(1, -1);
                if (seqDiffLocus === templateDiffLocus) {
                    return template;
                }
            }
        }

        return 'unknown';
    };

    // Get parent for a sequence
    const getParent = (seqId: string) => {
        if (!currentRound.input_templates) return 'unknown';
        const inputTemplates = currentRound.input_templates.split(',').map(id => id.trim()).filter(id => id.length > 0);
        return findParent(seqId, inputTemplates);
    };

    // Combine slate sequences with activity data (including extra sequences from activity)
    const allSequences = new Set([...slateSeqIds]);
    if (activityData) {
        activityData.forEach(item => {
            const baseName = item.seq_id;
            allSequences.add(baseName);
        });
    }

    const tableData = Array.from(allSequences).map(seqId => ({
        seq_id: seqId,
        in_slate: slateSeqIds.includes(seqId),
        parent: getParent(seqId),
        mean_activity: getMeanActivity(seqId)
    }));

    return (
        <Card
            title={
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span>Selected Slate & Results</span>
                    {currentRound.slate_seq_ids && (
                        <Text type="secondary">({currentRound.slate_seq_ids.split(',').length} in slate)</Text>
                    )}
                </div>
            }
        >
            {hasSlate || hasActivityFile ? (
                <div>
                    <div style={{ marginBottom: '16px' }}>
                        <Text strong>Sequence Overview</Text>
                    </div>
                    <Table
                        dataSource={tableData}
                        columns={[
                            {
                                title: 'Sequence ID',
                                dataIndex: 'seq_id',
                                key: 'seq_id',
                                render: (seqId: string) => (
                                    <Text style={{
                                        fontFamily: 'Monaco, Consolas, "Courier New", monospace',
                                        fontSize: '12px',
                                        backgroundColor: '#f5f5f5',
                                        padding: '2px 6px',
                                        borderRadius: '4px',
                                        color: '#595959'
                                    }}>
                                        {seqId}
                                    </Text>
                                )
                            },
                            {
                                title: 'In Slate',
                                dataIndex: 'in_slate',
                                key: 'in_slate',
                                render: (inSlate: boolean) => (
                                    <input
                                        type="checkbox"
                                        checked={inSlate}
                                        readOnly
                                        style={{ cursor: 'not-allowed' }}
                                    />
                                ),
                                width: 80,
                                align: 'center'
                            },
                            {
                                title: 'Parent',
                                dataIndex: 'parent',
                                key: 'parent',
                                render: (parent: string) => (
                                    <Text style={{
                                        fontFamily: 'Monaco, Consolas, "Courier New", monospace',
                                        fontSize: '12px',
                                        backgroundColor: parent === 'unknown' ? '#fff2e6' : '#f0f7ff',
                                        padding: '2px 6px',
                                        borderRadius: '4px',
                                        color: parent === 'unknown' ? '#d46b08' : '#0050b3'
                                    }}>
                                        {parent}
                                    </Text>
                                ),
                                sorter: (a, b) => a.parent.localeCompare(b.parent),
                            },
                        ]}
                        rowKey="seq_id"
                        pagination={false}
                        size="small"
                        scroll={{ y: 300 }}
                    />

                    {/* DNA Build Section - only show if Teselagen is enabled */}
                    {hasSlate && foldId && import.meta.env.VITE_TESELAGEN_BACKEND_URL && (
                        <div style={{ marginTop: '24px', padding: '16px', backgroundColor: '#fafafa', borderRadius: '6px' }}>
                            <div style={{ marginBottom: '12px' }}>
                                <Text strong>DNA Build & Teselagen Integration</Text>
                                <br />
                                <Text type="secondary">Send selected mutants to Teselagen for automated primer design and Gibson assembly.</Text>
                            </div>
                            <Button
                                icon={<ExperimentOutlined />}
                                onClick={() => setDnaBuildModalOpen(true)}
                                type="primary"
                            >
                                Send DNA Build to Teselagen
                            </Button>
                        </div>
                    )}

                    {/* Activity Upload Section */}
                    {hasActivityFile ? null : (
                        <div style={{ marginTop: '24px', padding: '16px', backgroundColor: '#fafafa', borderRadius: '6px' }}>
                            <div style={{ marginBottom: '12px' }}>
                                <Text strong>Upload Activity Results</Text>
                                <br />
                                <Text type="secondary">Upload experimental results. Note that <strong>you should include all compatible data thusfar, not just the mutants tested in this round!</strong> This data will be used to train a model in the next round.</Text>
                                <br />
                                <a
                                    href="/example_activity_file.xlsx"
                                    download="example_activity_file.xlsx"
                                    style={{
                                        color: '#1890ff',
                                        textDecoration: 'none',
                                        fontSize: '13px',
                                        marginTop: '8px',
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        gap: '4px'
                                    }}
                                >
                                    <DownloadOutlined style={{ fontSize: '12px' }} />
                                    Download template activity file
                                </a>
                            </div>
                            <Form.Item style={{ marginBottom: 0 }}>
                                <Upload
                                    beforeUpload={onActivityFileUpload}
                                    accept=".xlsx,.xls"
                                    maxCount={1}
                                    showUploadList={false}
                                >
                                    <Button
                                        icon={<UploadOutlined />}
                                        loading={uploadingActivity}
                                        type="primary"
                                    >
                                        {uploadingActivity ? 'Uploading...' : 'Upload Activity Results (.xlsx)'}
                                    </Button>
                                </Upload>
                            </Form.Item>
                        </div>
                    )}

                    {/* Activity Plot */}
                    {showActivityPlot && activityData && activityData.length > 0 && (
                        <div style={{ marginTop: '24px' }}>
                            <Title level={5}>Activity Visualization</Title>
                            <Plot
                                data={(() => {
                                    // Group sequences by base name
                                    const groups = activityData.reduce((acc, item) => {
                                        const baseName = item.seq_id;
                                        if (!acc[baseName]) {
                                            acc[baseName] = [];
                                        }
                                        acc[baseName].push(item);
                                        return acc;
                                    }, {} as Record<string, Array<{ seq_id: string, activity: number }>>);

                                    // Calculate averages and sort by average activity
                                    const groupAverages = Object.entries(groups).map(([baseName, items]) => ({
                                        baseName,
                                        items,
                                        avgActivity: items.reduce((sum, item) => sum + item.activity, 0) / items.length
                                    })).sort((a, b) => b.avgActivity - a.avgActivity);

                                    return [
                                        // Bar chart for averages
                                        {
                                            x: groupAverages.map(g => g.baseName),
                                            y: groupAverages.map(g => g.avgActivity),
                                            type: 'bar',
                                            name: 'Average Activity',
                                            marker: {
                                                color: groupAverages.map(g =>
                                                    g.baseName.toUpperCase().includes('WT') ? '#fa8c16' : '#1890ff'
                                                ),
                                                opacity: 0.8
                                            },
                                            text: groupAverages.map(g => g.avgActivity.toFixed(3)),
                                            textposition: 'outside',
                                            hovertemplate: '<b>%{x}</b><br>Average Activity: %{y:.4f}<br>Count: ' +
                                                groupAverages.map(g => g.items.length).join(',').split(',')[0] + '<extra></extra>',
                                        },
                                        // Scatter plot for individual points
                                        {
                                            x: activityData.map(d => d.seq_id),
                                            y: activityData.map(d => d.activity),
                                            mode: 'markers',
                                            type: 'scatter',
                                            name: 'Individual Measurements',
                                            marker: {
                                                color: activityData.map(d =>
                                                    d.seq_id.toUpperCase().includes('WT') ? '#fa8c16' : '#1890ff'
                                                ),
                                                size: 6,
                                                opacity: 0.6,
                                                line: { color: 'white', width: 1 }
                                            },
                                            hovertemplate: '<b>%{text}</b><br>Activity: %{y:.4f}<extra></extra>',
                                            text: activityData.map(d => d.seq_id)
                                        }
                                    ];
                                })()}
                                layout={{
                                    title: {
                                        text: 'Activity by Sequence',
                                        font: { size: 14, color: '#262626' }
                                    },
                                    xaxis: {
                                        title: { text: 'Sequence ID', font: { size: 12, color: '#595959' } },
                                        tickangle: -45,
                                        tickfont: { size: 10, color: '#8c8c8c' },
                                        gridcolor: '#f0f0f0',
                                        linecolor: '#d9d9d9'
                                    },
                                    yaxis: {
                                        title: { text: 'Activity', font: { size: 12, color: '#595959' } },
                                        tickfont: { size: 10, color: '#8c8c8c' },
                                        gridcolor: '#f0f0f0',
                                        linecolor: '#d9d9d9'
                                    },
                                    legend: {
                                        orientation: 'h',
                                        x: 0.5,
                                        xanchor: 'center',
                                        y: -0.25,
                                        yanchor: 'top',
                                        font: { size: 11, color: '#595959' }
                                    },
                                    margin: { l: 60, r: 30, t: 50, b: 120 },
                                    plot_bgcolor: '#fafafa',
                                    paper_bgcolor: 'white',
                                    font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' }
                                }}
                                config={{
                                    displayModeBar: true,
                                    displaylogo: false,
                                    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d']
                                }}
                                style={{ width: '100%', height: '400px' }}
                            />
                        </div>
                    )}

                    {/* Re-upload Activity Data Button */}
                    {hasActivityFile && (
                        <div style={{ marginTop: '16px', textAlign: 'center' }}>
                            <Upload
                                beforeUpload={onActivityFileUpload}
                                accept=".xlsx,.xls"
                                maxCount={1}
                                showUploadList={false}
                            >
                                <Button
                                    loading={uploadingActivity}
                                    icon={<UploadOutlined />}
                                    disabled={hasNextRound}
                                    title={hasNextRound ? "Cannot re-upload activity data when next round exists" : "Re-upload activity data"}
                                >
                                    {uploadingActivity ? 'Uploading...' : 'Re-upload Activity Data'}
                                </Button>
                            </Upload>
                        </div>
                    )}
                </div>
            ) : (
                <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                    <FaRocket style={{ fontSize: '48px', color: '#d9d9d9' }} />
                    <Title level={5} type="secondary">No slate selected yet.</Title>
                </div>
            )}

            {/* DNA Build Modal - only show if Teselagen is enabled */}
            {foldId && import.meta.env.VITE_TESELAGEN_BACKEND_URL && (
                <DnaBuildModal
                    open={dnaBuildModalOpen}
                    onClose={() => setDnaBuildModalOpen(false)}
                    foldId={foldId}
                    defaultSeqIds={slateSeqIds}
                />
            )}
        </Card>
    );
};

export default MutantSlateCard;
