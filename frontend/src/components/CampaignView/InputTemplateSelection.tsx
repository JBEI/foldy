import React, { useState } from 'react';
import {
    Card,
    Typography,
    Button,
    Alert,
    Spin,
    Input,
    Tooltip
} from 'antd';
import Plot from 'react-plotly.js';
import { Data } from 'plotly.js';

const { Title, Text, Paragraph } = Typography;

interface InputTemplateSelectionProps {
    allPriorRoundsActivity: Array<{ seq_id: string, activity: number, round_number: number }>;
    loadingActivityData: boolean;
    savingTemplates: boolean;
    selectedSeqIds: string[];
    setSelectedSeqIds?: (seqIds: string[]) => void;
    onSaveTemplateSelection: () => void;
    isEditable?: boolean;
}

const InputTemplateSelection: React.FC<InputTemplateSelectionProps> = ({
    allPriorRoundsActivity,
    loadingActivityData,
    savingTemplates,
    selectedSeqIds,
    setSelectedSeqIds,
    onSaveTemplateSelection,
    isEditable = true
}) => {
    const [isManualEditing, setIsManualEditing] = useState(false);
    const [manualTemplateInput, setManualTemplateInput] = useState('');

    // Create bar chart data sorted by round, then by activity
    const chartData = allPriorRoundsActivity
        .slice()
        .sort((a, b) => {
            if (a.round_number !== b.round_number) {
                return a.round_number - b.round_number;
            }
            return b.activity - a.activity; // Highest activity first within round
        });

    // Color scheme for rounds
    const roundColors: Record<number, string> = {
        1: '#1f77b4',  // blue
        2: '#ff7f0e',  // orange
        3: '#2ca02c',  // green
        4: '#d62728',  // red
        5: '#9467bd',  // purple
        6: '#8c564b',  // brown
    };

    // Create plotly data
    const plotData: Data[] = Object.entries(
        chartData.reduce((acc, item) => {
            const roundKey = `Round ${item.round_number}`;
            if (!acc[roundKey]) acc[roundKey] = [];
            acc[roundKey].push(item);
            return acc;
        }, {} as Record<string, typeof chartData>)
    ).map(([roundName, roundData]) => {
        const roundNumber = parseInt(roundName.split(' ')[1]);

        return {
            type: 'bar',
            name: roundName,
            x: roundData.map(item => item.seq_id),
            y: roundData.map(item => item.activity),
            marker: {
                color: roundData.map((item) => {
                    const baseColor = roundColors[roundNumber] || '#1f77b4';
                    return selectedSeqIds.includes(item.seq_id) ? baseColor : baseColor + '80'; // Add transparency for unselected
                }),
                line: {
                    color: roundData.map(item => selectedSeqIds.includes(item.seq_id) ? '#000' : 'transparent'),
                    width: 2
                }
            },
            text: roundData.map(item => selectedSeqIds.includes(item.seq_id) ? '★' : ''),
            textposition: 'outside',
            hovertemplate: '<b>%{x}</b><br>Activity: %{y:.4f}<br>Round: ' + roundNumber + '<extra></extra>'
        } as Data;
    });

    const handleBarClick = (event: any) => {
        if (!setSelectedSeqIds || !event.points || !event.points[0]) return;

        const clickedSeqId = event.points[0].x;
        setSelectedSeqIds(prev =>
            prev.includes(clickedSeqId)
                ? prev.filter(id => id !== clickedSeqId)
                : [...prev, clickedSeqId]
        );
    };

    return (
        <div>
            <Card>
                <div style={{ marginBottom: '24px' }}>
                    <Title level={4}>Click mutants to pass into this round</Title>
                    <Paragraph>
                        Select which mutants you'd like to serve as the starting points for the next round of mutagenesis. We will evaluate all single mutant modifications of these starting points. Common choices are:
                    </Paragraph>
                    <ol style={{ marginBottom: '16px' }}>
                        <li><strong>The best performer:</strong> If you want to take your prior best performer, and add single mutations to it: this is for you.</li>
                        <li><strong>Just the wild type ("WT"):</strong> If you want to make the next round focused only on mutants that are one-away from the wild type sequence, choose just WT.</li>
                        <li><strong>Best performers from each round:</strong> If you want the algorithm to have the flexibility to make derivatives of your best performer, or second best performer, or go back and try a mutation from a simpler... this option is for you. <strong>This is the recommended option, as it gives FolDE the most flexibility in choosing mutants, and might decrease the risk of getting stuck in a local minimum.</strong></li>
                    </ol>
                </div>

                {loadingActivityData ? (
                    <div style={{ textAlign: 'center', padding: '40px' }}>
                        <Spin size="large" />
                        <div style={{ marginTop: '16px' }}>Loading activity data from prior rounds...</div>
                    </div>
                ) : chartData.length > 0 ? (
                    <div>
                        <Text strong style={{ marginBottom: '16px', display: 'block' }}>
                            {setSelectedSeqIds
                                ? 'Click mutants from prior rounds to use them as starting points for the next round of mutants:'
                                : 'Selected mutants for this round (read-only):'}
                        </Text>
                        <div style={{ marginBottom: '24px' }}>
                            <Plot
                                data={plotData}
                                layout={{
                                    title: '',
                                    xaxis: { title: 'Sequence ID' },
                                    yaxis: { title: 'Activity' },
                                    height: 400,
                                    margin: { t: 20, b: 100 },
                                    showlegend: true,
                                    hovermode: 'closest'
                                }}
                                config={{ displayModeBar: false }}
                                onClick={handleBarClick}
                                style={{ width: '100%' }}
                            />
                        </div>

                        <div style={{ marginBottom: '24px' }}>
                            <Text strong style={{ marginBottom: '8px', display: 'block' }}>
                                Selected mutants ({selectedSeqIds.length}):
                            </Text>

                            {!isManualEditing ? (
                                <div style={{ position: 'relative' }}>
                                    <div style={{
                                        padding: '12px',
                                        backgroundColor: '#f5f5f5',
                                        borderRadius: '4px',
                                        fontFamily: 'Monaco, Consolas, "Courier New", monospace',
                                        fontSize: '12px',
                                        minHeight: '24px',
                                        color: selectedSeqIds.length === 0 ? '#999' : 'inherit'
                                    }}>
                                        {selectedSeqIds.length > 0 ? selectedSeqIds.join(', ') : 'No mutants selected'}
                                    </div>
                                    {setSelectedSeqIds && (
                                        <Button
                                            size="small"
                                            type="text"
                                            style={{
                                                position: 'absolute',
                                                right: '8px',
                                                top: '50%',
                                                transform: 'translateY(-50%)',
                                                fontSize: '12px',
                                                height: '20px',
                                                padding: '0 6px'
                                            }}
                                            onClick={() => {
                                                setManualTemplateInput(selectedSeqIds.join(', '));
                                                setIsManualEditing(true);
                                            }}
                                        >
                                            edit
                                        </Button>
                                    )}
                                </div>
                            ) : (
                                <div style={{ display: 'flex', gap: '8px' }}>
                                    <Input.TextArea
                                        value={manualTemplateInput}
                                        onChange={(e) => setManualTemplateInput(e.target.value)}
                                        placeholder="Enter mutant IDs separated by commas (e.g., WT, mutant1, mutant2)"
                                        rows={3}
                                        style={{
                                            fontFamily: 'Monaco, Consolas, "Courier New", monospace',
                                            fontSize: '12px'
                                        }}
                                    />
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                        <Button
                                            size="small"
                                            type="primary"
                                            onClick={() => {
                                                if (!setSelectedSeqIds) return;
                                                const newSeqIds = manualTemplateInput
                                                    .split(',')
                                                    .map(s => s.trim())
                                                    .filter(s => s.length > 0);
                                                setSelectedSeqIds(newSeqIds);
                                                setIsManualEditing(false);
                                            }}
                                        >
                                            Save
                                        </Button>
                                        <Button
                                            size="small"
                                            onClick={() => {
                                                setIsManualEditing(false);
                                                setManualTemplateInput('');
                                            }}
                                        >
                                            Cancel
                                        </Button>
                                    </div>
                                </div>
                            )}

                            {!isManualEditing && selectedSeqIds.some(id => !allPriorRoundsActivity.some(item => item.seq_id === id)) && (
                                <Alert
                                    message="Manual entries detected"
                                    description="Some selected mutants were not found in prior screening rounds. This is allowed but not recommended."
                                    type="warning"
                                    style={{ marginTop: '8px', fontSize: '12px' }}
                                    showIcon
                                />
                            )}
                        </div>
                    </div>
                ) : (
                    <Alert
                        message="No Activity Data Available"
                        description="No activity data was found from previous rounds. You may need to upload activity data to prior rounds first."
                        type="info"
                    />
                )}
            </Card>

            {/* Save Selection Button */}
            {!loadingActivityData && allPriorRoundsActivity.length > 0 && (
                <div style={{
                    marginTop: '16px',
                    display: 'flex',
                    justifyContent: 'flex-end'
                }}>
                    <Tooltip
                        title={!isEditable ? "Cannot change starting mutants once a slate has been selected" : ""}
                    >
                        <Button
                            type="primary"
                            size="large"
                            loading={savingTemplates}
                            onClick={onSaveTemplateSelection}
                            disabled={selectedSeqIds.length === 0 || !isEditable}
                        >
                            Save Selection ({selectedSeqIds.length} mutants)
                        </Button>
                    </Tooltip>
                </div>
            )}
        </div>
    );
};

export default InputTemplateSelection;
