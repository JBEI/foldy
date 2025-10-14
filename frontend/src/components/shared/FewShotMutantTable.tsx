import React, { useState, useMemo } from 'react';
import { Select, Spin } from 'antd';
import ProposedSlateTable, { ProposedSlateTableColumn } from './ProposedSlateTable';
import { Selection } from '../FoldView/StructurePane';
import Plot from 'react-plotly.js';
import { SlateData } from '../../api/fewShotApi';


interface FewShotMutantTableProps {
    yamlConfig: string | null;
    slateData: SlateData[] | null;
    setSelectedSubsequence: (selection: Selection | null) => void;
    sortOptions: { [key: string]: string[] } | null;
    onBuildSlate?: (seqIds: string[]) => void;
    disableSlateBuilder?: boolean;
}

const applySortOrderToSlateData = (
    slateData: SlateData[],
    seqIdOrder: string[] | null
): SlateData[] => {
    if (!seqIdOrder) return slateData;

    return slateData.map(item => ({
        ...item,
        order: seqIdOrder.indexOf(item.seqId)
    })).sort((a, b) => {
        if (a.order === null && b.order === null) return 0;
        if (a.order === null) return 1;
        if (b.order === null) return -1;
        return a.order - b.order;
    });
};

const seqIdListToLociList = (seqIdList: string[]): number[] => {
    const lociToHighlightList: number[] = [];
    seqIdList.forEach(seqId => {
        seqId.split('_').forEach(alleleId => {
            const match = alleleId.match(/[A-Z](\d+)[A-Z]/);
            if (match) {
                lociToHighlightList.push(parseInt(match[1]));
            }
        });
    });
    return Array.from(new Set(lociToHighlightList));
}

const seqIdListToAlleleIdCount = (seqIdList: string[]): { [alleleId: string]: number } => {
    const mutatationCount: { [alleleId: string]: number } = {};
    seqIdList.forEach(seqId => {
        seqId.split('_').forEach(alleleId => {
            mutatationCount[alleleId] = (mutatationCount[alleleId] || 0) + 1;
        })
    })
    return mutatationCount;
}

const FewShotMutantTable: React.FC<FewShotMutantTableProps> = ({
    yamlConfig,
    slateData,
    setSelectedSubsequence,
    sortOptions,
    onBuildSlate,
    disableSlateBuilder,
}) => {
    if (!slateData) {
        return <div style={{ textAlign: 'center', padding: '60px 0' }}>
            <Spin size="large" />
        </div>;
    }

    const [selectedSeqIds, setSelectedSeqIds] = useState<string[]>([]);
    const [seqIdOrderChoice, setSeqIdOrderChoice] = useState<string | null>("selection_order");

    const tableData: SlateData[] | null = useMemo(() => {
        const sortOrder = (seqIdOrderChoice && sortOptions) ? sortOptions[seqIdOrderChoice] : null;
        return applySortOrderToSlateData(slateData, sortOrder);
    }, [slateData, seqIdOrderChoice, sortOptions]);

    const correlationData = useMemo(() => {
        if (!tableData || tableData.length === 0 || !tableData[0].modelPredictions) {
            return null;
        }

        const sequenceCount = tableData.length;
        if (sequenceCount <= 1) return null;

        // Create a matrix to store correlations between sequences
        const matrix: number[][] = Array(sequenceCount).fill(0).map(() => Array(sequenceCount).fill(0));

        // For each pair of sequences, we need to calculate correlation of their values
        // across the different metrics (mean, stddev, score)
        for (let i = 0; i < sequenceCount; i++) {
            for (let j = 0; j < sequenceCount; j++) {
                // For identical sequences, correlation is 1
                if (i === j) {
                    matrix[i][j] = 1;
                    continue;
                }

                // Calculate correlation coefficient between model predictions
                const seqIData = tableData[i].modelPredictions;
                const seqJData = tableData[j].modelPredictions;

                if (!seqIData || !seqJData || seqIData.length === 0 || seqJData.length === 0) {
                    matrix[i][j] = 0;
                    continue;
                }

                // Calculate means
                const iMean = seqIData.reduce((sum, val) => sum + val, 0) / seqIData.length;
                const jMean = seqJData.reduce((sum, val) => sum + val, 0) / seqJData.length;

                // Calculate correlation coefficient (normalized covariance)
                let numerator = 0;
                let iVariance = 0;
                let jVariance = 0;

                for (let k = 0; k < seqIData.length; k++) {
                    const iDiff = seqIData[k] - iMean;
                    const jDiff = seqJData[k] - jMean;
                    numerator += iDiff * jDiff;
                    iVariance += iDiff * iDiff;
                    jVariance += jDiff * jDiff;
                }

                // Correlation coefficient = covariance / (stddev_i * stddev_j)
                const correlation = numerator / (Math.sqrt(iVariance) * Math.sqrt(jVariance));

                matrix[i][j] = correlation;
            }
        }

        return matrix;
    }, [tableData]);

    if (!tableData) return null;

    const columns: ProposedSlateTableColumn[] = [
        {
            key: "order",
            name: "Order",
            sortable: true,
            resizable: true,
            width: 60,
            sortDescendingFirst: false,
            formatter: ({ row }: { row: any }) => (
                <div style={{ textAlign: 'center' }}>
                    {row.order !== null ? row.order : ''}
                </div>
            )
        },
        {
            key: "seqId",
            name: "Sequence ID",
            sortable: true,
            resizable: true,
            sortDescendingFirst: true,
            formatter: ({ row }: { row: any }) => (
                <div
                    uk-tooltip={row.seqId}
                    style={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        paddingLeft: '5px'
                    }}
                >
                    {row.seqId}
                </div>
            )
        },
        {
            key: 'relevantMeasuredMutants',
            name: "Measured",
            resizable: true,
            formatter: ({ row }: { row: any }) => (
                <div
                    uk-tooltip={row.relevantMeasuredMutants}
                    style={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                    }}
                >
                    {row.relevantMeasuredMutants}
                </div>
            )
        },
        {
            key: 'predictionMean',
            name: "Mean",
            sortable: true,
            resizable: true,
            width: 70,
            formatter: ({ row }: { row: any }) => (
                <div uk-tooltip={row.predictionMean.toFixed(4)} style={{ textAlign: 'left' }}>
                    {row.predictionMean.toFixed(2)}
                </div>
            )
        },
        {
            key: "predictionStddev",
            name: "STD",
            sortable: true,
            resizable: true,
            width: 70,
            formatter: ({ row }: { row: any }) => (
                <div uk-tooltip={row.predictionStddev.toFixed(4)} style={{ textAlign: 'left' }}>
                    {row.predictionStddev.toFixed(2)}
                </div>
            )
        }
    ];

    const alleleIdCount = useMemo(() => {
        return seqIdListToAlleleIdCount(tableData.map(row => row.seqId));
    }, [tableData]);

    return (
        <div>
            <h3>Proposed Slate</h3>
            <div style={{ marginBottom: '10px' }}>
                <Select
                    value={seqIdOrderChoice}
                    onChange={setSeqIdOrderChoice}
                    placeholder="Select sequence order"
                    style={{ width: '200px' }}
                    allowClear
                >
                    {Object.keys(sortOptions || {}).map(key => (
                        <Select.Option key={key} value={key}>
                            {key}
                        </Select.Option>
                    ))}
                </Select>
            </div>

            <ProposedSlateTable
                description="These are the top mutants selected by the model for evaluating in the next round. Here you can view the mean activity prediction of the mutants (unitless), click the sequence ID to highlight the residues on the structure, and view the standard deviation of the ensemble of predictors, which is a measure of model confidence in these predictions. Try changing the sort order to cluster, to simplify the slate self-correlation heatmap!"
                data={tableData}
                columns={columns}
                yamlConfig={yamlConfig}
                setSelectedSubsequence={setSelectedSubsequence}
                enableRowSelection={true}
                enableRowClick={true}
                onSelectedRowsChange={setSelectedSeqIds}
                showHighlightOnModelButton={false}
                showSlateBuilderButton={!!onBuildSlate}
                disableSlateBuilderButton={disableSlateBuilder}
                onBuildSlate={onBuildSlate}
            />

            <div>
                <h3 style={{ marginTop: '20px' }}>Slate Stats</h3>
                <p>
                    Slate contains {Object.keys(alleleIdCount).length} mutations
                    at {seqIdListToLociList(tableData.map(row => row.seqId)).length} loci, including
                </p>

                <ul>
                    {Object.entries(alleleIdCount)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 10)
                        .map(([alleleId, count]) => (
                            <li key={alleleId}>
                                {alleleId}: {count} mutants
                            </li>
                        ))}
                    {Object.keys(alleleIdCount).length > 10 && (
                        <li>... and {Object.keys(alleleIdCount).length - 10} more</li>
                    )}
                </ul>

                <h3>Slate Self-Correlation Heatmap</h3>
                <p>
                    This heatmap shows the correlation between predicted activities of different mutants in the slate.
                    High correlation (red) means mutants have similar predicted activities across ensemble members.
                    Low correlation (blue) means mutants have different predicted activities.
                </p>

                {useMemo(() => {
                    if (!correlationData || !tableData) return null;

                    return (
                        <span>
                            <h3>Slate Self Correlation</h3>
                            <div style={{
                                height: '600px', // Increased height for better visibility
                                backgroundColor: 'white',  //'#f9f9f9',
                                padding: '15px',
                                borderRadius: '4px',
                                marginTop: '20px',
                                overflowX: 'auto', // Add horizontal scroll for many sequences
                                overflowY: 'auto'  // Add vertical scroll too
                            }}>
                                <Plot
                                    data={[{
                                        z: correlationData,
                                        x: tableData.map(row => row.seqId),
                                        y: tableData.map(row => row.seqId),
                                        type: 'heatmap',
                                        colorscale: 'RdBu',
                                        zmin: -1,
                                        zmax: 1,
                                        hovertemplate: '%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>',
                                        showscale: true,
                                        colorbar: {
                                            title: 'Correlation',
                                            titleside: 'right'
                                        }
                                    }]}
                                    layout={{
                                        title: 'Sequence Prediction Correlation',
                                        autosize: true,
                                        // Increase margins to accommodate sequence IDs
                                        margin: { l: 150, r: 50, t: 60, b: 100 },
                                        xaxis: {
                                            title: 'Sequence ID',
                                            tickangle: 45,
                                            tickfont: { size: 10 }
                                        },
                                        yaxis: {
                                            title: 'Sequence ID',
                                            autorange: 'reversed',
                                            tickfont: { size: 10 }
                                        },
                                        plot_bgcolor: 'white',  // '#f9f9f9',
                                        paper_bgcolor: 'white',  // '#f9f9f9',
                                        font: { family: 'Arial, sans-serif' }
                                    }}
                                    style={{ width: '100%', height: '100%' }}
                                    useResizeHandler={true}
                                    config={{
                                        responsive: true,
                                        displayModeBar: true,
                                        displaylogo: false,
                                        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                                        toImageButtonOptions: {
                                            format: 'png',
                                            filename: 'sequence_correlation_heatmap',
                                            height: 800,
                                            width: 800,
                                            scale: 2
                                        }
                                    }}
                                />
                            </div>
                        </span>
                    );
                }, [correlationData, tableData])}
            </div>
        </div>
    );
};

export default FewShotMutantTable;
