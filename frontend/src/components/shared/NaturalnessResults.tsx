import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Data } from 'plotly.js';
import Papa from 'papaparse';
import { CheckboxControl, NumberInputControl } from '../../util/controlComponents';
import ProposedSlateTable, { ProposedSlateTableColumn } from './ProposedSlateTable';
import { Selection } from '../FoldView/StructurePane';
import { notify } from '../../services/NotificationService';
import { BoltzYamlHelper } from '../../util/boltzYamlHelper';

// Define standard amino acid residues
const RESIDUES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'];

const NATURALNESS_COLUMN = 'probability';
const WT_MARGINAL_COLUMN = 'wt_marginal';

// Calculate Approximate Pseudo Log Likelihood (PLL)
// Reference: Gordon, Cade, Amy X. Lu, and Pieter Abbeel. 2024.
// "Protein Language Model Fitness Is a Matter of Preference." Bioinformatics. bioRxiv.
// https://www.biorxiv.org/content/10.1101/2024.10.03.616542v1.full.pdf
const calculatePLL = (naturalnessCsvDataString: string, alpha: number, beta: number, epsilon: number): number | null => {
    const { data, errors } = Papa.parse<Record<string, string>>(naturalnessCsvDataString, {
        header: true,
        delimiter: ',',
        skipEmptyLines: true,
        dynamicTyping: true
    });

    if (errors.length > 0 || !data.length) {
        return null;
    }

    // Filter for wild-type sequences (where mutant residue equals WT residue)
    const wtSequences = data.filter(row => {
        const seqId = row['seq_id'];
        if (!seqId) return false;

        const match = seqId.match(/([A-Z])(\d+)([A-Z])/);
        if (!match) return false;

        const wtResidue = match[1];
        const mutantResidue = match[3];
        return wtResidue === mutantResidue;
    });

    if (wtSequences.length === 0) {
        return null;
    }

    // Calculate PLL = 1/L * sum(log(P(wt residue) + epsilon))
    const L = wtSequences.length;
    let logSum = 0;

    wtSequences.forEach(row => {
        // Calculate PLL using the formula: p' ← max((α+β)/α*p - β/α, ε)
        let prob = parseFloat(row[NATURALNESS_COLUMN]) || 0;
        prob = Math.max(((alpha + beta) / alpha * prob - beta / alpha), epsilon);
        logSum += Math.log(prob);
    });

    return logSum / L;
};

interface NaturalnessResultsProps {
    naturalnessCsvData: string;
    yamlConfig: string | null;
    setSelectedSubsequence: (selection: Selection | null) => void;
    runName?: string;
    onClose?: () => void;
    onBuildSlate?: (seqIds: string[]) => void;
    disableSlateBuilder?: boolean;
    disableRowSelection?: boolean;
}

type RowData = {
    seqId: string;
    score: number;
    model: number | null;
}

const parseSeqId = (seqId: string): { wtResidue: string, locus: number, mutantResidue: string } => {
    if (seqId.includes('_')) {
        throw new Error(`Invalid seqId: "${seqId}"`);
    }
    const match = seqId.match(/([A-Z])(\d+)([A-Z])/);
    if (!match) {
        throw new Error(`Invalid seqId: "${seqId}"`);
    }
    return { wtResidue: match[1], locus: parseInt(match[2]), mutantResidue: match[3] };
}

const parseCsvDataIntoRowData = (naturalnessCsvDataString: string, useWtMarginalAsScore: boolean, zeroWildType: boolean, maxMutationsPerLocus: number | undefined, topPerformersToDisplay: number | undefined): RowData[] | null => {
    const { data, errors } = Papa.parse<Record<string, string>>(naturalnessCsvDataString, {
        header: true,
        delimiter: ',',
        skipEmptyLines: true,
        dynamicTyping: true
    });

    if (errors.length > 0) {
        notify.error(`Error parsing naturalness CSV: ${errors.map(error => error.message).join(', ')}`);
        return null;
    }

    const interiorTableRows = data.filter((row) => {
        const endsInSpecialCharacter = row['seq_id'].match(/.*<.*>/);
        const endsInDot = row['seq_id'].match(/.*\..*/);
        const endsInHyphen = row['seq_id'].match(/.*-.*/);
        const endsInBar = row['seq_id'].match(/.*\|.*/);
        if (endsInSpecialCharacter || endsInDot || endsInHyphen || endsInBar) {
            return false;
        }
        return true;
    }).map((row) => {
        var score;
        if (useWtMarginalAsScore) {
            score = parseFloat(row[WT_MARGINAL_COLUMN]);
            score = score ? Math.log(score || 1e-7) : null;
        } else {
            score = parseFloat(row[NATURALNESS_COLUMN]) || 0;
        }
        score = score || 0;

        const { wtResidue, locus, mutantResidue } = parseSeqId(row['seq_id']);

        if (zeroWildType && wtResidue == mutantResidue) {
            score = 0;
        }
        var model = null;
        if (row['model'] != null) {
            model = parseInt(row['model']);
        }

        return {
            seqId: row['seq_id'],
            score: score,
            model: model
        };
    });

    if (maxMutationsPerLocus !== undefined || topPerformersToDisplay !== undefined) {
        const filteredRows = new Map<number, RowData[]>();

        interiorTableRows.forEach(row => {
            const { locus } = parseSeqId(row.seqId);
            if (!filteredRows.has(locus)) {
                filteredRows.set(locus, []);
            }
            filteredRows.get(locus)!.push(row);
        });

        const finalRows: RowData[] = [];
        filteredRows.forEach((locusRows, locus) => {
            locusRows.sort((a, b) => b.score - a.score);
            const limit = maxMutationsPerLocus || locusRows.length;
            finalRows.push(...locusRows.slice(0, limit));
        });

        finalRows.sort((a, b) => b.score - a.score);
        const displayLimit = topPerformersToDisplay || finalRows.length;
        return finalRows.slice(0, displayLimit);
    }

    return interiorTableRows;
};

const NaturalnessTable: React.FC<{
    naturalnessCsvData: string;
    useWtMarginalAsScore: boolean;
    zeroWildType: boolean;
    maxMutationsPerLocus: number;
    topPerformersToDisplay: number;
    yamlConfig: string | null;
    setSelectedSubsequence: (selection: Selection | null) => void;
    onBuildSlate?: (seqIds: string[]) => void;
    disableSlateBuilder?: boolean;
    disableRowSelection?: boolean;
}> = ({
    naturalnessCsvData,
    useWtMarginalAsScore,
    zeroWildType,
    maxMutationsPerLocus,
    topPerformersToDisplay,
    yamlConfig,
    setSelectedSubsequence,
    onBuildSlate,
    disableSlateBuilder,
    disableRowSelection,
}) => {
        if (!naturalnessCsvData) return null;

        const tableData: RowData[] | null = useMemo(() => {
            return parseCsvDataIntoRowData(naturalnessCsvData, useWtMarginalAsScore, zeroWildType, maxMutationsPerLocus, topPerformersToDisplay);
        }, [naturalnessCsvData, useWtMarginalAsScore, zeroWildType, maxMutationsPerLocus, topPerformersToDisplay]);

        if (!tableData) return null;

        const columns: ProposedSlateTableColumn[] = [
            {
                key: "seqId",
                name: "Sequence ID",
                sortable: true,
                resizable: true,
                sortDescendingFirst: true
            },
            {
                key: "score",
                name: useWtMarginalAsScore ? "Log(WT Marginal Likelihood)" : "Probability",
                sortable: true,
                resizable: true,
                formatter: ({ row }: { row: any }) => row.score.toFixed(4),
                sortDescendingFirst: true
            }
        ];

        return (
            <ProposedSlateTable
                description="These mutants have the highest naturalness scores. Click on a sequence ID to highlight the residues on the structure."
                data={tableData}
                columns={columns}
                yamlConfig={yamlConfig}
                setSelectedSubsequence={setSelectedSubsequence}
                rowSelection={false}
                enableRowClick={true}
                showCopyButton={true}
                showHighlightButton={true}
                showHighlightOnModelButton={false}
                showSlateBuilderButton={!!onBuildSlate}
                disableSlateBuilderButton={disableSlateBuilder}
                disableRowSelection={disableRowSelection}
                onBuildSlate={onBuildSlate}
            />
        );
    };

const NaturalnessResults: React.FC<NaturalnessResultsProps> = ({
    naturalnessCsvData,
    yamlConfig,
    setSelectedSubsequence,
    runName = "Naturalness Results",
    onClose,
    onBuildSlate,
    disableSlateBuilder,
    disableRowSelection = false
}) => {
    const [maskWildType, setMaskWildType] = useState<boolean>(false);
    const [zeroWildType, setZeroWildType] = useState<boolean>(false);
    const [showWTMarginalLikelihood, setShowWTMarginalLikelihood] = useState<boolean>(true);
    const [maxMutationsPerLocus, setMaxMutationsPerLocus] = useState<number>(3);
    const [topPerformersToDisplay, setTopPerformersToDisplay] = useState<number>(24);
    const [alpha, setAlpha] = useState<number>(0.1);
    const [alphaInput, setAlphaInput] = useState<string>('0.1');
    const [beta, setBeta] = useState<number>(0.1);
    const [betaInput, setBetaInput] = useState<string>('0.1');
    const [epsilon, setEpsilon] = useState<number>(0.001);
    const [epsilonInput, setEpsilonInput] = useState<string>('0.00x1');

    const pllValue = useMemo(() => {
        if (!naturalnessCsvData) return null;
        return calculatePLL(naturalnessCsvData, alpha, beta, epsilon);
    }, [naturalnessCsvData, alpha, beta, epsilon]);

    const naturalnessPlot = useMemo(() => {
        if (!naturalnessCsvData) return null;

        const rowData = parseCsvDataIntoRowData(naturalnessCsvData, showWTMarginalLikelihood, zeroWildType, undefined, undefined);
        if (!rowData) return null;

        // Process data for heatmap
        const locusSet = new Set<number>();
        const scoreHeatmapData: { [key: string]: number } = {};
        const wtResidues: { [key: number]: string } = {};

        var ensembleMembers = 1.0;

        rowData.forEach(row => {
            if (row.model != null) {
                ensembleMembers = Math.max(ensembleMembers, row.model);
            }
            const seqId = row.seqId;

            const match = seqId.match(/([A-Z])(\d+)([A-Z])/);
            if (match) {
                const wtResidue = match[1];
                const locus = parseInt(match[2]);
                const mutantResidue = match[3];
                locusSet.add(locus);
                wtResidues[locus] = wtResidue;
                const key = `${locus}-${mutantResidue}`;
                if (key in scoreHeatmapData) {
                    scoreHeatmapData[key] += row.score;
                } else {
                    scoreHeatmapData[key] = row.score;
                }
            }
        });
        Object.keys(scoreHeatmapData).forEach(key => {
            scoreHeatmapData[key] /= ensembleMembers;
        });

        const loci = Array.from(locusSet).sort((a, b) => a - b);
        const zValues = RESIDUES.map(res =>
            loci.map(locus => {
                if (res === wtResidues[locus]) {
                    if (maskWildType) {
                        return null;
                    }
                }
                const key = `${locus}-${res}`;
                return key in scoreHeatmapData ? scoreHeatmapData[key] : null;
            })
        );
        const zmin = showWTMarginalLikelihood ? 0 : 0;
        // const zmin = showWTMarginalLikelihood ? Math.min(...zValues.flat(2).filter(val => val !== null) as number[]) : 0;
        const zmax = showWTMarginalLikelihood ? Math.max(...zValues.flat(2).filter(val => val !== null) as number[]) : 1;

        // Create customdata to match the z-values structure (RESIDUES x loci)
        const customData = RESIDUES.map(residue =>
            loci.map(locus => wtResidues[locus])
        );

        const hoverTemplate = showWTMarginalLikelihood ? '%{customdata}%{x}%{y}<br>Score: 10^%{z}<extra></extra>' : '%{customdata}%{x}%{y}<br>Probability: %{z}<extra></extra>';



        const plotlyData: Array<Partial<Data>> = [{
            type: 'heatmap',
            z: zValues,
            x: loci,
            y: RESIDUES,
            colorscale: 'Viridis',
            zmin: zmin,
            zmax: zmax,
            hovertemplate: hoverTemplate,
            customdata: customData,
            showscale: true,  // Ensure colorbar is shown
        }];

        return (
            <div style={{ marginBottom: '20px', width: '100%', maxWidth: '100%', height: '400px', overflow: 'hidden' }}>
                <Plot
                    data={plotlyData}
                    layout={{
                        title: {
                            text: showWTMarginalLikelihood ? 'Log(WT Marginal Likelihood) Heatmap' : 'Naturalness Probability Heatmap',
                            font: { size: 14, color: '#262626' }
                        },
                        xaxis: {
                            title: { text: 'Locus', font: { size: 12, color: '#595959' } },
                            tickfont: { size: 10, color: '#8c8c8c' }
                        },
                        yaxis: {
                            title: { text: 'Amino Acid', font: { size: 12, color: '#595959' } },
                            tickfont: { size: 10, color: '#8c8c8c' }
                        },
                        autosize: true,
                        margin: { l: 50, r: 50, t: 50, b: 50 },
                        plot_bgcolor: 'white',
                        paper_bgcolor: 'white',
                        font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' }
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '100%' }}
                    config={{
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d']
                    }}
                    onHover={(event) => {
                        if (event.points && event.points.length > 0 && yamlConfig) {
                            const point = event.points[0];
                            // Convert fractional coordinates to discrete indices
                            const xIndex = Math.floor(point.x as number);
                            const yIndex = Math.floor(point.y as number);
                            console.log(`xIndex: ${xIndex}, yIndex: ${yIndex}, point.x: ${point.x}`);

                            // Get the locus from x-axis (which contains the loci array)
                            if (xIndex >= 0 && xIndex < loci.length) {
                                const locus = xIndex; //loci[xIndex];

                                // Get chain ID from yaml config
                                const configHelper = new BoltzYamlHelper(yamlConfig);
                                if (configHelper.getProteinSequences().length > 1) {
                                    return; // Don't highlight on multimers
                                }
                                const chainId = configHelper.getProteinSequences()[0][0];

                                setSelectedSubsequence({
                                    data: [{
                                        struct_asym_id: chainId,
                                        start_residue_number: locus,
                                        end_residue_number: locus,
                                        color: "#FFD700", // Gold color for hover
                                    }],
                                    nonSelectedColor: "white",
                                });
                            }
                        }
                    }}
                    onUnhover={() => {
                        setSelectedSubsequence(null);
                    }}
                />
            </div>
        );
    }, [naturalnessCsvData, showWTMarginalLikelihood, zeroWildType, maskWildType]);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {onClose && (
                <div style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: "10px"
                }}>
                    <h2 style={{ margin: 0, overflowWrap: 'anywhere' }}>
                        {runName}
                    </h2>
                    <button
                        onClick={onClose}
                        style={{
                            background: "none",
                            border: "none",
                            cursor: "pointer",
                            fontSize: "20px",
                            padding: "5px",
                            color: "#666"
                        }}
                        aria-label="Close"
                    >
                        ✕
                    </button>
                </div>
            )}

            {/* PLL Widget
            <h3 style={{ marginBottom: '12px' }}>Calculated Pseudo Log Likelihood (In Development)</h3>
            <p style={{
                margin: 0,
                fontSize: '14px',
                color: '#6c757d',
                lineHeight: '1.4'
            }}>
                Protein language models can give a general sense of how "natural" it considers your protein, called the <strong>Pseudo Log Likelihood (PLL)</strong>. Very low PLL proteins (e.g., less than -1.4) tend to have worse naturalness predictions.
                Based on <a href="https://doi.org/10.1101/2024.10.03.616542" target="_blank" rel="noopener noreferrer" style={{ color: '#007bff', textDecoration: 'none' }}>
                    Gordon et al. (2024) "Protein Language Model Fitness Is a Matter of Preference."
                </a>

                <br /><br />
                <strong style={{ color: '#d4351c' }}>NOTE:</strong> Alpha and beta parameters are not provided for ESM-C in the Gordon paper. We assume both equal 0.1, as was true for ESM-2. Epsilon is not provided in the Gordon paper, and has a huge effect on PLL calculation. Altogether, this PLL is a very rough estimate, and is not yet ratified as being an accurate tool.
            </p>

            <div style={{
                background: '#f8f9fa',
                border: '1px solid #e9ecef',
                borderRadius: '8px',
                padding: '20px',
                marginBottom: '12px'
            }}>
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '16px'
                }}>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '20px',
                        flexWrap: 'wrap'
                    }}>
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px'
                        }}>
                            <label style={{
                                fontWeight: '600',
                                color: '#495057',
                                minWidth: '50px'
                            }}>
                                Alpha:
                            </label>
                            <input
                                type="text"
                                value={alphaInput}
                                onChange={(e) => {
                                    const value = e.target.value;
                                    setAlphaInput(value);
                                    const parsed = parseFloat(value);
                                    if (!isNaN(parsed) && parsed > 0) {
                                        setAlpha(parsed);
                                    }
                                }}
                                onBlur={() => {
                                    const parsed = parseFloat(alphaInput);
                                    if (isNaN(parsed) || parsed <= 0) {
                                        setAlphaInput('0.1');
                                        setAlpha(0.1);
                                    }
                                }}
                                style={{
                                    padding: '4px 8px',
                                    border: '1px solid #ced4da',
                                    borderRadius: '4px',
                                    width: '80px',
                                    fontSize: '14px'
                                }}
                            />
                        </div>
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px'
                        }}>
                            <label style={{
                                fontWeight: '600',
                                color: '#495057',
                                minWidth: '50px'
                            }}>
                                Beta:
                            </label>
                            <input
                                type="text"
                                value={betaInput}
                                onChange={(e) => {
                                    const value = e.target.value;
                                    setBetaInput(value);
                                    const parsed = parseFloat(value);
                                    if (!isNaN(parsed) && parsed > 0) {
                                        setBeta(parsed);
                                    }
                                }}
                                onBlur={() => {
                                    const parsed = parseFloat(betaInput);
                                    if (isNaN(parsed) || parsed <= 0) {
                                        setBetaInput('0.1');
                                        setBeta(0.1);
                                    }
                                }}
                                style={{
                                    padding: '4px 8px',
                                    border: '1px solid #ced4da',
                                    borderRadius: '4px',
                                    width: '80px',
                                    fontSize: '14px'
                                }}
                            />
                        </div>
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px'
                        }}>
                            <label style={{
                                fontWeight: '600',
                                color: '#495057',
                                minWidth: '60px'
                            }}>
                                Epsilon:
                            </label>
                            <input
                                type="text"
                                value={epsilonInput}
                                onChange={(e) => {
                                    const value = e.target.value;
                                    setEpsilonInput(value);
                                    const parsed = parseFloat(value);
                                    if (!isNaN(parsed) && parsed > 0) {
                                        setEpsilon(parsed);
                                    }
                                }}
                                onBlur={() => {
                                    const parsed = parseFloat(epsilonInput);
                                    if (isNaN(parsed) || parsed <= 0) {
                                        setEpsilonInput('0.01');
                                        setEpsilon(0.01);
                                    }
                                }}
                                style={{
                                    padding: '4px 8px',
                                    border: '1px solid #ced4da',
                                    borderRadius: '4px',
                                    width: '80px',
                                    fontSize: '14px'
                                }}
                            />
                        </div>
                    </div>
                    <div style={{
                        fontSize: '24px',
                        fontWeight: 'bold',
                        color: pllValue !== null && pllValue < -1.4 ? '#dc3545' : '#28a745'
                    }}>
                        PLL: {pllValue !== null ? pllValue.toFixed(4) : 'N/A'}
                    </div>
                </div>
            </div> */}

            <h3 style={{ marginBottom: '8px' }}>Single Mutant Naturalness</h3>
            <p style={{
                margin: '0 0 16px 0',
                fontSize: '14px',
                color: '#6c757d',
                lineHeight: '1.4'
            }}>
                This displays the naturalness of every possible single mutation of the protein. Higher naturalness (aka wt-marginal likelihood) is correlated with higher activity.
            </p>

            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '16px',
                marginBottom: '16px'
            }}>
                <div>
                    <NumberInputControl
                        label="Max number of mutants per locus"
                        value={maxMutationsPerLocus}
                        onChange={setMaxMutationsPerLocus}
                        min={1}
                    />
                    <NumberInputControl
                        label="Top mutants to display"
                        value={topPerformersToDisplay}
                        onChange={setTopPerformersToDisplay}
                        min={1}
                    />
                </div>
                <div>
                    <CheckboxControl
                        label="Mask wild-type amino acids"
                        checked={maskWildType}
                        onChange={setMaskWildType}
                    />
                    <CheckboxControl
                        label="Zero out wild-type amino acids"
                        checked={zeroWildType}
                        onChange={setZeroWildType}
                    />
                    <CheckboxControl
                        label="Display Log(WT Marginal Likelihood)"
                        checked={showWTMarginalLikelihood}
                        onChange={setShowWTMarginalLikelihood}
                    />
                </div>
            </div>

            <div>
                {naturalnessPlot}
            </div>

            <div>
                <h3>Proposed Slate</h3>
                <NaturalnessTable
                    naturalnessCsvData={naturalnessCsvData}
                    useWtMarginalAsScore={showWTMarginalLikelihood}
                    zeroWildType={zeroWildType}
                    maxMutationsPerLocus={maxMutationsPerLocus}
                    topPerformersToDisplay={topPerformersToDisplay}
                    yamlConfig={yamlConfig}
                    setSelectedSubsequence={setSelectedSubsequence}
                    onBuildSlate={onBuildSlate}
                    disableSlateBuilder={disableSlateBuilder}
                    disableRowSelection={disableRowSelection}
                />
            </div>
        </div>
    );
};

export default NaturalnessResults;
