import React, { useState, ChangeEvent, useMemo } from 'react';
import UIkit from 'uikit';
import { FileInfo, Logit, Invokation } from 'src/types/types';
import { evolve } from '../../api/evolveApi';
import { FaDownload, FaEye, FaRedo } from 'react-icons/fa';
import fileDownload from 'js-file-download';
import { removeLeadingSlash } from '../../api/commonApi';
import { getFile } from '../../api/fileApi';
import { startLogits } from '../../api/embedApi';
import Plot from 'react-plotly.js';
import { Data } from 'plotly.js';
import Papa from 'papaparse';
import { useTable, useGlobalFilter, useSortBy, TableInstance } from 'react-table';
import { ESMModelPicker } from './ESMModelPicker';
import { Selection } from './StructurePane';
import DataGrid from 'react-data-grid';
import ReactDataGrid from 'react-data-grid';
// import 'react-data-grid/lib/styles.css';  // Don't forget the styles!


const NATURALNESS_COLUMN = 'probability';
const WT_MARGINAL_COLUMN = 'wt_marginal';

// Define standard amino acid residues
const RESIDUES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'];

interface NaturalnessTabProps {
    foldId: number;
    foldName: string | null;
    foldChainIds: string[] | null;
    jobs: Invokation[] | null;
    logits: Logit[] | null;
    setSelectedSubsequence: (selection: Selection | null) => void;
    setErrorText: (error: string) => void;
}



type RowData = {
    seqId: string;
    score: number;
    model: number | null;
}

const parseSeqId = (seqId: string): { wtResidue: string, locus: number, mutantResidue: string } => {
    // If there is an underscore in seq id, we bail.
    if (seqId.includes('_')) {
        throw new Error(`Invalid seqId: "${seqId}"`);
    }
    const match = seqId.match(/([A-Z])(\d+)([A-Z])/);
    if (!match) {
        throw new Error(`Invalid seqId: "${seqId}"`);
    }
    return { wtResidue: match[1], locus: parseInt(match[2]), mutantResidue: match[3] };
}

const parseCsvDataIntoRowData = (logitCsvDataString: string, useWtMarginalAsScore: boolean, zeroWildType: boolean, maxMutationsPerLocus: number | undefined, topPerformersToDisplay: number | undefined): RowData[] | null => {
    const { data, errors } = Papa.parse<Record<string, string>>(logitCsvDataString, {
        header: true,
        delimiter: ',',
        skipEmptyLines: true,
        dynamicTyping: true
    });

    if (errors.length > 0) {
        UIkit.notification({ message: `Error parsing logit CSV: ${errors.map(error => error.message).join(', ')}`, status: 'danger' });
        return null;
    }

    const interiorTableRows = data.filter((row) => {
        // Filter out rows that end in special characters like <cls>.
        const endsInSpecialCharacter = row['seq_id'].match(/.*<.*>/);
        const endsInDot = row['seq_id'].match(/.*\..*/);
        const endsInHyphen = row['seq_id'].match(/.*-.*/);
        const endsInBar = row['seq_id'].match(/.*\|.*/);
        if (endsInSpecialCharacter || endsInDot || endsInHyphen || endsInBar) {
            console.log(`Filtering out row: ${row['seq_id']}`);
            return false;
        }
        return true;
    }).map((row) => {
        var score;
        if (useWtMarginalAsScore) {
            score = parseFloat(row[WT_MARGINAL_COLUMN]);
            // Take the log of the score.
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
            model = parseFloat(row['model']);
        }

        return {
            seqId: row['seq_id'],
            score: score,
            model: model,
        };
    });

    const allRows = interiorTableRows.sort((a, b) => b.score - a.score);

    // Filter out mutations where we've already seen that locus N times.
    const locusCounts: { [key: number]: number } = {};
    const relevantRows = [];
    for (const row of allRows) {
        const { wtResidue, locus, mutantResidue } = parseSeqId(row.seqId);

        locusCounts[locus] = (locusCounts[locus] || 0) + 1;

        if (maxMutationsPerLocus && locusCounts[locus] > maxMutationsPerLocus) {
            continue;
        }
        relevantRows.push(row);
        if (topPerformersToDisplay && relevantRows.length >= topPerformersToDisplay) {
            break;
        }
    }
    return relevantRows;
}

// const LogitTable: React.FC<LogitTableProps> = ({ logitCsvData, useWtMarginalAsScore, zeroWildType, topPerformersToDisplay }) => {
//     if (!logitCsvData) return null;

//     const tableData: RowData[] | null = useMemo(() => {
//         return parseCsvDataIntoRowData(logitCsvData, useWtMarginalAsScore, zeroWildType)?.slice(0, topPerformersToDisplay) || null;
//     }, [logitCsvData, useWtMarginalAsScore, topPerformersToDisplay]);
//     if (!tableData) return null;

//     const tableSubset = tableData.slice(0, topPerformersToDisplay);

//     let columnDefs: ColDef<RowData>[] = [
//         { field: 'seqId', headerName: 'Sequence ID', sortable: true, filter: true },
//         {
//             field: 'score',
//             headerName: useWtMarginalAsScore ? 'WT Marginal Likelihood' : 'Probability',
//             sortable: true,
//             filter: 'agNumberColumnFilter',
//             valueFormatter: (params: any) => params.value.toExponential(6),
//             sort: 'desc',
//             sortIndex: 0
//         },
//         { field: 'model', headerName: 'Model', sortable: true, filter: true }
//     ];

//     return (
//         <div
//             className="ag-theme-alpine"
//             style={{
//                 width: '100%',
//                 height: '500px',
//                 marginTop: '20px'
//             }}
//         >
//             <AgGridReact
//                 modules={[ClientSideRowModelModule]}
//                 rowData={tableSubset}
//                 columnDefs={columnDefs}
//                 defaultColDef={{
//                     flex: 1,
//                     minWidth: 100,
//                     resizable: true,
//                     sortable: true,
//                     filter: true,
//                     suppressMovable: true,
//                     // cellStyle: { userSelect: 'text' }
//                 }}
//                 enableCellTextSelection={true}
//                 copyHeadersToClipboard={true}
//                 domLayout='autoHeight'
//                 ensureDomOrder={true}
//             // suppressCellFocus={true}
//             // suppressRowClickSelection={true}
//             />
//         </div>
//     );
// };
interface LogitTableProps {
    logitCsvData: string | null;
    useWtMarginalAsScore: boolean;
    zeroWildType: boolean;
    maxMutationsPerLocus: number;
    topPerformersToDisplay: number;
}

const LogitTable: React.FC<LogitTableProps> = ({
    logitCsvData,
    useWtMarginalAsScore,
    zeroWildType,
    maxMutationsPerLocus,
    topPerformersToDisplay,
}) => {
    if (!logitCsvData) return null;

    const [sortColumn, setSortColumn] = useState<string | null>(null);
    const [sortDirection, setSortDirection] = useState<'ASC' | 'DESC'>('DESC');

    const tableData: RowData[] | null = useMemo(() => {
        const data = parseCsvDataIntoRowData(logitCsvData, useWtMarginalAsScore, zeroWildType, maxMutationsPerLocus, topPerformersToDisplay);
        if (!data) return null;


        if (data && sortColumn) {
            return [...data].sort((a, b) => {
                const aValue = a[sortColumn as keyof RowData];
                const bValue = b[sortColumn as keyof RowData];
                if (aValue === null) return 1;
                if (bValue === null) return -1;
                return sortDirection === 'ASC'
                    ? (aValue < bValue ? -1 : 1)
                    : (aValue > bValue ? -1 : 1);
            });
        }
        return data;
    }, [logitCsvData, useWtMarginalAsScore, zeroWildType, maxMutationsPerLocus, topPerformersToDisplay, sortColumn, sortDirection]);

    if (!tableData) return null;

    const handleSort = (sortCol: string) => {
        if (sortColumn === sortCol) {
            setSortDirection(sortDirection === 'ASC' ? 'DESC' : 'ASC');
        } else {
            setSortColumn(sortCol);
            setSortDirection('ASC');
        }
    };

    const columns = [
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
        <div style={{ width: "auto", height: "auto", marginTop: "20px" }}>
            <ReactDataGrid
                columns={columns}
                rowGetter={i => tableData[i]}
                rowsCount={tableData.length}
                enableCellSelect={true}
                onGridSort={(sortCol, direction) => {
                    setSortColumn(sortCol);
                    setSortDirection(direction.toUpperCase() as 'ASC' | 'DESC');
                }}
            />
        </div>
    );
};

const NaturalnessTab: React.FC<NaturalnessTabProps> = ({ foldId, foldName, foldChainIds, jobs, logits, setSelectedSubsequence, setErrorText }) => {
    const [runName, setRunName] = useState<string>('');
    const [logitModel, setLogitModel] = useState<string>('esmc_600m');
    const [useStructure, setUseStructure] = useState<boolean>(false);
    const [showForm, setShowForm] = useState<boolean>(false);

    const [displayedLogitId, setDisplayedLogitId] = useState<number | null>(null);
    const [logitCsvData, setLogitCsvData] = useState<string | null>(null);
    const [maskWildType, setMaskWildType] = useState<boolean>(false);
    const [zeroWildType, setZeroWildType] = useState<boolean>(false);
    const [showWTMarginalLikelihood, setShowWTMarginalLikelihood] = useState<boolean>(true);

    const [maxMutationsPerLocus, setMaxMutationsPerLocus] = useState<number>(2);
    const [topPerformersToDisplay, setTopPerformersToDisplay] = useState<number>(24);


    const handleStartLogit = async () => {
        try {
            UIkit.notification({ message: 'Starting naturalness run...', timeout: 2000 });
            const logitRun = await startLogits(foldId, runName, useStructure, logitModel);
            console.log(`logitRun: ${logitRun}`);
            console.log(`logitRun keys: ${Object.keys(logitRun)}`);
            UIkit.notification({
                message: `Logit run started with id ${logitRun.id} and name ${logitRun.name}`,
                status: 'success'
            });
        } catch (error) {
            UIkit.notification({
                message: `Failed to start logit run: ${error}`,
                status: 'danger'
            });
        }
    };

    const getLogitStatus = (logit: Logit): string => {
        const job = jobs?.find(job => job.id === logit.invokation_id);
        return job?.state || 'Unknown';
    };

    const downloadLogitCsv = (logit: Logit) => {
        if (!foldName) {
            setErrorText('Fold name is not set.');
            return;
        }
        const logitPath = `naturalness/logits_${logit.name}_melted.csv`;
        console.log(`Downloading logits for ${logit.name} at path ${logitPath}`);
        getFile(logit.fold_id, logitPath).then(
            (fileBlob: Blob) => {
                const newFname = `logits_${foldName}_${logit.name}_melted.csv`;
                UIkit.notification(`Downloading ${logitPath} with file name ${newFname}!`);
                fileDownload(fileBlob, newFname);
            },
            (e) => {
                console.log(e);
                setErrorText(e.toString());
            }
        );
    };

    const rerunLogit = async (logit: Logit) => {
        UIkit.notification({ message: `Repopulating "New Logit Run" with parameters from ${logit.name}.`, timeout: 2000 });
        setRunName(logit.name);
        setShowForm(true);
        setUseStructure(logit.use_structure || false);
        setLogitModel(logit.logit_model);
    };

    const loadLogit = (logitId: number) => {
        const logit = logits?.find(logit => logit.id === logitId);
        if (!logit) {
            UIkit.notification({ message: `Logit ${logitId} not found.`, status: 'danger' });
            return;
        }
        setDisplayedLogitId(logitId);
        console.log(`Loading logit ${logit.name}...`);

        getFile(foldId, `naturalness/logits_${logit.name}_melted.csv`).then(
            (fileBlob: Blob) => {
                // Create a FileReader to read the blob as text
                const reader = new FileReader();
                reader.onload = (e) => {
                    const fileString = e.target?.result as string;
                    setLogitCsvData(fileString);
                };
                reader.readAsText(fileBlob);
            },
            (e) => {
                console.log(e);
                setErrorText(e.toString());
            }
        );
    }

    const logitPlot = useMemo(() => {
        if (!logitCsvData) return null;

        const rowData = parseCsvDataIntoRowData(logitCsvData, showWTMarginalLikelihood, zeroWildType, undefined, undefined);
        if (!rowData) return null;


        // Process data for heatmap
        const locusSet = new Set<number>();
        const scoreHeatmapData: { [key: string]: number } = {};
        const wtResidues: { [key: number]: string } = {};  // Store wild-type residues by position

        var ensembleMembers = 1.0;

        rowData.forEach(row => {
            if (row.model != null) {
                ensembleMembers = Math.max(ensembleMembers, row.model);
            }
            const seqId = row.seqId;

            // Extract locus and mutant residue using regex
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
        console.log(scoreHeatmapData);

        const loci = Array.from(locusSet).sort((a, b) => a - b);
        const zValues = RESIDUES.map(res =>
            loci.map(locus => {
                // If masking is enabled and this is the wild-type residue, return null
                if (res === wtResidues[locus]) {
                    if (maskWildType) {
                        return null;
                    }
                }
                const key = `${locus}-${res}`;
                // Use explicit check to handle zero values correctly
                return key in scoreHeatmapData ? scoreHeatmapData[key] : null;
            })
        );
        const zmin = showWTMarginalLikelihood ? 0 : 0;
        const zmax = showWTMarginalLikelihood ? Math.max(...zValues.flat(2).filter(val => val !== null) as number[]) : 1;

        const hoverTemplate = showWTMarginalLikelihood ? '%{customdata}%{x}%{y}<br>Score: 10^%{z}<extra></extra>' : '%{customdata}%{x}%{y}<br>Probability: %{z}<extra></extra>';

        const plotlyData: Array<Partial<Data>> = [{
            type: 'heatmap',
            z: zValues,
            x: loci,
            y: RESIDUES,
            colorscale: 'Viridis',
            hoverongaps: false,
            zmin: zmin,
            zmax: zmax,
            zauto: false,
            hovertemplate: hoverTemplate,
            customdata: RESIDUES.map(() => loci.map(locus => wtResidues[locus])),
            showscale: true,
        }];

        return (
            <div style={{ width: '100%', maxWidth: '900px' }}>
                <Plot
                    data={plotlyData}
                    layout={{
                        title: `Naturalness${showWTMarginalLikelihood ? ' (WT Marginal Likelihood, log scale)' : ' (Residue Probability)'}`,
                        xaxis: { title: 'Position in Sequence' },
                        yaxis: { title: 'Mutant Residue' },
                        height: 500,
                        autosize: true,
                        margin: { l: 50, r: 50, t: 50, b: 50 }
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '100%' }}
                />
            </div>
        );
    }, [logitCsvData, maskWildType, zeroWildType, showWTMarginalLikelihood]);

    const highlightResiduesOnModel = () => {
        if (!logitCsvData) return;

        const tableData: RowData[] | null = parseCsvDataIntoRowData(logitCsvData, showWTMarginalLikelihood, zeroWildType, maxMutationsPerLocus, topPerformersToDisplay) || null;
        if (!tableData) return null;

        const lociToHighlight = tableData.map(row => {
            const { wtResidue, locus, mutantResidue } = parseSeqId(row.seqId);
            return locus;
        }).filter(residue => residue !== null);

        // Get unique residues to highlight
        const uniqueLociToHighlight = Array.from(new Set(lociToHighlight));

        const selection = uniqueLociToHighlight.map(locus => {
            return {
                struct_asym_id: foldChainIds?.[0] || 'A',
                start_residue_number: locus,
                end_residue_number: locus,
                color: "#FFD700",
            }
        })

        setSelectedSubsequence({
            data: selection,
            // nonSelectedColor: "white",
        });
    }

    return (
        <div style={{ padding: '20px', backgroundColor: '#f8f9fa', boxShadow: '0 2px 6px rgba(0, 0, 0, 0.1)', borderRadius: '8px' }}>
            {/* Description Section */}
            <section style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                <h3 style={{ marginBottom: '10px' }}>Naturalness Overview</h3>
                <div>
                    Naturalness (TODO: describe PLMs, naturalness, logits, etc)
                    <ul>
                        <li><code>logit model</code> which PLM you want to use to predict logits</li>
                    </ul>
                    <p>
                        Once complete, you can download the "naturalness" scores for all mutants from the Files tab.
                    </p>
                    <p>
                        <code>Estimated cost:</code>~$1 per run.
                    </p>
                </div>
            </section>

            {/* Evolution Runs Table */}
            <section style={{ marginBottom: '30px', padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                <h3>Logit Runs</h3>
                <table className="uk-table uk-table-striped">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {logits?.map(logit => (
                            <tr key={logit.id}>
                                <td>{logit.name}</td>
                                <td>{getLogitStatus(logit)}</td>
                                <td>
                                    {
                                        getLogitStatus(logit) == 'finished' ?
                                            <>
                                                <FaEye
                                                    uk-tooltip="View results"
                                                    onClick={() => loadLogit(logit.id)} />
                                                <FaDownload
                                                    uk-tooltip="Download logit CSV."
                                                    onClick={() => downloadLogitCsv(logit)} />
                                            </> : null
                                    }
                                    <FaRedo uk-tooltip="Retry the logit run."
                                        onClick={() => rerunLogit(logit)} />
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </section>

            {/* Display logit info, if requested. */}
            {
                displayedLogitId ?
                    <>
                        <div style={{ marginBottom: '10px' }}>
                            <label>
                                <input
                                    type="checkbox"
                                    className="uk-checkbox"
                                    checked={maskWildType}
                                    onChange={(e) => setMaskWildType(e.target.checked)}
                                /> Mask wild-type amino acids
                            </label>
                        </div>
                        <div style={{ marginBottom: '10px' }}>
                            <label>
                                <input
                                    type="checkbox"
                                    className="uk-checkbox"
                                    checked={zeroWildType}
                                    onChange={(e) => setZeroWildType(e.target.checked)}
                                /> Zero out wild-type amino acids
                            </label>
                        </div>
                        <div style={{ marginBottom: '10px' }}>
                            <label>
                                <input
                                    type="checkbox"
                                    className="uk-checkbox"
                                    checked={showWTMarginalLikelihood}
                                    onChange={(e) => setShowWTMarginalLikelihood(e.target.checked)}
                                /> Display Log(WT Marginal Likelihood)
                            </label>
                        </div>
                        {logitPlot}
                        <div>
                            <label>
                                Max number of mutants per locus:
                                <input
                                    type="number"
                                    className="uk-input"
                                    value={maxMutationsPerLocus}
                                    onChange={(e) => setMaxMutationsPerLocus(parseInt(e.target.value))}
                                    style={{ width: '100px', marginLeft: '10px' }}
                                    min="1"
                                />
                            </label>
                        </div>
                        <div>
                            <label>
                                Top mutants to display:
                                <input
                                    type="number"
                                    className="uk-input"
                                    value={topPerformersToDisplay}
                                    onChange={(e) => setTopPerformersToDisplay(parseInt(e.target.value))}
                                    style={{ width: '100px', marginLeft: '10px' }}
                                    min="1"
                                />
                            </label>
                        </div>
                        <LogitTable
                            logitCsvData={logitCsvData}
                            useWtMarginalAsScore={showWTMarginalLikelihood}
                            zeroWildType={zeroWildType}
                            maxMutationsPerLocus={maxMutationsPerLocus}
                            topPerformersToDisplay={topPerformersToDisplay}
                        />
                        <div style={{ marginTop: '10px', display: 'flex', gap: '10px' }}>
                            <button
                                className="uk-button uk-button-default"
                                onClick={() => {
                                    if (!logitCsvData) return;

                                    const tableData = parseCsvDataIntoRowData(logitCsvData, showWTMarginalLikelihood, zeroWildType, maxMutationsPerLocus, topPerformersToDisplay)

                                    if (!tableData) return;

                                    const mutations = tableData
                                        .map(row => row.seqId)
                                        .join('\n');

                                    navigator.clipboard.writeText(mutations);
                                    UIkit.notification({ message: 'Mutations copied to clipboard!', status: 'success' });
                                }}
                            >
                                Copy mutations to clipboard
                            </button>
                            <button className="uk-button uk-button-primary" onClick={() => highlightResiduesOnModel()}>
                                Highlight residues on model
                            </button>
                        </div>
                    </>
                    : null
            }

            {/* Collapsible New Run Section */}
            <div>
                <div
                    className='uk-margin-top uk-margin-bottom'
                    style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "10px 15px",
                        backgroundColor: "#f8f9fa",
                        border: "1px solid #e0e0e0",
                        borderRadius: "8px",
                        boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
                        cursor: "pointer",
                        fontWeight: "bold",
                    }}
                    onClick={() => setShowForm(!showForm)}
                >
                    <span>New Logit Run</span>
                    <span>{showForm ? "▲" : "▼"}</span>
                </div>
                {showForm && (
                    <section style={{ padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <h3>Start New Logit Run</h3>
                        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                            {/* Name Input */}
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <label className="uk-form-label">Name</label>
                                <input
                                    type="text"
                                    className="uk-input"
                                    value={runName}
                                    onChange={(e) => setRunName(e.target.value)}
                                />
                            </div>

                            {/* Add Model Selection Dropdown */}
                            <ESMModelPicker
                                value={logitModel}
                                onChange={setLogitModel}
                            />

                            {/* Use Structure Checkbox */}
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <label className="uk-form-label">
                                    <input
                                        type="checkbox"
                                        className="uk-checkbox uk-margin-small-right"
                                        checked={useStructure}
                                        onChange={(e) => setUseStructure(e.target.checked)}
                                    />
                                    Use Structure (experimental)
                                </label>
                            </div>
                        </div>

                        <button
                            className="uk-button uk-button-primary uk-margin-top"
                            onClick={handleStartLogit}
                            disabled={runName === ''}
                        >
                            Start Logit Run
                        </button>
                    </section>
                )}
            </div>
        </div >
    );
};

export default NaturalnessTab;