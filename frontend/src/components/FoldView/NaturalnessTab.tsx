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

interface NaturalnessTabProps {
    foldId: number;
    foldName: string | null;
    jobs: Invokation[] | null;
    logits: Logit[] | null;
    setErrorText: (error: string) => void;
}

const NaturalnessTab: React.FC<NaturalnessTabProps> = ({ foldId, foldName, jobs, logits, setErrorText }) => {
    const [runName, setRunName] = useState<string>('');
    const [logitModel, setLogitModel] = useState<string>('esmc_600m');
    const [useStructure, setUseStructure] = useState<boolean>(false);
    const [showForm, setShowForm] = useState<boolean>(false);

    const [displayedLogitId, setDisplayedLogitId] = useState<number | null>(null);
    const [logitCsvData, setLogitCsvData] = useState<string | null>(null);
    const [maskWildType, setMaskWildType] = useState<boolean>(false);
    const [zeroWildType, setZeroWildType] = useState<boolean>(false);
    const [showWTMarginalLikelihood, setShowWTMarginalLikelihood] = useState<boolean>(false);


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

        const { data, errors, meta } = Papa.parse<Record<string, string>>(logitCsvData, {
            header: true,
            delimiter: ',',
            skipEmptyLines: true,
            dynamicTyping: true
        });

        if (errors.length > 0) {
            return <div>Error parsing logit CSV: {errors.map(error => error.message).join(', ')}</div>;
        }

        // Extract the naturalness column name (any column that's not seq_id)
        const columns = Object.keys(data[0]);
        const naturalnessColumn = 'probability';
        const wtMarginalColumn = 'wt_marginal';

        // Define standard amino acid residues
        const residues = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'];

        // Process data for heatmap
        const locusSet = new Set<number>();
        const probabilityHeatmapData: { [key: string]: number } = {};
        const wtMarginalHeatmapData: { [key: string]: number } = {};
        const wtResidues: { [key: number]: string } = {};  // Store wild-type residues by position

        data.forEach(row => {
            const seqId = row['seq_id'];
            // Extract locus and mutant residue using regex
            const match = seqId.match(/([A-Z])(\d+)([A-Z])/);
            if (match) {
                const wtResidue = match[1];
                const locus = parseInt(match[2]);
                const mutantResidue = match[3];
                locusSet.add(locus);
                probabilityHeatmapData[`${locus}-${mutantResidue}`] = parseFloat(row[naturalnessColumn]);
                wtMarginalHeatmapData[`${locus}-${mutantResidue}`] = parseFloat(row[wtMarginalColumn]);
                wtResidues[locus] = wtResidue;
            }
        });
        console.log(probabilityHeatmapData);
        console.log(wtMarginalHeatmapData);

        const loci = Array.from(locusSet).sort((a, b) => a - b);
        const zValues = residues.map(res =>
            loci.map(locus => {
                // If masking is enabled and this is the wild-type residue, return null
                if (res === wtResidues[locus]) {
                    if (zeroWildType) {
                        return 0.0;
                    } else if (maskWildType) {
                        return null;
                    }
                }
                if (showWTMarginalLikelihood) {
                    return wtMarginalHeatmapData[`${locus}-${res}`] || null;
                } else {
                    return probabilityHeatmapData[`${locus}-${res}`] || null;
                }
            })
        );

        const plotlyData: Array<Partial<Data>> = [{
            type: 'heatmap',
            z: zValues,
            x: loci,
            y: residues,
            colorscale: 'Viridis',
            hoverongaps: false,
            zmin: showWTMarginalLikelihood ? undefined : 0,  // Set minimum value for color scale
            zmax: showWTMarginalLikelihood ? undefined : 1,  // Set maximum value for color scale
            hovertemplate: '%{customdata}%{x}%{y}<br>' +
                'Score: 10^(%{z})<extra></extra>',
            customdata: residues.map(() => loci.map(locus => wtResidues[locus])),
            showscale: true,
        }];

        return (
            <div style={{ width: '100%', maxWidth: '900px' }}>
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
                        /> Display WT Marginal Likelihood
                    </label>
                </div>
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
                displayedLogitId ? logitPlot : null
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
                                    Use Structure
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