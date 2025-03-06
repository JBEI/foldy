import React, { useState, ChangeEvent } from 'react';
import UIkit from 'uikit';
import { FileInfo, Evolution, Invokation } from 'src/types/types';
import { evolve } from '../../api/evolveApi';
import { FaDownload, FaRedo } from 'react-icons/fa';
import fileDownload from 'js-file-download';
import { removeLeadingSlash } from '../../api/commonApi';
import { getFile } from '../../api/fileApi';
interface EvolveTabProps {
    foldId: number;
    jobs: Invokation[] | null;
    files: FileInfo[] | null;
    evolutions: Evolution[] | null;
    setErrorText: (error: string) => void;
}

const EvolveTab: React.FC<EvolveTabProps> = ({ foldId, jobs, files, evolutions, setErrorText }) => {
    const [evolutionName, setEvolutionName] = useState<string>('');
    const [showForm, setShowForm] = useState<boolean>(false);
    const [activityFile, setActivityFile] = useState<File | null>(null);
    const [mode, setMode] = useState<'randomforest' | 'mlp' | 'finetuning'>('randomforest');
    const [selectedEmbeddingPaths, setSelectedEmbeddingPaths] = useState<string[]>([]);
    const [finetuningModelCheckpoint, setFinetuningModelCheckpoint] = useState<string>('facebook/esm2_t6_8M_UR50D');

    const embeddingFiles = files?.filter(file =>
        file.key.includes('embed')
    ) || [];

    const handleFileSelection = (event: ChangeEvent<HTMLSelectElement>) => {
        const selectedOptions = Array.from(event.target.selectedOptions).map(option => option.value);
        setSelectedEmbeddingPaths(selectedOptions);
    };

    const handleActivityFileUpload = (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            if (!file.name.match(/\.(xlsx|xls)$/i)) {
                UIkit.notification({
                    message: 'Please upload an Excel file (.xlsx or .xls)',
                    status: 'danger'
                });
                return;
            }
            setActivityFile(file);
        }
    };

    const handleEvolve = async () => {
        if (!activityFile || ((mode === 'randomforest' || mode === 'mlp') && selectedEmbeddingPaths.length === 0)) {
            UIkit.notification({
                message: 'Please fill in all required fields',
                status: 'warning'
            });
            return;
        }

        try {
            UIkit.notification({ message: 'Starting evolution...', timeout: 2000 });
            const foldEvolution = await evolve(
                evolutionName,
                foldId,
                activityFile,
                mode,
                (mode === 'randomforest' || mode === 'mlp') ? selectedEmbeddingPaths : undefined,
                mode === 'finetuning' ? finetuningModelCheckpoint : undefined
            );
            UIkit.notification({
                message: `Evolution process started with id ${foldEvolution.id} and name ${foldEvolution.name}`,
                status: 'success'
            });
        } catch (error) {
            UIkit.notification({
                message: `Failed to start evolution process: ${error}`,
                status: 'danger'
            });
        }
    };

    const getEvolutionStatus = (evolution: Evolution): string => {
        const job = jobs?.find(job => job.id === evolution.invokation_id);
        return job?.state || 'Unknown';
    };

    const downloadPredictedActivity = (evolution: Evolution) => {
        const predictedActivityPath = `evolve/${evolution.name}/predicted_activity.csv`;
        console.log(`Downloading predicted activity for evolution ${evolution.id} at path ${predictedActivityPath}`);
        getFile(evolution.fold_id, predictedActivityPath).then(
            (fileBlob: Blob) => {
                const newFname = `${evolution.name}_predicted_activity.csv`;
                UIkit.notification(`Downloading ${predictedActivityPath} with file name ${newFname}!`);
                fileDownload(fileBlob, newFname);
            },
            (e) => {
                console.log(e);
                setErrorText(e.toString());
            }
        );
    };

    const rerunEvolution = async (evolution: Evolution) => {
        UIkit.notification({ message: `Repopulating "New Evolution Run" with parameters from ${evolution.name}. Make sure to add the activity file, you can download the previous one from Files tab.`, timeout: 2000 });
        setEvolutionName(evolution.name);
        setMode(evolution.mode);
        if (evolution.embedding_files) {
            setSelectedEmbeddingPaths(evolution.embedding_files.split(','));
        }
        if (evolution.finetuning_model_checkpoint) {
            console.log(`Setting finetuning model checkpoint to ${evolution.finetuning_model_checkpoint}`);
            setFinetuningModelCheckpoint(evolution.finetuning_model_checkpoint);
        }
        setShowForm(true);
    };

    return (
        <div style={{ padding: '20px', backgroundColor: '#f8f9fa', boxShadow: '0 2px 6px rgba(0, 0, 0, 0.1)', borderRadius: '8px' }}>
            {/* Description Section */}
            <section style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                <h3 style={{ marginBottom: '10px' }}>Evolution Runs Overview</h3>
                <div>
                    This section allows you to run a version of
                    <a href="https://www.biorxiv.org/content/10.1101/2024.07.17.604015v1"> EvolvePro </a>
                    on your protein. This tool facilitates low-N directed evolution of proteins,
                    with as little as 16 screened mutants per round. Please see the paper for more
                    details. Each run takes in an
                    <ul>
                        <li><code>activity excel file</code> with columns seq_id and activity</li>
                        <li><code>embedding files</code> embeddings run in the excel tab, containing embeddings for both the mutants with activity measurements as well as all mutants you wish to screen.</li>
                    </ul>
                    <p>
                        Once complete, you can download the predicted activities for all mutants from the Files tab.
                    </p>
                    <h4>
                        Example activity file
                    </h4>
                    <img
                        style={{
                            width: "200px",
                        }}
                        src={`/evolve_activity_excel_example.png`}
                        alt=""
                    />
                    <p>
                        <code>Estimated cost:</code>~$0.05 per evolution round.
                    </p>
                </div>
            </section>

            {/* Evolution Runs Table */}
            <section style={{ marginBottom: '30px', padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                <h3>Evolution Runs</h3>
                <table className="uk-table uk-table-striped">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {evolutions?.map(evolution => (
                            <tr key={evolution.id}>
                                <td>{evolution.name}</td>
                                <td>{getEvolutionStatus(evolution)}</td>
                                <td>
                                    <FaDownload
                                        uk-tooltip="Download predicted activity CSV."
                                        onClick={() => downloadPredictedActivity(evolution)} />
                                    <FaRedo uk-tooltip="Retry the evolution run."
                                        onClick={() => rerunEvolution(evolution)} />
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </section>

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
                    <span>New Evolution Run</span>
                    <span>{showForm ? "▲" : "▼"}</span>
                </div>
                {showForm && (
                    <section style={{ padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <h3>Start New Evolution Run</h3>
                        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>

                            {/* Name Input */}
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <label className="uk-form-label">Name</label>
                                <input
                                    type="text"
                                    className="uk-input"
                                    value={evolutionName}
                                    onChange={(e) => setEvolutionName(e.target.value)}
                                />
                            </div>

                            {/* Activity File Upload */}
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <label className="uk-form-label">Upload Activity File</label>
                                <input
                                    type="file"
                                    accept=".xlsx,.xls"
                                    onChange={handleActivityFileUpload}
                                    className="uk-input"
                                />
                                {activityFile && (
                                    <p className="uk-text-meta">Selected file: {activityFile.name}</p>
                                )}
                            </div>

                            {/* Mode Selection */}
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <label className="uk-form-label">Mode</label>
                                <select
                                    className="uk-select"
                                    value={mode}
                                    onChange={(e) => setMode(e.target.value as 'finetuning' | 'randomforest')}
                                >
                                    <option value="randomforest">Random Forest</option>
                                    <option value="mlp">Multi-Layer Perceptron</option>
                                    <option value="finetuning">Finetuning</option>
                                </select>
                            </div>

                            {/* Conditional inputs based on mode */}
                            {mode === 'finetuning' && (
                                <div style={{ flex: 1, minWidth: '200px' }}>
                                    <label className="uk-form-label">Model Checkpoint</label>
                                    <select
                                        className="uk-select"
                                        value={finetuningModelCheckpoint}
                                        onChange={(e) => setFinetuningModelCheckpoint(e.target.value)}
                                    >
                                        <option value="facebook/esm2_t6_8M_UR50D">ESM2 (8M params)</option>
                                        <option value="facebook/esm2_t33_650M_UR50D">ESM2 (650M params)</option>
                                        <option value="facebook/esm2_t48_15B_UR50D">ESM2 (15B params)</option>
                                    </select>
                                </div>
                            )}

                            {/* Show embedding files selection only for randomforest mode */}
                            {(mode === 'randomforest' || mode === 'mlp') && (
                                <div style={{ flex: '0 0 auto', width: '100%' }}>
                                    <label className="uk-form-label">Select Embedding Files</label>
                                    <select
                                        className="uk-select"
                                        multiple
                                        size={Math.min(10, embeddingFiles.length || 1)}
                                        value={selectedEmbeddingPaths}
                                        onChange={handleFileSelection}
                                    >
                                        {embeddingFiles.map(file => (
                                            <option key={file.key} value={file.key}>
                                                {file.key.split('/').pop()}
                                            </option>
                                        ))}
                                    </select>
                                    <p className="uk-text-meta">
                                        Selected {selectedEmbeddingPaths.length} embedding file(s)
                                    </p>
                                </div>
                            )}
                        </div>

                        <button
                            className="uk-button uk-button-primary uk-margin-top"
                            onClick={handleEvolve}
                            disabled={
                                evolutionName === '' ||
                                !activityFile ||
                                ((mode === 'randomforest' || mode === 'mlp') && selectedEmbeddingPaths.length === 0) ||
                                (mode === 'finetuning' && !finetuningModelCheckpoint)
                            }
                        >
                            Start Evolution
                        </button>
                    </section>
                )}
            </div>
        </div >
    );
};

export default EvolveTab;