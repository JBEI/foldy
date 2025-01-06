// EmbeddingTab.tsx

import React, { useState, ChangeEvent } from 'react';
// If UIkit has no TypeScript definitions, you can declare it as any
import UIkit from 'uikit';
import { FileInfo, Evolution, Invokation } from 'src/types/types';
import { evolve } from '../../api/evolveApi';

// Define the props interface
interface EvolveTabProps {
    foldId: number;
    jobs: Invokation[] | null;
    files: FileInfo[] | null;
    evolutions: Evolution[] | null;
}

// Define the EmbeddingTab functional component
const EvolveTab: React.FC<EvolveTabProps> = ({ foldId, jobs, files, evolutions }) => {
    const [evolutionName, setEvolutionName] = useState<string>('');
    const [selectedEmbeddingPaths, setSelectedEmbeddingPaths] = useState<string[]>([]);
    const [activityFile, setActivityFile] = useState<File | null>(null);

    // Filter files to only show embedding files
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
            // Only accept excel files
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
        if (!activityFile || selectedEmbeddingPaths.length === 0) {
            UIkit.notification({
                message: 'Please select both embedding files and an activity file',
                status: 'warning'
            });
            return;
        }

        try {
            UIkit.notification({
                message: `Starting evolution...`,
                timeout: 2000 // milliseconds
            });
            const foldEvolution = await evolve(evolutionName, foldId, selectedEmbeddingPaths, activityFile);
            console.log(foldEvolution);
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

    // Function to get status from jobs array
    const getEvolutionStatus = (evolution: Evolution): string => {
        const job = jobs?.find(job => job.id === evolution.invokation_id);
        return job?.state || 'Unknown';
    };

    return (
        <div className="uk-margin">
            <h3>Evolution Runs</h3>
            <table className="uk-table uk-table-striped">
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {evolutions?.map((evolution) => (
                        <tr key={evolution.id}>
                            <td>{evolution.name}</td>
                            <td>{getEvolutionStatus(evolution)}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
            <hr />
            <h3>Start New Evolution Run</h3>
            <label className="uk-form-label">Name</label>
            <input
                type="text"
                className="uk-input"
                value={evolutionName}
                onChange={(e) => setEvolutionName(e.target.value)}
            />
            <label className="uk-form-label">Select Embedding Files</label>
            <select
                className="uk-select"
                multiple
                size={Math.min(10, embeddingFiles.length || 1)}
                value={selectedEmbeddingPaths}
                onChange={handleFileSelection}
            >
                {embeddingFiles.map((file) => (
                    <option key={file.key} value={file.key}>
                        {file.key.split('/').pop()}
                    </option>
                ))}
            </select>

            <div className="uk-margin">
                <p className="uk-text-meta">
                    Selected {selectedEmbeddingPaths.length} embedding file(s)
                </p>
            </div>

            <div className="uk-margin">
                <label className="uk-form-label">Upload Activity File</label>
                <div className="uk-form-controls">
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
            </div>

            <button
                className="uk-button uk-button-primary"
                onClick={handleEvolve}
                disabled={!activityFile || selectedEmbeddingPaths.length === 0}
            >
                Start Evolution
            </button>
        </div>
    );
};

export default EvolveTab;
