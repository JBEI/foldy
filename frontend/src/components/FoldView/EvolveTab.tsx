// EmbeddingTab.tsx

import React, { useState, ChangeEvent } from 'react';
import { Invokation, startEmbeddings, evolve } from "../../services/backend.service";
// If UIkit has no TypeScript definitions, you can declare it as any
import UIkit from 'uikit';
import { FileInfo } from '../../services/backend.service';

// Define the props interface
interface EvolveTabProps {
    foldId: number;
    jobs: Invokation[] | null;
    files: FileInfo[] | null;
}

// Define the EmbeddingTab functional component
const EvolveTab: React.FC<EvolveTabProps> = ({ foldId, jobs, files }) => {
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
            await evolve(foldId, selectedEmbeddingPaths, activityFile);
            UIkit.notification({
                message: 'Evolution process started',
                status: 'success'
            });
        } catch (error) {
            UIkit.notification({
                message: 'Failed to start evolution process',
                status: 'danger'
            });
        }
    };

    return (
        <div className="uk-margin">
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
