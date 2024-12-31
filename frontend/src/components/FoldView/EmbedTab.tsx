// EmbeddingTab.tsx

import React, { useState, ChangeEvent } from 'react';
import { Invokation, startEmbeddings } from "../../services/backend.service";
// If UIkit has no TypeScript definitions, you can declare it as any
import UIkit from 'uikit';

// Define the props interface
interface EmbedTabProps {
    foldId: number;
    jobs: Invokation[] | null;
}

// Define the EmbeddingTab functional component
const EmbedTab: React.FC<EmbedTabProps> = ({ foldId, jobs }) => {
    // State to hold the content of the textarea
    const [batchName, setBatchName] = useState<string | null>(null);
    const [dmsStartingSeqIds, setDmsStartingSeqIds] = useState<string>('WT');
    const [extraSequenceIDs, setExtraSequenceIDs] = useState<string>('');

    /**
     * Handles changes in the textarea.
     * @param event - The change event from the textarea.
     */
    const handleDmsStartingSeqIDsTextareaChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
        setDmsStartingSeqIds(event.target.value);
    };

    /**
     * Handles changes in the textarea.
     * @param event - The change event from the textarea.
     */
    const handleExtraSeqIDsTextareaChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
        setExtraSequenceIDs(event.target.value);
    };

    /**
     * Initiates the DMS Embeddings process.
     * @param model - The model identifier (e.g., "esmc_300m", "esmc_600m").
     */
    const handleStartDmsEmbeddings = async (model: string) => {
        const dmsStartingSeqIdsArray: string[] = dmsStartingSeqIds
            .split('\n')
            .map(line => line.trim())
            .filter(line => line !== '')
            .map(line => line === 'WT' ? '' : line);
        // Process the textarea input into an array of non-empty, trimmed strings
        const extraIDsArray: string[] = extraSequenceIDs
            .split('\n')
            .map(line => line.trim())
            .filter(line => line !== '');
        console.log(dmsStartingSeqIdsArray);
        console.log(extraIDsArray);

        if (!batchName) {
            // Show error notification
            UIkit.notification({
                message: 'Batch name is required.',
                status: 'danger',
                pos: 'top-right',
                timeout: 3000
            });
            return;
        }

        try {
            // Await the asynchronous startDmsEmbeddings function
            await startEmbeddings(foldId, batchName, dmsStartingSeqIdsArray, extraIDsArray, model);

            // Show success notification
            UIkit.notification({
                message: 'Started embedding run.',
                status: 'success',
                pos: 'top-right',
                timeout: 3000
            });
        } catch (error) {
            console.error('Error starting embedding run:', error);

            // Show error notification
            UIkit.notification({
                message: 'Failed to start embedding run.',
                status: 'danger',
                pos: 'top-right',
                timeout: 3000
            });
        }
    };

    return (
        <li key="DMSli">
            <div className="uk-margin">
                <label htmlFor="batch-name" className="uk-form-label">
                    Batch Name
                </label>
                <div className="uk-form-controls">
                    <input
                        id="batch-name"
                        className="uk-input"
                        type="text"
                        placeholder="Enter batch name"
                        value={batchName || ''}
                        onChange={(e) => setBatchName(e.target.value)}
                    />
                </div>
            </div>
            {/* Extra Sequence IDs Textarea */}
            <div className="uk-margin">
                <label htmlFor="extra-sequence-ids" className="uk-form-label">
                    DMS Starting Sequence IDs
                </label>
                <div className="uk-form-controls">
                    <textarea
                        id="extra-sequence-ids"
                        className="uk-textarea"
                        rows={5}
                        placeholder={`Enter one mutation per line, eg:
WT (this is a special entry that means we do a DMS from the WT sequence)
W100C_T431G (will also do a DMS scan starting with this mutant)`}
                        value={dmsStartingSeqIds}
                        onChange={handleDmsStartingSeqIDsTextareaChange}
                    ></textarea>
                    {/* Future Help Text */}
                    {/* <p className="uk-text-small uk-text-meta">
                    Add your help text here.
                </p> */}
                </div>
            </div>
            {/* Extra Sequence IDs Textarea */}
            <div className="uk-margin">
                <label htmlFor="extra-sequence-ids" className="uk-form-label">
                    Extra Sequence IDs
                </label>
                <div className="uk-form-controls">
                    <textarea
                        id="extra-sequence-ids"
                        className="uk-textarea"
                        rows={5}
                        placeholder={`Enter one mutation per line, eg:
A37T
W100C_T431G`}
                        value={extraSequenceIDs}
                        onChange={handleExtraSeqIDsTextareaChange}
                    ></textarea>
                    {/* Future Help Text */}
                    {/* <p className="uk-text-small uk-text-meta">
                        Add your help text here.
                    </p> */}
                </div>
            </div>

            {/* DMS Embedding Buttons */}
            <button
                type="button"
                className="uk-button uk-button-primary uk-margin-left uk-margin-small-bottom uk-form-small"
                onClick={() => handleStartDmsEmbeddings("esmc_300m")}
            >
                Start DMS Embedding (300M model)
            </button>
            <button
                type="button"
                className="uk-button uk-button-primary uk-margin-left uk-margin-small-bottom uk-form-small"
                onClick={() => handleStartDmsEmbeddings("esmc_600m")}
            >
                Start DMS Embedding (600M model)
            </button>

            {/* Ongoing batches */}
            <hr></hr>
            <div style={{ display: "flex", flexDirection: "row" }}>
                <div style={{ overflowX: "scroll", flexGrow: 1 }}>
                    <table className="uk-table uk-table-striped uk-table-small">
                        <thead>
                            <tr>
                                <th>Batch Name</th>
                                <th>Batch Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {jobs
                                ? [...jobs].filter((job) => job.type?.startsWith('embed')).map((embedJob) => (
                                    <tr key={embedJob.id}>
                                        <td>{embedJob.type}</td>
                                        <td>{embedJob.state}</td>
                                    </tr>))
                                : <div />}
                        </tbody>
                    </table>
                </div>
            </div>
        </li >
    );
};

export default EmbedTab;
