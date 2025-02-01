import React, { useState, ChangeEvent } from 'react';
import { startEmbeddings } from "../../services/backend.service";
import UIkit from 'uikit';
import { Embedding, Invokation } from '../../types/types';
import { FaDownload, FaRedo } from 'react-icons/fa';
import { getFile } from '../../api/fileApi';
import fileDownload from 'js-file-download';
import { startLogits } from '../../api/embedApi';

interface EmbedTabProps {
    foldId: number;
    jobs: Invokation[] | null;
    embeddings: Embedding[] | null;
    setErrorText: (text: string) => void;
}

const EmbedTab: React.FC<EmbedTabProps> = ({ foldId, jobs, embeddings, setErrorText }) => {
    const [batchName, setBatchName] = useState<string | null>(null);
    const [dmsStartingSeqIds, setDmsStartingSeqIds] = useState<string>('WT');
    const [extraSequenceIDs, setExtraSequenceIDs] = useState<string>('');
    const [showEmbeddingSection, setShowEmbeddingSection] = useState<boolean>(false);

    const handleDmsStartingSeqIDsTextareaChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
        setDmsStartingSeqIds(event.target.value);
    };

    const handleExtraSeqIDsTextareaChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
        setExtraSequenceIDs(event.target.value);
    };

    const handleStartDmsEmbeddings = async (model: string) => {
        const dmsStartingSeqIdsArray: string[] = dmsStartingSeqIds
            .split('\n')
            .map(line => line.trim())
            .filter(line => line !== '');
        const extraIDsArray: string[] = extraSequenceIDs
            .split('\n')
            .map(line => line.trim())
            .filter(line => line !== '');

        if (!batchName) {
            UIkit.notification({
                message: 'Batch name is required.',
                status: 'danger',
                pos: 'top-right',
                timeout: 3000
            });
            return;
        }

        try {
            await startEmbeddings(foldId, batchName, dmsStartingSeqIdsArray, extraIDsArray, model);
            UIkit.notification({
                message: 'Started embedding run.',
                status: 'success',
                pos: 'top-right',
                timeout: 3000
            });
        } catch (error) {
            console.error('Error starting embedding run:', error);
            UIkit.notification({
                message: 'Failed to start embedding run.',
                status: 'danger',
                pos: 'top-right',
                timeout: 3000
            });
        }
    };

    const getEmbeddingStatus = (embedding: Embedding): string => {
        const job = jobs?.find(job => job.id === embedding.invokation_id);
        return job?.state || 'Unknown';
    };

    const downloadEmbedding = (embedding: Embedding) => {
        const paddedFoldId = foldId.toString().padStart(6, '0');
        const embeddingPath = `embed/${paddedFoldId}_embeddings_${embedding.embedding_model}_${embedding.name}.csv`;
        console.log(`Downloading embedding ${embedding.id} at path ${embeddingPath}`);
        getFile(embedding.fold_id, embeddingPath).then(
            (fileBlob: Blob) => {
                const newFname = embeddingPath.split('/').pop() || 'embeddings.csv';
                UIkit.notification(`Downloading ${embeddingPath} with file name ${newFname}!`);
                fileDownload(fileBlob, newFname);
            },
            (e) => {
                console.log(e);
                setErrorText(e.toString());
            }
        );
    };

    const rerunEmbedding = async (embedding: Embedding) => {
        UIkit.notification({ message: `Repopulating "New Embedding Run" with parameters from ${embedding.name}.`, timeout: 2000 });
        console.log(embedding);
        setBatchName(embedding.name);
        setDmsStartingSeqIds(embedding.dms_starting_seq_ids.split(',').join('\n'));
        setExtraSequenceIDs(embedding.extra_seq_ids.split(',').join('\n'));
        setShowEmbeddingSection(true);
    };

    const handleStartLogits = async () => {
        await startLogits(foldId);
        UIkit.notification({ message: `Started logits run.`, timeout: 2000 });
    };

    return (
        <div style={{ padding: '20px', backgroundColor: '#f8f9fa', boxShadow: '0 2px 6px rgba(0, 0, 0, 0.1)', borderRadius: '8px' }}>
            {/* Description Section */}
            <section style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                <h3 style={{ marginBottom: '10px' }}>DMS Embedding Overview</h3>
                <div>
                    This tab allows you to embed protein sequences using large language
                    models like <a href="https://github.com/evolutionaryscale/esm">ESMC</a>.
                    These embeddings can be used to do low-N directed evolution, as in the
                    Evolve tab. Each run takes in:
                    <ul>
                        <li>
                            <code>Extra Sequence IDs</code>: "WT" to embed the WT sequence, as well as other
                            variants to embed. Eg, "A43W_T67G" to embed the mutant with those two mutations.
                        </li>
                        <li>
                            <code>DMS Starting Sequence IDs</code>: For each line in this
                            field, all possible single amino acid mutants will be embedded.
                            For each input here, this produces a large number of embeddings,
                            ~19X the number of amino acids in the protein.
                        </li>
                    </ul>
                    You can embed just the wild type sequence by entering "WT" in
                    the "Extra Sequence IDs" field, as well as any other variants of interest.
                    Additionally you can get embeddings for a large number of mutants with the
                    "DMS Starting Sequence IDs" field - for each line in this field, all
                    possible single amino acid mutants will be embedded.
                    <p>
                        <code>Estimated cost:</code>~$100 for a DMS of a 500AA protein.
                    </p>
                </div>
            </section>

            {/* Batch Status Section */}
            <section style={{ padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                <h4>Ongoing Batches</h4>
                <div style={{ overflowX: 'auto' }}>
                    <table className="uk-table uk-table-striped">
                        <thead>
                            <tr>
                                <th>Batch Name</th>
                                <th>Batch Status</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {embeddings?.map(embedding => (
                                <tr key={embedding.id}>
                                    <td>{embedding.name}</td>
                                    <td>{getEmbeddingStatus(embedding)}</td>
                                    <td>
                                        <FaDownload
                                            uk-tooltip="Download embeddings CSV."
                                            onClick={() => downloadEmbedding(embedding)} />
                                        <FaRedo
                                            uk-tooltip="Rerun embedding."
                                            onClick={() => rerunEmbedding(embedding)} />
                                    </td>
                                </tr>
                            ))
                                || <tr><td colSpan={2}>No embeddings available</td></tr>}
                        </tbody>
                    </table>
                </div>
            </section>

            {/* Collapsible Section */}
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
                    onClick={() => setShowEmbeddingSection(!showEmbeddingSection)}
                >
                    <span>New Embedding Run</span>
                    <span>{showEmbeddingSection ? "▲" : "▼"}</span>
                </div>
                {showEmbeddingSection && (
                    <div style={{ padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <h4>Start a New Embedding Run</h4>
                        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                            {/* Batch Name */}
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <label htmlFor="batch-name" className="uk-form-label">Batch Name</label>
                                <input
                                    id="batch-name"
                                    className="uk-input"
                                    type="text"
                                    placeholder="Enter batch name"
                                    value={batchName || ''}
                                    onChange={(e) => setBatchName(e.target.value)}
                                />
                            </div>

                            {/* Extra Sequence IDs */}
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <label htmlFor="extra-sequence-ids" className="uk-form-label">Extra Sequence IDs</label>
                                <textarea
                                    id="extra-sequence-ids"
                                    className="uk-textarea"
                                    rows={5}
                                    placeholder="Enter one mutation per line, e.g., A37T, W100C_T431G"
                                    value={extraSequenceIDs}
                                    onChange={handleExtraSeqIDsTextareaChange}
                                ></textarea>
                            </div>

                            {/* DMS Starting Sequence IDs */}
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <label htmlFor="dms-starting-seq-ids" className="uk-form-label">DMS Starting Sequence IDs</label>
                                <textarea
                                    id="dms-starting-seq-ids"
                                    className="uk-textarea"
                                    rows={5}
                                    placeholder="Enter one mutation per line, e.g., WT, W100C_T431G"
                                    value={dmsStartingSeqIds}
                                    onChange={handleDmsStartingSeqIDsTextareaChange}
                                ></textarea>
                            </div>
                        </div>
                        <div style={{ marginTop: '20px' }}>
                            <button
                                className="uk-button uk-button-primary"
                                onClick={() => handleStartDmsEmbeddings("esmc_300m")}
                            >
                                Start Embedding (300M model)
                            </button>
                            <button
                                className="uk-button uk-button-primary uk-margin-left"
                                onClick={() => handleStartDmsEmbeddings("esmc_600m")}
                            >
                                Start Embedding (600M model)
                            </button>

                        </div>
                        <hr />
                        <h4>Logits</h4>
                        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                            <button
                                className="uk-button uk-button-primary"
                                onClick={() => handleStartLogits()}
                            >
                                Start Logits
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default EmbedTab;