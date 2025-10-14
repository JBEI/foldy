import React, { useState } from 'react';
import { Modal, Alert, Button as AntButton, Typography, Upload } from 'antd';
import { QuestionCircleOutlined, UploadOutlined } from '@ant-design/icons';
import { startEmbeddings } from '../../api/embedApi';
import { notify } from '../../services/NotificationService';
import { ESMModelPicker } from '../FoldView/ESMModelPicker';
import { FormRow, FormField } from '../../util/tabComponents';
import { TextInputControl, TextAreaControl } from '../../util/controlComponents';
import { Embedding } from '../../types/types';

const { Text, Paragraph, Title } = Typography;

interface EmbeddingModalProps {
    open: boolean;
    onClose: () => void;
    foldIds: number[];
    title?: string;
    templateEmbedding?: Embedding;
    disableSequenceFields?: boolean;
}

const isNullOrUndefined = (value: string | null | undefined) => value === null || value === undefined;

export const EmbeddingModal: React.FC<EmbeddingModalProps> = ({
    open,
    onClose,
    foldIds,
    title = "New Embedding Run",
    templateEmbedding,
    disableSequenceFields = false
}) => {
    const [batchName, setBatchName] = useState<string>(templateEmbedding?.name || '');
    const [dmsStartingSeqIds, setDmsStartingSeqIds] = useState<string>(isNullOrUndefined(templateEmbedding?.dms_starting_seq_ids) ? 'WT' : templateEmbedding.dms_starting_seq_ids.split(',').join('\n'));
    const [extraSequenceIDs, setExtraSequenceIDs] = useState<string>(isNullOrUndefined(templateEmbedding?.extra_seq_ids) ? '' : templateEmbedding.extra_seq_ids.split(',').join('\n'));
    const [homologFasta, setHomologFasta] = useState<string | null>(templateEmbedding?.homolog_fasta || null);
    const [homologFile, setHomologFile] = useState<File | null>(null);
    const [extraLayers, setExtraLayers] = useState<string>(templateEmbedding?.extra_layers || '');
    const [domainBoundaries, setDomainBoundaries] = useState<string>(templateEmbedding?.domain_boundaries || '');
    const [model, setModel] = useState<string>(templateEmbedding?.embedding_model || 'esmc_300m');
    const [showHelpModal, setShowHelpModal] = useState<boolean>(false);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const handleStartEmbeddings = async () => {
        if (!batchName.trim()) {
            notify.error('Batch name is required.');
            return;
        }

        const dmsStartingSeqIdsArray: string[] = dmsStartingSeqIds
            .split('\n')
            .map(line => line.trim())
            .filter(line => line !== '');
        const extraIDsArray: string[] = extraSequenceIDs
            .split('\n')
            .map(line => line.trim())
            .filter(line => line !== '');
        const extraLayersArray: string[] = extraLayers
            .split(',')
            .map(line => line.trim())
            .filter(line => line !== '');
        const domainBoundariesArray: string[] = domainBoundaries
            .split(',')
            .map(line => line.trim())
            .filter(line => line !== '');

        setIsLoading(true);

        try {
            const promises = foldIds.map(foldId =>
                startEmbeddings(foldId, batchName, dmsStartingSeqIdsArray, extraIDsArray, extraLayersArray, model, homologFasta, domainBoundariesArray)
            );

            await Promise.all(promises);

            notify.success(`Started embedding runs for ${foldIds.length} fold(s)`);

            // Reset form
            setBatchName('');
            setDmsStartingSeqIds('WT');
            setExtraSequenceIDs('');
            setExtraLayers('');
            setDomainBoundaries('');
            setModel('esmc_300m');
            setHomologFasta(null);
            setHomologFile(null);

            onClose();
        } catch (error) {
            notify.error(`Failed to start embedding runs: ${error}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleHomologFileChange = (file: File | null) => {
        setHomologFile(file);
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const content = e.target?.result as string;
                setHomologFasta(content);
            };
            reader.readAsText(file);
        } else {
            setHomologFasta(null);
        }
    };

    return (
        <>
            <Modal
                title={title}
                open={open}
                onCancel={onClose}
                footer={[
                    <AntButton key="cancel" onClick={onClose}>
                        Cancel
                    </AntButton>,
                    <AntButton
                        key="start"
                        type="primary"
                        onClick={handleStartEmbeddings}
                        disabled={!batchName.trim()}
                        loading={isLoading}
                    >
                        Start Embedding Run{foldIds.length > 1 ? 's' : ''}
                    </AntButton>
                ]}
                width={700}
            >
                {/* Help Alert */}
                <Alert
                    message="What are Protein Embeddings?"
                    description={
                        <div>
                            <Paragraph>
                                Generate high-dimensional vector representations of protein sequences using large language models.
                                These embeddings capture structural and functional information for use in machine learning models.
                            </Paragraph>
                            <AntButton
                                type="link"
                                icon={<QuestionCircleOutlined />}
                                onClick={() => setShowHelpModal(true)}
                                style={{ padding: 0 }}
                            >
                                View detailed embedding guide
                            </AntButton>
                        </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: '20px' }}
                />

                <div style={{ marginBottom: '16px' }}>
                    <Text strong>Target Folds:</Text> {foldIds.length} fold{foldIds.length > 1 ? 's' : ''}
                </div>

                <TextInputControl
                    label="Batch Name"
                    value={batchName}
                    onChange={setBatchName}
                    placeholder="Enter batch name"
                />

                <FormRow>
                    <FormField>
                        <TextAreaControl
                            label="Extra Sequence IDs"
                            value={extraSequenceIDs}
                            onChange={setExtraSequenceIDs}
                            placeholder="Enter one mutation per line, e.g., A37T, W100C_T431G"
                            disabled={disableSequenceFields}
                        />
                    </FormField>

                    <FormField>
                        <TextAreaControl
                            label="DMS Starting Sequence IDs"
                            value={dmsStartingSeqIds}
                            onChange={setDmsStartingSeqIds}
                            placeholder="Enter one mutation per line, e.g., WT, W100C_T431G"
                            disabled={disableSequenceFields}
                        />
                    </FormField>
                </FormRow>

                <ESMModelPicker
                    value={model}
                    onChange={setModel}
                />

                <div style={{ marginBottom: '24px' }}></div>

                <div style={{ marginBottom: '16px' }}>
                    <Typography.Text strong style={{ marginBottom: '8px', display: 'block' }}>
                        Homolog FASTA File (Optional)
                    </Typography.Text>
                    <Upload
                        beforeUpload={(file) => {
                            handleHomologFileChange(file);
                            return false; // Prevent auto upload
                        }}
                        accept=".fasta,.fa,.txt"
                        maxCount={1}
                        fileList={homologFile ? [{
                            uid: '1',
                            name: homologFile.name,
                            status: 'done'
                        }] : []}
                        onRemove={() => handleHomologFileChange(null)}
                    >
                        <AntButton icon={<UploadOutlined />}>
                            Select FASTA File
                        </AntButton>
                    </Upload>
                </div>
                {homologFasta && (
                    <>
                        <div style={{ fontSize: '12px', color: '#666', marginTop: '-8px', marginBottom: '8px' }}>
                            Upload a FASTA file containing homolog sequences with IDs like HOM-name (e.g., HOM-ortho1, HOM-para2)
                        </div>
                        {homologFasta && (
                            <div style={{
                                marginBottom: '16px',
                                padding: '8px',
                                backgroundColor: '#f5f5f5',
                                border: '1px solid #d9d9d9',
                                borderRadius: '4px',
                                fontSize: '11px',
                                fontFamily: 'monospace',
                                color: '#666',
                                maxHeight: '120px',
                                overflow: 'hidden'
                            }}>
                                <div style={{ marginBottom: '4px', fontSize: '10px', fontWeight: 500, textTransform: 'uppercase' }}>
                                    FASTA Preview (first 5 lines)
                                </div>
                                <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                                    {homologFasta.split('\n').slice(0, 5).join('\n')}
                                    {homologFasta.split('\n').length > 5 && '\n...'}
                                </pre>
                            </div>
                        )}
                    </>
                )}

                <TextAreaControl
                    label="Extra Layers"
                    value={extraLayers}
                    onChange={setExtraLayers}
                    placeholder="Enter extra embedding layers to extract like 5,10,15"
                    rows={1}
                    inputStyle={{ resize: 'vertical' }}
                />

                <TextAreaControl
                    label="Domain Boundaries (Optional)"
                    value={domainBoundaries}
                    onChange={setDomainBoundaries}
                    placeholder="Enter domain boundary positions like 80,150 for a 200AA protein with domains at positions 1-80, 81-150, 151-200"
                    rows={1}
                    inputStyle={{ resize: 'vertical' }}
                />
            </Modal>

            {/* Detailed Help Modal */}
            <Modal
                title="Protein Embedding Guide"
                open={showHelpModal}
                onCancel={() => setShowHelpModal(false)}
                footer={[
                    <AntButton key="close" onClick={() => setShowHelpModal(false)}>
                        Close
                    </AntButton>
                ]}
                width={700}
            >
                <div>
                    <Title level={4}>What are Protein Embeddings?</Title>
                    <Paragraph>
                        Protein embeddings are high-dimensional vector representations that capture the structural
                        and functional properties of protein sequences. Generated using large language models like
                        <a href="https://github.com/evolutionaryscale/esm" target="_blank" rel="noopener noreferrer"> ESMC</a>,
                        these embeddings can be used for downstream machine learning tasks.
                    </Paragraph>

                    <Title level={4}>Input Types Explained</Title>
                    <Paragraph>
                        There are two main input types for embedding generation:
                    </Paragraph>
                    <ul>
                        <li>
                            <Text strong>Extra Sequence IDs:</Text> Specific variants you want to embed individually:
                            <ul style={{ marginTop: '8px', marginLeft: '16px' }}>
                                <li>"WT" - embeds the wild-type sequence</li>
                                <li>"A43W_T67G" - embeds a specific double mutant</li>
                                <li>"K150R" - embeds a single point mutation</li>
                                <li>Use one sequence ID per line</li>
                            </ul>
                        </li>
                        <li style={{ marginTop: '12px' }}>
                            <Text strong>DMS Starting Sequence IDs:</Text> Base sequences for comprehensive mutational scanning:
                            <ul style={{ marginTop: '8px', marginLeft: '16px' }}>
                                <li>For each sequence listed, ALL possible single amino acid mutants will be generated and embedded</li>
                                <li>Creates ~19x the protein length in embeddings (19 amino acids x each position)</li>
                                <li>Example: "WT" will generate embeddings for A1C, A1D, A1E... through to the last position</li>
                                <li>Use this for deep mutational scanning experiments</li>
                            </ul>
                        </li>
                    </ul>

                    <Title level={4}>Homolog FASTA Files (Optional)</Title>
                    <Paragraph>
                        You can optionally include homolog sequences (orthologs, paralogs, indel mutants) to be embedded:
                    </Paragraph>
                    <ul>
                        <li><Text strong>Format:</Text> Standard FASTA format with sequence IDs prefixed with "HOM-" (e.g., HOM-ortho1, HOM-para2). Homolog IDs can include letters, numbers, and hyphens</li>
                        <li><Text strong>Usage:</Text> These <Text strong>must also be listed in Extra Sequence IDs</Text> to be embedded</li>
                        <li><Text strong>File types:</Text> Accepts .fasta, .fa, or .txt files</li>
                    </ul>

                    <Alert
                        message="Example Usage"
                        description={
                            <div>
                                <Text strong>Extra Sequence IDs:</Text> WT, A43W_T67G, K150R<br />
                                <Text strong>DMS Starting Sequence IDs:</Text> WT<br />
                                <Text>This will embed the wild-type, two specific mutants, plus all single mutants of the wild-type.</Text>
                            </div>
                        }
                        type="success"
                        showIcon
                        style={{ marginTop: '12px', marginBottom: '12px' }}
                    />

                    <Title level={4}>Domain Boundaries (Optional)</Title>
                    <Paragraph>
                        Domain boundaries allow you to pool embeddings within protein domains instead of averaging across the entire protein:
                    </Paragraph>
                    <ul>
                        <li><Text strong>Format:</Text> Comma-separated list of domain boundary positions (0-indexed)</li>
                        <li><Text strong>Example:</Text> For a 200AA protein with domains at positions 1-80, 81-150, 151-200, enter: "80,150"</li>
                        <li><Text strong>Result:</Text> Instead of one averaged embedding, you get concatenated embeddings from each domain</li>
                        <li><Text strong>Use case:</Text> Preserves domain-specific information in multi-domain proteins</li>
                    </ul>

                    <Title level={4}>Model Selection</Title>
                    <Paragraph>
                        Choose a pLM model:
                    </Paragraph>
                    <ul>
                        <li><Text strong>ESMC 300M:</Text> Fast, what was evaluated in the FolDE paper. Available for academic use.</li>
                        <li><Text strong>ESM2 15B:</Text> Slower, used in the EvolvePro paper.</li>
                    </ul>

                    <Alert
                        message="Cost Consideration"
                        description="~$10 for a DMS of a 500AA protein."
                        type="warning"
                        showIcon
                        style={{ marginTop: '16px' }}
                    />

                    <Paragraph style={{ marginTop: '16px' }}>
                        <Text strong>Output:</Text> Completed embeddings can be downloaded as CSV files from the Files tab and are also available for download in the Evolution tab.
                    </Paragraph>
                </div>
            </Modal>
        </>
    );
};
