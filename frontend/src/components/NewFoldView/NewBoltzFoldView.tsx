import React, { useState, useEffect, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import BoltzYamlBuilder, { BoltzYamlBuilderRef } from "../../util/boltzYamlBuilder";
import { Row, Col, Form, Input, Switch, Alert, InputNumber, Modal, Button as AntButton, Typography, Upload, Tooltip } from "antd";
import { QuestionCircleOutlined, BookOutlined, InboxOutlined, CheckCircleFilled } from '@ant-design/icons';
import { postFolds } from "../../api/foldApi";
import { FoldInput } from "../../types/types";
import UIkit from "uikit";
import { notify } from "../../services/NotificationService";
import CitationModal from "../shared/CitationModal";
import { EditableTagList } from "../../util/editableTagList";
import Papa from "papaparse";
import { BoltzYamlHelper } from "../../util/boltzYamlHelper";
import { initInputToken } from "node_modules/antd/lib/input/style";

const { Text, Paragraph, Title } = Typography;


interface AdvancedSettings {
    diffusionSamples: number;
    startFoldJob: boolean;
    emailOnCompletion: boolean;
    skipDuplicateEntries: boolean;
    stayOnPage: boolean;
}

async function createFold(
    foldName: string,
    yamlData: string,
    isDryRun: boolean,
    options: {
        userType: string | null;
        tags?: string[];
        diffusionSamples?: number;
        startFoldJob?: boolean;
        emailOnCompletion?: boolean;
        skipDuplicateEntries?: boolean;
    }
): Promise<void> {
    if (options.userType === "viewer") {
        throw new Error("Viewers cannot create folds");
    }

    const fold: FoldInput = {
        name: foldName,
        tags: options.tags || [],
        yaml_config: yamlData,
        diffusion_samples: options.diffusionSamples || null,
        yaml_helper: null,
        sequence: null,
        af2_model_preset: "boltz",
        disable_relaxation: false,
    };

    return await postFolds([fold], {
        startJob: options.startFoldJob || false,
        emailOnCompletion: options.emailOnCompletion || false,
        skipDuplicates: options.skipDuplicateEntries || false,
        isDryRun: isDryRun,
    });
}

interface NewFoldState {
    foldName: string
    yamlString: string
}

interface NewFoldBoxProps {
    initialFoldState: NewFoldState
    setNewFoldState: (newFoldState: NewFoldState | null) => void
    builderRef?: React.RefObject<BoltzYamlBuilderRef>
}

const NewFoldBox: React.FC<NewFoldBoxProps> = ({
    initialFoldState,
    setNewFoldState,
    builderRef,
}) => {
    const [isBeingEdited, setIsBeingEdited] = useState<boolean>(true);
    const [foldName, setFoldName] = useState<string>(initialFoldState.foldName);
    const [yamlString, setYamlString] = useState<string>(initialFoldState.yamlString);

    // Update state when initialFoldState changes (e.g., from CSV import)
    useEffect(() => {
        setFoldName(initialFoldState.foldName);
        setYamlString(initialFoldState.yamlString);
    }, [initialFoldState.foldName, initialFoldState.yamlString]);

    const onSave = (yamlString: string) => {
        setYamlString(yamlString);
        setNewFoldState({
            foldName: foldName,
            yamlString: yamlString,
        })
        setIsBeingEdited(false);
    }

    return (
        <div style={{ borderColor: 'black', borderWidth: '2px', borderStyle: 'dashed', borderRadius: '8px', marginBottom: '1rem' }}>
            {
                isBeingEdited ?
                    <BoltzYamlBuilder
                        ref={builderRef}
                        key={yamlString}
                        foldName={foldName}
                        setFoldName={setFoldName}
                        initialYaml={yamlString}
                        onSave={onSave}
                        submitButtonText="Validate Input"
                    /> :
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '1rem'
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <h3 style={{ margin: 0 }}>{foldName}</h3>
                            <CheckCircleFilled style={{ color: '#52c41a' }} />
                        </div>
                        <AntButton
                            type="primary"
                            ghost
                            onClick={() => {
                                setNewFoldState(null);
                                setIsBeingEdited(!isBeingEdited);
                            }}
                        >
                            Edit
                        </AntButton>
                    </div>
            }
        </div>
    );
}

interface NewBoltzFoldViewProps {
    userType: string | null;
}

const NewBoltzFoldView: React.FC<NewBoltzFoldViewProps> = ({ userType }) => {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const [tags, setTags] = useState<string[]>([]);
    const [newFoldTemplateList, setNewFoldTemplateList] = useState<NewFoldState[]>([{
        foldName: "",
        yamlString: `version: 1
sequences:
  - protein:
      id: [A]
      sequence: ""
`}]);
    const [newFoldStateList, setNewFoldStateList] = useState<(NewFoldState | null)[]>([null]);
    const builderRefs = useRef<Array<React.RefObject<BoltzYamlBuilderRef>>>([]);

    const [isSubmitting, setIsSubmitting] = useState(false);
    const [showHelpModal, setShowHelpModal] = useState<boolean>(false);
    const [showCitationsModal, setShowCitationsModal] = useState<boolean>(false);
    const [showBulkInputModal, setShowBulkInputModal] = useState<boolean>(false);
    const [advancedSettings, setAdvancedSettings] = useState<AdvancedSettings>({
        diffusionSamples: 1,
        startFoldJob: true,
        emailOnCompletion: true,
        skipDuplicateEntries: false,
        stayOnPage: false,
    });

    // Check for URL parameters to prepopulate the form
    useEffect(() => {
        const nameParam = searchParams.get('name');
        const yamlParam = searchParams.get('yaml');
        const tagsParam = searchParams.get('tags');

        // Only process URL params if they exist and we haven't already processed them
        if (nameParam || yamlParam || tagsParam) {
            let newNewFoldTemplate;
            if (newFoldTemplateList) {
                newNewFoldTemplate = newFoldTemplateList[0];
            } else {
                newNewFoldTemplate = {
                    foldName: "",
                    yamlString: ""
                }
            }

            if (nameParam) {
                newNewFoldTemplate.foldName = decodeURIComponent(nameParam);
            }

            if (yamlParam) {
                try {
                    notify.info(`YAML prepopulated from URL parameter`);
                    console.log(decodeURIComponent(yamlParam));
                    newNewFoldTemplate.yamlString = decodeURIComponent(yamlParam);
                } catch (error) {
                    console.error('Error decoding YAML parameter:', error);
                }
            }

            if (nameParam || yamlParam) {
                setNewFoldTemplateList([newNewFoldTemplate]);
            }

            if (tagsParam) {
                try {
                    const decodedTags = JSON.parse(decodeURIComponent(tagsParam));
                    if (Array.isArray(decodedTags)) {
                        setTags(decodedTags);
                    }
                } catch (error) {
                    console.error('Error decoding tags parameter:', error);
                }
            }

            // Clear the search params immediately after processing
            navigate(window.location.pathname, { replace: true });
        }
    }, [searchParams, navigate]); // Include dependencies

    const addTag = (tagToAdd: string) => {
        if (!tags.includes(tagToAdd)) {
            setTags([...tags, tagToAdd]);
        }
    };

    const deleteTag = (tagToDelete: string) => {
        setTags(tags.filter(tag => tag !== tagToDelete));
    };

    const handleTagClick = (tag: string) => {
        // For new fold view, we don't navigate to tag pages
        console.log(`Tag clicked: ${tag}`);
    };

    const addAnotherNewFold = () => {
        setNewFoldStateList([...newFoldStateList, null]);
        // Add a new ref for the new fold
        builderRefs.current.push(React.createRef<BoltzYamlBuilderRef>());
    };

    const removeLastNewFold = () => {
        if (newFoldStateList.length > 1) {
            setNewFoldStateList(newFoldStateList.slice(0, -1));
            // Remove the last ref
            builderRefs.current.pop();
        }
    };

    // Initialize refs to match the initial fold state list
    useEffect(() => {
        builderRefs.current = newFoldStateList.map(() => React.createRef<BoltzYamlBuilderRef>());
    }, [newFoldStateList.length]);

    const saveAllEditingFolds = () => {
        builderRefs.current.forEach((ref, index) => {
            // Only submit if the fold is currently being edited (i.e., newFoldStateList[index] is null)
            if (ref.current && newFoldStateList[index] === null) {
                ref.current.submit();
            }
        });
    };

    const processBulkCSV = (csvData: Papa.ParseResult<any>) => {
        const newFolds: NewFoldState[] = [];

        let failed = false;

        csvData.data.forEach((row, rowIdx) => {
            if (!row.fold_name) {
                notify.error(`Row ${rowIdx} is missing a fold_name.`)
                return;
            }

            try {
                const helper = new BoltzYamlHelper(`version: 1\nsequences:\n`);
                helper.setVersion(1);

                // Parse chain columns dynamically
                const chainData: { [key: string]: any } = {};
                for (const [key, value] of Object.entries(row)) {
                    if (typeof key === 'string' && key.startsWith('chain_')) {
                        const parts = key.split('_');
                        if (parts.length >= 3) {
                            const chainNum = parts[1];
                            const property = parts.slice(2).join('_');

                            if (!chainData[chainNum]) {
                                chainData[chainNum] = {};
                            }
                            chainData[chainNum][property] = value;
                        } else {
                            notify.error(`Surprising column name: ${key}`)
                            failed = true;
                            return;
                        }
                    }
                }

                // Process each chain
                for (const [chainNum, chain] of Object.entries(chainData)) {
                    const { id, type, sequence, smiles, ccd } = chain as any;

                    if (!id || !type) continue;

                    switch (type.toLowerCase()) {
                        case 'protein':
                            if (!sequence) {
                                notify.error(`Row ${rowIdx} Chain ${id} is a ${type} and is missing a sequence.`)
                                failed = true;
                                return;
                            }
                            helper.addProtein({ id, sequence });
                            break;
                        case 'dna':
                            if (!sequence) {
                                notify.error(`Row ${rowIdx} Chain ${id} is a ${type} and is missing a sequence.`)
                                failed = true;
                                return;
                            }
                            helper.addDNA({ id, sequence });
                            break;
                        case 'rna':
                            if (!sequence) {
                                notify.error(`Row ${rowIdx} Chain ${id} is a ${type} and is missing a sequence.`)
                                failed = true;
                                return;
                            }
                            helper.addRNA({ id, sequence });
                            break;
                        case 'ligand':
                            if (!smiles && !ccd) {
                                notify.error(`Row ${rowIdx} Chain ${id} is a ${type} and is missing a smiles or ccd.`)
                                failed = true;
                                return;
                            }
                            helper.addLigand({ id, smiles, ccd });
                            break;
                    }
                }

                // Add affinity property if specified
                if (row.affinity_chain_id) {
                    helper.addAffinityProperty({ binder: row.affinity_chain_id });
                }

                newFolds.push({
                    foldName: row.fold_name,
                    yamlString: helper.toYAML()
                });

            } catch (error) {
                console.error(`Error processing row for fold ${row.fold_name}:`, error);
                notify.error(`Error processing fold ${row.fold_name}: ${error}`);
                failed = true;
                return;
            }
        });

        if (newFolds.length === 0) {
            notify.error(`No folds were imported from CSV.`)
            return;
        }

        if (failed) {
            notify.error('Failed to import some folds from CSV.');
            return;
        }

        // Clear existing forms and replace with CSV data
        setNewFoldTemplateList(newFolds);
        setNewFoldStateList(new Array(newFolds.length).fill(null));
        // Update refs array to match new fold count
        builderRefs.current = new Array(newFolds.length).fill(null).map(() => React.createRef<BoltzYamlBuilderRef>());
        notify.success(`Successfully imported ${newFolds.length} folds from CSV`);

        setShowBulkInputModal(false);
    };

    const downloadExampleCSV = () => {
        const exampleCSV = `fold_name,chain_1_id,chain_1_type,chain_1_sequence,chain_2_id,chain_2_type,chain_2_sequence,chain_2_smiles,affinity_chain_id
        protein_dimer,A,protein,MVTPELVKPLMEKDKMVGQKVRQIRNAQMYKNQLRVTPRSQCGVSSLQGFYRHCLVQARRRVVPSSVSTVYHNYTPNRSYLQKHQRAFDLVHYQHPVKPHVNLPLGDVLVHVHKQQVVLTQWREVVLHCQSVTFYSGIMTLTQVDKHGREVHYSQKFLQHQRLFQIHQYVPLQSRRHK,B,protein,GPTLKELQ,,
        ligand_protein,B,protein,GPTLKELQEIQKELQIQRRQVMQTQQTRQVSQNLQKLQKQHQMHQHQGQKVRQKQHQRQVYQKQGQKVRQKQHQRQVYQKQGQKVRQKQHQRQVYQKQ,C,ligand,,COC(=O)c1ccc(cc1)C(C)(C)C,C
        dna_protein,D,protein,MAAAAAAAAAAAAAAAAAAAAAAAAKQKLQKQKQKQQKQKQKQKQQKQKQKQKQQKQKQKQKQQKQKQKQKQQKQKQKQKQQKQKQKQKQQKQKQKQKQ,E,dna,ATCGATCGATCGATCG,,
        rna_complex,F,rna,AUCGAUCGAUCGAUCG,G,protein,GPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTKGPTK,,`;

        const blob = new Blob([exampleCSV], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'example_bulk_input.csv';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const saveAllFolds = async () => {
        if (newFoldStateList.length === 0) {
            notify.error(`Must add at least one fold before submitting.`)
            return;
        }
        if (newFoldStateList.some(newFoldState => newFoldState === null)) {
            notify.error(`One of the folds is not saved: must save yaml configs before submitting folds.`)
            return;
        }
        // Check if any of the fold names are the same.
        if (newFoldStateList.some((newFoldState1, ii) => {
            if (newFoldState1 === null) return false;
            return newFoldStateList.slice(0, ii).some(newFoldState2 => newFoldState2 && newFoldState2.foldName === newFoldState1.foldName);
        })) {
            notify.error(`All fold names must be unique.`)
            return;
        }
        setIsSubmitting(true);

        // First run all dry runs and wait for them to complete
        const dryRunResults = await Promise.all(newFoldStateList.map(async (newFoldState) => {
            if (!newFoldState) {
                return { success: false, error: 'Must save yaml configs before submitting folds.' };
            }
            const foldName = newFoldState.foldName;
            const yamlString = newFoldState.yamlString;

            if (!foldName.trim()) {
                return { success: false, error: 'Please enter a fold name' };
            }

            // Check for weird characters in the fold name
            if (/[^a-zA-Z0-9_ -]/.test(foldName)) {
                return { success: false, error: 'Fold name contains invalid characters. Please use only letters, numbers, underscores, hyphens, and spaces.' };
            }

            // Run the dry run
            try {
                await createFold(foldName, yamlString, true, {
                    userType,
                    tags,
                    ...advancedSettings,
                });
                return { success: true };
            } catch (err) {
                return { success: false, error: `Failed to create fold ${foldName}: ${String(err)}` };
            }
        }));

        // Check if any dry runs failed
        const failedDryRun = dryRunResults.find(result => !result.success);
        if (failedDryRun) {
            notify.error(failedDryRun.error || 'Failed to create fold');
            setIsSubmitting(false);
            return;
        }

        // If we get here, all dry runs succeeded, so we can proceed with actual creation
        const createResults = await Promise.all(newFoldStateList.map(async (newFoldState) => {
            if (!newFoldState) return { success: true }; // This was already checked above
            try {
                await createFold(newFoldState.foldName, newFoldState.yamlString, false, {
                    userType,
                    tags,
                    ...advancedSettings,
                });
                return { success: true };
            } catch (err) {
                return { success: false, error: `Failed to create fold ${newFoldState.foldName}: ${String(err)}` };
            }
        }));

        // Check if any creations failed
        const failedCreate = createResults.find(result => !result.success);
        if (failedCreate) {
            notify.error(failedCreate.error || 'Failed to create fold');
            setIsSubmitting(false);
            return;
        }

        notify.success("Fold successfully created!");
        setIsSubmitting(false);

        if (!advancedSettings.stayOnPage) {
            navigate("/");
        }
    }


    return (
        <div
            data-testid="About"
            style={{
                flexGrow: 1,
                overflowY: "scroll",
                paddingTop: "10px",
                paddingBottom: "10px",
            }}>
            {/* Fixed Header */}
            <div style={{ padding: "1rem", borderBottom: "1px solid #f0f0f0" }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <h1 style={{ margin: 0 }}>New Boltz Fold</h1>
                    <AntButton onClick={() => setShowBulkInputModal(true)}>Bulk Input</AntButton>
                </div>

                {userType === "viewer" && (
                    <Alert
                        message="You do not have permissions to submit folds on this instance."
                        type="error"
                        style={{ marginBottom: "1rem" }}
                    />
                )}

                {/* Clean overview with help buttons */}
                <Alert
                    message="Create a new protein structure prediction"
                    description={
                        <div>
                            <Paragraph>
                                Foldy uses Boltz-2x to predict protein structures with exceptional accuracy for multimers,
                                small molecule docking, DNA/RNA interactions, and post-translational modifications.
                            </Paragraph>
                            <div style={{ display: 'flex', gap: '12px', marginTop: '12px' }}>
                                <AntButton
                                    type="link"
                                    icon={<QuestionCircleOutlined />}
                                    onClick={() => setShowHelpModal(true)}
                                    style={{ padding: 0 }}
                                >
                                    View new fold guide
                                </AntButton>
                                <AntButton
                                    type="link"
                                    icon={<BookOutlined />}
                                    onClick={() => setShowCitationsModal(true)}
                                    style={{ padding: 0 }}
                                >
                                    View citations
                                </AntButton>
                            </div>
                        </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: '20px' }}
                />

                {/* Setup Instructions Modal */}
                <Modal
                    title="Boltz Fold Guide"
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
                        <Title level={4}>What is Boltz-2x?</Title>
                        <Paragraph>
                            Boltz-2x is an open-source protein structure prediction model with exceptional accuracy for complex scenarios:
                        </Paragraph>
                        <ul>
                            <li>Protein multimers (multi-chain complexes)</li>
                            <li>Small molecule docking</li>
                            <li>DNA/RNA interactions</li>
                            <li>Post-translational modifications</li>
                        </ul>
                        <Paragraph>
                            <a href="https://github.com/jwohlwend/boltz" target="_blank" rel="noopener noreferrer">GitHub Repository</a> |
                            <a href="https://www.biorxiv.org/content/10.1101/2024.11.19.624167v4" target="_blank" rel="noopener noreferrer"> Research Paper</a>
                        </Paragraph>

                        <Title level={4}>Required Inputs</Title>
                        <ul>
                            <li>
                                <Text strong>Fold Name:</Text> Unique identifier (max 80 characters, use only letters, numbers, underscores, hyphens, and spaces)
                            </li>
                            <li>
                                <Text strong>YAML Configuration:</Text> Define sequences and structure
                            </li>
                            <li>
                                <Text strong>Chain IDs:</Text> Single uppercase letters for each molecule (A, B, C, etc.)
                            </li>
                        </ul>

                        <Title level={4}>YAML Structure</Title>
                        <ul>
                            <li><Text strong>Version:</Text> Always set to 1</li>
                            <li><Text strong>Sequences:</Text> Define protein, DNA, RNA, and ligand sequences</li>
                            <li><Text strong>Multiple copies:</Text> Use multiple chain IDs for the same sequence</li>
                        </ul>

                        <Alert
                            message="Need help with YAML?"
                            description="Use the interactive YAML builder below to construct your configuration step-by-step."
                            type="success"
                            showIcon
                            style={{ marginTop: '16px' }}
                        />
                    </div>
                </Modal>

                <CitationModal
                    open={showCitationsModal}
                    onClose={() => setShowCitationsModal(false)}
                />

                {/* Bulk Input Modal */}
                <Modal
                    title="Bulk Input from CSV"
                    open={showBulkInputModal}
                    onCancel={() => setShowBulkInputModal(false)}
                    footer={[
                        <AntButton key="close" onClick={() => setShowBulkInputModal(false)}>
                            Close
                        </AntButton>
                    ]}
                    width={800}
                >
                    <div style={{ marginBottom: '20px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                            <Title level={4} style={{ margin: 0 }}>Upload CSV File</Title>
                            <AntButton onClick={downloadExampleCSV}>Download Example CSV</AntButton>
                        </div>
                        <Paragraph>
                            Upload a CSV file to create multiple folds at once. The CSV should have the following column structure:
                        </Paragraph>
                        <ul>
                            <li><Text code>fold_name</Text> - Name for the fold</li>
                            <li><Text code>chain_N_id</Text> - Chain ID (A, B, C, etc.)</li>
                            <li><Text code>chain_N_type</Text> - Chain type (protein, dna, rna, ligand)</li>
                            <li><Text code>chain_N_sequence</Text> - Sequence for protein/DNA/RNA</li>
                            <li><Text code>chain_N_smiles</Text> - SMILES string for ligands</li>
                            <li><Text code>chain_N_ccd</Text> - CCD code for ligands</li>
                            <li><Text code>affinity_chain_id</Text> - (Optional) Chain ID for affinity binding</li>
                        </ul>
                        <Paragraph>
                            Where <Text code>N</Text> is the chain number (1, 2, 3, etc.).
                        </Paragraph>
                    </div>

                    <Upload.Dragger
                        name="csvFile"
                        accept=".csv"
                        multiple={false}
                        showUploadList={false}
                        beforeUpload={(file) => {
                            const reader = new FileReader();
                            reader.onload = (e) => {
                                const csvText = e.target?.result as string;
                                Papa.parse(csvText, {
                                    header: true,
                                    skipEmptyLines: true,
                                    complete: (results) => {
                                        if (results.errors.length > 0) {
                                            notify.error(`CSV parsing error: ${results.errors[0].message}`);
                                        } else {
                                            processBulkCSV(results);
                                        }
                                    },
                                    error: (error: any) => {
                                        notify.error(`CSV parsing error: ${error.message}`);
                                    }
                                });
                            };
                            reader.readAsText(file);
                            return false; // Prevent default upload
                        }}
                    >
                        <p className="ant-upload-drag-icon">
                            <InboxOutlined />
                        </p>
                        <p className="ant-upload-text">Click or drag CSV file to this area to upload</p>
                        <p className="ant-upload-hint">
                            Support for single CSV file upload only. CSV file will be parsed and folds will be created automatically.
                        </p>
                    </Upload.Dragger>
                </Modal>
            </div>

            {/* Scrollable Content */}
            <div style={{ flex: 1, overflow: "auto", padding: "1rem" }}>
                <Row gutter={24}>
                    <Col xs={24} sm={24} md={18} lg={18} xl={18}>
                        {
                            [...newFoldStateList].map((_, ii) => {
                                const setNewFoldState = (newFoldState: NewFoldState | null) => {
                                    const newNewFoldStateList = [...newFoldStateList];
                                    newNewFoldStateList[ii] = newFoldState;
                                    setNewFoldStateList(newNewFoldStateList);
                                }
                                const newFoldTemplate = newFoldTemplateList.length > ii ?
                                    newFoldTemplateList[ii] : {
                                        foldName: "",
                                        yamlString: ""
                                    }
                                return <NewFoldBox
                                    key={`${ii}-${newFoldTemplate.foldName}-${newFoldTemplate.yamlString}`}
                                    initialFoldState={newFoldTemplate}
                                    setNewFoldState={setNewFoldState}
                                    builderRef={builderRefs.current[ii]}
                                />;
                            })
                        }
                        <div style={{
                            display: 'flex',
                            flexWrap: 'wrap',
                            gap: '12px',
                            marginBottom: '20px',
                            marginRight: '1rem' // Prevent overlap with right column
                        }}>
                            <AntButton onClick={addAnotherNewFold}>Add Another Fold</AntButton>
                            <AntButton onClick={removeLastNewFold}>Remove Last Fold</AntButton>
                            <AntButton onClick={saveAllEditingFolds}>Validate All</AntButton>
                            {(() => {
                                const hasUnvalidatedFolds = newFoldStateList.some(newFoldState => newFoldState === null);
                                const isSubmitDisabled = isSubmitting || hasUnvalidatedFolds;
                                console.log(`isSubmitDisabled: ${isSubmitDisabled} newFoldStateList length: ${newFoldStateList.length} is null ${newFoldStateList.map(newFoldState => newFoldState === null)}`)

                                if (hasUnvalidatedFolds) {
                                    return (
                                        <Tooltip title="Cannot submit until all new folds are validated">
                                            <AntButton type="primary" disabled={isSubmitDisabled} loading={isSubmitting}>
                                                Submit
                                            </AntButton>
                                        </Tooltip>
                                    );
                                }

                                return (
                                    <AntButton type="primary" onClick={saveAllFolds} disabled={isSubmitDisabled} loading={isSubmitting}>
                                        Submit
                                    </AntButton>
                                );
                            })()}
                        </div>
                    </Col>

                    {/* Advanced settings column - will scroll with content */}
                    <Col xs={24} sm={24} md={6} lg={6} xl={6} style={{
                        position: "sticky",
                        top: "1rem",
                        marginTop: "1rem" // Add margin on mobile when stacked
                    }}>
                        <div style={{
                            backgroundColor: "#f5f5f5",
                            padding: "1rem",
                            borderRadius: "8px"
                        }}>
                            <h3>Advanced Settings</h3>
                            <Form layout="vertical">
                                <div style={{ marginBottom: "8px" }}>
                                    <span style={{ fontSize: "14px", color: "#666", marginRight: "8px" }}>Tags:</span>
                                    <EditableTagList
                                        tags={tags}
                                        addTag={addTag}
                                        deleteTag={deleteTag}
                                        handleTagClick={handleTagClick}
                                    />
                                </div>
                                <Form.Item label="Diffusion Samples">
                                    <InputNumber
                                        value={advancedSettings.diffusionSamples}
                                        onChange={(value) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                diffusionSamples: value || 1,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>


                                <Form.Item label="Start Fold Job Immediately">
                                    <Switch
                                        checked={advancedSettings.startFoldJob}
                                        onChange={(checked) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                startFoldJob: checked,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>

                                <Form.Item label="Email on Completion">
                                    <Switch
                                        checked={advancedSettings.emailOnCompletion}
                                        onChange={(checked) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                emailOnCompletion: checked,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>

                                <Form.Item label="Skip Duplicate Entries">
                                    <Switch
                                        checked={advancedSettings.skipDuplicateEntries}
                                        onChange={(checked) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                skipDuplicateEntries: checked,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>

                                <Form.Item label="Stay on this page after fold creation">
                                    <Switch
                                        checked={advancedSettings.stayOnPage}
                                        onChange={(checked) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                stayOnPage: checked,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>
                            </Form>
                        </div>
                    </Col>
                </Row>
            </div>
        </div >
    );
};

export default NewBoltzFoldView;
