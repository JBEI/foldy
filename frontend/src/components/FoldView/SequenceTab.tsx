import React, { useState, useMemo } from "react";
import { EditableTagList } from "../../util/editableTagList";
import SeqViz from "seqviz";
import { BoltzYamlHelper, ChainSequence, LigandData } from "../../util/boltzYamlHelper";
import BoltzYamlBuilder from "../../util/boltzYamlBuilder";
import UIkit from "uikit";
import { Selection } from "./StructurePane";
import { notify } from "../../services/NotificationService";
import { TabContainer, SectionCard, CollapsibleSection } from "../../util/tabComponents";
import { Alert, Modal, Button as AntButton, Typography, Input, Switch, Space } from 'antd';
import { QuestionCircleOutlined, EditOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { getFile } from "../../api/fileApi";
import { getFoldAffinityPrediction } from "../../api/foldApi";
import { AffinityPrediction, RenderableAnnotations } from "src/types/types";

const { Text, Paragraph, Title } = Typography;

export interface SubsequenceSelection {
    chainIdx: number;
    startResidue: number;
    endResidue: number;
    subsequence: string;
}


interface SequenceTabProps {
    foldId: number;
    foldName: string;
    foldTags: string[];
    foldOwner: string;
    foldCreateDate: string;
    foldPublic: boolean | null;
    yamlConfig: string | null;
    foldDiffusionSamples: number | null;

    // Old AlphaFold inputs.
    sequence: string | null;

    renderablePfamAnnotations: RenderableAnnotations | null;

    setPublic: (is_public: boolean) => void;
    setFoldName: () => void;
    addTag: (tagToAdd: string) => void;
    deleteTag: (tagToDelete: string) => void;
    handleTagClick: (tagToOpen: string) => void;

    setSelectedSubsequence: (selection: Selection | null) => void;

    userType: string | null;
    setYamlConfig: (yaml: string) => void;
}

// Utility function to format molar concentration with appropriate units
const formatMolarConcentration = (valueInMolar: number): string => {
    if (valueInMolar >= 1) {
        return `${(valueInMolar).toFixed(1)} M`;
    } else if (valueInMolar >= 1e-3) {
        return `${(valueInMolar * 1e3).toFixed(1)} mM`;
    } else if (valueInMolar >= 1e-6) {
        return `${(valueInMolar * 1e6).toFixed(1)} μM`;
    } else if (valueInMolar >= 1e-9) {
        return `${(valueInMolar * 1e9).toFixed(1)} nM`;
    } else {
        return `${(valueInMolar * 1e12).toFixed(1)} pM`;
    }
};

// Convert RenderableAnnotations to SeqViz format
const convertToSeqVizAnnotations = (renderableAnnotations: RenderableAnnotations | null, chainName: string) => {
    if (!renderableAnnotations || !renderableAnnotations[chainName]) {
        return [];
    }

    return renderableAnnotations[chainName].map(annotation => ({
        name: annotation.type,
        start: annotation.start,
        end: annotation.end,
        direction: 1 as const,
        color: annotation.color
    }));
};

const SequenceTab = React.memo((props: SequenceTabProps) => {
    const [showYamlSection, setShowYamlSection] = useState<boolean>(false);
    const [affinityPrediction, setAffinityPrediction] = useState<AffinityPrediction | null>(null);

    // Handler to color structure by pfam annotations
    const colorStructureByPfam = () => {
        if (!props.renderablePfamAnnotations) return;

        const selectionData: {
            struct_asym_id: string;
            start_residue_number: number;
            end_residue_number: number;
            color: string;
        }[] = [];

        // Convert RenderableAnnotations to Selection format
        for (const [chainName, annotations] of Object.entries(props.renderablePfamAnnotations)) {
            annotations.forEach(annotation => {
                selectionData.push({
                    struct_asym_id: chainName,
                    start_residue_number: annotation.start,
                    end_residue_number: annotation.end,
                    color: annotation.color
                });
            });
        }

        const selection: Selection = {
            data: selectionData,
            nonSelectedColor: "#CCCCCC" // Light gray for non-pfam regions
        };

        console.log("Setting structure selection for pfam coloring:", selection);
        props.setSelectedSubsequence(selection);
    };

    const configHelper = props.yamlConfig ? new BoltzYamlHelper(props.yamlConfig) : null;

    // Add a memo to load affinity predictions
    useMemo(async () => {
        getFoldAffinityPrediction(props.foldId).then(
            (affinityPrediction: AffinityPrediction) => {
                setAffinityPrediction(affinityPrediction);
            },
            (e: any) => {
                console.log(e);
            }
        )
    }, [props.foldId]);

    var sequenceNames: string[];
    var sequences: string[];
    if (configHelper) {
        sequenceNames = configHelper.getProteinSequences().map((e) => e[0]);
        sequences = configHelper.getProteinSequences().map((e) => e[1]);
    } else if (props.sequence) {
        const oldSequenceStrs = props.sequence.split(";");
        sequenceNames = oldSequenceStrs.map((ss) => ss.includes(":") ? ss.split(":")[0] : props.foldName);
        sequences = oldSequenceStrs.map((ss) => ss.includes(":") ? ss.split(":")[1] : ss);
    } else {
        return <div>No sequence found.</div>
    }

    const renderSequenceViewer = () => {
        // Get affinity binder if specified
        const affinityProperties = configHelper?.getProperties() || [];
        const affinityBinder = affinityProperties[0]?.affinity?.binder;

        // Organize sequences by type
        const ligands = configHelper?.getLigands() || [];
        const proteins = configHelper?.getProteinSequences() || [];
        const dnaSequences = configHelper?.getDNASequences() || [];
        const rnaSequences = configHelper?.getRNASequences() || [];

        // Find the affinity ligand if it exists
        const affinityLigand = ligands.find(ligand =>
            ligand.chain_ids.includes(affinityBinder || '')
        );

        return <>
            {/* Render affinity ligand first if it exists */}
            {affinityLigand && (
                <div key="affinity-ligand" style={{
                    marginBottom: "20px",
                    backgroundColor: "#f0f8ff", // Light blue background
                    padding: "15px",
                    borderRadius: "8px",
                    border: "1px solid #d1e8ff"
                }}>
                    <h3>{affinityLigand.chain_ids.join(", ")} (Ligand - Affinity Target)</h3>
                    <div style={{
                        border: "1px solid #d1e8ff",
                        borderRadius: "4px",
                        padding: "12px",
                        backgroundColor: "#ffffff",
                        fontFamily: "monospace",
                        fontSize: "14px",
                        overflowWrap: "break-word",
                        wordBreak: "break-all"
                    }}>
                        {affinityLigand.smiles || affinityLigand.ccd}
                    </div>
                    {affinityPrediction && (
                        <div style={{ marginTop: "10px" }}>
                            <div>
                                <strong>Predicted Affinity:</strong>{' '}
                                {' '}
                                {affinityPrediction.affinity_pred_value}
                                {' '}
                                ({formatMolarConcentration(Math.pow(10, affinityPrediction.affinity_pred_value - 6))}, {((6 - affinityPrediction.affinity_pred_value) * 1.364).toFixed(2)} kJ/mol)
                                <a
                                    href="https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                >
                                    <InfoCircleOutlined style={{ color: '#1890ff', marginLeft: '4px' }} />
                                </a>
                            </div>
                            <div>
                                <strong>Binding Probability:</strong>{' '}
                                {(affinityPrediction.affinity_probability_binary * 100).toFixed(1)}%
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Render other ligands */}
            {ligands
                .filter(ligand => ligand !== affinityLigand)
                .map((ligand: LigandData, idx: number) => (
                    <div key={idx} style={{ marginBottom: "20px" }}>
                        <h3>{ligand.chain_ids.join(", ")} (Ligand)</h3>
                        <div style={{
                            border: "1px solid #d9d9d9",
                            borderRadius: "4px",
                            padding: "12px",
                            backgroundColor: "#fafafa",
                            fontFamily: "monospace",
                            fontSize: "14px",
                            overflowWrap: "break-word",
                            wordBreak: "break-all"
                        }}>
                            {ligand.smiles || ligand.ccd}
                        </div>
                    </div>
                ))
            }

            {/* Render protein sequences */}
            {proteins.map((ss: ChainSequence, idx: number) => {
                const chainName = ss[0];
                const chainSeq = ss[1];

                const onSelectionHandler = (chainName: string, selection: any) => {
                    if (!configHelper) {
                        console.log('Config helper is underfined, cannot show residues.')
                        return;
                    }
                    const chainIndex = configHelper.getProteinSequences().findIndex(x => x[0] === chainName);
                    if (chainIndex === -1) {
                        console.log(`Could not find chain ${chainName} in boltz config: ${configHelper?.getProteinSequences()}`);
                        return;
                    }
                    if (selection.start && selection.end) {
                        console.log(selection);
                        var start = Math.min(selection.start, selection.end);
                        var end = Math.max(selection.start, selection.end);
                        if (start >= end) {
                            start = -1;
                            end = 0;
                        }
                        console.log(`${start}, ${end}`)
                        props.setSelectedSubsequence({
                            data: [{
                                struct_asym_id: chainName,
                                start_residue_number: start + 1,
                                end_residue_number: end,
                                color: "white",
                            }],
                            // nonSelectedColor: "white",
                        });
                    }
                };

                return (
                    <div key={`sequence-${chainName}-${idx}`} style={{ marginBottom: "20px" }}>
                        <h3>{chainName}</h3>
                        <SeqViz
                            key={`seqviz-${chainName}-${idx}`}
                            name={chainName}
                            seq={chainSeq}
                            seqType="aa"
                            viewer="linear"
                            showComplement={false}
                            zoom={{ linear: 10 }}
                            annotations={convertToSeqVizAnnotations(props.renderablePfamAnnotations, chainName)}
                            style={{
                                width: "100%",
                                marginBottom: "20px",
                                border: "1px solid #e0e0e0",
                                borderRadius: "8px",
                            }}
                            onSelection={(selection: any) => onSelectionHandler(chainName, selection)}
                        />
                    </div>
                );
            })}

            {/* Render DNA sequences */}
            {dnaSequences.map((dna: ChainSequence, idx: number) => (
                <div key={idx} style={{ marginBottom: "20px" }}>
                    <h3>{dna[0]} (DNA)</h3>
                    <div>
                        <SeqViz
                            name={dna[0]}
                            seq={dna[1]}
                            seqType="dna"
                            viewer="linear"
                            style={{
                                width: "100%",
                                marginBottom: "20px",
                                border: "1px solid #e0e0e0",
                                borderRadius: "8px",
                            }}
                        />
                    </div>
                </div>
            ))}

            {/* Render RNA sequences */}
            {rnaSequences.map((rna: ChainSequence, idx: number) => (
                <div key={idx} style={{ marginBottom: "20px" }}>
                    <h3>{rna[0]} (RNA)</h3>
                    <div>
                        <SeqViz
                            name={rna[0]}
                            seq={rna[1]}
                            seqType="rna"
                            viewer="linear"
                            style={{
                                width: "100%",
                                marginBottom: "20px",
                                border: "1px solid #e0e0e0",
                                borderRadius: "8px",
                            }}
                        />
                        {rna[1]}
                    </div>
                </div>
            ))}
        </>
    };

    const canEditYaml = props.userType !== "viewer";

    const [showHelpModal, setShowHelpModal] = useState<boolean>(false);

    return (
        <TabContainer>
            {/* Sequence Viewer */}
            <SectionCard>
                {renderSequenceViewer()}
                {/* Color by Pfam button */}
                {props.renderablePfamAnnotations && Object.keys(props.renderablePfamAnnotations).length > 0 && (
                    <div style={{ marginBottom: '16px' }}>
                        <AntButton
                            onClick={colorStructureByPfam}
                        >
                            Color Structure by Pfam Domains
                        </AntButton>
                    </div>
                )}
            </SectionCard>

            {/* YAML Builder Section - only show if user has permission */}
            {canEditYaml && (
                <CollapsibleSection
                    title="Edit YAML Configuration"
                    isOpen={showYamlSection}
                    onToggle={() => setShowYamlSection(!showYamlSection)}
                    style={{ marginBottom: '20px' }}
                >
                    <BoltzYamlBuilder
                        initialYaml={props.yamlConfig || undefined}
                        onSave={(yaml) => {
                            console.log(`YAML: ${yaml}`);
                            UIkit.modal
                                .confirm(
                                    `Are you sure you want to update the YAML configuration?`
                                )
                                .then(async () => {
                                    await props.setYamlConfig(yaml);
                                    notify.info("Updated YAML configuration. You can refold the protein from Actions > Refold.");
                                });
                        }}
                        foldName={props.foldName}
                        setFoldName={null}
                    />
                </CollapsibleSection>
            )}

            {/* Form Section */}
            <SectionCard>
                {/* Help Alert */}
                <Alert
                    message="Protein Properties & Settings"
                    description={
                        <div>
                            <Paragraph>
                                Configure basic protein properties, tags, and folding parameters.
                                Use the edit buttons to make changes.
                            </Paragraph>
                            <AntButton
                                type="link"
                                icon={<QuestionCircleOutlined />}
                                onClick={() => setShowHelpModal(true)}
                                style={{ padding: 0 }}
                            >
                                View detailed property guide
                            </AntButton>
                        </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: '20px' }}
                />

                {/* Detailed Help Modal */}
                <Modal
                    title="Protein Properties Guide"
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
                        <Title level={4}>Basic Properties</Title>
                        <ul>
                            <li><Text strong>Name:</Text> Descriptive identifier for your protein</li>
                            <li><Text strong>Owner:</Text> User who created this fold</li>
                            <li><Text strong>Created:</Text> Timestamp of fold creation</li>
                            <li><Text strong>Diffusion Samples:</Text> Number of samples used in structure generation</li>
                        </ul>

                        <Title level={4}>Visibility & Organization</Title>
                        <ul>
                            <li><Text strong>Visibility:</Text> Make this fold visible to other users</li>
                            <li><Text strong>Tags:</Text> Add descriptive labels for organization and search</li>
                        </ul>

                        <Alert
                            message="Parameter Changes"
                            description="Changes to folding parameters require refolding the protein to take effect."
                            type="warning"
                            showIcon
                            style={{ marginTop: '16px' }}
                        />
                    </div>
                </Modal>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px 24px', maxWidth: '800px' }}>
                    {/* Row 1 */}
                    <div>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Name</Text>
                        <Space.Compact style={{ width: '100%' }}>
                            <Input
                                value={props.foldName}
                                disabled
                                style={{ width: 'calc(100% - 40px)' }}
                                size="small"
                            />
                            <AntButton
                                icon={<EditOutlined />}
                                onClick={props.setFoldName}
                                disabled={props.userType === "viewer"}
                                title="Edit name"
                                size="small"
                            />
                        </Space.Compact>
                    </div>

                    <div>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Diffusion Samples</Text>
                        <Input
                            value={props.foldDiffusionSamples || ''}
                            disabled
                            size="small"
                        />
                    </div>

                    {/* Row 2 */}
                    <div>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Owner</Text>
                        <Input
                            value={props.foldOwner}
                            disabled
                            size="small"
                        />
                    </div>

                    <div>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Created</Text>
                        <Input
                            value={props.foldCreateDate}
                            disabled
                            size="small"
                        />
                    </div>

                    {/* Row 3 */}
                    <div>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Visibility</Text>
                        <Switch
                            checked={props.foldPublic || false}
                            onChange={(checked) => props.setPublic(checked)}
                            checkedChildren="Public"
                            unCheckedChildren="Private"
                            size="small"
                        />
                    </div>

                    {/* Tags section - spans both columns */}
                    <div>
                        <Text strong style={{ display: 'block', marginBottom: '8px' }}>Tags</Text>
                        <EditableTagList
                            tags={props.foldTags || []}
                            addTag={props.addTag}
                            deleteTag={props.deleteTag}
                            handleTagClick={props.handleTagClick}
                        />
                    </div>
                </div>
            </SectionCard>
        </TabContainer>
    );
});

export default SequenceTab;
