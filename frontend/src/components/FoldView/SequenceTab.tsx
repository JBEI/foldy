import React, { useState } from "react";
import { EditableTagList } from "../../util/editableTagList";
import SeqViz from "seqviz";
import { AiFillEdit } from "react-icons/ai";
import { BoltzYamlHelper, ChainSequence, LigandData } from "../../util/boltzYamlHelper";
import BoltzYamlBuilder from "../../util/boltzYamlBuilder";
import UIkit from "uikit";

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
    foldModelPreset: string | null;
    foldDisableRelaxation: boolean | null;
    yaml_config: string | null;
    sequence: string | null;
    colorScheme: string;

    setPublic: (is_public: boolean) => void;
    setDisableRelaxation: (disable_relaxation: boolean) => void;
    setFoldName: () => void;
    setFoldModelPreset: () => void;
    addTag: (tagToAdd: string) => void;
    deleteTag: (tagToDelete: string) => void;
    handleTagClick: (tagToOpen: string) => void;

    setSelectedSubsequence: (sele: SubsequenceSelection) => void;

    userType: string | null;
    setYamlConfig: (yaml: string) => void;
}

const SequenceTab = React.memo((props: SequenceTabProps) => {
    const [showYamlSection, setShowYamlSection] = useState<boolean>(false);

    const config_helper = props.yaml_config ? new BoltzYamlHelper(props.yaml_config) : null;

    var sequenceNames: string[];
    var sequences: string[];
    if (config_helper) {
        sequenceNames = config_helper.getProteinSequences().map((e) => e[0]);
        sequences = config_helper.getProteinSequences().map((e) => e[1]);
    } else if (props.sequence) {
        const oldSequenceStrs = props.sequence.split(";");
        sequenceNames = oldSequenceStrs.map((ss) => ss.includes(":") ? ss.split(":")[0] : props.foldName);
        sequences = oldSequenceStrs.map((ss) => ss.includes(":") ? ss.split(":")[1] : ss);
    } else {
        return <div>No sequence found.</div>
    }

    const renderSequenceViewer = () => {
        return <>
            {sequences.map((ss: string, idx: number) => {
                const chainName = sequenceNames[idx];
                const chainSeq = ss;

                const onSelectionHandler = (selection: any) => {
                    if (selection.start && selection.end) {
                        const start = Math.min(selection.start, selection.end);
                        const end = Math.max(selection.start, selection.end);
                        props.setSelectedSubsequence({
                            chainIdx: idx,
                            startResidue: start + 1,
                            endResidue: end + 1,
                            subsequence: chainSeq.substring(start, end),
                        });
                    }
                };

                return (
                    <div key={idx} style={{ marginBottom: "20px" }}>
                        <h3>{chainName}</h3>
                        <SeqViz
                            name={chainName}
                            seq={chainSeq}
                            seqType="aa"
                            viewer="linear"
                            showComplement={false}
                            zoom={{ linear: 10 }}
                            style={{
                                width: "100%",
                                marginBottom: "20px",
                                border: "1px solid #e0e0e0",
                                borderRadius: "8px",
                            }}
                            onSelection={onSelectionHandler}
                        />
                    </div>
                );
            })}
            {config_helper?.getLigands().map((ligand: LigandData, idx: number) => {
                return <div key={idx} style={{ marginBottom: "20px" }}>
                    <h3>{ligand.chain_ids.join(", ")} (Ligand)</h3>
                    <div>
                        {ligand.smiles || ligand.ccd}
                    </div>
                </div>
            })}
            {config_helper?.getDNASequences().map((dna: ChainSequence, idx: number) => {
                return <div key={idx} style={{ marginBottom: "20px" }}>
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
            })}
            {config_helper?.getRNASequences().map((rna: ChainSequence, idx: number) => {
                return <div key={idx} style={{ marginBottom: "20px" }}>
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
            })}
        </>
    };

    const canEditYaml = props.userType !== "viewer";

    return (
        <div style={{ padding: "20px" }}>
            {/* Sequence Viewer */}
            <section
                style={{
                    backgroundColor: "#f8f9fa",
                    borderRadius: "8px",
                    padding: "15px",
                    boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
                }}
            >
                {renderSequenceViewer()}
            </section>

            {/* YAML Builder Section - only show if user has permission */}
            {canEditYaml && (
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
                        onClick={() => setShowYamlSection(!showYamlSection)}
                    >
                        <span>Edit YAML Configuration</span>
                        <span>{showYamlSection ? "▲" : "▼"}</span>
                    </div>
                    {showYamlSection && (
                        <div style={{
                            padding: '15px',
                            backgroundColor: '#ffffff',
                            borderRadius: '8px',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                            marginBottom: '20px'
                        }}>
                            <BoltzYamlBuilder
                                initialYaml={props.yaml_config || undefined}
                                onSave={(yaml) => {
                                    console.log(`YAML: ${yaml}`);
                                    UIkit.modal
                                        .confirm(
                                            `Are you sure you want to update the YAML configuration?`
                                        )
                                        .then(async () => {
                                            await props.setYamlConfig(yaml);
                                            UIkit.notification("Updated YAML configuration. You can refold the protein from Actions > Refold.");
                                        });
                                }}
                            />
                        </div>
                    )}
                </div>
            )}

            {/* Form Section */}
            <form
                style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 2fr",
                    gap: "15px",
                    marginTop: "20px",
                    backgroundColor: "#ffffff",
                    borderRadius: "8px",
                    padding: "15px",
                    boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
                }}
            >
                {/* Name */}
                <label>Name</label>
                <div style={{ display: "flex", alignItems: "center" }}>
                    <input
                        className="uk-input"
                        value={props.foldName}
                        disabled
                        style={{ flex: 1 }}
                    />
                    <button
                        className="uk-button uk-button-default"
                        onClick={(e) => {
                            e.preventDefault();
                            props.setFoldName();
                        }}
                        style={{
                            marginLeft: "10px",
                            border: "1px solid #ccc",
                            borderRadius: "5px",
                        }}
                        disabled={props.userType === "viewer"}
                    >
                        <AiFillEdit />
                    </button>
                </div>

                {/* Owner */}
                <label>Owner</label>
                <input
                    className="uk-input"
                    value={props.foldOwner}
                    disabled
                />

                {/* Created */}
                <label>Created</label>
                <input
                    className="uk-input"
                    value={props.foldCreateDate}
                    disabled
                />

                {/* Public */}
                <label>Public</label>
                <div style={{ display: "flex", alignItems: "center" }}>
                    <input
                        type="checkbox"
                        checked={props.foldPublic || false}
                        onChange={(e) => props.setPublic(!props.foldPublic)}
                        style={{
                            width: "20px",
                            height: "20px",
                            marginRight: "10px",
                        }}
                    />
                    <span>{props.foldPublic ? "Yes" : "No"}</span>
                </div>

                {/* Tags */}
                <label>Tags</label>
                <EditableTagList
                    tags={props.foldTags || []}
                    addTag={props.addTag}
                    deleteTag={props.deleteTag}
                    handleTagClick={props.handleTagClick}
                />

                {/* Model Preset */}
                <label>Model Preset</label>
                <div style={{ display: "flex", alignItems: "center" }}>
                    <input
                        className="uk-input"
                        value={props.foldModelPreset || "unset"}
                        disabled
                    />
                    <button
                        className="uk-button uk-button-default"
                        onClick={(e) => {
                            e.preventDefault();
                            props.setFoldModelPreset();
                        }}
                        style={{
                            marginLeft: "10px",
                            border: "1px solid #ccc",
                            borderRadius: "5px",
                        }}
                    >
                        <AiFillEdit />
                    </button>
                </div>

                {/* Disable Relaxation */}
                <label>Disable Relaxation</label>
                <div style={{ display: "flex", alignItems: "center" }}>
                    <input
                        type="checkbox"
                        checked={props.foldDisableRelaxation !== null ? props.foldDisableRelaxation : true}
                        onChange={(e) =>
                            props.setDisableRelaxation(!props.foldDisableRelaxation)
                        }
                        style={{
                            width: "20px",
                            height: "20px",
                            marginRight: "10px",
                        }}
                    />
                    <span>{props.foldDisableRelaxation ? "Yes" : "No"}</span>
                </div>
            </form>
        </div>
    );
});

export default SequenceTab;