import React from "react";
import { EditableTagList } from "../../util/editableTagList";
import SeqViz from "seqviz";
import { AiFillEdit } from "react-icons/ai";

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
    sequence: string;
    colorScheme: string;

    setPublic: (is_public: boolean) => void;
    setDisableRelaxation: (disable_relaxation: boolean) => void;
    setFoldName: () => void;
    addTag: (tagToAdd: string) => void;
    deleteTag: (tagToDelete: string) => void;
    handleTagClick: (tagToOpen: string) => void;

    setSelectedSubsequence: (sele: SubsequenceSelection) => void;

    userType: string | null;
}

const SequenceTab = React.memo((props: SequenceTabProps) => {
    const renderSequenceViewer = () => {
        return props.sequence.split(";").map((ss: string, idx: number) => {
            const [chainName, chainSeq] = ss.includes(":")
                ? ss.split(":")
                : [props.foldName, ss];

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
        });
    };

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
                <input
                    className="uk-input"
                    value={props.foldModelPreset || "unset"}
                    disabled
                />

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