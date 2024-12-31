import React from "react";
import { EditableTagList } from "../../util/editableTagList";
import { SequenceAnnotation, VariousColorSchemes } from "../../util/plots";
import { AiFillEdit } from "react-icons/ai";
import SeqViz from 'seqviz';
import { Selection } from "node_modules/seqviz/dist/selectionContext";
import SelectionColormaker from "node_modules/react-ngl/dist/@types/ngl/declarations/color/selection-colormaker";
// const ReactSequenceViewer = require("react-sequence-viewer");
// import ReactSequenceViewer from "react-sequence-viewer";

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
    pfamColors: VariousColorSchemes | null;

    setPublic: (is_public: boolean) => void;
    setDisableRelaxation: (disable_relaxation: boolean) => void;
    setFoldName: () => void;
    addTag: (tagToAdd: string) => void;
    deleteTag: (tagToDelete: string) => void;
    handleTagClick: (tagToOpen: string) => void;

    setSelectedSubsequence: (sele: SubsequenceSelection) => void;

    userType: string | null;
}

const SequenceTab = React.memo(
    (props: SequenceTabProps) => {
        const getSequenceViewerCoverage = (chainIdx: number): SequenceAnnotation[] => {
            if (props.colorScheme === "pfam") {
                return props.pfamColors?.sVCoverage
                    ? props.pfamColors.sVCoverage[chainIdx]
                    : [];
            } else {
                return [];
            }
        };

        const getSequenceViewerLegend = (chainIdx: number) => {
            if (props.colorScheme === "pfam") {
                return props.pfamColors?.sVLegend
                    ? props.pfamColors.sVLegend[chainIdx]
                    : [];
            } else {
                return [];
            }
        };
        console.log(`FOLD ${props.foldDisableRelaxation}`);

        return (
            <div style={{ display: 'flex', flexDirection: 'column' }}>
                {props.sequence.split(";").map((ss: string, idx: number) => {
                    var chainSeq: string;
                    var chainName: string;
                    if (ss.includes(":")) {
                        chainName = ss.split(":")[0];
                        chainSeq = ss.split(":")[1];
                    } else {
                        chainName = props.foldName;
                        chainSeq = ss;
                    }

                    const annotations = getSequenceViewerCoverage(idx).map((v) => {
                        return {
                            start: v.start,
                            end: v.end,
                            name: v.tooltip,
                            color: v.bgcolor,
                        }
                    })

                    const onSelectionHandler = (selection: Selection) => {
                        if (selection.start && selection.end) {
                            const start = Math.min(selection.start, selection.end);
                            const end = Math.max(selection.start, selection.end);
                            props.setSelectedSubsequence({
                                chainIdx: idx,
                                startResidue: start + 1,
                                endResidue: end + 1,
                                subsequence: chainSeq.substring(start, end)
                            });
                        }
                    }

                    return <>
                        <h2 key={`${idx}_heading`}>{chainName}</h2>
                        <SeqViz
                            key={idx}
                            name={chainName}
                            seq={chainSeq}
                            seqType="aa"
                            annotations={annotations}
                            viewer="linear"
                            showComplement={false}
                            zoom={{ linear: 10 }} // Adjust zoom level as needed
                            style={{ width: '100%', marginBottom: '20px' }}  // Customize styles as needed , height: '400px',
                            onSelection={onSelectionHandler}
                        /></>;
                })}
                <form className="uk-form-horizontal">
                    <div className="uk-margin">
                        <label className="uk-form-label" htmlFor="form-horizontal-text">
                            Name
                        </label>
                        <div className="uk-form-controls">
                            <input
                                className="uk-input uk-width-3-4"
                                value={props.foldName}
                                disabled
                            ></input>
                            <span className="uk-width-1-4">
                                <button
                                    className="uk-button uk-button-default uk-width-auto uk-margin-small-left"
                                    onClick={(e) => {
                                        e.preventDefault();
                                        e.stopPropagation();
                                        props.setFoldName();
                                    }}
                                    disabled={props.userType === "viewer"}
                                >
                                    <AiFillEdit />
                                </button>
                            </span>
                        </div>
                    </div>
                    <div className="uk-margin">
                        <label className="uk-form-label" htmlFor="form-horizontal-text">
                            Owner
                        </label>
                        <div className="uk-form-controls">
                            <input
                                className="uk-input uk-form-width-large"
                                id="form-horizontal-text"
                                type="text"
                                value={props.foldOwner}
                                disabled
                            />
                        </div>
                    </div>
                    <div className="uk-margin">
                        <label className="uk-form-label" htmlFor="form-horizontal-text">
                            Created
                        </label>
                        <div className="uk-form-controls">
                            <input
                                className="uk-input uk-form-width-large"
                                id="form-horizontal-text"
                                type="text"
                                value={props.foldCreateDate}
                                disabled
                            />
                        </div>
                    </div>
                    <div className="uk-margin">
                        <label className="uk-form-label" htmlFor="form-horizontal-text">
                            Public
                        </label>
                        <div
                            className="uk-form-controls"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <input
                                className="uk-input uk-form-width-large uk-checkbox"
                                type="checkbox"
                                checked={props.foldPublic || false}
                                onChange={(e) => {
                                    e.stopPropagation();
                                    props.setPublic(!props.foldPublic);
                                }}
                            />
                        </div>
                    </div>
                    <div className="uk-margin">
                        <label className="uk-form-label" htmlFor="form-horizontal-text">
                            Tags
                        </label>
                        <div className="uk-form-controls">
                            <EditableTagList
                                tags={props.foldTags || []}
                                addTag={props.addTag}
                                deleteTag={props.deleteTag}
                                handleTagClick={props.handleTagClick}
                            />
                        </div>
                    </div>
                    <div className="uk-margin">
                        <label className="uk-form-label" htmlFor="form-horizontal-text">
                            Model Preset
                        </label>
                        <div className="uk-form-controls">
                            <input
                                className="uk-input uk-form-width-large"
                                id="form-horizontal-text"
                                type="text"
                                value={props.foldModelPreset || "unset"}
                                disabled
                            />
                        </div>
                    </div>
                    <div className="uk-margin">
                        <label className="uk-form-label" htmlFor="form-horizontal-text">
                            Disable Relaxation {typeof props.foldDisableRelaxation}
                        </label>
                        <div
                            className="uk-form-controls"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <input
                                className="uk-input uk-form-width-large uk-checkbox"
                                type="checkbox"
                                checked={(props.foldDisableRelaxation !== null) ? props.foldDisableRelaxation : true}
                                onChange={(e) => {
                                    e.stopPropagation();
                                    props.setDisableRelaxation(!props.foldDisableRelaxation);
                                }}
                            />
                        </div>
                    </div>
                </form>
            </div>
        );
    },
    (prevProps: SequenceTabProps, nextProps: SequenceTabProps) => {
        return (
            prevProps.foldName === nextProps.foldName &&
            prevProps.foldTags.length === nextProps.foldTags.length &&
            prevProps.foldTags.every((ee, ii) => nextProps.foldTags[ii] === ee) &&
            prevProps.foldDisableRelaxation === nextProps.foldDisableRelaxation &&
            prevProps.foldOwner === nextProps.foldOwner &&
            prevProps.foldCreateDate === nextProps.foldCreateDate &&
            prevProps.foldPublic === nextProps.foldPublic &&
            prevProps.foldModelPreset === nextProps.foldModelPreset &&
            prevProps.sequence === nextProps.sequence &&
            prevProps.colorScheme === nextProps.colorScheme &&
            prevProps.pfamColors === nextProps.pfamColors
        );
    }
);

export default SequenceTab;
