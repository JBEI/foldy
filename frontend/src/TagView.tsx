import fileDownload from "js-file-download";
import React, { useEffect, useState } from "react";
import { CSVLink } from "react-csv";
import { useParams } from "react-router-dom";
import UIkit from "uikit";
import {
    getFoldFileZip,
    getJobStatus,
    queueJob,
} from "./services/backend.service";
import { makeFoldTable } from "./util/foldTable";
import { NewDockPrompt } from "./util/newDockPrompt";
import { getFolds, updateFold } from "./api/foldApi";
import { Dock, Fold } from "./types/types";

function TagView(props: { setErrorText: (a: string) => void }) {
    let { tagStringParam } = useParams();
    const [tagString] = useState<string>(tagStringParam || "");
    const [folds, setFolds] = useState<Fold[] | null>(null);
    const [relativeFpathToDownload, setRelativeFpathToDownload] = useState<
        string | null
    >(null);
    const [stageToStart, setStageToStart] = useState<string | null>(null);
    const [showJobManagement, setShowJobManagement] = useState(true);

    if (!tagStringParam) {
        throw Error("Somehow wound up with an invalid tagstring.");
    }

    useEffect(() => {
        getFolds(null, tagString, null, null).then(setFolds, (e) => {
            props.setErrorText(e.toString());
        });
    }, [props]);

    const restartWholePipelineForAnyFailedJob = () => {
        if (!folds) {
            return;
        }

        var numFoldsChanged = 0;
        for (const fold of folds) {
            if (!fold.id) {
                continue;
            }

            if (
                getJobStatus(fold, "features") === "failed" ||
                getJobStatus(fold, "models") === "failed" ||
                getJobStatus(fold, "decompress_pkls") === "failed"
            ) {
                const stageToRun = "both";
                queueJob(fold.id, stageToRun, false).then(
                    () => {
                        UIkit.notification(
                            `Successfully started stage ${stageToRun} for ${fold.name}.`
                        );
                    },
                    (e) => {
                        props.setErrorText(e);
                    }
                );
                numFoldsChanged += 1;
            }
        }

        if (numFoldsChanged === 0) {
            UIkit.notification("No folds needed a restart.");
        }
    };

    const startStageForAllFolds = () => {
        if (!folds) {
            return;
        }

        if (!stageToStart) {
            UIkit.notification("No stage selected.");
            return;
        }

        var numChanged = 0;
        for (const fold of folds) {
            if (!fold.id) {
                continue;
            }
            if (getJobStatus(fold, stageToStart) === "finished") {
                continue;
            }
            ++numChanged;
            queueJob(fold.id, stageToStart, false).then(
                () => {
                    UIkit.notification(`Successfully started stage(s) for ${fold.name}.`);
                },
                (e) => {
                    props.setErrorText(e);
                }
            );
        }
        if (numChanged === 0) {
            UIkit.notification(`All folds have finished stage ${stageToStart}.`);
        }
    };

    const getFoldsDataForCsv = () => {
        if (!folds) {
            return "";
        }
        return folds?.map((fold) => {
            const copy: any = structuredClone(fold);
            delete copy["docks"];
            delete copy["jobs"];
            if (fold.docks) {
                fold.docks.forEach((dock: Dock) => {
                    copy[`dock_${dock.ligand_name}_smiles`] = dock.ligand_smiles;
                    const energy = dock.pose_energy === null ? NaN : dock.pose_energy;
                    copy[`dock_${dock.ligand_name}_dg`] = energy;
                    const confidences =
                        dock.pose_confidences === null ? NaN : dock.pose_confidences;
                    copy[`dock_${dock.ligand_name}_confidences`] = confidences;
                });
            }
            return copy;
        });
    };

    const downloadFoldPdbZip = () => {
        if (!folds) {
            return;
        }
        if (folds.some((fold) => fold.id === null)) {
            console.error("Some fold has a null ID...");
            return;
        }
        const fold_ids = folds.map((fold) => fold.id || 0);
        const output_dirname = `${tagString}_pdbs`;
        getFoldFileZip(fold_ids, "ranked_0.pdb", output_dirname).then(
            (fold_pdb_blob) => {
                fileDownload(fold_pdb_blob, `${output_dirname}.zip`);
            },
            (e) => {
                props.setErrorText(e);
            }
        );
    };

    const downloadFoldFileZip = () => {
        if (!folds) {
            return;
        }
        if (folds.some((fold) => fold.id === null)) {
            props.setErrorText("Some fold has a null ID... Weird.");
            return;
        }
        if (!relativeFpathToDownload) {
            props.setErrorText("No path set.");
            return;
        }
        const fold_ids = folds.map((fold) => fold.id || 0);
        const output_dirname = `${tagString}_bulk_download`;
        getFoldFileZip(fold_ids, relativeFpathToDownload, output_dirname).then(
            (file_blob) => {
                fileDownload(file_blob, `${output_dirname}.zip`);
            },
            (e) => {
                console.log(e);
                props.setErrorText(e);
            }
        );
    };

    const makeAllFoldsPublic = () => {
        if (!folds) {
            return;
        }
        UIkit.modal
            .confirm(
                `Are you sure you want to make all folds with tag ${tagString} public?`
            )
            .then(() => {
                folds.forEach((fold) => {
                    if (!fold.id) {
                        console.error("Some fold has a null ID...");
                        return;
                    }
                    updateFold(fold.id, { public: true }).then(() => {
                        UIkit.notification(`Successfully made ${fold.name} public.`);
                    });
                });
            });
    };

    return (
        <div style={{ padding: "20px" }}>
            {/* Tag Header */}
            <h2 style={{ textAlign: "center", marginBottom: "20px" }}>
                Tag: <b>{tagString}</b>
            </h2>

            {/* Folds Table */}
            {folds ? (
                <div key="loadedDiv">{makeFoldTable(folds)}</div>
            ) : (
                <div className="uk-text-center" key="unloadedDiv">
                    <div uk-spinner="ratio: 4" key="spinner"></div>
                </div>
            )}

            {/* Downloads Section */}
            <section style={sectionStyle} className="uk-width-1-3">
                <h3>Downloads</h3>
                <div style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "10px",
                }}>
                    <CSVLink
                        data={folds ? folds : []}
                        className="uk-button uk-button-primary"
                        filename={`${tagString}_metadata.csv`}
                    >
                        Download Metadata as CSV
                    </CSVLink>
                    <button
                        className="uk-button uk-button-primary"
                        onClick={downloadFoldPdbZip}
                    >
                        Download Fold PDBs in ZIP
                    </button>
                    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                        <input
                            type="text"
                            placeholder="ranked_0/plddt.npy"
                            value={relativeFpathToDownload || ""}
                            onChange={(e) => setRelativeFpathToDownload(e.target.value)}
                            style={inputStyle}
                        />
                        <button
                            className="uk-button uk-button-primary"
                            onClick={downloadFoldFileZip}
                        >
                            Download File
                        </button>
                    </div>
                </div>
            </section>

            {/* Visibility Section */}
            <section style={sectionStyle} className="uk-width-1-3">
                <h3>Visibility</h3>
                <button
                    className="uk-button uk-button-primary"
                    onClick={makeAllFoldsPublic}
                >
                    Make All Structures Public
                </button>
            </section>

            {/* Job Management Section */}
            <section style={sectionStyle} className="uk-width-1-3">
                <div
                    style={{ display: "flex", justifyContent: "space-between", cursor: "pointer" }}
                    onClick={() => setShowJobManagement(!showJobManagement)}
                >
                    <h3>Job Management</h3>
                    <span>{showJobManagement ? "▲" : "▼"}</span>
                </div>
                {showJobManagement && (
                    <div>
                        <button
                            className="uk-button uk-button-primary"
                            onClick={() => restartWholePipelineForAnyFailedJob()}
                        >
                            Restart Whole Pipeline for Failed Jobs
                        </button>
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }} className="uk-margin-small-top">
                            <select
                                value={stageToStart || ""}
                                onChange={(e) => setStageToStart(e.target.value)}
                                style={inputStyle}
                            >
                                <option value="">Select a Stage...</option>
                                <option value="both">Both</option>
                                <option value="annotate">Annotate</option>
                                <option value="write_fastas">Write FASTAs</option>
                                <option value="features">Features</option>
                                <option value="models">Models</option>
                                <option value="decompress_pkls">Decompress PKLs</option>
                            </select>
                            <button
                                className="uk-button uk-button-primary"
                                onClick={startStageForAllFolds}
                            >
                                Start Stage for All Folds
                            </button>
                        </div>
                    </div>
                )}
            </section>

            {/* Docking Section */}
            <section style={sectionStyle} className="uk-width-1-3">
                <h3>Docking</h3>
                {folds && (
                    <NewDockPrompt
                        setErrorText={props.setErrorText}
                        foldIds={folds.map((fold) => fold.id ?? -1)}
                        existingLigands={{
                            ...(folds.reduce((acc, fold) => {
                                acc[fold.id] = fold.docks?.map((dock: Dock) => dock.ligand_name) || [];
                                return acc;
                            }, {} as Record<number, string[]>)),
                        }}
                    />
                )}
            </section>
        </div>
    );
}

const sectionStyle = {
    backgroundColor: "#ffffff",
    borderRadius: "8px",
    padding: "15px",
    marginBottom: "20px",
    boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
};

const buttonContainerStyle = {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
};

const inputStyle = {
    padding: "8px",
    borderRadius: "5px",
    border: "1px solid #ccc",
    flex: 1,
};

export default TagView;
