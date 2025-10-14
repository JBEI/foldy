import fileDownload from "js-file-download";
import React, { useEffect, useState } from "react";
import { CSVLink } from "react-csv";
import { useParams } from "react-router-dom";
import UIkit from "uikit";
import { Button, Space, Input, Select, Spin, Checkbox } from "antd";
import { queueJob } from "./api/commonApi";
import { getFoldFileZip, getFoldsWithPagination, getJobStatus } from "./api/foldApi";
import { makeFoldTable } from "./util/foldTable";
import { NewDockPrompt } from "./util/newDockPrompt";
import { updateFold, getFoldAffinityPrediction } from "./api/foldApi";
import { Dock, Fold } from "./types/types";
import { notify } from "./services/NotificationService";
import { NaturalnessModal } from "./components/shared/NaturalnessModal";
import { EmbeddingModal } from "./components/shared/EmbeddingModal";

function TagView() {
    let { tagStringParam } = useParams();
    const [tagString] = useState<string>(tagStringParam || "");
    const [folds, setFolds] = useState<Fold[] | null>(null);
    const [relativeFpathToDownload, setRelativeFpathToDownload] = useState<
        string | null
    >(null);
    const [stageToStart, setStageToStart] = useState<string | null>(null);
    const [affinityData, setAffinityData] = useState<any[]>([]);
    const [showNaturalnessModal, setShowNaturalnessModal] = useState<boolean>(false);
    const [showEmbeddingModal, setShowEmbeddingModal] = useState<boolean>(false);
    const [flattenFilepath, setFlattenFilepath] = useState<boolean>(false);
    const [useFoldName, setUseFoldName] = useState<boolean>(false);

    if (!tagStringParam) {
        throw Error("Somehow wound up with an invalid tagstring.");
    }

    const fetchAllFoldData = () => {
        getFoldsWithPagination(null, tagString, null, null).then(
            (v) => {
                setFolds(v.data);
            }, (e) => {
                notify.error(e.toString());
            });
    };

    useEffect(() => {
        fetchAllFoldData();
    }, [tagString]);

    const refoldAnyFailedFolds = () => {
        if (!folds) {
            return;
        }

        var numFoldsChanged = 0;
        for (const fold of folds) {
            if (!fold.id) {
                continue;
            }

            if (
                getJobStatus(fold, "boltz") === "failed"
            ) {
                const stageToRun = "both";
                queueJob(fold.id, stageToRun, false).then(
                    () => {
                        notify.success(
                            `Successfully started stage ${stageToRun} for ${fold.name}.`
                        );
                    },
                    (e) => {
                        notify.error(e);
                    }
                );
                numFoldsChanged += 1;
            }
        }

        if (numFoldsChanged === 0) {
            notify.info("No folds needed a restart.");
        }
    };

    const startStageForAllFolds = () => {
        if (!folds) {
            return;
        }

        if (!stageToStart) {
            notify.warning("No stage selected.");
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
                    notify.success(`Successfully started stage(s) for ${fold.name}.`);
                },
                (e) => {
                    notify.error(e);
                }
            );
        }
        if (numChanged === 0) {
            notify.info(`All folds have finished stage ${stageToStart}.`);
        }
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
        const output_dirname = `${tagString}_cifs`;
        // Enable both flatten_filepath and use_fold_name for PDB downloads
        getFoldFileZip(fold_ids, "ranked_0.cif", output_dirname, true, true).then(
            (fold_pdb_blob) => {
                fileDownload(fold_pdb_blob, `${output_dirname}.zip`);
            },
            (e) => {
                notify.error(e);
            }
        );
    };

    const loadAffinityData = async () => {
        if (!folds) {
            return;
        }

        const affinityData = await Promise.all(
            folds.map(async (fold) => {
                const baseRow = {
                    fold_id: fold.id,
                    fold_name: fold.name,
                    fold_tags: fold.tags,
                    affinity_pred_value: '',
                    affinity_probability_binary: '',
                    affinity_pred_value1: '',
                    affinity_probability_binary1: '',
                    affinity_pred_value2: '',
                    affinity_probability_binary2: ''
                };

                if (!fold.id) {
                    return baseRow;
                }

                try {
                    const predictedAffinity = await getFoldAffinityPrediction(fold.id);
                    return {
                        fold_id: fold.id,
                        fold_name: fold.name,
                        fold_tags: fold.tags,
                        affinity_pred_value: predictedAffinity.affinity_pred_value,
                        affinity_probability_binary: predictedAffinity.affinity_probability_binary,
                        affinity_pred_value1: predictedAffinity.affinity_pred_value1,
                        affinity_probability_binary1: predictedAffinity.affinity_probability_binary1,
                        affinity_pred_value2: predictedAffinity.affinity_pred_value2,
                        affinity_probability_binary2: predictedAffinity.affinity_probability_binary2
                    };
                } catch (e) {
                    console.log(`Failed to get affinity prediction for fold ${fold.id}:`, e);
                    return baseRow;
                }
            })
        );

        setAffinityData(affinityData);
    };

    const downloadFoldFileZip = () => {
        if (!folds) {
            return;
        }
        if (folds.some((fold) => fold.id === null)) {
            notify.error("Some fold has a null ID... Weird.");
            return;
        }
        if (!relativeFpathToDownload) {
            notify.warning("No path set.");
            return;
        }
        const fold_ids = folds.map((fold) => fold.id || 0);
        const output_dirname = `${tagString}_bulk_download`;
        getFoldFileZip(fold_ids, relativeFpathToDownload, output_dirname, flattenFilepath, useFoldName).then(
            (file_blob) => {
                fileDownload(file_blob, `${output_dirname}.zip`);
            },
            (e) => {
                console.log(e);
                notify.error(e);
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
                        notify.success(`Successfully made ${fold.name} public.`);
                    });
                });
            });
    };

    return (
        <div style={{ padding: "20px", overflowY: "scroll" }}>
            {/* Tag Header */}
            <h2 style={{ textAlign: "center", marginBottom: "20px" }}>
                Tag: <b>{tagString}</b>
            </h2>

            {/* Folds Table */}
            {folds ? (
                <div key="loadedDiv">{makeFoldTable(folds, {
                    editable: true,
                    onTagsChange: fetchAllFoldData
                })}</div>
            ) : (
                <div style={{ textAlign: 'center', padding: '60px 0' }} key="unloadedDiv">
                    <Spin size="large" key="spinner" />
                </div>
            )}

            {/* Container for all sections */}
            <div style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "20px",
                justifyContent: "flex-start",
            }}>
                {/* Downloads Section */}
                <section style={sectionStyle}>
                    <h3 style={{ marginBottom: "16px", color: "#1890ff", fontWeight: 600 }}>Downloads</h3>
                    <Space direction="vertical" style={{ width: "100%" }}>
                        <CSVLink
                            data={folds ? folds : []}
                            filename={`${tagString}_metadata.csv`}
                            style={{ display: "block", textAlign: "center", textDecoration: "none" }}
                        >
                            <Button block>Download Metadata as CSV</Button>
                        </CSVLink>
                        <Button
                            onClick={downloadFoldPdbZip}
                            block
                        >
                            Download Fold CIFs in ZIP
                        </Button>
                        {
                            affinityData.length > 0 ?
                                <CSVLink
                                    data={affinityData}
                                    className="ant-btn ant-btn-primary"
                                    filename={`${tagString}_affinity.csv`}
                                    style={{ display: "block", textAlign: "center", textDecoration: "none" }}
                                >
                                    <Button block>Download Affinity CSV</Button>
                                </CSVLink> : <Button onClick={loadAffinityData} block>Prepare Affinity Data for Download</Button>
                        }

                        <Space direction="vertical" style={{ width: "100%" }}>
                            <Space.Compact style={{ width: "100%" }}>
                                <Input
                                    placeholder="ranked_0/plddt.npy"
                                    value={relativeFpathToDownload || ""}
                                    onChange={(e) => setRelativeFpathToDownload(e.target.value)}
                                />
                                <Button
                                    type="primary"
                                    onClick={downloadFoldFileZip}
                                >
                                    Download File
                                </Button>
                            </Space.Compact>
                            <Space direction="vertical" size="small">
                                <Checkbox
                                    checked={flattenFilepath}
                                    onChange={(e) => setFlattenFilepath(e.target.checked)}
                                >
                                    Flatten file structure (no folders)
                                </Checkbox>
                                <Checkbox
                                    checked={useFoldName}
                                    onChange={(e) => setUseFoldName(e.target.checked)}
                                >
                                    Use fold names instead of IDs
                                </Checkbox>
                            </Space>
                        </Space>
                    </Space>
                </section>

                {/* Visibility Section */}
                <section style={sectionStyle}>
                    <h3 style={{ marginBottom: "16px", color: "#1890ff", fontWeight: 600 }}>Visibility</h3>
                    <Button
                        type="primary"
                        onClick={makeAllFoldsPublic}
                        block
                    >
                        Make All Structures Public
                    </Button>
                </section>

                {/* Job Management Section */}
                <section style={sectionStyle}>
                    <h3 style={{ marginBottom: "16px", color: "#1890ff", fontWeight: 600 }}>Job Management</h3>
                    <Space direction="vertical" style={{ width: "100%" }}>
                        <Button
                            type="primary"
                            onClick={() => refoldAnyFailedFolds()}
                            block
                        >
                            Refold Failed Folds
                        </Button>
                        <Space.Compact style={{ width: "100%" }}>
                            <Select
                                value={stageToStart || undefined}
                                onChange={setStageToStart}
                                placeholder="Select a Stage..."
                                style={{ width: "60%" }}
                            >
                                <Select.Option value="both">Fold and Annotate</Select.Option>
                                <Select.Option value="annotate">Annotate</Select.Option>
                                <Select.Option value="write_fastas">Write FASTAs</Select.Option>
                            </Select>
                            <Button
                                type="primary"
                                onClick={startStageForAllFolds}
                            >
                                Start Stage
                            </Button>
                        </Space.Compact>
                    </Space>
                </section>

                {/* Docking Section */}
                <section style={sectionStyle}>
                    <h3 style={{ marginBottom: "16px", color: "#1890ff", fontWeight: 600 }}>Docking</h3>
                    {folds && (
                        <NewDockPrompt
                            foldIds={folds.map((fold) => fold.id ?? -1)}
                            existingLigands={{
                                ...(folds.reduce((acc, fold) => {
                                    if (fold.id !== null) {
                                        acc[fold.id] = fold.docks?.map((dock: Dock) => dock.ligand_name) || [];
                                    }
                                    return acc;
                                }, {} as Record<number, string[]>)),
                            }}
                        />
                    )}
                </section>

                {/* ML Analysis Section */}
                <section style={sectionStyle}>
                    <h3 style={{ marginBottom: "16px", color: "#1890ff", fontWeight: 600 }}>ML Analysis</h3>
                    <Space direction="vertical" style={{ width: "100%" }}>
                        <Button
                            type="primary"
                            onClick={() => setShowNaturalnessModal(true)}
                            block
                            disabled={!folds || folds.length === 0}
                        >
                            Run Naturalness Analysis
                        </Button>
                        <Button
                            type="primary"
                            onClick={() => setShowEmbeddingModal(true)}
                            block
                            disabled={!folds || folds.length === 0}
                        >
                            Generate Embeddings
                        </Button>
                    </Space>
                </section>
            </div>

            {/* Modal Components */}
            {folds && (
                <>
                    <NaturalnessModal
                        open={showNaturalnessModal}
                        onClose={() => setShowNaturalnessModal(false)}
                        foldIds={folds.map((fold) => fold.id).filter((id): id is number => id !== null)}
                        title={`Run Naturalness Analysis (${tagString})`}
                    />
                    <EmbeddingModal
                        open={showEmbeddingModal}
                        onClose={() => setShowEmbeddingModal(false)}
                        foldIds={folds.map((fold) => fold.id).filter((id): id is number => id !== null)}
                        title={`Generate Embeddings (${tagString})`}
                        disableSequenceFields={true}
                    />
                </>
            )}
        </div>
    );
}

const sectionStyle = {
    backgroundColor: "#ffffff",
    borderRadius: "12px",
    padding: "20px",
    marginBottom: "20px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
    border: "1px solid #f0f0f0",
    width: "350px",  // Fixed width for each section
    flex: "0 0 auto",  // Prevent sections from growing or shrinking
};


export default TagView;
