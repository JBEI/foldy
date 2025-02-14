import jquery from "jquery";
import ParsePdb, { ParsedPdb } from "parse-pdb";
import React, { Component, RefObject } from "react";
import { AiOutlineFolder, AiOutlineFolderOpen } from "react-icons/ai";
import { FaDownload } from "react-icons/fa";
import FileBrowser from "react-keyed-file-browser";
import { useParams } from "react-router-dom";
import UIkit from "uikit";
import {
    deleteDock,
    getDockSdf,
    getFoldPdb,
    getFoldPfam,
    getInvokation,
    queueJob,
} from "../../services/backend.service";
import { FoldyMascot } from "../../util/foldyMascot";
import { VariousColorSchemes, getColorsForAnnotations } from "../../util/plots";
import ContactTab from "./ContactTab";
import DockTab from "./DockTab";
import "./FoldView.scss";
import PaeTab from "./PaeTab";
import JobsTab from "./JobsTab";
import SequenceTab, { SubsequenceSelection } from "./SequenceTab";
import fileDownload from "js-file-download";
import NaturalnessTab from "./NaturalnessTab";
import EmbedTab from "./EmbedTab";
import EvolveTab from "./EvolveTab";
import { Annotations, FileInfo, Fold, FoldPdb, Invokation } from "../../types/types";
import { removeLeadingSlash } from "../../api/commonApi";
import { downloadFileStraightToFilesystem, getFile, getFileList } from "../../api/fileApi";
import { getFold, updateFold } from "../../api/foldApi";
import StructurePane from "./StructurePane";

const REFRESH_STATE_PERIOD = 5000;
const REFRESH_STATE_MAX_ITERS = 200;

const getResidueCenter = (
    pdb: ParsedPdb,
    residueName: string
): number[] | null => {
    const found = residueName.match(/[A-Z](\d+)/);

    if (!found || found.length !== 2) {
        console.error(
            `Residue did not match regex for getResidueCenter ${residueName} ${found}`
        );
        return null;
    }

    console.log(pdb);
    const residueIdx = parseInt(found[1]);

    const matchingResidues = pdb.atoms.filter((atom) => {
        // TODO: Don't assume we are docking on chain A.
        return atom.resSeq === residueIdx && atom.chainID === "A";
    });
    const numAtms = matchingResidues.length;
    if (!numAtms) {
        console.error(`No atoms found for residue ${residueName}`);
        return null;
    }
    const loc = [
        matchingResidues.map((e) => e.x / (1.0 * numAtms)).reduce((a, b) => a + b),
        matchingResidues.map((e) => e.y / (1.0 * numAtms)).reduce((a, b) => a + b),
        matchingResidues.map((e) => e.z / (1.0 * numAtms)).reduce((a, b) => a + b),
    ];

    return loc;
};

const getCubeEdges = (
    center: number[],
    rad: number
): { start: number[]; end: number[] }[] => {
    if (center.length !== 3) {
        console.error(`Invalid cube center: ${center}`);
    }
    const x = center[0];
    const y = center[1];
    const z = center[2];

    // Define the four corners.
    const p000 = [x - rad, y - rad, z - rad];
    const p001 = [x - rad, y - rad, z + rad];
    const p010 = [x - rad, y + rad, z - rad];
    const p011 = [x - rad, y + rad, z + rad];
    const p100 = [x + rad, y - rad, z - rad];
    const p101 = [x + rad, y - rad, z + rad];
    const p110 = [x + rad, y + rad, z - rad];
    const p111 = [x + rad, y + rad, z + rad];

    var out: { start: number[]; end: number[] }[] = [];
    // Define the back face.
    out.push({ start: p000, end: p001 });
    out.push({ start: p001, end: p011 });
    out.push({ start: p011, end: p010 });
    out.push({ start: p010, end: p000 });
    // Define the front face.
    out.push({ start: p100, end: p101 });
    out.push({ start: p101, end: p111 });
    out.push({ start: p111, end: p110 });
    out.push({ start: p110, end: p100 });
    // Define the connecting edges.
    out.push({ start: p000, end: p100 });
    out.push({ start: p001, end: p101 });
    out.push({ start: p010, end: p110 });
    out.push({ start: p011, end: p111 });
    return out;
};

interface FoldProps {
    foldId: number;
    setErrorText: (a: string) => void;
    userType: string | null;
}

// interface DisplayedDock {
//     sdf: Blob;
//     frame: number;
//     nglComponent: StructureComponent;
//     boxComponents: NGLComponent[];
// }

interface FoldState {
    foldData: Fold | null;

    // Note that a subset of job data is also in foldData.
    files: FileInfo[];
    jobs: Invokation[] | null;
    pdb: FoldPdb | null;
    parsedPdb: ParsedPdb | null;

    // Defines our current color "mode".
    colorScheme: string;

    pfamAnnotations: Annotations | null;
    pfamColors: VariousColorSchemes | null;

    // // Docking stuff.
    // displayedDocks: { [ligandName: string]: DisplayedDock };

    // // Nglviewer and other view management.
    // pdbRepr: NGLRepresentationCollection | null;
    // selectionRepr: NGLRepresentationCollection[] | null;
    pdbFailedToLoad: boolean;
    paeIsOnScreen: boolean;
    contactIsOnScreen: boolean;
    showSplitScreen: boolean;
    numRefreshes: number;
}

// From UIkit's definition of a "medium" window: https://getuikit.com/docs/visibility
const WINDOW_WIDTH_FOR_SPLIT_SCREEN = 960;
const MAX_JOBS_TO_REFRESH = 2;
class InternalFoldView extends Component<FoldProps, FoldState> {
    interval: NodeJS.Timeout | null = null;

    constructor(props: FoldProps) {
        super(props);

        this.state = {
            foldData: null,
            files: [],
            jobs: null,
            pdb: null,
            parsedPdb: null,

            colorScheme: "pfam",  // pLDDT

            pfamAnnotations: null,
            pfamColors: null,

            // displayedDocks: {},

            // pdbRepr: null,
            // selectionRepr: null,
            pdbFailedToLoad: false,
            paeIsOnScreen: false,
            contactIsOnScreen: false,
            showSplitScreen: window.innerWidth >= WINDOW_WIDTH_FOR_SPLIT_SCREEN,
            numRefreshes: 0,
        };
    }

    preventDefault = (e: any) => e.preventDefault();

    handleResize = () => {
        const newShowSplitScreen =
            window.innerWidth >= WINDOW_WIDTH_FOR_SPLIT_SCREEN;
        if (newShowSplitScreen !== this.state.showSplitScreen) {
            this.setState({ showSplitScreen: newShowSplitScreen });
        }
    };

    refreshFoldDataFromBackend = () => {
        getFold(this.props.foldId).then((new_fold_data) => {
            console.log(`Got new fold with tags ${new_fold_data.tags}`);
            this.setState({ foldData: new_fold_data });
            if (this.state.foldData?.jobs) {
                // Get current state of jobs as a map
                const currentJobStates = new Map(
                    this.state.jobs?.map(job => [job.id, job]) || []
                );

                // For each job in foldData, determine if we need to refresh it
                const jobsToRefresh = this.state.foldData.jobs
                    .filter(foldJob => {
                        const currentJob = currentJobStates.get(foldJob.id);
                        return foldJob.state === "running" ||
                            currentJob?.state === "running" ||
                            (currentJob && currentJob.state !== foldJob.state);
                    })
                    .map(job => job.id);

                if (jobsToRefresh.length > MAX_JOBS_TO_REFRESH) {
                    UIkit.notification(`Not streaming job logs because there are too many jobs (${jobsToRefresh.length} > ${MAX_JOBS_TO_REFRESH}))`, {
                        status: 'warning',
                        timeout: 1000,
                    });
                    return;
                }

                // Create final job list
                const finalJobs = [...this.state.foldData.jobs].map(foldJob => {
                    // Use existing job data if we have it and don't need to refresh
                    if (!jobsToRefresh.includes(foldJob.id)) {
                        return currentJobStates.get(foldJob.id) || foldJob;
                    }
                    return foldJob;
                });

                // Only fetch jobs that need refreshing
                if (jobsToRefresh.length > 0) {
                    Promise.all(
                        jobsToRefresh.map(jobId => getInvokation(jobId))
                    ).then(
                        (refreshedJobs) => {
                            // Update the jobs that were refreshed
                            refreshedJobs.forEach(refreshedJob => {
                                const index = finalJobs.findIndex(j => j.id === refreshedJob.id);
                                if (index !== -1) {
                                    finalJobs[index] = refreshedJob;
                                }
                            });
                            this.setState({ jobs: finalJobs });
                        },
                        (e) => {
                            this.props.setErrorText(e.toString());
                        }
                    );
                } else {
                    // If no jobs need refreshing, just update state
                    this.setState({ jobs: finalJobs });
                }
            }
        });
    };

    componentDidMount() {
        this.interval = setInterval(() => {
            if (
                this.state.numRefreshes > REFRESH_STATE_MAX_ITERS &&
                this.interval
            ) {
                clearInterval(this.interval);
            }
            this.setState({
                numRefreshes: this.state.numRefreshes + 1,
            });
            this.refreshFoldDataFromBackend();
        }, REFRESH_STATE_PERIOD);

        // ReactSequenceViewer requires jQuery, and who are we to deny them?
        // @ts-ignore
        window.$ = window.jQuery = jquery;

        // @ts-ignore
        UIkit.util.on(document, "beforeshow", "#paeli", (e: any) =>
            this.setState({ paeIsOnScreen: true })
        );
        // @ts-ignore
        UIkit.util.on(document, "beforehide", "#paeli", (e: any) =>
            this.setState({ paeIsOnScreen: false })
        );
        // @ts-ignore
        UIkit.util.on(document, "beforeshow", "#contactli", (e: any) =>
            this.setState({ contactIsOnScreen: true })
        );
        // @ts-ignore
        UIkit.util.on(document, "beforehide", "#contactli", (e: any) =>
            this.setState({ contactIsOnScreen: false })
        );

        window.addEventListener("resize", this.handleResize);

        getFileList(this.props.foldId).then((files: FileInfo[]) => {
            this.setState({ files: files });
        });

        getFold(this.props.foldId).then(
            (new_fold_data) => {
                this.setState({ foldData: new_fold_data });

                if (this.state.foldData?.jobs) {
                    Promise.all(
                        this.state.foldData.jobs.map((inv) => getInvokation(inv.id))
                    ).then(
                        (fullInvs) => {
                            this.setState({ jobs: fullInvs });
                        },
                        (e) => {
                            this.props.setErrorText(e.toString());
                        }
                    );
                }

                getFoldPfam(this.props.foldId).then(
                    (pfam) => {
                        if (!this.state.foldData) {
                            return;
                        }
                        this.setState({
                            pfamAnnotations: pfam,
                            // pfamColors: getColorsForAnnotations(
                            //     this.state.foldData.sequence,
                            //     pfam
                            // ),
                        });
                    },
                    (e) => {
                        console.log(e.toString());
                    }
                );

                // NOTE: This is where you can switch tabs to the structure view, once loaded up.
                // const switcher = document.getElementById('tab');
                // if (switcher) {
                //   UIkit.tab(switcher).show(3);
                // }

                return getFoldPdb(this.props.foldId, 0).then(
                    (pdb) => {
                        const parsedPdb = ParsePdb(pdb.pdb_string);
                        console.log(parsedPdb);

                        this.setState({ parsedPdb: parsedPdb, pdb: pdb });

                        if (!this.state.foldData || !this.state.foldData.id) {
                            return;
                        }

                        console.log(`PDB is ${pdb.pdb_string.length} characters long.`);
                    },
                    (e) => {
                        // TODO(jbr): In this case, have Foldy pop up saying the structure isn't available.
                        // console.log('in the right place');
                        // this.props.setErrorText(e.toString());
                        this.setState({ pdbFailedToLoad: true });
                    }
                );
            },
            (e) => {
                this.props.setErrorText(e.toString());
            }
        );
    }

    componentWillUnmount() {
        window.removeEventListener("resize", this.handleResize);

        if (this.interval) {
            clearInterval(this.interval);
        }
    }



    render() {
        var structurePane = (
            <div key="structure" style={{ height: "100%" }}>
                <StructurePane
                    pdbString={this.state.pdb?.pdb_string ?? null}
                    pdbFailedToLoad={this.state.pdbFailedToLoad}
                />
            </div>
        );

        var toolViewHeader = (
            <ul
                className="uk-tab"
                data-uk-tab="connect: #switcher; swiping: false"
                id="tab"
                style={{
                    marginBottom: "0px",
                }}
            >
                <li>
                    <a>Inputs</a>
                </li>
                <li>
                    <a>Logs</a>
                </li>
                <li>
                    <a>Files</a>
                </li>
                {/* TODO(jbr): Figure out why we can't pass displayStructure here... */}
                <li>
                    <a>PAE</a>
                </li>
                <li>
                    <a>Contacts</a>
                </li>
                <li>
                    <a>Dock</a>
                </li>
                <li>
                    <a>Naturalness</a>
                </li>
                <li>
                    <a>Embed</a>
                </li>
                <li>
                    <a>Evolve</a>
                </li>
                <li>
                    <a>Actions</a>
                </li>
            </ul>
        );
        var toolViewContentPane = (
            <ul className="uk-switcher uk-margin uk-padding-small" id="switcher">
                <li key="sequenceli">
                    {this.state.foldData ? (
                        <SequenceTab
                            foldId={this.props.foldId}
                            foldName={this.state.foldData?.name}
                            foldTags={this.state.foldData?.tags}
                            foldOwner={this.state.foldData?.owner}
                            foldCreateDate={this.state.foldData?.create_date}
                            foldPublic={this.state.foldData?.public}
                            foldModelPreset={this.state.foldData?.af2_model_preset}
                            foldDisableRelaxation={this.state.foldData?.disable_relaxation}
                            yaml_config={this.state.foldData.yaml_config}
                            sequence={this.state.foldData.sequence}
                            colorScheme={this.state.colorScheme}
                            setPublic={this.setPublic}
                            setDisableRelaxation={this.setDisableRelaxation}
                            setFoldName={this.setFoldName}
                            setFoldModelPreset={this.setFoldModelPreset}
                            setYamlConfig={this.setYamlConfig}
                            addTag={this.addTag}
                            deleteTag={this.deleteTag}
                            handleTagClick={this.handleTagClick}
                            setSelectedSubsequence={this.setSelectedSubsequence}
                            userType={this.props.userType}
                        ></SequenceTab>
                    ) : null}
                </li>

                <li key="jobsli">
                    <JobsTab jobs={this.state.jobs} />
                </li>

                <li key="filesli">
                    <h3>Quick Access</h3>
                    <form className="uk-margin-bottom">
                        <fieldset className="uk-fieldset">
                            <div>
                                <button
                                    type="button"
                                    className="uk-button uk-button-primary uk-margin-left uk-form-small"
                                    onClick={this.maybeDownloadPdb}
                                    disabled={
                                        !(this.state.foldData?.name && this.state.pdb?.pdb_string)
                                    }
                                >
                                    Download Best PDB
                                </button>
                            </div>
                        </fieldset>
                    </form>

                    <h3>Files</h3>
                    <FileBrowser
                        files={this.state.files}
                        icons={{
                            Folder: <AiOutlineFolder />,
                            FolderOpen: <AiOutlineFolderOpen />,
                            Download: <FaDownload />,
                        }}
                        onDownloadFile={this.downloadFile}
                    />
                </li>

                <li key="paeli" id="paeli">
                    {this.state.paeIsOnScreen ? (
                        <PaeTab
                            foldId={this.props.foldId}
                            foldSequence={this.state.foldData?.sequence}
                        />
                    ) : null}
                </li>

                <li key="contactli" id="contactli">
                    {this.state.contactIsOnScreen ? (
                        <ContactTab
                            foldId={this.props.foldId}
                            foldSequence={this.state.foldData?.sequence}
                        ></ContactTab>
                    ) : null}
                </li>

                <li key="dock">
                    <DockTab
                        foldId={this.props.foldId}
                        foldName={this.state.foldData?.name || null}
                        foldSequence={this.state.foldData?.sequence}
                        docks={this.state.foldData ? this.state.foldData.docks : null}
                        jobs={this.state.foldData ? this.state.foldData.jobs : null}
                        setErrorText={this.props.setErrorText}
                        displayedLigandNames={[]}  // Object.keys(this.state.displayedDocks)
                        // ranks={Object.fromEntries(
                        //     Object.entries(this.state.displayedDocks).map(([key, value]) => [
                        //         key,
                        //         value.frame + 1,
                        //     ])
                        // )}
                        ranks={{}}
                        displayLigandPose={this.displayLigandPose}
                        shiftFrame={this.shiftFrame}
                        deleteLigandPose={this.deleteLigandPose}
                    />
                </li>

                <li key="Logitli">
                    <NaturalnessTab
                        foldId={this.props.foldId}
                        foldName={this.state.foldData?.name || null}
                        jobs={this.state.jobs}
                        logits={this.state.foldData?.logits || null}
                        setErrorText={this.props.setErrorText}
                    />
                </li>

                <li key="Embedli">
                    <EmbedTab
                        foldId={this.props.foldId}
                        jobs={this.state.jobs}
                        embeddings={this.state.foldData?.embeddings || null}
                        setErrorText={this.props.setErrorText}
                    />
                </li>

                <li key="Evolveli">
                    <EvolveTab
                        foldId={this.props.foldId}
                        jobs={this.state.jobs}
                        files={this.state.files}
                        evolutions={this.state.foldData?.evolutions || null}
                        setErrorText={this.props.setErrorText}
                    />
                </li>

                <li key="actionsli">
                    <form>
                        <fieldset className="uk-fieldset uk-margin">
                            <h3>Job Management</h3>
                            {[...this.actionToStageName].map((actionAndStageName) => {
                                return (
                                    <div key={actionAndStageName[1]}>
                                        <button
                                            type="button"
                                            className="uk-button uk-button-primary uk-margin-left uk-margin-small-bottom uk-form-small"
                                            onClick={() => this.startStage(actionAndStageName[1])}
                                        >
                                            {actionAndStageName[0]}
                                        </button>
                                    </div>
                                );
                            })}
                        </fieldset>
                    </form>
                </li>
            </ul>
        );

        return (
            <div className="tool-page">
                <h2
                    className="uk-margin-left uk-margin-right uk-text-center" // uk-heading-line 
                    style={{
                        marginBottom: "0px",
                        // paddingBottom: "20px",
                    }}
                    id="foldname"
                >
                    <b>{this.state.foldData ? this.state.foldData.name : "Loading..."}</b>
                </h2>
                <div className="uk-flex uk-flex-center uk-flex-wrap">
                    {[...(this.state.foldData?.jobs || [])].map((job: Invokation) => {
                        // If it's (dock, embedding, evolve) and it's not running or queued, don't show it.
                        if (
                            (job.type?.startsWith("dock_") || job.type?.startsWith("embed_") || job.type?.startsWith("evolve_")) &&
                            (job.state !== 'running' && job.state !== 'queued')) {
                            return null;
                        }
                        return (
                            <div key={job.id}>
                                {this.renderBadge(job.type || "misc", job.state, job.starttime)}
                                {/* <br /> */}
                            </div>
                        );
                    })}
                </div>

                <div
                    className="uk-grid uk-margin-top tool-panel-container"
                    data-uk-tab="margin: 20"
                >
                    <div className="uk-width-1-1 uk-width-1-2@m structure-panel">
                        {structurePane}
                    </div>

                    <div
                        className="uk-width-1-1 uk-width-1-2@m tool-panel"
                        style={{ height: "100%", display: "flex", flexDirection: "column" }}
                    >
                        {toolViewHeader}
                        <div
                            className="tool-panel-contents"
                        // style={{ flexGrow: 1, overflowY: "scroll" }}
                        >
                            {toolViewContentPane}
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    ////////////////////////////////////////////////////////////////////////////////
    // UTILITY FUNCTIONS BELOW.
    ////////////////////////////////////////////////////////////////////////////////

    getNglColorSchemeName = (colorScheme: string): string => {
        if (colorScheme === "pLDDT") {
            return "bFactor";
        } else if (colorScheme === "chainname") {
            return "chainname";
        } else if (colorScheme === "pfam") {
            return this.state.pfamColors?.nglColorscheme || "chainname";
        }
        console.error("Got invalid color scheme...");
        return "unknown";
    };

    renderBadge = (
        stageName: string,
        state: string | null | undefined,
        starttime: string | null
    ) => {
        if (!state) {
            return null;
        }

        var jobIsSuspiciouslyLongRunning = false;
        if (starttime) {
            const hoursElapsed =
                (new Date().getTime() - new Date(starttime).getTime()) / 36e5;
            jobIsSuspiciouslyLongRunning = hoursElapsed > 24;
        }

        var badgeColor;
        if (state === "failed") {
            badgeColor = "#f0506e";
        } else if (state === "finished") {
            badgeColor = "#777"; // Too light grey: "#E5E5E5";  // green: "#32d296";
        } else if (state === "deferred") {
            badgeColor = "#999999";
        } else {
            if (jobIsSuspiciouslyLongRunning) {
                badgeColor = "eed202";
            } else {
                badgeColor = "#1C87EF";
            }
        }

        return (
            <span
                className="uk-button-small uk-button-default uk-button-badge uk-margin-small-left"
                style={{ color: badgeColor, borderColor: badgeColor }}
            >
                <span>
                    {stageName}: {state}
                    {jobIsSuspiciouslyLongRunning ? (
                        <div uk-tooltip="This job has been marked running for more than 24 hours. It may have failed. You should restart this stage from the 'Actions' tab below.">
                            ⚠️
                        </div>
                    ) : null}
                </span>
            </span>
        );
    };

    changeColor = () => {
        var newColorScheme: string;
        if (this.state.colorScheme === "pLDDT") {
            newColorScheme = "chainname";
        } else if (this.state.colorScheme === "chainname") {
            newColorScheme = "pfam";
        } else {
            newColorScheme = "pLDDT";
        }

        var nglViewerColorScheme = this.getNglColorSchemeName(newColorScheme);

        // if (this.state.pdbRepr) {
        //     this.state.pdbRepr.setColor(nglViewerColorScheme);
        // }
        this.setState({ colorScheme: newColorScheme });
    };

    startStage = (stage: string) => {
        queueJob(this.props.foldId, stage, true).then(
            () => {
                UIkit.notification(`Successfully started ${stage}.`);
            },
            (e) => {
                this.props.setErrorText(e.toString());
            }
        );
    };

    maybeDownloadPdb = () => {
        if (!this.state.pdb || !this.state.foldData) {
            return;
        }
        fileDownload(this.state.pdb.pdb_string, `${this.state.foldData.name}.pdb`);
    };

    actionToStageName = [
        ["Rewrite fasta files", "write_fastas"],
        ["Rerun Sequence Annotation", "annotate"],
        ["Refold", "both"],
        ["AlphaFold2: Rerun MSA computation", "features"],
        ["AlphaFold2: Rerun Structure Prediction", "models"],
        ["AlphaFold2: Rerun Decompress Pickles job", "decompress_pkls"],
        ["Send notification email", "email"],
    ];

    // displayLigandPose = (ligandName: string) => {
    //     if (ligandName in this.state.displayedDocks) {
    //         UIkit.notification(`Hiding ${ligandName}`);
    //         this.state.stage?.removeComponent(
    //             this.state.displayedDocks[ligandName].nglComponent
    //         );

    //         for (const boxComponent of this.state.displayedDocks[ligandName]
    //             .boxComponents) {
    //             this.state.stage?.removeComponent(boxComponent);
    //         }

    //         const newDisplayedDocks = this.state.displayedDocks;
    //         delete newDisplayedDocks[ligandName];
    //         this.setState({ displayedDocks: newDisplayedDocks });
    //         return;
    //     }

    //     const dock = this.state.foldData?.docks?.find(
    //         (e) => e.ligand_name === ligandName
    //     );
    //     if (!dock) {
    //         console.error(`No ligand found with name ${ligandName}`);
    //         return;
    //     }

    //     UIkit.notification(`Displaying SDF file for ${ligandName}`);
    //     getDockSdf(this.props.foldId, ligandName).then(
    //         (sdf: Blob) => {
    //             if (!this.state.stage || !this.state.parsedPdb) {
    //                 return;
    //             }

    //             var boxComponents = new Array<NGLComponent>();

    //             if (dock.bounding_box_residue && dock.bounding_box_radius_angstrom) {
    //                 // TODO: Parse PDB to get the residues selected:
    //                 // https://www.npmjs.com/package/parse-pdb

    //                 const resCenter = getResidueCenter(
    //                     this.state.parsedPdb,
    //                     dock.bounding_box_residue
    //                 );
    //                 if (resCenter) {
    //                     for (const edge of getCubeEdges(
    //                         resCenter,
    //                         dock.bounding_box_radius_angstrom
    //                     )) {
    //                         // https://nglviewer.org/ngl/api/class/src/stage/stage.js~Stage.html
    //                         // https://nglviewer.org/ngl/api/class/src/component/component.js~Component.html
    //                         // https://nglviewer.org/ngl/api/class/src/component/shape-component.js~ShapeComponent.html
    //                         // https://nglviewer.org/ngl/api/class/src/geometry/shape.js~Shape.html

    //                         var shape = new NGL.Shape("shape");
    //                         shape.addCylinder(edge.start, edge.end, [0.5, 0.5, 0.5], 0.1);
    //                         var shapeComponent =
    //                             this.state.stage.addComponentFromObject(shape);
    //                         if (shapeComponent) {
    //                             // @ts-ignore
    //                             shapeComponent.addRepresentation("buffer");
    //                             boxComponents.push(shapeComponent);
    //                         }
    //                     }
    //                 }
    //             }

    //             this.state.stage // @ts-ignore
    //                 .loadFile(sdf, { ext: "sdf", asTrajectory: true }) // @ts-ignore
    //                 .then((o: any) => {
    //                     o.addRepresentation("ball+stick");
    //                     o.signals.trajectoryAdded.add((e: any) => {
    //                         console.log("TRAJECTORY ADDED");
    //                     });
    //                     // How to set a frame: https://github.com/nglviewer/ngl/blob/4ab8753c38995da675e9efcae2291a298948ccca/examples/js/gui.js
    //                     // All I have to do is get the trajectories: https://github.com/nglviewer/ngl/blob/4ab8753c38995da675e9efcae2291a298948ccca/src/trajectory/trajectory.ts
    //                     // The above example uses an trajectoryadded listener, or something like that, which does exist on my repr...
    //                     // After figuring it out I found a working example...: https://nglviewer.org/mdsrv/embedded.html
    //                     o.addTrajectory(null, {});

    //                     const newDisplayedDocks = this.state.displayedDocks;
    //                     newDisplayedDocks[ligandName] = {
    //                         sdf: sdf,
    //                         frame: 0,
    //                         nglComponent: o,
    //                         boxComponents: boxComponents,
    //                     };

    //                     this.setState({ displayedDocks: newDisplayedDocks });
    //                 });
    //         },
    //         (e) => {
    //             this.props.setErrorText(e.toString());
    //         }
    //     );
    // };

    deleteLigandPose = (ligandId: number, ligandName: string) => {
        UIkit.modal
            .confirm(
                `Are you sure you want to delete the docking result for ${ligandName}?`
            )
            .then(
                () => {
                    // if (ligandName in this.state.displayedDocks) {
                    //     this.state.stage?.removeComponent(
                    //         this.state.displayedDocks[ligandName].nglComponent
                    //     );

                    //     for (const boxComponent of this.state.displayedDocks[ligandName]
                    //         .boxComponents) {
                    //         this.state.stage?.removeComponent(boxComponent);
                    //     }

                    //     const newDisplayedDocks = this.state.displayedDocks;
                    //     delete newDisplayedDocks[ligandName];
                    //     this.setState({ displayedDocks: newDisplayedDocks });
                    //     return;
                    // }

                    deleteDock(ligandId).then(
                        () => {
                            UIkit.notification(`Successfully deleted ligand ${ligandName}.`);
                        },
                        (e) => {
                            UIkit.alert(
                                `Failed to delete dock ${ligandName}... something went wrong.`
                            );
                        }
                    );
                },
                () => { }
            );
    };

    shiftFrame = (ligandName: string, shift: number) => {
        // if (ligandName in this.state.displayedDocks) {
        //     const disp = this.state.displayedDocks[ligandName];
        //     var newFrame = disp.frame + shift;
        //     if (disp.nglComponent.trajList.length) {
        //         if (newFrame < 0) {
        //             newFrame = 0;
        //         }
        //         if (newFrame > disp.nglComponent.structure.frames.length) {
        //             newFrame = disp.nglComponent.structure.frames.length - 1;
        //         }
        //         disp.nglComponent.trajList[0].setFrame(newFrame);
        //     }

        //     const newDisplayedDocks = this.state.displayedDocks;
        //     newDisplayedDocks[ligandName].frame = newFrame;
        //     this.setState({ displayedDocks: newDisplayedDocks });
        // }
    };

    setPublic = (is_public: boolean) => {
        UIkit.modal
            .confirm(
                `Are you sure you want to make this fold and associated data ${is_public ? "" : "in"
                }visible to the public?`
            )
            .then(() => {
                updateFold(this.props.foldId, { public: is_public }).then(
                    () => {
                        this.refreshFoldDataFromBackend();
                        UIkit.notification("Updated public status.");
                    },
                    (e) => {
                        this.props.setErrorText(e);
                    }
                );
            });
    };

    setDisableRelaxation = (new_disable_relaxation: boolean) => {
        UIkit.modal
            .confirm(
                `Are you sure you want to set "disable relaxation" to ${new_disable_relaxation
                } for future runs of this fold?`
            )
            .then(() => {
                updateFold(this.props.foldId, { disable_relaxation: new_disable_relaxation }).then(
                    () => {
                        this.refreshFoldDataFromBackend();
                        UIkit.notification("Updated disable relaxation setting.");
                    },
                    (e) => {
                        this.props.setErrorText(e);
                    }
                );
            });
    };

    setFoldName = () => {
        UIkit.modal
            .prompt("New fold name:", "")
            .then((newFoldName: string | null) => {
                if (!newFoldName) {
                    return;
                }
                UIkit.modal
                    .confirm(
                        `Are you sure you want to rename this fold to ${newFoldName}?`
                    )
                    .then(() => {
                        updateFold(this.props.foldId, { name: newFoldName }).then(
                            () => {
                                this.refreshFoldDataFromBackend();
                                UIkit.notification("Updated fold name.");
                            },
                            (e) => {
                                this.props.setErrorText(e);
                            }
                        );
                    });
            });
    };

    setFoldModelPreset = () => {
        UIkit.modal
            .prompt("New fold model:", "")
            .then((newFoldModelPreset: string | null) => {
                if (!newFoldModelPreset) {
                    return;
                }
                updateFold(this.props.foldId, { af2_model_preset: newFoldModelPreset }).then(
                    () => {
                        this.refreshFoldDataFromBackend();
                        UIkit.notification("Updated fold model.");
                    },
                    (e) => {
                        this.props.setErrorText(e);
                    }
                );
            });
    };

    setYamlConfig = async (yaml: string) => {
        await updateFold(this.props.foldId, { yaml_config: yaml }).then(
            () => {
                this.refreshFoldDataFromBackend();
            },
            (e) => {
                this.props.setErrorText(e);
            }
        );
    };

    addTag = (tagToAdd: string) => {
        const tags = this.state.foldData?.tags;
        if (!tags) {
            return;
        }
        tags.push(tagToAdd);
        updateFold(this.props.foldId, { tags: tags }).then(
            () => {
                this.refreshFoldDataFromBackend();
                UIkit.notification("Updated tags.");
            },
            (e) => {
                this.props.setErrorText(e);
            }
        );
    };

    deleteTag = (tagToDelete: string) => {
        UIkit.modal.confirm("Delete tag?").then(
            () => {
                const tags = this.state.foldData?.tags;
                if (!tags) {
                    return;
                }
                const newTags = tags.filter((tag, index) => tag !== tagToDelete);
                updateFold(this.props.foldId, { tags: newTags }).then(
                    () => {
                        this.refreshFoldDataFromBackend();
                        UIkit.notification("Updated tags.");
                    },
                    (e) => {
                        this.props.setErrorText(e);
                    }
                );
            },
            () => {
                console.log("Tag deletion cancelled.");
            }
        );
    };

    handleTagClick = (tagToOpen: string) => {
        window.open(`/tag/${tagToOpen}`, "_self");
    };

    setSelectedSubsequence = (sele: SubsequenceSelection) => {
        // // Selection language described here:
        // // https://nglviewer.org/ngl/api/manual/usage/selection-language.html
        // // Convert chain index to a chain name, which is just "A" or "B" for NGL,
        // // for some reason.
        // const nglChainName = String.fromCharCode(65 + sele.chainIdx);
        // const selectionString = `${sele.startResidue}-${sele.endResidue}:${nglChainName}`;
        // if (this.state.selectionRepr) {
        //     console.log(`Settings selection to "${selectionString}"`);
        //     for (const singleRepr of this.state.selectionRepr) {
        //         singleRepr.setSelection(selectionString); // and :${chain}`);
        //     }
        // } else {
        //     console.log("No selectionRepr found.");
        // }
    };

    downloadFile = (keys: string[]) => {
        console.log(keys);
        for (let key of keys) {
            UIkit.notification(`Getting ${key} from server...`);
            downloadFileStraightToFilesystem(this.props.foldId, removeLeadingSlash(key), (progress: number) => {
                console.log(`Downloading ${key}: ${progress}%`);
            });
            // getFile(this.props.foldId, removeLeadingSlash(key)).then(
            //     (fileBlob: Blob) => {
            //         const newFname = key.split("/").pop();
            //         if (!newFname) {
            //             this.props.setErrorText(`No file name found for ${key}`);
            //             return;
            //         }
            //         console.log(`Downloading ${key} with file name ${newFname}!!!`);
            //         fileDownload(fileBlob, newFname);
            //     },
            //     (e) => {
            //         this.props.setErrorText(e.toString());
            //     }
            // );
        }
    };

    formatStartTime = (jobstarttime: string | null) => {
        return jobstarttime
            ? new Date(jobstarttime).toLocaleString("en-US", {
                timeStyle: "short",
                dateStyle: "short",
                timeZone: "America/Los_Angeles"
            })
            : "Not Started / Unknown";
    };

    formatRunTime = (jobRunTime: number | null) => {
        return jobRunTime
            ? `${Math.floor(jobRunTime / (60 * 60))} hr ${Math.floor(jobRunTime / 60) % 60
            } min ${Math.floor(jobRunTime) % 60} sec`
            : "NA";
    };
}

function FoldView(props: {
    setErrorText: (a: string | null) => void;
    userType: string | null;
}) {
    let { foldId } = useParams();
    if (!foldId) {
        return null;
    }
    return (
        <InternalFoldView
            setErrorText={props.setErrorText}
            foldId={parseInt(foldId)}
            userType={props.userType}
        />
    );
}

export default FoldView;
