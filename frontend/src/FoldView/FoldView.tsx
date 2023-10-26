import React, { Component, RefObject } from "react";
import "./FoldView.scss";
import UIkit from "uikit";
import {
  Annotations,
  deleteDock,
  FileInfo,
  Fold,
  FoldPdb,
  getDockSdf,
  getFile,
  getFileList,
  getFold,
  getFoldPae,
  getFoldPdb,
  getFoldPfam,
  getInvokation,
  Invokation,
  queueJob,
  updateFold,
} from "../services/backend.service";
import jquery from "jquery";
import { getColorsForAnnotations, VariousColorSchemes } from "../util/plots";
import { AiOutlineFolder, AiOutlineFolderOpen } from "react-icons/ai";
import {
  Stage,
  StructureComponent,
  Component as NGLComponent,
  RepresentationCollection as NGLRepresentationCollection,
} from "react-ngl/dist/@types/ngl/declarations/ngl";
import { useParams } from "react-router-dom";
import SequenceTab, { SubsequenceSelection } from "./SequenceTab";
import ContactTab from "./ContactTab";
import PaeTab from "./PaeTab";
import DockTab from "./DockTab";
import FileBrowser from "react-keyed-file-browser";
import { FaDownload } from "react-icons/fa";
import ParsePdb, { ParsedPdb } from "parse-pdb";
import { FoldyMascot } from "./../util/foldyMascot";
const NGL = require("./../../node_modules/ngl/dist/ngl");
const fileDownload = require("js-file-download");

const REFRESH_STATE_PERIOD = 3000;
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

interface DisplayedDock {
  sdf: Blob;
  frame: number;
  nglComponent: StructureComponent;
  boxComponents: NGLComponent[];
}

interface FoldState {
  foldData: Fold | null;

  // Note that a subset of job data is also in foldData.
  files: FileInfo[];
  jobs: Invokation[] | null;
  pdb: FoldPdb | null;
  parsedPdb: ParsedPdb | null;
  stage: Stage | null;
  stageRef: RefObject<any>;

  // Defines our current color "mode".
  colorScheme: string;

  pfamAnnotations: Annotations | null;
  pfamColors: VariousColorSchemes | null;

  // Docking stuff.
  displayedDocks: { [ligandName: string]: DisplayedDock };

  // Nglviewer and other view management.
  pdbRepr: NGLRepresentationCollection | null;
  selectionRepr: NGLRepresentationCollection | null;
  pdbFailedToLoad: boolean;
  paeIsOnScreen: boolean;
  contactIsOnScreen: boolean;
  showSplitScreen: boolean;
  numRefreshes: number;
  isRocking: boolean;
}

// From UIkit's definition of a "medium" window: https://getuikit.com/docs/visibility
const WINDOW_WIDTH_FOR_SPLIT_SCREEN = 960;

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
      stage: null,
      stageRef: React.createRef(),

      colorScheme: "pLDDT",

      pfamAnnotations: null,
      pfamColors: null,

      displayedDocks: {},

      pdbRepr: null,
      selectionRepr: null,
      pdbFailedToLoad: false,
      paeIsOnScreen: false,
      contactIsOnScreen: false,
      showSplitScreen: window.innerWidth >= WINDOW_WIDTH_FOR_SPLIT_SCREEN,
      numRefreshes: 0,
      isRocking: true,
    };
  }

  preventDefault = (e: any) => e.preventDefault();

  handleResize = () => {
    if (this.state.stage) {
      this.state.stage.handleResize();
    }

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
    });
  };

  componentDidMount() {
    this.interval = setInterval(() => {
      getFold(this.props.foldId).then((new_fold_data) => {
        if (
          this.state.numRefreshes > REFRESH_STATE_MAX_ITERS &&
          this.interval
        ) {
          clearInterval(this.interval);
        }
        this.setState({
          foldData: new_fold_data,
          numRefreshes: this.state.numRefreshes + 1,
        });
      });
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
              pfamColors: getColorsForAnnotations(
                this.state.foldData.sequence,
                pfam
              ),
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
            var stage = new NGL.Stage("viewport", { backgroundColor: "white" });

            var stringBlob = new Blob([pdb.pdb_string], { type: "text/plain" });

            // Load this here, since apparently it's not accessible within the below code...
            const nglColorScheme = this.getNglColorSchemeName(
              this.state.colorScheme
            );
            stage.loadFile(stringBlob, { ext: "pdb" }).then((o: any) => {
              const pdbRepr = o.addRepresentation("cartoon", {
                colorScheme: nglColorScheme,
              });
              var duration = 1000; // optional duration for animation, defaults to zero
              o.autoView(duration);

              const selectionRepr = o.addRepresentation("cartoon", {
                // Start out with nothing selected - so make an impossible selection.
                sele: "1 and 2",
                // color: "#F866AF",  // Hot pink.
                color: "red",
              });

              this.setState({ pdbRepr: pdbRepr, selectionRepr: selectionRepr });
            });
            stage.setRock(false);
            this.state.stageRef.current.addEventListener(
              "wheel",
              this.preventDefault,
              { passive: false }
            );
            this.setState({ stage: stage });
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
    this.state.stageRef.current.removeEventListener(
      "wheel",
      this.preventDefault
    );
    window.removeEventListener("resize", this.handleResize);

    if (this.interval) {
      clearInterval(this.interval);
    }
  }

  render() {
    var structurePane = (
      <div key="structure" style={{ height: "100%" }}>
        {this.state.pdb ? null : (
          <div className="uk-text-center">
            {this.state.pdbFailedToLoad ? (
              <FoldyMascot
                text={"Looks like your structure isn't ready."}
                moveTextAbove={false}
              />
            ) : (
              <div uk-spinner="ratio: 4"></div>
            )}
          </div>
        )}

        <div
          className="uk-text-center"
          id="viewport"
          style={{ width: "100%", height: "100%" }}
          ref={this.state.stageRef}
        >
          <div
            style={{
              display: "inline",
              position: "absolute",
              top: "5px",
              left: "25px",
              zIndex: 99,
              borderRadius: "4px",
            }}
          >
            <button
              className="uk-button uk-button-small uk-button-default uk-margin-small-right"
              onClick={this.changeColor}
              style={{ backgroundColor: "white" }}
            >
              Color: {this.state.colorScheme.split("|").pop()}
            </button>
            <button
              className="uk-button uk-button-small uk-button-default"
              style={{ backgroundColor: "white" }}
              onClick={() => this.toggleRocking()}
            >
              Rock
            </button>
          </div>
        </div>
      </div>
    );

    var toolViewHeader = (
      <ul
        className="uk-tab"
        data-uk-tab="connect: #switcher; swiping: false"
        id="tab"
      >
        {/* {
    this.state.showSplitScreen ?
    null :
    <li className={displayStructure ? undefined : 'uk-disabled'}>
      <a>Structure</a>
    </li>
  } */}
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
              sequence={this.state.foldData.sequence}
              colorScheme={this.state.colorScheme}
              pfamColors={this.state.pfamColors}
              setPublic={this.setPublic}
              setFoldName={this.setFoldName}
              addTag={this.addTag}
              deleteTag={this.deleteTag}
              handleTagClick={this.handleTagClick}
              setSelectedSubsequence={this.setSelectedSubsequence}
              userType={this.props.userType}
            ></SequenceTab>
          ) : null}
        </li>

        <li key="logsli">
          {this.state.jobs ? (
            <span>
              <h2>Invokations</h2>
              <div className="uk-overflow-auto">
                <table
                  className="uk-table uk-table-hover uk-table-small"
                  style={{ tableLayout: "fixed" }}
                >
                  <thead>
                    <tr>
                      <th className="uk-table-shrink uk-text-nowrap">Type</th>
                      <th className="uk-table-shrink uk-text-nowrap">State</th>
                      <th className="uk-table-shrink uk-text-nowrap">
                        Start time
                      </th>
                      <th className="uk-table-shrink uk-text-nowrap">
                        Runtime
                      </th>
                      <th className="uk-table-shrink uk-text-nowrap">Logs</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...this.state.jobs].map((job: Invokation) => {
                      return (
                        <tr key={`${job.job_id}_${job.id}`}>
                          <td
                            className="uk-text-nowrap uk-text-truncate"
                            uk-tooltip={job.type}
                          >
                            {job.type}
                          </td>
                          <td
                            className="uk-text-nowrap uk-text-truncate"
                            uk-tooltip={job.state}
                          >
                            {job.state}
                          </td>
                          <td
                            className="uk-text-nowrap uk-text-truncate"
                            uk-tooltip={this.formatStartTime(job.starttime)}
                          >
                            {this.formatStartTime(job.starttime)}
                          </td>
                          <td
                            className="uk-text-nowrap uk-text-truncate"
                            uk-tooltip={this.formatRunTime(job.timedelta_sec)}
                          >
                            {this.formatRunTime(job.timedelta_sec)}
                          </td>
                          <td className="uk-text-nowrap uk-text-truncate">
                            <a href={`#logs_${job.id.toString()}`}>View</a>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <span>
                {[...this.state.jobs].map((job: Invokation) => {
                  return (
                    <div
                      id={`logs_${job.id.toString()}`}
                      key={job.id || "jobid should not be null"}
                    >
                      <h3>{job.type} Logs</h3>
                      <pre>Command: {job.command}</pre>
                      <pre>{job.log}</pre>
                    </div>
                  );
                })}
              </span>
            </span>
          ) : null}
        </li>

        <li key="filesli">
          <h3>Quick Access</h3>
          <form className="uk-margin-bottom">
            <fieldset className="uk-fieldset">
              <div>
                <button
                  type="button"
                  className="uk-button uk-button-default uk-margin-left uk-form-small"
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

          {/* TODO: Implement a file download.
      https://github.com/TimboKZ/chonky-website/blob/master/2.x_storybook/src/demos/S3Browser.tsx
      https://uptick.github.io/react-keyed-file-browser/
      */}
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
            displayedLigandNames={Object.keys(this.state.displayedDocks)}
            ranks={Object.fromEntries(
              Object.entries(this.state.displayedDocks).map(([key, value]) => [
                key,
                value.frame + 1,
              ])
            )}
            displayLigandPose={this.displayLigandPose}
            shiftFrame={this.shiftFrame}
            deleteLigandPose={this.deleteLigandPose}
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
          className="uk-heading-line uk-margin-left uk-margin-right uk-text-center"
          style={{ marginBottom: "0px", paddingBottom: "20px" }}
          id="foldname"
        >
          <b>{this.state.foldData ? this.state.foldData.name : "Loading..."}</b>
        </h2>
        {/* <hr className="uk-divider-icon" /> */}
        <div className="uk-flex uk-flex-center uk-flex-wrap">
          {[...(this.state.foldData?.jobs || [])].map((job: Invokation) => {
            if (job.type?.startsWith("dock_")) {
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
    if (!this.state.stage) {
      return;
    }

    var newColorScheme: string;
    if (this.state.colorScheme === "pLDDT") {
      newColorScheme = "chainname";
    } else if (this.state.colorScheme === "chainname") {
      newColorScheme = "pfam";
    } else {
      newColorScheme = "pLDDT";
    }

    var nglViewerColorScheme = this.getNglColorSchemeName(newColorScheme);

    if (this.state.pdbRepr) {
      this.state.pdbRepr.setColor(nglViewerColorScheme);
    }
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
    ["Rerun whole pipeline", "both"],
    ["Rewrite fasta files", "write_fastas"],
    ["Rerun MSA computation", "features"],
    ["Rerun Structure Prediction", "models"],
    ["Rerun Decompress Pickles job", "decompress_pkls"],
    ["Rerun PFAM Sequence Annotation", "annotate"],
    ["Send notification email", "email"],
  ];

  displayLigandPose = (ligandName: string) => {
    if (ligandName in this.state.displayedDocks) {
      UIkit.notification(`Hiding ${ligandName}`);
      this.state.stage?.removeComponent(
        this.state.displayedDocks[ligandName].nglComponent
      );

      for (const boxComponent of this.state.displayedDocks[ligandName]
        .boxComponents) {
        this.state.stage?.removeComponent(boxComponent);
      }

      const newDisplayedDocks = this.state.displayedDocks;
      delete newDisplayedDocks[ligandName];
      this.setState({ displayedDocks: newDisplayedDocks });
      return;
    }

    const dock = this.state.foldData?.docks?.find(
      (e) => e.ligand_name === ligandName
    );
    if (!dock) {
      console.error(`No ligand found with name ${ligandName}`);
      return;
    }

    UIkit.notification(`Displaying SDF file for ${ligandName}`);
    getDockSdf(this.props.foldId, ligandName).then(
      (sdf: Blob) => {
        if (!this.state.stage || !this.state.parsedPdb) {
          return;
        }

        var boxComponents = new Array<NGLComponent>();

        if (dock.bounding_box_residue && dock.bounding_box_radius_angstrom) {
          // TODO: Parse PDB to get the residues selected:
          // https://www.npmjs.com/package/parse-pdb

          const resCenter = getResidueCenter(
            this.state.parsedPdb,
            dock.bounding_box_residue
          );
          if (resCenter) {
            for (const edge of getCubeEdges(
              resCenter,
              dock.bounding_box_radius_angstrom
            )) {
              // https://nglviewer.org/ngl/api/class/src/stage/stage.js~Stage.html
              // https://nglviewer.org/ngl/api/class/src/component/component.js~Component.html
              // https://nglviewer.org/ngl/api/class/src/component/shape-component.js~ShapeComponent.html
              // https://nglviewer.org/ngl/api/class/src/geometry/shape.js~Shape.html

              var shape = new NGL.Shape("shape");
              shape.addCylinder(edge.start, edge.end, [0.5, 0.5, 0.5], 0.1);
              var shapeComponent =
                this.state.stage.addComponentFromObject(shape);
              if (shapeComponent) {
                // @ts-ignore
                shapeComponent.addRepresentation("buffer");
                boxComponents.push(shapeComponent);
              }
            }
          }
        }

        this.state.stage // @ts-ignore
          .loadFile(sdf, { ext: "sdf", asTrajectory: true }) // @ts-ignore
          .then((o: any) => {
            o.addRepresentation("ball+stick");
            o.signals.trajectoryAdded.add((e: any) => {
              console.log("TRAJECTORY ADDED");
            });
            // How to set a frame: https://github.com/nglviewer/ngl/blob/4ab8753c38995da675e9efcae2291a298948ccca/examples/js/gui.js
            // All I have to do is get the trajectories: https://github.com/nglviewer/ngl/blob/4ab8753c38995da675e9efcae2291a298948ccca/src/trajectory/trajectory.ts
            // The above example uses an trajectoryadded listener, or something like that, which does exist on my repr...
            // After figuring it out I found a working example...: https://nglviewer.org/mdsrv/embedded.html
            o.addTrajectory(null, {});

            const newDisplayedDocks = this.state.displayedDocks;
            newDisplayedDocks[ligandName] = {
              sdf: sdf,
              frame: 0,
              nglComponent: o,
              boxComponents: boxComponents,
            };

            this.setState({ displayedDocks: newDisplayedDocks });
          });
      },
      (e) => {
        this.props.setErrorText(e.toString());
      }
    );
  };

  deleteLigandPose = (ligandId: number, ligandName: string) => {
    UIkit.modal
      .confirm(
        `Are you sure you want to delete the docking result for ${ligandName}?`
      )
      .then(
        () => {
          if (ligandName in this.state.displayedDocks) {
            this.state.stage?.removeComponent(
              this.state.displayedDocks[ligandName].nglComponent
            );

            for (const boxComponent of this.state.displayedDocks[ligandName]
              .boxComponents) {
              this.state.stage?.removeComponent(boxComponent);
            }

            const newDisplayedDocks = this.state.displayedDocks;
            delete newDisplayedDocks[ligandName];
            this.setState({ displayedDocks: newDisplayedDocks });
            return;
          }

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
        () => {}
      );
  };

  shiftFrame = (ligandName: string, shift: number) => {
    if (ligandName in this.state.displayedDocks) {
      const disp = this.state.displayedDocks[ligandName];
      var newFrame = disp.frame + shift;
      if (disp.nglComponent.trajList.length) {
        if (newFrame < 0) {
          newFrame = 0;
        }
        if (newFrame > disp.nglComponent.structure.frames.length) {
          newFrame = disp.nglComponent.structure.frames.length - 1;
        }
        disp.nglComponent.trajList[0].setFrame(newFrame);
      }

      const newDisplayedDocks = this.state.displayedDocks;
      newDisplayedDocks[ligandName].frame = newFrame;
      this.setState({ displayedDocks: newDisplayedDocks });
    }
  };

  toggleRocking = () => {
    const newIsRocking = !this.state.isRocking;
    if (this.state.stage) {
      this.state.stage.setRock(newIsRocking);
    }
    this.setState({ isRocking: newIsRocking });
  };

  setPublic = (is_public: boolean) => {
    UIkit.modal
      .confirm(
        `Are you sure you want to make this fold and associated data ${
          is_public ? "" : "in"
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
    // Selection language described here:
    // https://nglviewer.org/ngl/api/manual/usage/selection-language.html
    // Convert chain index to a chain name, which is just "A" or "B" for NGL,
    // for some reason.
    const nglChainName = String.fromCharCode(65 + sele.chainIdx);
    const selectionString = `${sele.startResidue}-${sele.endResidue}:${nglChainName}`;
    if (this.state.selectionRepr) {
      console.log(`Settings selection to "${selectionString}"`);
      this.state.selectionRepr.setSelection(selectionString); // and :${chain}`);
    } else {
      console.log("No selectionRepr found.");
    }
  };

  downloadFile = (keys: string[]) => {
    function removeLeadingSlash(val: string) {
      return val.startsWith("/") ? val.substring(1) : val;
    }
    console.log(keys);
    for (let key of keys) {
      console.log(`Getting ${key} from server...`);
      getFile(this.props.foldId, removeLeadingSlash(key)).then(
        (fileBlob: Blob) => {
          console.log(`Downloading ${key}!!!`);
          const newFname = key.split("/").slice(-1);
          fileDownload(fileBlob, newFname);
        },
        (e) => {
          this.props.setErrorText(e.toString());
        }
      );
    }
  };

  formatStartTime = (jobstarttime: string | null) => {
    return jobstarttime
      ? new Date(jobstarttime).toLocaleString("en-US", {
          timeStyle: "short",
          dateStyle: "short",
        })
      : "Not Started / Unknown";
  };

  formatRunTime = (jobRunTime: number | null) => {
    return jobRunTime
      ? `${Math.floor(jobRunTime / (60 * 60))} hr ${
          Math.floor(jobRunTime / 60) % 60
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
