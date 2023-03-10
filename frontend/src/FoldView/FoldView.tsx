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
  getFoldAntismash,
  getFoldPdb,
  getFoldPfam,
  getInvokation,
  Invokation,
  queueJob,
  updateFold,
} from "../services/backend.service";
import jquery from "jquery";
import { getColorsForAnnotations, VariousColorSchemes } from "../helpers/plots";
import { AiOutlineFolder, AiOutlineFolderOpen } from "react-icons/ai";
import {
  Stage,
  StructureComponent,
  Component as NGLComponent,
} from "react-ngl/dist/@types/ngl/declarations/ngl";
import { useParams } from "react-router-dom";
import SequenceTab from "./SequenceTab";
import ContactTab from "./ContactTab";
import PaeTab from "./PaeTab";
import DockTab from "./DockTab";
import FileBrowser from "react-keyed-file-browser";
import { FaDownload } from "react-icons/fa";
import ParsePdb, { ParsedPdb } from "parse-pdb";
import { Foldy } from "./../Util";
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

  antismashAnnotations: Annotations | null;
  antismashColors: VariousColorSchemes | null;

  pfamAnnotations: Annotations | null;
  pfamColors: VariousColorSchemes | null;

  // Docking stuff.
  displayedDocks: { [ligandName: string]: DisplayedDock };

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

      antismashAnnotations: null,
      antismashColors: null,

      pfamAnnotations: null,
      pfamColors: null,

      displayedDocks: {},

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

        getFoldAntismash(this.props.foldId).then(
          (antismash) => {
            if (!this.state.foldData) {
              return;
            }
            this.setState({
              antismashAnnotations: antismash,
              antismashColors: getColorsForAnnotations(
                this.state.foldData.sequence,
                antismash
              ),
            });
          },
          (e) => {
            console.log(e.toString());
          }
        );

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
            const nglColorScheme = this.getNglColorSchemeName(this.state.colorScheme);
            stage.loadFile(stringBlob, { ext: "pdb" }).then(function (o: any) {
              o.addRepresentation("cartoon", { colorScheme: nglColorScheme });
              var duration = 1000; // optional duration for animation, defaults to zero
              o.autoView(duration);
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

  getNglColorSchemeName = (colorScheme: string): string  => {
    if (colorScheme === "pLDDT") {
      return "bFactor";
    } else if (colorScheme === "chainname") {
      return "chainname";
    } else if (colorScheme === "antismash") {
      return this.state.antismashColors?.nglColorscheme || "chainname";
    } else if (colorScheme === "pfam") {
      return this.state.pfamColors?.nglColorscheme || "chainname";
    }
    console.error('Got invalid color scheme...');
    return "unknown";
  }

  render() {
    const renderBadge = (
      stageName: string,
      state: string | null | undefined
    ) => {
      if (!state) {
        return null;
      }

      var badgeColor;
      if (state === "failed") {
        badgeColor = "#f0506e";
      } else if (state === "finished") {
        badgeColor = "#777"; // Too light grey: "#E5E5E5";  // green: "#32d296";
      } else if (state === "deferred") {
        badgeColor = "#999999";
      } else {
        badgeColor = "#1C87EF";
      }

      return (
        <span
          className="uk-button-small uk-button-default uk-button-badge uk-margin-small-left"
          style={{ color: badgeColor, borderColor: badgeColor }}
        >
          <span>
            {stageName}: {state}
          </span>
        </span>
      );
    };

    const changeColor = () => {
      if (!this.state.stage) {
        return;
      }

      var newColorScheme: string;
      if (this.state.colorScheme === "pLDDT") {
        newColorScheme = "chainname";
      } else if (this.state.colorScheme === "chainname") {
        newColorScheme = "antismash";
      } else if (this.state.colorScheme === "antismash") {
        newColorScheme = "pfam";
      } else {
        newColorScheme = "pLDDT";
      }

      var nglViewerColorScheme = this.getNglColorSchemeName(newColorScheme);

      this.state.stage.removeAllComponents();
      if (this.state.pdb) {
        var stringBlob = new Blob([this.state.pdb.pdb_string], {
          type: "text/plain",
        });
        this.state.stage
          .loadFile(stringBlob, { ext: "pdb" })
          .then(function (o: any) {
            o.addRepresentation("cartoon", {
              colorScheme: nglViewerColorScheme,
            });
            o.autoView();
          });
      }
      this.setState({ colorScheme: newColorScheme });
    };

    const startStage = (stage: string) => {
      queueJob(this.props.foldId, stage, true).then(
        () => {
          UIkit.notification(`Successfully started ${stage}.`);
        },
        (e) => {
          this.props.setErrorText(e.toString());
        }
      );
    };

    const maybeDownloadPdb = () => {
      if (!this.state.pdb || !this.state.foldData) {
        return;
      }
      fileDownload(
        this.state.pdb.pdb_string,
        `${this.state.foldData.name}.pdb`
      );
    };

    const actionToStageName = [
      ["Start All Stages", "both"],
      ["Write Fastas", "write_fastas"],
      ["Start Features Job", "features"],
      ["Start Models job", "models"],
      ["Start Decompress Pickles job", "decompress_pkls"],
      ["Start Annotate Job", "annotate"],
      ["Send notification email", "email"],
    ];

    var structureView = (
      <div key="structure">
        {this.state.pdb ? null : (
          <div className="uk-text-center">
            {
              this.state.pdbFailedToLoad ?
              <Foldy 
                text={"Looks like your structure isn't ready."}
                moveTextAbove={false}
              /> :
              <div uk-spinner="ratio: 4"></div>
            }
          </div>
        )}

        <div
          className="uk-text-center"
          id="viewport"
          style={{ width: "100%", height: "500px" }}
          ref={this.state.stageRef}
        >
          {/* <div style={{
          position: 'fixed',
          bottom: '20px',
          right: '30px',
          zIndex: 99,
          borderRadius: '4px',
        }} */}
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
              onClick={changeColor}
              style={{ backgroundColor: "white" }}
            >
              {this.state.colorScheme.split("|").pop()}
            </button>
            <button
              className="uk-button uk-button-small uk-button-default"
              style={{ backgroundColor: "white" }}
              onClick={() => toggleRocking()}
            >
              Rock
            </button>
            {/* <a
            uk-scroll={1}
            href="#foldname"
            className="uk-button uk-button-default uk-margin-small-left uk-padding-remove-vertical"
            style={{backgroundColor: 'white'}}
            >
            up
          </a>
          <a
            uk-scroll={1}
            href="#switcher"
            className="uk-button uk-button-default uk-margin-small-left uk-padding-remove-vertical"
            style={{backgroundColor: 'white'}}
            >
            down
          </a> */}
          </div>
        </div>
      </div>
    );

    const displayLigandPose = (ligandName: string) => {
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
                frame: 1,
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

    const deleteLigandPose = (ligandId: number, ligandName: string) => {
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
                UIkit.notification(
                  `Successfully deleted ligand ${ligandName}.`
                );
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

    const shiftFrame = (ligandName: string, shift: number) => {
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

    const toggleRocking = () => {
      const newIsRocking = !this.state.isRocking;
      if (this.state.stage) {
        this.state.stage.setRock(newIsRocking);
      }
      this.setState({ isRocking: newIsRocking });
    };

    const addTag = (tagToAdd: string) => {
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
    const deleteTag = (tagToDelete: string) => {
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
    const handleTagClick = (tagToOpen: string) => {
      window.open(`/tag/${tagToOpen}`, "_self");
    };

    const downloadFile = (keys: string[]) => {
      console.log(keys);
      for (let key of keys) {
        getFile(this.props.foldId, key).then(
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

    return (
      <div>
        <h2
          className="uk-heading-line uk-margin-left uk-margin-right uk-text-center"
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
                {renderBadge(job.type || "misc", job.state)}
                {/* <br /> */}
              </div>
            );
          })}
        </div>

        <div className="uk-grid uk-margin-top" data-uk-tab="margin: 20">
          <div className="uk-width-1-1 uk-width-1-2@m">{structureView}</div>

          <div className="uk-width-1-1 uk-width-1-2@m">
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
                <a>Sequence</a>
              </li>
              <li>
                <a>Logging</a>
              </li>
              <li>
                <a>Actions</a>
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
            </ul>

            <ul
              className="uk-switcher uk-margin uk-padding-small"
              id="switcher"
            >
              <li key="sequenceli">
                {this.state.foldData ? (
                  <SequenceTab
                    foldId={this.props.foldId}
                    foldName={this.state.foldData?.name}
                    foldTags={this.state.foldData?.tags}
                    foldOwner={this.state.foldData?.owner}
                    foldModelPreset={this.state.foldData?.af2_model_preset}
                    foldDisableRelaxation={this.state.foldData?.disable_relaxation}
                    sequence={this.state.foldData.sequence}
                    colorScheme={this.state.colorScheme}
                    antismashColors={this.state.antismashColors}
                    pfamColors={this.state.pfamColors}
                    addTag={addTag}
                    deleteTag={deleteTag}
                    handleTagClick={handleTagClick}
                  ></SequenceTab>
                ) : null}
              </li>

              <li key="logsli">
                {this.state.jobs
                  ? [...this.state.jobs].map((job: Invokation) => {
                      return (
                        <div key={job.id || "jobid should not be null"}>
                          <h3>
                            {job.type} Logs [
                            {job.timedelta_sec
                              ? (job.timedelta_sec / 60).toFixed(0)
                              : "??"}{" "}
                            min]
                          </h3>
                          <pre>{job.log}</pre>
                        </div>
                      );
                    })
                  : null}
              </li>

              <li key="actionsli">
                <h3>Quick Actions</h3>
                <form className="uk-margin-bottom">
                  <fieldset className="uk-fieldset">
                    <div>
                      <button
                        type="button"
                        className="uk-button uk-button-default uk-margin-left uk-form-small"
                        onClick={maybeDownloadPdb}
                        disabled={
                          !(
                            this.state.foldData?.name &&
                            this.state.pdb?.pdb_string
                          )
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
                  onDownloadFile={downloadFile}
                />

                <form>
                  <fieldset className="uk-fieldset uk-margin">
                    <h3>Job Management</h3>
                    {[...actionToStageName].map((actionAndStageName) => {
                      return (
                        <div key={actionAndStageName[1]}>
                          <button
                            type="button"
                            className="uk-button uk-button-primary uk-margin-left uk-margin-small-bottom uk-form-small"
                            onClick={() => startStage(actionAndStageName[1])}
                          >
                            {actionAndStageName[0]}
                          </button>
                        </div>
                      );
                    })}
                  </fieldset>
                </form>
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
                  displayLigandPose={displayLigandPose}
                  shiftFrame={shiftFrame}
                  deleteLigandPose={deleteLigandPose}
                />
              </li>
            </ul>
          </div>
        </div>
      </div>
    );
  }
}

function FoldView(props: { setErrorText: (a: string | null) => void }) {
  let { foldId } = useParams();
  if (!foldId) {
    return null;
  }
  return (
    <InternalFoldView
      setErrorText={props.setErrorText}
      foldId={parseInt(foldId)}
    />
  );
}

export default FoldView;
