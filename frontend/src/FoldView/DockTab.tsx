import fileDownload from "js-file-download";
import React, { useMemo, useState } from "react";
import {
  FaCheckCircle,
  FaChevronLeft,
  FaChevronRight,
  FaClock,
  FaDownload,
  FaEllipsisV,
  FaEye,
  FaFrownOpen,
  FaHamburger,
  FaRedo,
  FaTrash,
} from "react-icons/fa";
import ReactShowMoreText from "react-show-more-text";
import { NewDockPrompt } from "./../util/newDockPrompt";
import UIkit from "uikit";
import {
  Dock,
  getDockSdf,
  Invokation,
  postDock,
} from "../services/backend.service";
import { DockInput } from "../services/backend.service";

interface DockTabProps {
  foldId: number;
  foldName: string | null;
  foldSequence: string | undefined;
  docks: Dock[] | null;
  jobs: Invokation[] | null;
  setErrorText: (error: string) => void;

  // UI Commands managed by the FoldView.
  displayedLigandNames: string[];
  ranks: { [ligandname: string]: number };
  displayLigandPose: (ligandName: string) => void;
  shiftFrame: (ligandName: string, shift: number) => void;
  deleteLigandPose: (ligandId: number, ligandName: string) => void;
}

type SortConfig = {
  key: keyof Dock | "fit" | null;
  direction: "ascending" | "descending";
};

const getDockState = (dock: Dock, jobs: Invokation[] | null) => {
  if (!jobs) {
    return "queued";
  }
  for (const invokation of jobs) {
    if (invokation.id === dock.invokation_id) {
      return invokation.state;
    }
  }
  return "failed";
};

const downloadLigandPose = (
  foldId: number,
  foldName: string | null,
  ligandName: string,
  setErrorText: (error: string) => void
) => {
  UIkit.notification(`Downloading SDF file for ${ligandName}`);
  getDockSdf(foldId, ligandName).then(
    (sdf: Blob) => {
      console.log(sdf);
      if (!foldName) {
        return;
      }
      fileDownload(sdf, `${foldName}_${ligandName}.sdf`);
    },
    (e) => {
      setErrorText(e.toString());
    }
  );
};

const DockTab = React.memo((props: DockTabProps) => {
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: "ligand_name",
    direction: "ascending",
  });

  const getFit = (dock: Dock) => {
    if (dock.tool === "diffdock") {
      const confidenceStr =
        dock.pose_confidences?.split(",")[
          props.ranks[dock.ligand_name] - 1 || 0
        ];
      return confidenceStr ? parseFloat(confidenceStr) : null;
    } else {
      return (props.ranks[dock.ligand_name] - 1 || 0) == 0
        ? dock.pose_energy
        : null;
    }
  };

  function compareValues(
    key: keyof Dock | "fit",
    direction: "ascending" | "descending"
  ) {
    var aVal, bVal;
    return (a: Dock, b: Dock) => {
      if (key == "fit") {
        aVal = getFit(a);
        bVal = getFit(b);
      } else {
        aVal = a[key];
        bVal = b[key];
      }

      // Directly return 0 if both values are equal or both are null
      if (aVal === bVal) return 0;

      // If 'a' or 'b' is null, determine their sort order
      if (aVal === null) return direction === "ascending" ? -1 : 1;
      if (bVal === null) return direction === "ascending" ? 1 : -1;

      // If both are non-null and not equal, compare them as per direction
      return (aVal as any) < (bVal as any)
        ? direction === "ascending"
          ? -1
          : 1
        : direction === "ascending"
        ? 1
        : -1;
    };
  }

  const sortedDocks = useMemo(() => {
    const sortFunction = sortConfig.key
      ? compareValues(sortConfig.key, sortConfig.direction)
      : null;

    return sortFunction && props.docks
      ? [...props.docks].sort(sortFunction)
      : props.docks;
  }, [props.docks, sortConfig]);

  const requestSort = (key: keyof Dock | "fit") => {
    let direction: "ascending" | "descending" = "ascending";
    if (sortConfig.key === key && sortConfig.direction === "ascending") {
      direction = "descending";
    }
    setSortConfig({ key, direction });
  };

  const getSortDirectionSymbol = (columnName: string) => {
    if (sortConfig.key === columnName) {
      return sortConfig.direction === "ascending" ? " ↑" : " ↓";
    }
    return "";
  };

  const rerunDock = (dock: Dock) => {
    const dockCopy: DockInput = dock;
    dockCopy.fold_id = props.foldId;

    postDock(dockCopy).then(
      () => {
        UIkit.notification(
          `Successfully started docking run for ${dock.ligand_name}`
        );
      },
      (e) => {
        props.setErrorText(`Docking ${dock.ligand_name} failed: ${e}`);
      }
    );
  };

  return (
    <div>
      <h3>Small Molecule Docking</h3>
      Run Autodock Vina to find a ligand pose within the protein which minimizes
      ΔG (
      <a href="https://onlinelibrary.wiley.com/doi/pdf/10.1002/jcc.21334">
        Trott et al, 2010
      </a>
      ). The default behavior is to look for ligand poses anywhere within the
      protein.
      <div style={{ display: "flex", flexDirection: "row" }}>
        <div style={{ overflowX: "scroll", flexGrow: 1 }}>
          <table className="uk-table uk-table-striped uk-table-small">
            <thead>
              <tr>
                <th onClick={() => requestSort("ligand_name")}>
                  Name{getSortDirectionSymbol("ligand_name")}
                </th>
                <th
                  uk-tooltip={
                    "Vina- energy of pose (kJ/mol, lower is better); Diffdock- confidence (unitless, higher is better)"
                  }
                  onClick={() => requestSort("fit")}
                >
                  Fit{getSortDirectionSymbol("fit")}
                </th>
                <th uk-tooltip={"The rank of the pose being displayed."}>
                  Rank
                </th>
                <th
                  uk-tooltip={"Docking Tool"}
                  onClick={() => requestSort("tool")}
                >
                  Tool{getSortDirectionSymbol("tool")}
                </th>
                <th uk-tooltip={"Bounding Box"}>Box</th>
                <th onClick={() => requestSort("ligand_smiles")}>
                  SMILES{getSortDirectionSymbol("ligand_smiles")}
                </th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {sortedDocks
                ? [...sortedDocks].map((dock) => {
                    return (
                      <tr key={dock.ligand_name}>
                        <td>{dock.ligand_name}</td>
                        <td>
                          <span
                            uk-tooltip={
                              dock.tool === "diffdock"
                                ? "Confidence, higher is better."
                                : "kJ/mol"
                            }
                          >
                            {getFit(dock)}
                          </span>
                        </td>
                        <td>{props.ranks[dock.ligand_name]}</td>
                        <td>{dock.tool}</td>
                        <td>
                          {dock.bounding_box_residue &&
                          dock.bounding_box_radius_angstrom ? (
                            <FaCheckCircle
                              uk-tooltip={`Residue ${dock.bounding_box_residue}\n
                        Radius ${dock.bounding_box_radius_angstrom}A`}
                            />
                          ) : null}
                        </td>
                        <td style={{ overflowWrap: "anywhere" }}>
                          <ReactShowMoreText
                            lines={1}
                            more="show"
                            less="hide"
                            expanded={false}
                            truncatedEndingComponent={" "}
                            width={1}
                          >
                            {dock.ligand_smiles}
                          </ReactShowMoreText>
                        </td>
                        <td style={{ minWidth: "100px", userSelect: "none" }}>
                          {getDockState(dock, props.jobs) === "queued" ||
                          getDockState(dock, props.jobs) === "running" ? (
                            <FaClock
                              uk-tooltip={getDockState(dock, props.jobs)}
                            />
                          ) : null}
                          {getDockState(dock, props.jobs) === "failed" ? (
                            <FaFrownOpen
                              uk-tooltip={getDockState(dock, props.jobs)}
                            />
                          ) : null}
                          {getDockState(dock, props.jobs) === "finished" ? (
                            <span>
                              <FaEye
                                uk-tooltip="Toggle display at left."
                                onClick={() =>
                                  props.displayLigandPose(dock.ligand_name)
                                }
                              />
                            </span>
                          ) : null}
                          {props.displayedLigandNames.includes(
                            dock.ligand_name
                          ) ? (
                            <span>
                              <FaChevronLeft
                                onClick={() =>
                                  props.shiftFrame(dock.ligand_name, -1)
                                }
                                uk-tooltip="Previous prediction"
                              />
                              <FaChevronRight
                                onClick={() =>
                                  props.shiftFrame(dock.ligand_name, 1)
                                }
                                uk-tooltip="Next prediction"
                              />
                            </span>
                          ) : null}
                          <FaEllipsisV
                            uk-tooltip={"Click for more actions"}
                          ></FaEllipsisV>
                          <div uk-dropdown="mode: click; pos: bottom-right; boundary: !.boundary; shift: false; flip: false">
                            <ul className="uk-nav uk-dropdown-nav">
                              {getDockState(dock, props.jobs) === "finished" ? (
                                <li className="uk-active">
                                  <a
                                    uk-tooltip="Download SDF of ligand pose."
                                    onClick={() =>
                                      downloadLigandPose(
                                        props.foldId,
                                        props.foldName,
                                        dock.ligand_name,
                                        props.setErrorText
                                      )
                                    }
                                  >
                                    <FaDownload />
                                    Download
                                  </a>
                                </li>
                              ) : null}
                              <li className="uk-active">
                                <a
                                  uk-tooltip="Delete docking result."
                                  onClick={() =>
                                    props.deleteLigandPose(
                                      dock.id,
                                      dock.ligand_name
                                    )
                                  }
                                >
                                  <FaTrash />
                                  Delete
                                </a>
                              </li>
                              <li className="uk-active">
                                <a
                                  uk-tooltip="Rerun this dock."
                                  onClick={() => {
                                    rerunDock(dock);
                                  }}
                                >
                                  <FaRedo />
                                  Rerun
                                </a>
                              </li>
                            </ul>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                : null}
            </tbody>
          </table>
        </div>
      </div>
      <h3>Dock new ligands</h3>
      <NewDockPrompt
        setErrorText={props.setErrorText}
        foldIds={[props.foldId]}
        existingLigands={{
          [props.foldId]: [
            ...(props.docks ?? []).map((dock) => dock.ligand_name),
          ],
        }}
      />
    </div>
  );
});

export default DockTab;
