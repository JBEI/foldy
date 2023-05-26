import fileDownload from "js-file-download";
import React from "react";
import {
  FaCheckCircle,
  FaChevronLeft,
  FaChevronRight,
  FaClock,
  FaDownload,
  FaEye,
  FaFrownOpen,
  FaTrash,
} from "react-icons/fa";
import ReactShowMoreText from "react-show-more-text";
import { NewDockPrompt } from "./../util/newDockPrompt";
import UIkit from "uikit";
import { Dock, getDockSdf, Invokation } from "../services/backend.service";

interface DockTabProps {
  foldId: number;
  foldName: string | null;
  foldSequence: string | undefined;
  docks: Dock[] | null;
  jobs: Invokation[] | null;
  setErrorText: (error: string) => void;

  // UI Commands managed by the FoldView.
  displayedLigandNames: string[];
  displayLigandPose: (ligandName: string) => void;
  shiftFrame: (ligandName: string, shift: number) => void;
  deleteLigandPose: (ligandId: number, ligandName: string) => void;
}

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
      <table className="uk-table uk-table-striped uk-table-small">
        <thead>
          <tr>
            <th>Name</th>
            <th>SMILES</th>
            <th uk-tooltip={"[kJ/mol]"}>Energy</th>
            <th uk-tooltip={"Docking Tool"}>Tool</th>
            <th uk-tooltip={"Bounding Box"}>Box</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {props.docks
            ? [...props.docks].map((dock) => {
                return (
                  <tr key={dock.ligand_name}>
                    <td>{dock.ligand_name}</td>
                    <td style={{ overflowWrap: "anywhere", width: "300px" }}>
                      <ReactShowMoreText
                        lines={1}
                        more="..."
                        less="less"
                        expanded={false}
                        truncatedEndingComponent={" "}
                      >
                        {dock.ligand_smiles}
                      </ReactShowMoreText>
                    </td>
                    <td>
                      {dock.tool === "diffdock" ? (
                        <span
                          uk-tooltip={
                            "DiffDock does not compute energy of docking."
                          }
                        >
                          N/A
                        </span>
                      ) : (
                        dock.pose_energy
                      )}
                    </td>
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
                    <td style={{ width: "100px" }}>
                      {getDockState(dock, props.jobs) === "queued" ||
                      getDockState(dock, props.jobs) === "running" ? (
                        <FaClock uk-tooltip={getDockState(dock, props.jobs)} />
                      ) : null}
                      {getDockState(dock, props.jobs) === "failed" ? (
                        <FaFrownOpen
                          uk-tooltip={getDockState(dock, props.jobs)}
                        />
                      ) : null}
                      {getDockState(dock, props.jobs) === "finished" ? (
                        <span>
                          <FaDownload
                            uk-tooltip="Download SDF of ligand pose."
                            onClick={() =>
                              downloadLigandPose(
                                props.foldId,
                                props.foldName,
                                dock.ligand_name,
                                props.setErrorText
                              )
                            }
                          />
                          <FaEye
                            uk-tooltip="Toggle display at left."
                            onClick={() =>
                              props.displayLigandPose(dock.ligand_name)
                            }
                          />
                        </span>
                      ) : null}
                      {props.displayedLigandNames.includes(dock.ligand_name) ? (
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
                      <FaTrash
                        uk-tooltip="Delete docking result."
                        onClick={() =>
                          props.deleteLigandPose(dock.id, dock.ligand_name)
                        }
                      />
                    </td>
                  </tr>
                );
              })
            : null}
        </tbody>
      </table>
      <h3>Dock new ligands</h3>
      <NewDockPrompt
        setErrorText={props.setErrorText}
        foldIds={[props.foldId]}
      />
    </div>
  );
});

export default DockTab;
