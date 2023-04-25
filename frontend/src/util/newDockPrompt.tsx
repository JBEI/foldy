import React, { useState } from "react";
import UIkit from "uikit";
import { DockInput, postDock } from "../services/backend.service";

const getLigandNameErrorMessage = (ligandName: string | null) => {
  if (!ligandName) {
    return null;
  }
  const foldNameIsValid = ligandName.match(/^[0-9a-zA-Z]+$/);
  if (!foldNameIsValid) {
    return "Must be alphanumeric.";
  }
  return null;
};

const validateAndSetInput = (
  e: HTMLInputElement,
  validationFn: (input: string | null) => string | null,
  setFn: (input: string | null) => void,
  input: string | null
) => {
  const invalidMessage = validationFn(input);
  if (invalidMessage) {
    e.setCustomValidity(invalidMessage);
    e.reportValidity();
  } else {
    e.setCustomValidity("");
  }

  setFn(input);
};

const getBBResidueErrorMessage = (bboxResidue: string | null) => {
  if (!bboxResidue) {
    return null;
  }
  const bboxResidueIsValid = bboxResidue.match(/^[A-Z][0-9]+$/);
  if (!bboxResidueIsValid) {
    return "Must be of the form <amino acid><index>, like W81.";
  }
  return null;
};

interface newDockTextboxInterface {
  setErrorText: (error: string) => void;
  foldIds: number[];
}

export function NewDockPrompt(props: newDockTextboxInterface) {
  const [toolName, setToolName] = useState<string>("");
  const [ligandName, setLigandName] = useState<string | null>(null);
  const [ligandSmiles, setLigandSmiles] = useState<string | null>(null);
  const [boundingBoxResidue, setBoundingBoxResidue] = useState<string | null>(
    null
  );
  const [boundingBoxRadiusAngstrom, setBoundingBoxRadiusAngstrom] = useState<
    string | null
  >(null);
  const [showTextbox, setShowTextbox] = useState<boolean>(false);
  const [textboxContents, setTextboxContents] = useState<string | null>(null);

  const submitTextboxDocks = () => {
    if (!textboxContents) {
      props.setErrorText("Must provide a smiles string and ligand name!");
      return;
    }

    const errors: string[] = [];
    const newDocks: DockInput[] = [];

    textboxContents.split("\n").forEach((lineContents, lineNumber) => {
      const lineItems = lineContents.split(",");

      var bounding_box_residue: string | null = null;
      var bounding_box_radius_angstrom: number | null = null;
      if (lineItems.length === 4) {
        bounding_box_residue = lineItems[2];
        bounding_box_radius_angstrom = parseFloat(lineItems[3]);
      } else if (lineItems.length !== 2) {
        errors.push(
          `Lines can have either two or four arguments, got ${lineContents}`
        );
        return;
      }
      const name = lineItems[0].trim();
      const smiles = lineItems[1].trim();

      if (!name.match(/^[0-9a-zA-Z]+$/)) {
        errors.push(`All ligand names must be alphanumeric, "${name}" is not`);
        return;
      }

      if (toolName === "") {
        errors.push("Must select a docking tool.");
        return;
      }

      props.foldIds.forEach((foldId) => {
        newDocks.push({
          fold_id: foldId,
          ligand_name: name,
          ligand_smiles: smiles,
          tool: toolName,
          bounding_box_residue: bounding_box_residue,
          bounding_box_radius_angstrom: bounding_box_radius_angstrom,
        });
      });
    });

    console.log(errors);

    if (errors.length) {
      props.setErrorText(errors.join("\n"));
      return;
    }

    newDocks.forEach((newDock) => {
      postDock(newDock).then(
        () => {
          UIkit.notification(
            `Successfully started docking run for ${newDock.ligand_name}`
          );
        },
        (e) => {
          props.setErrorText(`Docking ${newDock.ligand_name} failed: ${e}`);
        }
      );
    });
  };

  const submitFormDocks = () => {
    if (!ligandName || !ligandSmiles) {
      props.setErrorText("Must provide a ligand name and SMILES string!");
      return;
    }

    if (!ligandName.match(/^[0-9a-zA-Z]+$/)) {
      props.setErrorText(
        `All ligand names must be alphanumeric, "${ligandName}" is not`
      );
      return;
    }

    if (toolName === "") {
      props.setErrorText("Must select a docking tool.");
      return;
    }

    const newDocks: DockInput[] = [];
    props.foldIds.forEach((foldId) => {
      newDocks.push({
        fold_id: foldId,
        ligand_name: ligandName,
        ligand_smiles: ligandSmiles,
        tool: toolName,
        bounding_box_residue: boundingBoxResidue,
        bounding_box_radius_angstrom: boundingBoxRadiusAngstrom
          ? parseFloat(boundingBoxRadiusAngstrom)
          : null,
      });
    });

    newDocks.forEach((newDock) => {
      postDock(newDock).then(
        () => {
          UIkit.notification(
            `Successfully started docking run for ${newDock.ligand_name}`
          );
        },
        (e) => {
          props.setErrorText(`Docking ${newDock.ligand_name} failed: ${e}`);
        }
      );
    });
  };

  const runDocks = () => {
    if (showTextbox) {
      submitTextboxDocks();
    } else {
      submitFormDocks();
    }
  };

  return (
    <div>
      <div className="uk-margin-small">
        <div className="uk-form-controls">
          <select
            className="uk-select"
            id="form-horizontal-select"
            style={{ borderRadius: "20px" }}
            onChange={(e) => setToolName(e.target.value)}
            value={toolName}
          >
            <option value={""}>Select a docking program...</option>
            <option value={"vina"}>Docking with Autodock Vina</option>
            <option value={"diffdock"}>Docking with Diffdock</option>
          </select>
        </div>
      </div>

      {showTextbox ? (
        <textarea
          className="uk-textarea"
          rows={5}
          placeholder={
            "ligand1_name,ligand1_smiles[,bbox_residue,bbox_radius]\nligand2_name,ligand2_smiles[,bbox_residue,bbox_radius]"
          }
          style={{ fontFamily: 'consolas,"Liberation Mono",courier,monospace' }}
          value={textboxContents || ""}
          onChange={(e) => setTextboxContents(e.target.value)}
        ></textarea>
      ) : (
        <form className="uk-grid-small" uk-grid={1}>
          <div className="uk-width-1-1">
            <input
              className={
                "uk-input " +
                (getLigandNameErrorMessage(ligandName)
                  ? "uk-form-danger"
                  : null)
              }
              type="text"
              placeholder="Ligand Name"
              id="name"
              uk-tooltip="Name this ligand, something alphanumeric."
              value={ligandName || ""}
              style={{ borderRadius: "500px" }}
              onChange={(e) =>
                validateAndSetInput(
                  e.target,
                  getLigandNameErrorMessage,
                  setLigandName,
                  e.target.value
                )
              }
            />
          </div>
          <div className="uk-width-1-1">
            <input
              className={"uk-input "}
              type="text"
              placeholder="Ligand SMILES"
              id="name"
              uk-tooltip="Ligand SMILES string."
              value={ligandSmiles || ""}
              style={{ borderRadius: "500px" }}
              onChange={(e) => setLigandSmiles(e.target.value)}
            />
          </div>
          <div className="uk-width-1-2">
            <input
              className={
                "uk-input " +
                (getBBResidueErrorMessage(boundingBoxResidue)
                  ? "uk-form-danger"
                  : null)
              }
              type="text"
              placeholder="[Bounding Box Residue Center]"
              id="name"
              uk-tooltip="Residue ID, like W73, around which to set bounding box.  Ignored by Diffdock."
              value={boundingBoxResidue || ""}
              style={{ borderRadius: "500px" }}
              onChange={(e) =>
                validateAndSetInput(
                  e.target,
                  getBBResidueErrorMessage,
                  setBoundingBoxResidue,
                  e.target.value
                )
              }
              disabled={toolName === "diffdock"}
            />
          </div>
          <div className="uk-width-1-2">
            <input
              className={"uk-input "}
              type="number"
              min="0"
              placeholder="[Bounding Box Radius (Angstroms)]"
              id="name"
              uk-tooltip="Radius of bounding box in Angstroms. Ignored by Diffdock."
              value={boundingBoxRadiusAngstrom || ""}
              style={{ borderRadius: "500px" }}
              onChange={(e) => setBoundingBoxRadiusAngstrom(e.target.value)}
              disabled={toolName === "diffdock"}
            />
          </div>
        </form>
      )}
      <button
        type="button"
        className="uk-button uk-button-default uk-margin-small uk-margin-small-right"
        onClick={() => setShowTextbox(!showTextbox)}
      >
        {showTextbox ? "Hide" : "Show"} Bulk Input
      </button>
      <button
        type="button"
        className="uk-button uk-button-primary"
        onClick={runDocks}
      >
        Dock
      </button>
    </div>
  );
}
