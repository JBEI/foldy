import React, { useState } from "react";
import { AiOutlineCloseCircle, AiOutlinePlus } from "react-icons/ai";
import { Link } from "react-router-dom";
import UIkit from "uikit";
import { DecodedJwt } from "./services/authentication.service";
import {
  describeFoldState,
  DockInput,
  Fold,
  postDock,
} from "./services/backend.service";

const KeyCodes = {
  comma: 188,
  enter: 13,
};
export const TAG_DELIMITERS = [KeyCodes.comma, KeyCodes.enter];

export const TAG_CLASS_NAMES = {
  tag: "uk-badge custom-tags",
  tagInputField: "uk-input uk-form uk-width-1-2",
};

function getTagBadge(tag: string) {
  const badgeStyle = { background: "#999999" };
  return (
    <Link to={`/tag/${tag}`} key={tag}>
      <span className="uk-badge" style={badgeStyle}>
        {tag}
      </span>
    </Link>
  );
}

export function makeFoldTable(folds: Fold[]) {
  return (
    <div className="uk-overflow-auto">
      <table
        className="uk-table uk-table-hover"
        style={{ tableLayout: "fixed" }}
      >
        <thead>
          <tr>
            <th className="uk-table-small">Name</th>
            <th className="uk-table-large">Length</th>
            <th className="uk-table-small">State</th>
            <th className="uk-table-small">Owner</th>
            <th className="uk-table-small">Tags</th>
          </tr>
        </thead>
        <tbody>
          {[...folds].map((fold) => {
            return (
              <tr key={fold.name}>
                <td
                  style={{ overflowX: "hidden" }}
                  uk-tooltip={`title: ${fold.name}`}
                >
                  <Link to={"/fold/" + fold.id}>
                    <div style={{height: '100%', width: '100%'}}>
                      {fold.name}
                    </div>
                  </Link>
                </td>
                <td>{fold.sequence.length}</td>
                <td>
                  {/* {foldIsFinished(fold) ? null : <div uk-spinner="ratio: 0.5"></div>}&nbsp;  */}
                  {describeFoldState(fold)}
                </td>
                <td>{fold.owner}</td>
                <td>{[...fold.tags].map(getTagBadge)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export const getJobStatus = (fold: Fold, job_type: string): string | null => {
  if (!fold.jobs) {
    return null;
  }
  for (const job of fold.jobs) {
    if (job.type === job_type) {
      return job.state;
    }
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

      props.foldIds.forEach((foldId) => {
        newDocks.push({
          fold_id: foldId,
          ligand_name: name,
          ligand_smiles: smiles,
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

    const newDocks: DockInput[] = [];
    props.foldIds.forEach((foldId) => {
      newDocks.push({
        fold_id: foldId,
        ligand_name: ligandName,
        ligand_smiles: ligandSmiles,
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
              uk-tooltip="Residue ID, like W73, around which to set bounding box."
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
            />
          </div>
          <div className="uk-width-1-2">
            <input
              className={"uk-input "}
              type="number"
              min="0"
              placeholder="[Bounding Box Radius (Angstroms)]"
              id="name"
              uk-tooltip="Radius of bounding box in Angstroms."
              value={boundingBoxRadiusAngstrom || ""}
              style={{ borderRadius: "500px" }}
              onChange={(e) => setBoundingBoxRadiusAngstrom(e.target.value)}
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

interface FoldyProps {
  decodedToken: DecodedJwt | null;
  isExpired: boolean;
  moveAbove: boolean;
}

export function Foldy(props: FoldyProps) {
  if (props.decodedToken && !props.isExpired) {
    return null;
  } else {
    return (
      <div>
        <div
          style={{
            position: "fixed",
            bottom: props.moveAbove ? "262px" : "210px",
            right: props.moveAbove ? "34px" : "180px",
          }}
          className={
            props.moveAbove ? "sbbox sbtriangleabove" : "sbbox sbtriangle"
          }
        >
          Welcome to {process.env.REACT_APP_INSTITUTION} Foldy!
        </div>
        <img
          style={{
            width: "250px",
            position: "fixed",
            bottom: "10px",
            right: "10px",
            zIndex: -10,
          }}
          src={`${process.env.PUBLIC_URL}/pksito.gif`}
          alt=""
        />
      </div>
    );
  }
}

export interface EditableTagListProps {
  tags: string[];
  addTag: (tag: string) => void;
  deleteTag: (tag: string) => void;
  handleTagClick: (tag: string) => void;
}

export function EditableTagList(props: EditableTagListProps) {
  const addNewTag = () => {
    UIkit.modal.prompt("New tag:", "").then(
      (newTag: string | null) => {
        if (newTag) {
          props.addTag(newTag);
        }
      },
      () => {
        console.log("No new tag.");
      }
    );
  };

  return (
    <div className="uk-input" onInput={(e) => console.log(e)}>
      {props.tags.map((tag: string) => {
        return (
          <span
            key={tag}
            className="uk-badge uk-badge-bigger uk-margin-small-right"
          >
            <span
              style={{ padding: "0 3px 0 8px" }}
              onClick={() => props.handleTagClick(tag)}
            >
              {tag}
            </span>
            <AiOutlineCloseCircle
              style={{ cursor: "pointer" }}
              onClick={() => props.deleteTag(tag)}
            />
          </span>
        );
      })}
      <span
        className="uk-badge uk-badge-bigger uk-margin-small-right"
        style={{ background: "#999999", cursor: "pointer" }}
        onClick={() => addNewTag()}
      >
        <AiOutlinePlus />
      </span>
    </div>
  );
}
