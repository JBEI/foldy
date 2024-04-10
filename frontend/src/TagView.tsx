import fileDownload from "js-file-download";
import React, { useEffect, useState } from "react";
import { CSVLink } from "react-csv";
import { useParams } from "react-router-dom";
import UIkit from "uikit";
import {
  Dock,
  Fold,
  getFoldPdbZip,
  getFolds,
  getJobStatus,
  queueJob,
  updateFold,
} from "./services/backend.service";
import { makeFoldTable } from "./util/foldTable";
import { NewDockPrompt } from "./util/newDockPrompt";

function TagView(props: { setErrorText: (a: string) => void }) {
  let { tagStringParam } = useParams();
  const [tagString] = useState<string>(tagStringParam || "");
  const [folds, setFolds] = useState<Fold[] | null>(null);
  const [stageToStart, setStageToStart] = useState<string | null>(null);

  if (!tagStringParam) {
    throw Error("Somehow wound up with an invalid tagstring.");
  }

  useEffect(() => {
    getFolds(null, tagString, null, null).then(setFolds, (e) => {
      props.setErrorText(e.toString());
    });
  }, [props]);

  const restartWholePipelineForAnyFailedjob = () => {
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
        fold.docks.forEach((dock) => {
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
    const dirname = `${tagString}_pdbs`;
    getFoldPdbZip(fold_ids, dirname).then(
      (fold_pdb_blob) => {
        fileDownload(fold_pdb_blob, `${dirname}.zip`);
      },
      (e) => {
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
    <div
      className="uk-margin-small-left uk-margin-small-right"
      style={{
        flexGrow: 1,
        overflowY: "scroll",
        paddingTop: "10px",
        paddingBottom: "10px",
      }}
    >
      <h2 className="uk-heading-line uk-margin-left uk-margin-right uk-text-center">
        <b>Tag: {tagString}</b>
      </h2>
      {folds ? (
        <div key="loadedDiv">{makeFoldTable(folds)}</div>
      ) : (
        <div className="uk-text-center" key="unloadedDiv">
          {/* We're setting key so that the table doesn't spin... */}
          <div uk-spinner="ratio: 4" key="spinner"></div>
        </div>
      )}
      <form>
        <fieldset className="uk-fieldset">
          <div className="uk-margin">
            <button
              type="button"
              className="uk-button uk-button-primary uk-form-small"
              onClick={() => restartWholePipelineForAnyFailedjob()}
            >
              Restart Whole Pipeline For Any Failed Jobs
            </button>
          </div>
        </fieldset>

        <fieldset className="uk-fieldset">
          <div className="uk-margin">
            <CSVLink
              data={getFoldsDataForCsv()}
              className="uk-button uk-button-primary uk-form-small"
              filename={`${tagString}_metadata`}
            >
              Download Metadata as CSV
            </CSVLink>
          </div>
        </fieldset>

        <fieldset className="uk-fieldset">
          <div className="uk-margin">
            <button
              type="button"
              className="uk-button uk-button-primary uk-form-small"
              onClick={() => downloadFoldPdbZip()}
            >
              Download Fold PDBs in Zip File
            </button>
          </div>
        </fieldset>

        <fieldset className="uk-fieldset">
          <div className="uk-margin">
            <button
              type="button"
              className="uk-button uk-button-primary uk-form-small"
              onClick={() => makeAllFoldsPublic()}
            >
              Make All Structures Public
            </button>
          </div>
        </fieldset>

        <fieldset className="uk-fieldset">
          <div className="uk-margin">
            <select
              className="uk-select uk-form-width-medium uk-form-small"
              id="form-horizontal-select"
              value={stageToStart || ""}
              onChange={(e) => {
                setStageToStart(e.target.value);
              }}
            >
              <option></option>
              <option>both</option>
              <option>annotate</option>
              <option>write_fastas</option>
              <option>features</option>
              <option>models</option>
              <option>decompress_pkls</option>
            </select>
            <button
              type="button"
              className="uk-button uk-button-primary uk-form-small"
              onClick={startStageForAllFolds}
            >
              Start stage for all folds
            </button>
          </div>
        </fieldset>
      </form>

      <h3>Docking</h3>
      {folds ? (
        <NewDockPrompt
          setErrorText={props.setErrorText}
          foldIds={folds.map((fold) => fold.id ?? -1)} // Should never happen, but null fold ids are replaced w/ invalid.
          existingLigands={Array.prototype.reduce(
            (acc, fold) => ({
              ...acc,
              [fold.id]: (fold.docks ?? []).map(
                (dock: Dock) => dock.ligand_name
              ),
            }),
            []
          )}
        />
      ) : null}
    </div>
  );
}

export default TagView;
