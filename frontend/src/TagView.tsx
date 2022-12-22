import React, { useEffect, useState } from "react";
import UIkit from "uikit";
import { Fold, getFolds, queueJob } from "./services/backend.service";
import { getJobStatus, makeFoldTable, NewDockPrompt } from "./Util";
import { CSVLink } from "react-csv";
import { useParams } from "react-router-dom";

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
        });
      }
      return copy;
    });
  };

  return (
    <div className="uk-margin-small-left uk-margin-small-right">
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
              Download as CSV
            </CSVLink>
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
        />
      ) : null}
    </div>
  );
}

export default TagView;
