import React, { useState } from "react";
import UIkit from "uikit";
import {
  addInvokationToAllJobs,
  bulkAddTag,
  createDbs,
  killFoldsInRange,
  killWorker,
  queueTestJob,
  removeFailedJobs,
  runUnrunStages,
  sendTestEmail,
  setAllUnsetModelPresets,
  stampDbs,
  upgradeDbs,
} from "../services/backend.service";

function SudoPage(props: { setErrorText: (a: string) => void }) {
  const [revision, setRevision] = useState<string>("");
  const [newJobQueue, setNewJobQueue] = useState<string>("");
  const [queueToClear, setQueueToClear] = useState<string>("");
  const [workerToKill, setWorkerToKill] = useState<string>("");
  const [newInvokationType, setNewInvokationType] = useState<string | null>(
    null
  );
  const [newInvokationState, setNewInvokationState] = useState<string | null>(
    null
  );
  const [stageToRun, setStageToRun] = useState<string | null>(null);
  const [foldsToKill, setFoldsToKill] = useState<string | null>(null);
  const [foldsToBulkAddTag, setFoldsToBulkAddTag] = useState<string | null>(
    null
  );
  const [tagToBulkAdd, setTagToBulkAdd] = useState<string | null>(null);

  const localCreateDbs = () => {
    createDbs().then(
      () => {
        UIkit.notification("Create successul.");
      },
      (e) => {
        props.setErrorText(e.toString());
      }
    );
  };

  const localUpgradeDbs = () => {
    upgradeDbs().then(
      () => {
        UIkit.notification("Upgrade successul.");
      },
      (e) => {
        props.setErrorText(e.toString());
      }
    );
  };

  const localStampDbs = () => {
    stampDbs(revision).then(
      () => {
        UIkit.notification("Upgrade successul.");
      },
      (e) => {
        props.setErrorText(e.toString());
      }
    );
  };

  const localQueueJob = () => {
    queueTestJob(newJobQueue).then(
      () => {
        UIkit.notification("Successfully queued.");
      },
      (e) => {
        props.setErrorText(e.toString());
      }
    );
  };

  const localRemoveFailedJobs = () => {
    removeFailedJobs(queueToClear).then(
      () => {
        UIkit.notification(
          `Successfully removed failed jobs from ${queueToClear}.`
        );
      },
      (e) => {
        props.setErrorText(e.toString());
      }
    );
  };

  const localKillWorker = () => {
    killWorker(workerToKill).then(
      () => {
        UIkit.notification(`Successfully killed worker ${workerToKill}.`);
      },
      (e) => {
        props.setErrorText(e.toString());
      }
    );
  };

  const localSendEmail = () => {
    sendTestEmail().then(
      () => {
        UIkit.notification("Sent email.");
      },
      (e) => {
        props.setErrorText(e.toString());
      }
    );
  };

  const localAddInvokationToAllJobs = () => {
    if (!newInvokationType || !newInvokationState) {
      return;
    }
    addInvokationToAllJobs(newInvokationType, newInvokationState).then(
      () => {
        UIkit.notification("Successfully added job type.");
      },
      (e) => {
        props.setErrorText(e.toString());
      }
    );
  };

  const localRunUnrunStages = () => {
    if (!stageToRun) {
      return;
    }
    runUnrunStages(stageToRun).then(
      () => {
        UIkit.notification("Successfully started all stages.");
      },
      (e) => {
        props.setErrorText(e.toString());
      }
    );
  };

  const localSetAllUnsetModelPresets = () => {
    setAllUnsetModelPresets().then(
      () => {
        UIkit.notification("Successfully set all model presets.");
      },
      (e) => {
        props.setErrorText(e.toString());
      }
    );
  };

  const localKillFoldsInRange = () => {
    if (!foldsToKill) {
      return;
    }
    killFoldsInRange(foldsToKill).then(
      () => {
        UIkit.notification("Successfully killed folds.");
      },
      (e) => {
        props.setErrorText(e.toString());
      }
    );
  };

  const localBulkAddTag = () => {
    if (!foldsToBulkAddTag || !tagToBulkAdd) {
      return;
    }
    bulkAddTag(foldsToBulkAddTag, tagToBulkAdd).then(
      () => {
        UIkit.notification("Successfully added tags.");
      },
      (e) => {
        props.setErrorText(e.toString());
      }
    );
  };

  return (
    <form
      data-testid="Workers"
      className="uk-margin-left uk-margin-right uk-form-horizontal"
    >
      <h3>Create the DBs</h3>
      <button
        type="button"
        className="uk-button uk-button-default"
        onClick={localCreateDbs}
      >
        Create DBs
      </button>
      <br />

      <h3>Upgrade the DBs</h3>
      <button
        type="button"
        className="uk-button uk-button-default"
        onClick={localUpgradeDbs}
      >
        Upgrade DBs
      </button>
      <br />

      <h3>Stamp DBs</h3>
      <label className="uk-form-label" htmlFor="name">
        Revision
      </label>
      <div className="uk-form-controls">
        <input
          className="uk-input"
          type="text"
          placeholder="Name"
          id="name"
          value={revision}
          onChange={(e) => setRevision(e.target.value)}
        />
      </div>
      <button
        type="button"
        className="uk-button uk-button-default uk-margin-small"
        onClick={localStampDbs}
      >
        Stamp Revision on DBs
      </button>

      <br />
      <h3>Queue a Job</h3>
      <div className="uk-margin-small">
        <label className="uk-form-label" htmlFor="name">
          Queue Name
        </label>
        <div className="uk-form-controls">
          <input
            className="uk-input"
            type="text"
            placeholder="Name"
            id="name"
            value={newJobQueue}
            onChange={(e) => setNewJobQueue(e.target.value)}
          />
        </div>
      </div>

      <button
        type="button"
        className="uk-button uk-button-default"
        onClick={localQueueJob}
      >
        Queue Job
      </button>

      <br />
      <h3>Remove Failed Jobs</h3>
      <div className="uk-margin-small">
        <label className="uk-form-label" htmlFor="name">
          Queue Name
        </label>
        <div className="uk-form-controls">
          <input
            className="uk-input"
            type="text"
            placeholder="Name"
            id="name"
            value={queueToClear}
            onChange={(e) => setQueueToClear(e.target.value)}
          />
        </div>
      </div>
      <button
        type="button"
        className="uk-button uk-button-default"
        onClick={localRemoveFailedJobs}
      >
        Remove Failed Jobs
      </button>

      <br />
      <h3>Kill Worker</h3>
      <div className="uk-margin-small">
        <label className="uk-form-label" htmlFor="name">
          Worker Name
        </label>
        <div className="uk-form-controls">
          <input
            className="uk-input"
            type="text"
            placeholder="Name"
            id="name"
            value={workerToKill}
            onChange={(e) => setWorkerToKill(e.target.value)}
          />
        </div>
      </div>
      <button
        type="button"
        className="uk-button uk-button-default"
        onClick={localKillWorker}
      >
        Kill Worker
      </button>

      <br />
      <h3>Send test email</h3>
      <button
        type="button"
        className="uk-button uk-button-default"
        onClick={localSendEmail}
      >
        Send Email
      </button>

      <br />
      <h3>Add Invokation To All Folds</h3>
      <div className="uk-margin-small">
        <label className="uk-form-label" htmlFor="name">
          Job Type
        </label>
        <div className="uk-form-controls">
          <input
            className="uk-input"
            type="text"
            placeholder="Name"
            id="name"
            value={newInvokationType || ""}
            onChange={(e) => setNewInvokationType(e.target.value)}
          />
        </div>
        <label className="uk-form-label" htmlFor="name">
          Job State
        </label>
        <div className="uk-form-controls">
          <input
            className="uk-input"
            type="text"
            placeholder="Name"
            id="name"
            value={newInvokationState || ""}
            onChange={(e) => setNewInvokationState(e.target.value)}
          />
        </div>
      </div>
      <button
        type="button"
        className="uk-button uk-button-default"
        onClick={localAddInvokationToAllJobs}
      >
        add invokation to all jobs
      </button>

      <br />
      <h3>Run Unrun (or failed) Stages</h3>
      <div className="uk-margin-small">
        <label className="uk-form-label" htmlFor="name">
          Stage Name
        </label>
        <div className="uk-form-controls">
          <input
            className="uk-input"
            type="text"
            placeholder="Name"
            id="name"
            value={stageToRun || ""}
            onChange={(e) => setStageToRun(e.target.value)}
          />
        </div>
      </div>
      <button
        type="button"
        className="uk-button uk-button-default"
        onClick={localRunUnrunStages}
      >
        run stage for all folds without that stage
      </button>

      <br />
      <h3>Set All Unset Model Presets</h3>
      <button
        type="button"
        className="uk-button uk-button-default"
        onClick={localSetAllUnsetModelPresets}
      >
        Set All Unset Model Presets
      </button>

      <br />
      <h3>Kill Folds in Range</h3>
      <div className="uk-margin-small">
        <label className="uk-form-label" htmlFor="name">
          Range of Folds
        </label>
        <div className="uk-form-controls">
          <input
            className="uk-input"
            type="text"
            placeholder="Range of folds to kill (eg, '10-60')"
            id="name"
            value={foldsToKill || ""}
            onChange={(e) => setFoldsToKill(e.target.value)}
          />
        </div>
      </div>
      <button
        type="button"
        className="uk-button uk-button-default"
        onClick={localKillFoldsInRange}
      >
        kill folds in range
      </button>

      <br />
      <h3>Bulk Add Tag To Folds in Range</h3>
      <div className="uk-margin-small">
        <label className="uk-form-label" htmlFor="name">
          Range of Folds
        </label>
        <div className="uk-form-controls">
          <input
            className="uk-input"
            type="text"
            placeholder="Range of folds (eg, '10-60')"
            id="name"
            value={foldsToBulkAddTag || ""}
            onChange={(e) => setFoldsToBulkAddTag(e.target.value)}
          />
        </div>
      </div>
      <div className="uk-margin-small">
        <label className="uk-form-label" htmlFor="name">
          Tag to Add
        </label>
        <div className="uk-form-controls">
          <input
            className="uk-input"
            type="text"
            placeholder="Tag to add (must be alphanumeric)"
            id="name"
            value={tagToBulkAdd || ""}
            onChange={(e) => setTagToBulkAdd(e.target.value)}
          />
        </div>
      </div>
      <button
        type="button"
        className="uk-button uk-button-default"
        onClick={localBulkAddTag}
      >
        bulk add tag
      </button>
    </form>
  );
}

export default SudoPage;
