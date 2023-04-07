import { authHeader, jsonBodyAuthHeader } from "../helpers/authHeader";
import { getJobStatus } from "../Util";
import { authenticationService } from "./authentication.service";

function handleFileResponse(response: Response) {
  if (!response.ok) {
    return Promise.reject(response.statusText);
  }
  if (response.status !== 200) {
    console.log("Here is the failed response:");
    console.log(response);
    return Promise.reject(`Request failed with error code ${response.status}`);
  }
  return response;
}

function handleResponse(response: Response) {
  if (!response.text) {
    return Promise.reject(response.statusText);
  }

  return response.text().then((text) => {
    const data = text && JSON.parse(text);
    if (!response.ok) {
      if ([401, 403].indexOf(response.status) !== -1) {
        // auto logout if 401 Unauthorized or 403 Forbidden response returned from api
        authenticationService.logout();
        // location.reload(true);
      }

      const error = (data && data.message) || response.statusText;
      console.log(error);
      return Promise.reject(error);
    }

    return data;
  });
}

export function getTestValue(): Promise<any> {
  const requestOptions = { method: "GET", headers: authHeader() };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/test`,
    requestOptions
  ).then(handleResponse);
}

export interface DockInput {
  fold_id: number;
  ligand_name: string;
  ligand_smiles: string;
  bounding_box_residue: string | null;
  bounding_box_radius_angstrom: number | null;
}

export interface Dock extends DockInput {
  id: number;
  invokation_id: number | null;
  pose_energy: number | null;
}

export interface FoldInput {
  name: string;
  tags: string[];
  sequence: string;
  af2_model_preset: string | null;
  disable_relaxation: boolean | null;
}

export interface Invokation {
  id: number;
  type: string | null;
  job_id: string | null;
  state: string | null;
  log: string | null;
  timedelta_sec: number | null;
}

export interface Fold extends FoldInput {
  id: number | null;
  owner: string;
  state: string | null;
  jobs: Invokation[] | null;
  docks: Dock[] | null;
}

export interface FoldPae {
  pae: number[][];
}

export interface FoldContactProb {
  contact_prob: number[][];
}

export interface Annotations {
  [chainName: string]: [
    {
      type: string;
      start: number;
      end: number;
    }
  ];
}

export interface FileInfo {
  key: string;
  size: number;
  modified: number;
}

export const describeFoldState = (fold: Fold) => {
  const featuresState = getJobStatus(fold, "features");
  const modelsState = getJobStatus(fold, "models");
  const decompressState = getJobStatus(fold, "decompress_pkls");

  // Special case: if anything hasn't been queued, just say unstarted.
  if (
    featuresState === null ||
    modelsState === null ||
    decompressState === null
  ) {
    return "unstarted";
  }

  // Another special case: before the beginning just say "queued".
  if (featuresState === "queued") {
    return "queued";
  }

  // Another special case: after the end just say "finished".
  if (decompressState === "finished") {
    return "finished";
  }

  // Normal case: print the state of the most recent stage.
  if (featuresState !== "finished") {
    return `features ${featuresState}`;
  }
  if (modelsState !== "finished") {
    return `models ${modelsState}`;
  }
  return `decompress_pkls ${decompressState}`;
};

export const foldIsFinished = (fold: Fold) => {
  return getJobStatus(fold, "models") === "finished";
};

export interface FoldPdb {
  pdb_string: string;
}

export interface FoldPkl {
  pkl_string: string;
}

export function getFolds(
  filter: string | null,
  tagString: string | null,
  page: number | null, // 1 indexed
  per_page: number | null
): Promise<Array<Fold>> {
  const requestOptions = { method: "GET", headers: authHeader() };

  const searchParams = new URLSearchParams();
  if (filter) {
    searchParams.append("filter", filter);
  }
  if (tagString) {
    searchParams.append("tag", tagString);
  }
  if (page && per_page) {
    searchParams.append("page", page.toString());
    searchParams.append("per_page", per_page.toString());
  }
  var url = `${process.env.REACT_APP_BACKEND_URL}/api/fold?${searchParams}`;
  return fetch(url, requestOptions).then(handleResponse);
}

export function postFolds(
  newFolds: FoldInput[],
  startFoldJob: boolean,
  emailOnCompletion: boolean,
  skipDuplicateEntries: boolean
): Promise<any> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({
      folds_data: newFolds,
      start_fold_job: startFoldJob,
      email_on_completion: emailOnCompletion,
      skip_duplicate_entries: skipDuplicateEntries,
    }),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/fold`,
    requestOptions
  ).then(handleResponse);
}

export function getFold(foldId: number): Promise<Fold> {
  const requestOptions = { method: "GET", headers: authHeader() };

  var url = `${process.env.REACT_APP_BACKEND_URL}/api/fold/${foldId}`;
  return fetch(url, requestOptions).then(handleResponse);
}

export function updateFold(
  foldId: number,
  fieldsToUpdate: Partial<Fold>
): Promise<boolean> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify(fieldsToUpdate),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/fold/${foldId}`,
    requestOptions
  ).then(handleResponse);
}

export function getInvokation(invokation_id: number): Promise<Invokation> {
  const requestOptions = { method: "GET", headers: authHeader() };

  var url = `${process.env.REACT_APP_BACKEND_URL}/api/invokation/${invokation_id}`;
  return fetch(url, requestOptions).then(handleResponse);
}

export function getFoldPdb(
  fold_id: number,
  model_number: number
): Promise<FoldPdb> {
  const requestOptions = { method: "GET", headers: authHeader() };

  var url = `${process.env.REACT_APP_BACKEND_URL}/api/fold_pdb/${fold_id}/${model_number}`;
  return fetch(url, requestOptions).then(handleResponse);
}

export function getFoldPkl(
  fold_id: number,
  model_number: number
): Promise<Blob> {
  const requestOptions = { method: "POST", headers: authHeader() };

  var url = `${process.env.REACT_APP_BACKEND_URL}/api/fold_pkl/${fold_id}/${model_number}`;
  return fetch(url, requestOptions).then((response) => {
    return response.blob();
  });
}

export function getDockSdf(
  fold_id: number,
  ligand_name: string
): Promise<Blob> {
  const requestOptions = { method: "POST", headers: authHeader() };

  var url = `${process.env.REACT_APP_BACKEND_URL}/api/dock_sdf/${fold_id}/${ligand_name}`;
  return fetch(url, requestOptions)
    .then(handleFileResponse)
    .then((response) => {
      return response.blob();
    });
}

export function getFoldPdbZip(
  fold_ids: number[],
  dirname: string
): Promise<Blob> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({
      fold_ids: fold_ids,
      dirname: dirname,
    }),
  };

  var url = `${process.env.REACT_APP_BACKEND_URL}/api/fold_pdb_zip`;
  return fetch(url, requestOptions)
    .then(handleFileResponse)
    .then((response) => {
      return response.blob();
    });
}

export function getFoldPae(
  fold_id: number,
  model_number: number
): Promise<FoldPae> {
  const requestOptions = { method: "GET", headers: authHeader() };

  var url = `${process.env.REACT_APP_BACKEND_URL}/api/pae/${fold_id}/${model_number}`;
  return fetch(url, requestOptions).then(handleResponse);
}

export function getFoldContactProb(
  fold_id: number,
  model_number: number
): Promise<FoldContactProb> {
  const requestOptions = { method: "GET", headers: authHeader() };

  var url = `${process.env.REACT_APP_BACKEND_URL}/api/contact_prob/${fold_id}/${model_number}`;
  return fetch(url, requestOptions).then(handleResponse);
}

export function getFoldPfam(fold_id: number): Promise<Annotations> {
  const requestOptions = { method: "GET", headers: authHeader() };

  var url = `${process.env.REACT_APP_BACKEND_URL}/api/pfam/${fold_id}`;
  return fetch(url, requestOptions).then(handleResponse);
}

export function getFileList(fold_id: number): Promise<FileInfo[]> {
  const requestOptions = { method: "GET", headers: authHeader() };

  var url = `${process.env.REACT_APP_BACKEND_URL}/api/file/list/${fold_id}`;
  return fetch(url, requestOptions).then(handleResponse);
}

export function getFile(fold_id: number, filePath: string): Promise<Blob> {
  const requestOptions = { method: "POST", headers: authHeader() };

  var url = `${process.env.REACT_APP_BACKEND_URL}/api/file/download/${fold_id}/${filePath}`;
  return fetch(url, requestOptions)
    .then(handleFileResponse)
    .then((response) => {
      return response.blob();
    });
}

export function postDock(newDock: DockInput): Promise<boolean> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify(newDock),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/dock`,
    requestOptions
  ).then(handleResponse);
}

export function deleteDock(dockId: number): Promise<boolean> {
  const requestOptions = {
    method: "DELETE",
    headers: jsonBodyAuthHeader(),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/dock/${dockId}`,
    requestOptions
  ).then(handleResponse);
}

// Administration functions....

export function createDbs(): Promise<any> {
  var headers = authHeader();
  headers.set("Content-Type", "application/json");
  const requestOptions = {
    method: "POST",
    headers: headers,
    body: JSON.stringify({}),
  };
  var url = `${process.env.REACT_APP_BACKEND_URL}/api/createdbs`;
  return fetch(url, requestOptions).then(handleResponse);
}

export function upgradeDbs(): Promise<any> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({}),
  };
  var url = `${process.env.REACT_APP_BACKEND_URL}/api/upgradedbs`;
  return fetch(url, requestOptions).then(handleResponse);
}

export function stampDbs(revision: string): Promise<any> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({ revision: revision }),
  };
  var url = `${process.env.REACT_APP_BACKEND_URL}/api/stampdbs`;
  return fetch(url, requestOptions).then(handleResponse);
}

export function queueJob(
  foldId: number,
  stage: string,
  emailOnCompletion: boolean
): Promise<any> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({
      fold_id: foldId,
      stage: stage,
      email_on_completion: emailOnCompletion,
    }),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/queuejob`,
    requestOptions
  ).then(handleResponse);
}

export function queueTestJob(queue: string): Promise<any> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({ queue: queue }),
  };
  var url = `${process.env.REACT_APP_BACKEND_URL}/api/queuetestjob`;
  return fetch(url, requestOptions).then(handleResponse);
}

export function removeFailedJobs(queue: string): Promise<any> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({ queue: queue }),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/remove_failed_jobs`,
    requestOptions
  ).then(handleResponse);
}

export function killWorker(workerToKill: string): Promise<any> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({ worker_id: workerToKill }),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/kill_worker`,
    requestOptions
  ).then(handleResponse);
}

export function sendTestEmail(): Promise<any> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({}),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/sendtestemail`,
    requestOptions
  ).then(handleResponse);
}

export function addInvokationToAllJobs(
  jobType: string,
  jobState: string
): Promise<any> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({}),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/addInvokationToAllJobs/${jobType}/${jobState}`,
    requestOptions
  ).then(handleResponse);
}

export function runUnrunStages(stageToRun: string): Promise<any> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({}),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/runUnrunStages/${stageToRun}`,
    requestOptions
  ).then(handleResponse);
}

export function setAllUnsetModelPresets(): Promise<boolean> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({}),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/set_all_unset_model_presets`,
    requestOptions
  ).then(handleResponse);
}

export function killFoldsInRange(foldRange: string): Promise<any> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({}),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/killFolds/${foldRange}`,
    requestOptions
  ).then(handleResponse);
}

export function bulkAddTag(foldRange: string, newTag: string): Promise<any> {
  const requestOptions = {
    method: "POST",
    headers: jsonBodyAuthHeader(),
    body: JSON.stringify({}),
  };
  return fetch(
    `${process.env.REACT_APP_BACKEND_URL}/api/bulkAddTag/${foldRange}/${newTag}`,
    requestOptions
  ).then(handleResponse);
}
