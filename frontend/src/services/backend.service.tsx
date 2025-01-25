// backend.service.tsx

import axios from 'axios';
import { authHeader, jsonBodyAuthHeader } from "../util/authHeader";
import { Annotations, DockInput, Fold, FoldContactProb, FoldPae, FoldPdb, Invokation } from 'src/types/types';

// ----- Axios instance setup -----
const api = axios.create({
    baseURL: import.meta.env.VITE_BACKEND_URL,
});

// Optional: Add request/response interceptors if needed.
api.interceptors.request.use(
    (config) => {
        // Insert any custom logic here. For example:
        // config.headers = { ...config.headers, ...authHeader() };
        return config;
    },
    (error) => Promise.reject(error)
);

api.interceptors.response.use(
    (response) => response,
    (error) => {
        // We can centralize error handling here.
        // For deeper customization, see `handleAxiosError` below or
        // simply throw the error to handle in each function.
        return Promise.reject(error);
    }
);

/**
 * Helper to transform axios errors to a consistent Error message.
 * You could also do this in interceptors if you prefer.
 */
function handleAxiosError(error: any): never {
    console.log(`HANDLING ERROR: ${error}`);
    if (error.response) {
        console.log(error);
        // The request was made and the server responded with a status code
        throw new Error(error.response.data?.message || error.response.data || error.message);
    } else if (error.request) {
        // The request was made but no response was received
        throw new Error("Network error: no response received");
    } else {
        // Something happened in setting up the request
        throw new Error(error.message);
    }
}

// ----- Utility helpers -----
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

export const describeFoldState = (fold: Fold) => {
    const boltzState = getJobStatus(fold, "boltz");
    const featuresState = getJobStatus(fold, "features");
    const modelsState = getJobStatus(fold, "models");
    const decompressState = getJobStatus(fold, "decompress_pkls");

    if (boltzState) {
        return boltzState;
    }

    if (
        featuresState === null ||
        modelsState === null ||
        decompressState === null
    ) {
        return "unstarted";
    }
    if (featuresState === "queued") {
        return "queued";
    }
    if (decompressState === "finished") {
        return "finished";
    }
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

// ----- API functions -----

export function getInvokation(invokation_id: number): Promise<Invokation> {
    return api
        .get(`/api/invokation/${invokation_id}`, { headers: authHeader() })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function getFoldPdb(
    fold_id: number,
    model_number: number
): Promise<FoldPdb> {
    return api
        .get(`/api/fold_pdb/${fold_id}/${model_number}`, { headers: authHeader() })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function getFoldPkl(
    fold_id: number,
    model_number: number
): Promise<Blob> {
    return api
        .post(`/api/fold_pkl/${fold_id}/${model_number}`, null, {
            headers: authHeader(),
            responseType: 'blob',
        })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function getDockSdf(
    fold_id: number,
    ligand_name: string
): Promise<Blob> {
    return api
        .post(`/api/dock_sdf/${fold_id}/${ligand_name}`, null, {
            headers: authHeader(),
            responseType: 'blob',
        })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function getFoldPdbZip(
    fold_ids: number[],
    dirname: string
): Promise<Blob> {
    return api
        .post('/api/fold_pdb_zip', { fold_ids, dirname }, {
            headers: jsonBodyAuthHeader(),
            responseType: 'blob',
        })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function getFoldFileZip(
    fold_ids: number[],
    relative_fpath: string,
    output_dirname: string
): Promise<Blob> {
    return api
        .post('/api/fold_file_zip', {
            fold_ids,
            relative_fpath,
            output_dirname,
        }, {
            headers: jsonBodyAuthHeader(),
            responseType: 'blob',
        })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function getFoldPae(
    fold_id: number,
    model_number: number
): Promise<FoldPae> {
    return api
        .get(`/api/pae/${fold_id}/${model_number}`, { headers: authHeader() })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function getFoldContactProb(
    fold_id: number,
    model_number: number
): Promise<FoldContactProb> {
    return api
        .get(`/api/contact_prob/${fold_id}/${model_number}`, { headers: authHeader() })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function getFoldPfam(fold_id: number): Promise<Annotations> {
    return api
        .get(`/api/pfam/${fold_id}`, { headers: authHeader() })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function startEmbeddings(
    fold_id: number,
    batch_name: string,
    dms_starting_seq_ids: string[],
    extra_seq_ids: string[],
    embedding_model: string
): Promise<boolean> {
    return api
        .post(
            `/api/embeddings/${fold_id}`,
            {
                batch_name,
                embedding_model,
                dms_starting_seq_ids,
                extra_seq_ids,
            },
            { headers: jsonBodyAuthHeader() }
        )
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function postDock(newDock: DockInput): Promise<boolean> {
    return api
        .post('/api/dock', newDock, { headers: jsonBodyAuthHeader() })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function deleteDock(dockId: number): Promise<boolean> {
    return api
        .delete(`/api/dock/${dockId}`, { headers: jsonBodyAuthHeader() })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

// ----- Administration functions -----

export function createDbs(): Promise<any> {
    return api
        .post('/api/createdbs', {}, { headers: jsonBodyAuthHeader() })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function upgradeDbs(): Promise<any> {
    return api
        .post('/api/upgradedbs', {}, { headers: jsonBodyAuthHeader() })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function stampDbs(revision: string): Promise<any> {
    return api
        .post('/api/stampdbs', { revision }, { headers: jsonBodyAuthHeader() })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function queueJob(
    foldId: number,
    stage: string,
    emailOnCompletion: boolean
): Promise<any> {
    return api
        .post(
            '/api/queuejob',
            {
                fold_id: foldId,
                stage,
                email_on_completion: emailOnCompletion,
            },
            { headers: jsonBodyAuthHeader() }
        )
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function queueTestJob(queue: string): Promise<any> {
    return api
        .post(
            '/api/queuetestjob',
            { queue },
            { headers: jsonBodyAuthHeader() }
        )
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function removeFailedJobs(queue: string): Promise<any> {
    return api
        .post(
            '/api/remove_failed_jobs',
            { queue },
            { headers: jsonBodyAuthHeader() }
        )
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function killWorker(workerToKill: string): Promise<any> {
    return api
        .post(
            '/api/kill_worker',
            { worker_id: workerToKill },
            { headers: jsonBodyAuthHeader() }
        )
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function sendTestEmail(): Promise<any> {
    return api
        .post('/api/sendtestemail', {}, { headers: jsonBodyAuthHeader() })
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function addInvokationToAllJobs(
    jobType: string,
    jobState: string
): Promise<any> {
    return api
        .post(
            `/api/addInvokationToAllJobs/${jobType}/${jobState}`,
            {},
            { headers: jsonBodyAuthHeader() }
        )
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function runUnrunStages(stageToRun: string): Promise<any> {
    return api
        .post(
            `/api/runUnrunStages/${stageToRun}`,
            {},
            { headers: jsonBodyAuthHeader() }
        )
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function setAllUnsetModelPresets(): Promise<boolean> {
    return api
        .post(
            '/api/set_all_unset_model_presets',
            {},
            { headers: jsonBodyAuthHeader() }
        )
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function killFoldsInRange(foldRange: string): Promise<any> {
    return api
        .post(
            `/api/killFolds/${foldRange}`,
            {},
            { headers: jsonBodyAuthHeader() }
        )
        .then((res) => res.data)
        .catch(handleAxiosError);
}

export function bulkAddTag(foldRange: string, newTag: string): Promise<any> {
    return api
        .post(
            `/api/bulkAddTag/${foldRange}/${newTag}`,
            {},
            { headers: jsonBodyAuthHeader() }
        )
        .then((res) => res.data)
        .catch(handleAxiosError);
}