import axiosInstance from '../services/axiosInstance';
import { Annotations, Fold, FoldContactProb, FoldInput, FoldPae, FoldCif, Invokation, AffinityPrediction } from '../types/types';
import { authenticationService } from "../services/authentication.service";
import { BoltzYamlHelper } from '../util/boltzYamlHelper';
import { getFile } from './fileApi';

// Add this helper function
function enhanceFoldWithYamlHelper(fold: Fold): Fold {
    // Try to parse the YAML config into a BoltzYamlHelper.
    try {
        if (fold.yaml_config && !fold.yaml_helper) {
            fold.yaml_helper = new BoltzYamlHelper(fold.yaml_config);
        }
    } catch (e) {
        console.error(e);
    }
    return fold;
}

interface PaginatedFoldsResponse {
    data: Fold[];
    pagination: {
        page: number | null;
        per_page: number | null;
        total: number | null;
        pages: number | null;
        has_prev: boolean | null;
        has_next: boolean | null;
    };
}

export const getFoldsWithPagination = async (
    filter: string | null,
    tagString: string | null,
    page: number | null,
    per_page: number | null
): Promise<PaginatedFoldsResponse> => {
    const params: Record<string, string | number> = {};
    if (filter) params.filter = filter;
    if (tagString) params.tag = tagString;
    if (page !== null) params.page = page;
    if (per_page !== null) params.per_page = per_page;

    const response = await axiosInstance.get<PaginatedFoldsResponse>('/api/paginated_fold', { params });
    return {
        data: response.data.data.map(enhanceFoldWithYamlHelper),
        pagination: response.data.pagination
    };
};

export const getFold = async (foldId: number): Promise<Fold> => {
    const response = await axiosInstance.get<Fold>(`/api/fold/${foldId}`).then((res) => enhanceFoldWithYamlHelper(res.data));
    return response;
};

export const postFolds = async (
    folds: FoldInput[],
    options: { startJob: boolean; emailOnCompletion: boolean; skipDuplicates: boolean, isDryRun: boolean }
): Promise<any> => {
    const body = {
        folds_data: folds,
        start_fold_job: options.startJob,
        email_on_completion: options.emailOnCompletion,
        skip_duplicate_entries: options.skipDuplicates,
        is_dry_run: options.isDryRun,
    };
    const response = await axiosInstance.post('/api/fold', body);
    return response.data;
};

export const updateFold = async (
    foldId: number,
    fieldsToUpdate: Partial<Fold>
): Promise<boolean> => {
    const response = await axiosInstance.post(`/api/fold/${foldId}`, fieldsToUpdate);
    return response.data;
};

/**
 * Gets an invokation by ID
 */
export const getInvokation = async (invokationId: number): Promise<Invokation> => {
    const response = await axiosInstance.get(`/api/invokation/${invokationId}`);
    return response.data;
};

/**
 * Downloads multiple fold files as a zip
 */
export const getFoldFileZip = async (
    foldIds: number[],
    relativeFpath: string,
    outputDirname: string,
    flattenFilepath?: boolean,
    useFoldName?: boolean
): Promise<Blob> => {
    const response = await axiosInstance.post(
        '/api/fold_file_zip',
        {
            fold_ids: foldIds,
            relative_fpath: relativeFpath,
            output_dirname: outputDirname,
            flatten_filepath: flattenFilepath || false,
            use_fold_name: useFoldName || false,
        },
        { responseType: 'blob' }
    );
    return response.data;
};

/**
 * Gets PAE data for a fold model
 */
export const getFoldPae = async (
    foldId: number,
    modelNumber: number
): Promise<FoldPae> => {
    const response = await axiosInstance.get(`/api/pae/${foldId}/${modelNumber}`);
    return response.data;
};

/**
 * Gets contact probability data for a fold model
 */
export const getFoldContactProb = async (
    foldId: number,
    modelNumber: number
): Promise<FoldContactProb> => {
    const response = await axiosInstance.get(`/api/contact_prob/${foldId}/${modelNumber}`);
    return response.data;
};

export const getFoldAffinityPrediction = async (foldId: number): Promise<AffinityPrediction> => {
    const predictedAffinityPath = `boltz/boltz_results_input/predictions/input/affinity_input.json`;
    const fileBlob = await getFile(foldId, predictedAffinityPath);
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const fileString = e.target?.result as string;
            resolve(JSON.parse(fileString));
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsText(fileBlob);
    });
};

/**
 * Gets PFAM annotations for a fold
 */
export const getFoldPfam = async (foldId: number): Promise<Annotations> => {
    const response = await axiosInstance.get(`/api/pfam/${foldId}`);
    return response.data;
};

/**
 * Get the job status for a specific type of job in a fold
 */
export const getJobStatus = (fold: Fold, jobType: string): string | null => {
    if (!fold.jobs) {
        return null;
    }
    for (const job of fold.jobs) {
        if (job.type === jobType) {
            return job.state;
        }
    }
    return null;
};

/**
 * Describes the overall fold state based on its jobs
 */
export const describeFoldState = (fold: Fold): string => {
    const boltzState = getJobStatus(fold, "boltz");

    return boltzState ?? "unstarted";
};

/**
 * Checks if a fold is finished
 */
export const foldIsFinished = (fold: Fold): boolean => {
    return getJobStatus(fold, "models") === "finished";
};

// Types for tags API
export interface TagInfo {
    tag: string;
    fold_count: number;
    contributors: string[];
    recent_folds: string[];
}

interface TagsResponse {
    tags: TagInfo[];
}

/**
 * Get all tags with their fold counts and contributors
 */
export const getAllTags = async (): Promise<TagInfo[]> => {
    const response = await axiosInstance.get<TagsResponse>('/api/tags');
    return response.data.tags;
};
