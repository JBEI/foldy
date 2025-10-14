import { authHeader } from '../util/authHeader';
import axiosInstance from '../services/axiosInstance';
import { FewShot } from '../types/types';
import { getFile } from './fileApi';

export const getFewShot = async (fewShotId: number): Promise<FewShot> => {
    const response = await axiosInstance.get<FewShot>(`/api/few_shots/${fewShotId}`);
    return response.data;
};

export async function runFewShot(
    fewShotName: string,
    foldId: number,
    activityFile: File | null,
    activityFileFromFewShotId: number | null,
    activityFileFromCampaignRoundId: number | null,
    mode: string,
    numMutants: number,
    embeddingFiles?: string[],
    naturalnessFiles?: string[],
    finetuningModelCheckpoint?: string,
    fewShotParams?: string,
): Promise<FewShot> {
    // Build request body
    const requestBody: any = {
        name: fewShotName,
        fold_id: foldId,
        mode: mode,
        num_mutants: numMutants,
        embedding_files: embeddingFiles?.join(',') ?? undefined,
        naturalness_files: naturalnessFiles?.join(',') ?? undefined,
        finetuning_model_checkpoint: finetuningModelCheckpoint ?? undefined,
        few_shot_params: fewShotParams ?? undefined,
    };

    // Handle activity file based on source
    if (activityFile) {
        // Convert file to base64
        const reader = new FileReader();
        const base64Promise = new Promise<string>((resolve, reject) => {
            reader.onload = () => {
                const base64 = (reader.result as string).split(',')[1]; // Remove data:...;base64, prefix
                resolve(base64);
            };
            reader.onerror = reject;
        });
        reader.readAsDataURL(activityFile);
        requestBody.activity_file_bytes = await base64Promise;
    } else if (activityFileFromFewShotId) {
        requestBody.activity_file_from_few_shot_id = activityFileFromFewShotId;
    } else if (activityFileFromCampaignRoundId) {
        requestBody.activity_file_from_round_id = activityFileFromCampaignRoundId;
    } else {
        throw new Error('No activity file source provided');
    }

    // Single POST request to create few shot
    const response = await axiosInstance.post<FewShot>('/api/few_shot', requestBody);
    return response.data;
}


export const deleteFewShot = async (fewShotId: number): Promise<void> => {
    await axiosInstance.delete(`/api/few_shots/${fewShotId}`);
};

/**
 * Interface for slate data item (individual mutant prediction)
 */
export interface SlateData {
    seqId: string;
    selected: boolean;
    order: number | null;
    relevantMeasuredMutants: string;
    predictionMean: number;
    predictionStddev: number;
    score: number;
    modelPredictions: number[];
}

/**
 * Interface for FewShot predicted slate response
 */
export interface FewShotPredictedSlateResponse {
    data: SlateData[];
    total_count: number;
    few_shot_id: number;
}

/**
 * Interface for FewShot debug information (training metrics only)
 */
export interface FewShotDebugInfo {
    debugData: any | null;
    sortOptions: { [key: string]: string[] } | null;
}

/**
 * Get predicted slate data for a FewShot run from the backend API
 *
 * @param fewShotId - ID of the FewShot run
 * @param options - Query options
 * @returns Promise containing slate data and metadata
 */
export const getFewShotPredictedSlate = async (
    fewShotId: number,
    options: {
        selectedOnly?: boolean;
        limit?: number;
    } = {}
): Promise<FewShotPredictedSlateResponse> => {
    const params = new URLSearchParams();

    if (options.selectedOnly !== undefined) {
        params.append('selected_only', options.selectedOnly.toString());
    }

    if (options.limit !== undefined) {
        params.append('limit', options.limit.toString());
    }

    const response = await axiosInstance.get<FewShotPredictedSlateResponse>(
        `/api/few_shot_predicted_slate/${fewShotId}?${params.toString()}`
    );

    return response.data;
};

/**
 * Load FewShot debug information (debug_info.json only) from a FewShot run
 * Uses the correct dynamic path resolution from output_fpath
 *
 * @param foldId - The fold ID
 * @param fewShotRun - The FewShot run object
 * @returns Promise containing debug data and sort options
 */
export const getFewShotDebugInfo = async (
    foldId: number,
    fewShotRun: FewShot
): Promise<FewShotDebugInfo> => {
    const result: FewShotDebugInfo = {
        debugData: null,
        sortOptions: null
    };

    try {
        // Load debug data using dynamic path resolution (the correct approach)
        if (fewShotRun.output_fpath) {
            const debugPath = fewShotRun.output_fpath.split('/').slice(0, -1).join('/') + '/debug_info.json';
            try {
                const debugBlob = await getFile(foldId, debugPath);
                const debugText = await debugBlob.text();
                // Replace NaN and Infinity with null for proper JSON parsing
                const cleanedString = debugText.replace(/NaN/g, 'null').replace(/Infinity/g, 'null');
                const jsonData = JSON.parse(cleanedString);
                result.debugData = jsonData;
                result.sortOptions = jsonData.sorts || null;
            } catch (debugError) {
                console.warn('Debug data not available:', debugError);
            }
        }
    } catch (error) {
        console.error('Error loading FewShot debug info:', error);
        throw error;
    }

    return result;
};
