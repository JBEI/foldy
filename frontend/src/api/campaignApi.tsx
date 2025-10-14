import axiosInstance from '../services/axiosInstance';
import { Campaign, CampaignRound } from '../types/types';

export interface PaginatedCampaignsResponse {
    campaigns: Campaign[];
    total: number;
    page: number;
    per_page: number;
    pages: number;
}

export interface CreateCampaignRequest {
    name: string;
    fold_id: number;
    description?: string;
    naturalness_model?: string;
    embedding_model?: string;
    domain_boundaries?: string;
}

export interface UpdateCampaignRequest {
    name?: string;
    description?: string;
    naturalness_model?: string;
    embedding_model?: string;
    domain_boundaries?: string;
}

export interface CreateCampaignRoundRequest {
    round_number?: number;
    date_started?: string;
}

export interface UpdateCampaignRoundRequest {
    mode?: string;
    naturalness_run_id?: number;
    slate_seq_ids?: string;
    result_activity_fpath?: string;
    input_templates?: string;
    few_shot_run_id?: number | null;
}

export const getCampaigns = async (
    page: number = 1,
    perPage: number = 20,
    foldId?: number
): Promise<PaginatedCampaignsResponse> => {
    const params = new URLSearchParams({
        page: page.toString(),
        per_page: perPage.toString(),
    });

    if (foldId) {
        params.append('fold_id', foldId.toString());
    }

    const response = await axiosInstance.get<PaginatedCampaignsResponse>(
        `/api/campaigns?${params.toString()}`
    );
    return response.data;
};

export const getCampaign = async (campaignId: number): Promise<Campaign> => {
    const response = await axiosInstance.get<Campaign>(`/api/campaigns/${campaignId}`);
    return response.data;
};

export const createCampaign = async (data: CreateCampaignRequest): Promise<Campaign> => {
    const response = await axiosInstance.post<Campaign>('/api/campaigns', data);
    return response.data;
};

export const updateCampaign = async (
    campaignId: number,
    data: UpdateCampaignRequest
): Promise<Campaign> => {
    const response = await axiosInstance.put<Campaign>(`/api/campaigns/${campaignId}`, data);
    return response.data;
};

export const deleteCampaign = async (campaignId: number): Promise<void> => {
    await axiosInstance.delete(`/api/campaigns/${campaignId}`);
};

export const getCampaignRounds = async (campaignId: number): Promise<CampaignRound[]> => {
    const response = await axiosInstance.get<CampaignRound[]>(`/api/campaigns/${campaignId}/rounds`);
    return response.data;
};

export const createCampaignRound = async (
    campaignId: number,
    data: CreateCampaignRoundRequest
): Promise<CampaignRound> => {
    const response = await axiosInstance.post<CampaignRound>(
        `/api/campaigns/${campaignId}/rounds`,
        data
    );
    return response.data;
};

export const getCampaignRound = async (
    campaignId: number,
    roundId: number
): Promise<CampaignRound> => {
    const response = await axiosInstance.get<CampaignRound>(
        `/api/campaigns/${campaignId}/rounds/${roundId}`
    );
    return response.data;
};

export const updateCampaignRound = async (
    campaignId: number,
    roundId: number,
    data: UpdateCampaignRoundRequest
): Promise<CampaignRound> => {
    const response = await axiosInstance.put<CampaignRound>(
        `/api/campaigns/${campaignId}/rounds/${roundId}`,
        data
    );
    return response.data;
};

export const deleteCampaignRound = async (
    campaignId: number,
    roundId: number
): Promise<void> => {
    await axiosInstance.delete(`/api/campaigns/${campaignId}/rounds/${roundId}`);
};

export interface ActivityDataResponse {
    data: Array<{seq_id: string, activity: number}>;
    count: number;
}

export const getCampaignRoundActivityData = async (
    campaignId: number,
    roundNumber: number
): Promise<ActivityDataResponse> => {
    const response = await axiosInstance.get<ActivityDataResponse>(
        `/api/campaigns/${campaignId}/${roundNumber}/activity_data`
    );
    return response.data;
};

export const uploadCampaignRoundActivityFile = async (
    campaignId: number,
    roundNumber: number,
    file: File
): Promise<{message: string, file_path: string}> => {
    const formData = new FormData();
    formData.append('activity_file', file);

    const response = await axiosInstance.post(
        `/api/campaigns/${campaignId}/${roundNumber}/activity_file`,
        formData,
        {
            headers: {
                'Content-Type': 'multipart/form-data',
            }
        }
    );
    return response.data;
};
