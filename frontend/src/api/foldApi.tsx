import axiosInstance from '../services/axiosInstance';
import { Fold, FoldInput } from '../types/types';

export const getFolds = async (
    filter: string | null,
    tagString: string | null,
    page: number | null,
    per_page: number | null
): Promise<Fold[]> => {
    const params: Record<string, string | number> = {};
    if (filter) params.filter = filter;
    if (tagString) params.tag = tagString;
    if (page !== null) params.page = page;
    if (per_page !== null) params.per_page = per_page;

    const response = await axiosInstance.get<Fold[]>('/api/fold', { params });
    return response.data;
};

export const getFold = async (foldId: number): Promise<Fold> => {
    const response = await axiosInstance.get<Fold>(`/api/fold/${foldId}`);
    return response.data;
};

export const postFolds = async (
    folds: FoldInput[],
    options: { startJob: boolean; emailOnCompletion: boolean; skipDuplicates: boolean }
): Promise<any> => {
    const body = {
        folds_data: folds,
        start_fold_job: options.startJob,
        email_on_completion: options.emailOnCompletion,
        skip_duplicate_entries: options.skipDuplicates,
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