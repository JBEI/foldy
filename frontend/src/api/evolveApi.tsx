import { authHeader } from '../util/authHeader';
import axiosInstance from '../services/axiosInstance';
import { Evolution } from '../types/types';

export const getEvolution = async (evolutionId: number): Promise<Evolution> => {
    const response = await axiosInstance.get<Evolution>(`/api/evolve/${evolutionId}`);
    return response.data;
};

export async function evolve(
    evolutionName: string,
    foldId: number,
    activityFile: File,
    mode: string,
    embeddingPaths?: string[],
    finetuningModelCheckpoint?: string
): Promise<Evolution> {
    const formData = new FormData();
    formData.append('name', evolutionName);
    formData.append('fold_id', foldId.toString());
    formData.append('activity_file', activityFile);

    formData.append('mode', mode);
    if (embeddingPaths) {
        formData.append('embedding_paths', JSON.stringify(embeddingPaths));
    }
    if (finetuningModelCheckpoint) {
        formData.append('finetuning_model_checkpoint', finetuningModelCheckpoint);
    }

    const response = await axiosInstance.post<Evolution>('/api/evolve', formData, {
        headers: {
            ...authHeader(),  // Keep auth headers
            'Content-Type': 'multipart/form-data',  // Override content type for this request
        }
    });
    return response.data;
}
