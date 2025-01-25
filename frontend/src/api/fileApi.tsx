import { FileInfo } from '../types/types';
import axiosInstance from '../services/axiosInstance';
import { authHeader } from '../util/authHeader';
import axios from 'axios';


export const getFileList = async (fold_id: number): Promise<FileInfo[]> => {
    const response = await axiosInstance.get(`/api/file/list/${fold_id}`);
    return response.data;
}

export const getFile = async (fold_id: number, filePath: string): Promise<Blob> => {
    try {
        const response = await axiosInstance.get(`/api/file/download/${fold_id}/${filePath}`, {
            headers: authHeader(),
            responseType: 'blob',
            timeout: 60000,
            onDownloadProgress: (progressEvent) => {
                // Optional: Add progress tracking
                console.log('Download Progress:', progressEvent);
            },
        });
        return response.data;
    } catch (error) {
        if (axios.isAxiosError(error)) {
            if (error.code === 'ECONNABORTED') {
                throw new Error('Download timed out');
            }
            console.error('Download failed:', error.message);
        }
        throw error;
    }
}