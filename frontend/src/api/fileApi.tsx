import { FileInfo } from '../types/types';
import axiosInstance from '../services/axiosInstance';
import { authHeader } from '../util/authHeader';


export const getFileList = async (fold_id: number): Promise<FileInfo[]> => {
    const response = await axiosInstance.get(`/api/file/list/${fold_id}`);
    return response.data;
}

export const getFile = async (fold_id: number, filePath: string): Promise<Blob> => {
    const response = await axiosInstance.get(`/api/file/download/${fold_id}/${filePath}`, {
        headers: authHeader(),
        responseType: 'blob',
    });
    return response.data;
}
