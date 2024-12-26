import axiosInstance from '../services/axiosInstance';

export const getBlobFile = async (url: string): Promise<Blob> => {
  const response = await axiosInstance.get(url, { responseType: 'blob' });
  return response.data;
};

export const postBlobFile = async (url: string, data: any): Promise<Blob> => {
  const response = await axiosInstance.post(url, data, { responseType: 'blob' });
  return response.data;
};

export const queueJob = async (
  foldId: number,
  stage: string,
  emailOnCompletion: boolean
): Promise<any> => {
  const body = { fold_id: foldId, stage, email_on_completion: emailOnCompletion };
  const response = await axiosInstance.post('/api/queuejob', body);
  return response.data;
};