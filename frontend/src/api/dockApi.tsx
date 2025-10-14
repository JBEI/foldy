import axiosInstance from '../services/axiosInstance';
import { DockInput } from '../types/types';

/**
 * Creates a new dock job
 */
export const postDock = async (newDock: DockInput): Promise<boolean> => {
  const response = await axiosInstance.post('/api/dock', newDock);
  return response.data;
};

/**
 * Deletes a dock by ID
 */
export const deleteDock = async (dockId: number): Promise<boolean> => {
  const response = await axiosInstance.delete(`/api/dock/${dockId}`);
  return response.data;
};

/**
 * Gets dock SDF file data
 */
export const getDockSdf = async (
  foldId: number,
  ligandName: string
): Promise<Blob> => {
  const response = await axiosInstance.post(
    `/api/dock_sdf/${foldId}/${ligandName}`,
    null,
    { responseType: 'blob' }
  );
  return response.data;
};
