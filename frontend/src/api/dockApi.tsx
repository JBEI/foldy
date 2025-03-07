import axiosInstance from '../services/axiosInstance';
import { DockInput } from '../types/types';

export const postDock = async (newDock: DockInput): Promise<boolean> => {
  const response = await axiosInstance.post('/api/dock', newDock);
  return response.data;
};

export const deleteDock = async (dockId: number): Promise<boolean> => {
  const response = await axiosInstance.delete(`/api/dock/${dockId}`);
  return response.data;
};