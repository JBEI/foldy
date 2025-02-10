import axiosInstance from "../services/axiosInstance";
import { Logit } from "../types/types";

export const startLogits = async (foldId: number, name: string, useStructure: boolean, logitModel: string): Promise<Logit> => {
    const response = await axiosInstance.post(`/api/startlogits/${foldId}`, { name: name, use_structure: useStructure, logit_model: logitModel });
    return response.data;
};