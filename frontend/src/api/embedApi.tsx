import axiosInstance from "../services/axiosInstance";
import { Logit } from "../types/types";

export const startLogits = async (foldId: number, name: string, logitModel: string, useStructure: boolean, getDepthTwoLogits: boolean): Promise<Logit> => {
    const response = await axiosInstance.post(
        `/api/startlogits/${foldId}`, {
        name: name,
        logit_model: logitModel,
        use_structure: useStructure,
        get_depth_two_logits: getDepthTwoLogits,
    });
    return response.data;
};