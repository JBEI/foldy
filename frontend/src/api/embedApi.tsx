import axiosInstance from "../services/axiosInstance";

export const startLogits = async (foldId: number, runName: string, logitModel: string) => {
    const response = await axiosInstance.post(`/api/startlogits/${foldId}`, { run_name: runName, logit_model: logitModel });
    return response.data;
};