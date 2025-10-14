import axiosInstance from "../services/axiosInstance";
import { Logit } from "../types/types";

/**
 * Starts logits computation
 */
export const startLogits = async (foldId: number, name: string, logitModel: string, useStructure: boolean, getDepthTwoLogits: boolean): Promise<Logit> => {
    const response = await axiosInstance.post(
        `/api/startnaturalness/${foldId}`, {
        name: name,
        logit_model: logitModel,
        use_structure: useStructure,
        get_depth_two_logits: getDepthTwoLogits,
    });
    return response.data;
};

/**
 * Starts embeddings computation
 */
export const startEmbeddings = async (
    foldId: number,
    batchName: string,
    dmsStartingSeqIds: string[],
    extraSeqIds: string[],
    extraLayers: string[],
    embeddingModel: string,
    homologFasta: string | null = null,
    domainBoundaries: string[] = []
): Promise<boolean> => {
    const response = await axiosInstance.post(
        `/api/embeddings`,
        {
            fold_id: foldId,
            name: batchName,
            embedding_model: embeddingModel,
            dms_starting_seq_ids: dmsStartingSeqIds.join(','),
            extra_seq_ids: extraSeqIds.join(','),
            extra_layers: extraLayers.join(','),
            domain_boundaries: domainBoundaries.join(','),
            homolog_fasta: homologFasta || undefined,
        }
    );
    return response.data;
};

/**
 * Deletes a naturalness run
 */
export const deleteNaturalness = async (naturalnessId: number): Promise<void> => {
    await axiosInstance.delete(`/api/naturalness/${naturalnessId}`);
};

/**
 * Deletes an embedding run
 */
export const deleteEmbedding = async (embeddingId: number): Promise<void> => {
    await axiosInstance.delete(`/api/embedding/${embeddingId}`);
};
