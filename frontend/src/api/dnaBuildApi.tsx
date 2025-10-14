import axiosInstance from "../services/axiosInstance";

export interface DnaBuildRequest {
    design_id: string;
    fold_id: string;
    genbank_files: { [filename: string]: string };
    seq_ids: string[];
    number_of_mutations: number;
    dry_run?: boolean;
    username?: string;
    otp?: string;
    project_id?: string;
}

export interface DnaBuildSeqResult {
    success: boolean;
    error_msg?: string;
    template_used?: string;
    teselagen_seq_id?: string;
}

export interface DnaBuildResponse {
    design_name: string;
    teselagen_id?: string;
    seq_id_results: { [seq_id: string]: DnaBuildSeqResult };
}

/**
 * Creates a DNA build design and optionally posts to Teselagen
 */
export const createDnaBuild = async (request: DnaBuildRequest): Promise<DnaBuildResponse> => {
    const response = await axiosInstance.post(
        `/api/dna-build`,
        request
    );
    return response.data;
};
