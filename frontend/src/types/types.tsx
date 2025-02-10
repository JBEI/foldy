import { BoltzYamlHelper } from "../util/boltzYamlHelper";

export interface FoldInput {
    name: string;
    tags: string[];

    yaml_config: string | null;
    yaml_helper: BoltzYamlHelper | null;

    // Old inputs.
    sequence: string | null;
    af2_model_preset: string | null;
    disable_relaxation: boolean | null;
}

export interface Fold extends FoldInput {
    id: number | null;
    owner: string;
    create_date: string; // ISO formatted datetime string
    public: boolean | null;
    state: string | null;
    jobs: Invokation[] | null;
    docks: Dock[] | null;
    logits: Logit[] | null;
    embeddings: Embedding[] | null;
    evolutions: Evolution[] | null;
}

export interface FoldPdb {
    pdb_string: string;
}

export interface FoldPae {
    pae: number[][];
}

export interface FoldContactProb {
    contact_prob: number[][];
}


export const getJobStatus = (fold: Fold, job_type: string): string | null => {
    if (!fold.jobs) {
        return null;
    }
    for (const job of fold.jobs) {
        if (job.type === job_type) {
            return job.state;
        }
    }
    return null;
};


export const describeFoldState = (fold: Fold) => {
    const featuresState = getJobStatus(fold, "features");
    const modelsState = getJobStatus(fold, "models");
    const decompressState = getJobStatus(fold, "decompress_pkls");

    // Special case: if anything hasn't been queued, just say unstarted.
    if (
        featuresState === null ||
        modelsState === null ||
        decompressState === null
    ) {
        return "unstarted";
    }

    // Another special case: before the beginning just say "queued".
    if (featuresState === "queued") {
        return "queued";
    }

    // Another special case: after the end just say "finished".
    if (decompressState === "finished") {
        return "finished";
    }

    // Normal case: print the state of the most recent stage.
    if (featuresState !== "finished") {
        return `features ${featuresState}`;
    }
    if (modelsState !== "finished") {
        return `models ${modelsState}`;
    }
    return `decompress_pkls ${decompressState}`;
};

export const foldIsFinished = (fold: Fold): boolean => {
    return getJobStatus(fold, "models") === "finished";
};

export interface DockInput {
    fold_id: number;
    ligand_name: string;
    ligand_smiles: string;
    tool: string | null;
    bounding_box_residue: string | null;
    bounding_box_radius_angstrom: number | null;
}

export interface Dock extends DockInput {
    id: number;
    invokation_id: number | null;
    pose_energy: number | null;
    pose_confidences: string | null;
}

export interface Logit {
    id: number;
    name: string;
    fold_id: number;
    logit_model: string;
    use_structure: boolean | null;
    invokation_id: number;
}

export interface Embedding {
    id: number;
    name: string;
    fold_id: number;
    embedding_model: string;
    extra_seq_ids: string;
    dms_starting_seq_ids: string;
    invokation_id: number;
}

export interface Evolution {
    id: number;
    name: string;
    fold_id: number;
    embedding_files: string | null;
    invokation_id: number | null;
}

export interface Invokation {
    id: number;
    type: string | null;
    job_id: string | null;
    state: string | null;
    command: string | null;
    log: string | null;
    timedelta_sec: number | null;
    starttime: string | null; // ISO formatted datetime string
}

export interface Annotations {
    [chainName: string]: Array<{
        type: string;
        start: number;
        end: number;
    }>;
}

export interface FileInfo {
    key: string;
    size: number;
    modified: number;
}






///////////////////


// // ----- Interfaces -----
// export interface DockInput {
//     fold_id: number;
//     ligand_name: string;
//     ligand_smiles: string;
//     tool: string | null;
//     bounding_box_residue: string | null;
//     bounding_box_radius_angstrom: number | null;
// }

// export interface Dock extends DockInput {
//     id: number;
//     invokation_id: number | null;
//     pose_energy: number | null;
//     pose_confidences: string | null;
// }

// export interface FoldInput {
//     name: string;
//     tags: string[];
//     sequence: string;
//     af2_model_preset: string | null;
//     disable_relaxation: boolean | null;
// }

// export interface Invokation {
//     id: number;
//     type: string | null;
//     job_id: string | null;
//     state: string | null;
//     command: string | null;
//     log: string | null;
//     timedelta_sec: number | null;
//     starttime: string | null; // ISO formatted datetime string.
// }

// export interface Fold extends FoldInput {
//     id: number | null;
//     owner: string;
//     create_date: string; // ISO formatted datetime string.
//     public: boolean | null;
//     state: string | null;
//     jobs: Invokation[] | null;
//     docks: Dock[] | null;
// }

// export interface FoldPae {
//     pae: number[][];
// }

// export interface FoldContactProb {
//     contact_prob: number[][];
// }

// export interface Annotations {
//     [chainName: string]: [
//         {
//             type: string;
//             start: number;
//             end: number;
//         }
//     ];
// }

// export interface FileInfo {
//     key: string;
//     size: number;
//     modified: number;
// }