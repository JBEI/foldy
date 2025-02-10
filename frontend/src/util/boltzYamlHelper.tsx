import { parse, stringify } from 'yaml'

/**
 * A protein chain is returned as a tuple [chainId, sequence].
 */
export type ChainSequence = [string, string]

/**
 * For modified residues inside a protein (or other polymer),
 * we store position + CCD code. We also keep track of which chain IDs
 * this modification applies to.
 */
export type ModificationData = {
    chain_ids: string[]
    position: number
    ccd: string
    [key: string]: any // For any additional fields
}

/**
 * For ligands, we can have either `smiles` or `ccd`.
 */
export type LigandData = {
    chain_ids: string[]
    smiles?: string
    ccd?: string
}

/**
 * The main helper class for reading and constructing
 * Boltz YAML configurations.
 */
export class BoltzYamlHelper {
    private data: any
    private version: number | null
    private sequences: any[]
    private constraints: any[]

    /**
     * Construct a new BoltzYamlHelper by parsing an existing YAML string.
     * @param yamlStr The YAML content as a string.
     */
    constructor(yamlStr: string) {
        this.data = parse(yamlStr)
        this.version = this.data?.version ?? null
        this.sequences = this.data?.sequences ?? []
        this.constraints = this.data?.constraints ?? []
    }

    /**
     * Return the version field from the YAML.
     */
    getVersion(): number | null {
        return this.version
    }

    /**
     * Set the version field.
     */
    setVersion(version: number) {
        this.version = version
    }

    getSequencesOfType(targetEntityType: string): ChainSequence[] {
        const results: ChainSequence[] = []

        for (const entry of this.sequences) {
            // Each entry is expected to have exactly one key,
            // e.g. "protein", "ligand", "rna", "dna", etc.
            const [entityType, entityData] = Object.entries(entry)[0] as [string, any]
            if (entityType === targetEntityType) {
                let chainIds = entityData.id
                if (!Array.isArray(chainIds)) {
                    chainIds = [chainIds]
                }
                const seq = entityData.sequence
                if (!seq) continue

                for (const cid of chainIds) {
                    results.push([cid, seq])
                }
            }
        }

        return results;
    }

    /**
     * Returns all protein chain sequences as [chainId, sequence].
     * Example: [ ["A","MVTPE"], ["B","MVTPE"], ... ]
     */
    getProteinSequences(): ChainSequence[] {
        return this.getSequencesOfType('protein');
    }

    getDNASequences(): ChainSequence[] {
        return this.getSequencesOfType('dna');
    }

    getRNASequences(): ChainSequence[] {
        return this.getSequencesOfType('rna');
    }

    /**
     * Returns an array of modifications across all polymer entities
     * (protein, dna, rna). Each object includes chain_ids, position, ccd, etc.
     */
    getModifications(): ModificationData[] {
        const modificationsList: ModificationData[] = []

        for (const entry of this.sequences) {
            const [entityType, entityData] = Object.entries(entry)[0] as [string, any]
            // Skip if not a polymer with possible modifications
            if (!['protein', 'dna', 'rna'].includes(entityType)) {
                continue
            }

            const mods = entityData.modifications ?? []
            if (mods.length > 0) {
                let chainIds = entityData.id
                if (!Array.isArray(chainIds)) {
                    chainIds = [chainIds]
                }

                for (const mod of mods) {
                    modificationsList.push({
                        ...mod,
                        chain_ids: chainIds
                    })
                }
            }
        }

        return modificationsList
    }

    /**
     * Returns the constraints array (both bond and pocket constraints).
     */
    getConstraints(): any[] {
        return this.constraints
    }

    /**
     * Returns all ligands as an array of { chain_ids, smiles?, ccd? }.
     */
    getLigands(): LigandData[] {
        const ligands: LigandData[] = []

        for (const entry of this.sequences) {
            const [entityType, entityData] = Object.entries(entry)[0] as [string, any]
            if (entityType === 'ligand') {
                let chainIds = entityData.id
                if (!Array.isArray(chainIds)) {
                    chainIds = [chainIds]
                }

                ligands.push({
                    chain_ids: chainIds,
                    smiles: entityData.smiles,
                    ccd: entityData.ccd
                })
            }
        }

        return ligands
    }

    /**
     * Helper method: push a new entry to this.sequences.
     */
    private pushSequenceEntry(entityType: string, entityData: any) {
        this.sequences.push({
            [entityType]: entityData
        })
    }

    /**
     * Add a new protein entry to sequences.
     * @param params.id One or more chain IDs
     * @param params.sequence The protein sequence
     * @param params.msa (Optional) Path to an MSA file or 'empty'
     * @param params.modifications (Optional) Array of modifications
     */
    addProtein(params: {
        id: string[] | string
        sequence: string
        msa?: string
        modifications?: { position: number; ccd: string }[]
    }) {
        const { id, sequence, msa, modifications } = params
        const chainIds = Array.isArray(id) ? id : [id]

        const proteinData: any = {
            id: chainIds,
            sequence
        }
        if (msa !== undefined) {
            proteinData.msa = msa
        }
        if (modifications !== undefined) {
            proteinData.modifications = modifications
        }

        this.pushSequenceEntry('protein', proteinData)
    }

    /**
     * Add a new ligand entry to sequences.
     * @param params.id One or more chain IDs
     * @param params.smiles (Optional) SMILES string
     * @param params.ccd (Optional) CCD code
     */
    addLigand(params: { id: string[] | string; smiles?: string; ccd?: string }) {
        const { id, smiles, ccd } = params
        const chainIds = Array.isArray(id) ? id : [id]

        const ligandData: any = {
            id: chainIds
        }
        if (smiles !== undefined) {
            ligandData.smiles = smiles
        }
        if (ccd !== undefined) {
            ligandData.ccd = ccd
        }

        this.pushSequenceEntry('ligand', ligandData)
    }

    /**
     * Add a bond constraint.
     * For example:
     *   addBondConstraint({ atom1: ["A", 42, "CA"], atom2: ["LIG", 1, "C1"] })
     */
    addBondConstraint(params: {
        atom1: [string, number, string]
        atom2: [string, number, string]
    }) {
        if (!Array.isArray(this.constraints)) {
            this.constraints = []
        }
        this.constraints.push({
            bond: {
                atom1: params.atom1,
                atom2: params.atom2
            }
        })
    }

    /**
     * Add a pocket constraint.
     *   addPocketConstraint({
     *     binder: "C", // The chain that binds the pocket
     *     contacts: [["A", 12], ["A", 14]]
     *   })
     */
    addPocketConstraint(params: {
        binder: string
        contacts: [string, number][]
    }) {
        if (!Array.isArray(this.constraints)) {
            this.constraints = []
        }
        this.constraints.push({
            pocket: {
                binder: params.binder,
                contacts: params.contacts
            }
        })
    }

    /**
     * Convert the current data back into a YAML string.
     */
    toYAML(): string {
        const obj = {
            version: this.version,
            sequences: this.sequences,
            constraints: this.constraints
        }
        return stringify(obj)
    }

    /**
     * Get all sequences with their full data, including modifications
     */
    getAllSequences(): Array<{
        entity_type: "protein" | "dna" | "rna" | "ligand";
        id: string[];
        sequence?: string;
        smiles?: string;
        ccd?: string;
        modifications?: Array<{
            position: number;
            ccd: string;
        }>;
    }> {
        const results = [];

        for (const entry of this.sequences) {
            const [entityType, entityData] = Object.entries(entry)[0] as [string, any];
            if (!["protein", "dna", "rna", "ligand"].includes(entityType)) {
                throw new Error(`Invalid entity type in sequence: ${entityType}`);
            }

            results.push({
                entity_type: entityType as "protein" | "dna" | "rna" | "ligand",
                id: Array.isArray(entityData.id) ? entityData.id : [entityData.id],
                sequence: entityData.sequence,
                smiles: entityData.smiles,
                ccd: entityData.ccd,
                modifications: entityData.modifications,
            });
        }

        return results;
    }

    /**
     * Get all constraints in their normalized form
     */
    getNormalizedConstraints(): Array<{
        constraint_type: "bond" | "pocket";
        bond_chain_id_1?: string;
        bond_res_idx_1?: number;
        bond_atom_name_1?: string;
        bond_chain_id_2?: string;
        bond_res_idx_2?: number;
        bond_atom_name_2?: string;
        binder?: string;
        contacts?: Array<{
            chain_id: string;
            res_idx: number;
        }>;
    }> {
        const normalized = [];

        for (const constraint of this.constraints) {
            const [constraintType, data] = Object.entries(constraint)[0] as [string, any];

            if (constraintType === 'bond') {
                normalized.push({
                    constraint_type: 'bond' as "bond" | "pocket",
                    bond_chain_id_1: data.atom1[0],
                    bond_res_idx_1: data.atom1[1],
                    bond_atom_name_1: data.atom1[2],
                    bond_chain_id_2: data.atom2[0],
                    bond_res_idx_2: data.atom2[1],
                    bond_atom_name_2: data.atom2[2],
                });
            } else if (constraintType === 'pocket') {
                normalized.push({
                    constraint_type: 'pocket' as "bond" | "pocket",
                    binder: data.binder,
                    contacts: data.contacts?.map((contact: [string, number]) => ({
                        chain_id: contact[0],
                        res_idx: contact[1],
                    })),
                });
            }
        }

        return normalized;
    }
}

// Example usage:
// const yamlStr = `
// version: 1
// sequences:
//   - protein:
//       id: [A, B]
//       sequence: MVTPE
// `
// const helper = new BoltzYamlHelper(yamlStr)
// console.log(helper.getProteinSequences())  // -> [['A','MVTPE'], ['B','MVTPE']]
// helper.addLigand({ id: 'LIG', ccd: 'SAH' })
// console.log(helper.toYAML())