import React, { useState } from "react";
import {
    AutoForm,
    AutoField,
    ListField,
    ListItemField,
    SubmitField,
    ErrorsField,
    // connectField,
    // useField
} from "uniforms-antd";
import { connectField, useField } from "uniforms";
import { JSONSchemaBridge } from "uniforms-bridge-json-schema";
import YAML from "yaml";
import Ajv, { ErrorObject } from 'ajv';
import addErrors from "ajv-errors";

/** Minimal shape we'll edit in Uniforms (internal model). */
interface BoltzFormModel {
    version: number;
    sequences: Array<{
        entity_type: "protein" | "dna" | "rna" | "ligand";
        id?: string;      // stored as comma-separated chain IDs
        sequence?: string;
        smiles?: string;
        ccd?: string;
    }>;
}

/** Props for our reusable builder */
export interface BoltzYamlBuilderProps {
    /**
     * Optional initial Boltz YAML. We'll parse it, transform to our simpler internal model,
     * and let the user edit. If omitted, we start with an empty default.
     */
    initialYaml?: string;

    /**
     * Callback when the user clicks "Save". We pass back a fully-formed
     * Boltz-format YAML string.
     */
    onSave?: (yamlString: string) => void;
}

/** 1) JSON Schema for our simpler internal shape (Uniforms). */
const simpleSchema = {
    title: "Boltz Config Editor",
    type: "object",
    required: ["version", "sequences"],
    properties: {
        version: {
            type: "number",
            default: 1,
            title: "YAML Version",
        },
        sequences: {
            type: "array",
            default: [],
            title: "Molecules / Chains",
            items: {
                type: "object",
                required: ["entity_type", "id"],
                properties: {
                    entity_type: {
                        type: "string",
                        title: "Entity Type",
                        enum: ["protein", "dna", "rna", "ligand"],
                        default: "protein",
                    },
                    id: {
                        type: "string",
                        title: "Chain ID(s) (comma-separated)",
                    },
                    sequence: {
                        type: "string",
                        title: "Sequence (for protein)",
                    },
                    smiles: {
                        type: "string",
                        title: "SMILES (for ligand)",
                    },
                    ccd: {
                        type: "string",
                        title: "CCD code (for ligand)",
                    },
                },
            },
        },
    },
};

// Define valid characters for each type
const VALID_AMINO_ACIDS = new Set('ACDEFGHIKLMNPQRSTVWY');
const VALID_DNA = new Set('ATCG');
const VALID_RNA = new Set('AUCG');

const ajv = new Ajv({ allErrors: true, allowUnionTypes: true });
addErrors(ajv);

const validate = ajv.compile(simpleSchema);

function transformAjvErrors(errors: ErrorObject[] | null | undefined) {
    if (!errors) return null;
    return errors.map((err) => {
        const field = err.instancePath.replace(/\//g, ".").replace(/^\./, "");
        return { name: field, message: err.message || "Validation error" };
    });
}

const schemaValidator = (model: Record<string, any>) => {
    // 1) First, let AJV do its standard validation
    validate(model);

    // 2) If AJV found errors, transform them
    if (validate.errors && validate.errors.length) {
        const details = transformAjvErrors(validate.errors);
        return details ? { details } : null;
    }

    // 3) Return nothing (no errors) if AJV is happy
    return null;
}

function additionalChecks(model: BoltzFormModel, errors: { details: [{ name: string, message: string }] } /* Uniforms error list */) {
    model.sequences?.forEach((seq: any, index: number) => {
        if (!seq?.entity_type) {
            errors.details.push({
                name: `sequences.${index}.entity_type`,
                message: 'Entity type is required'
            });
            return;
        }
        if (seq.entity_type === 'ligand') {
            const hasSmiles = Boolean(seq.smiles?.trim());
            const hasCcd = Boolean(seq.ccd?.trim());

            if (hasSmiles && hasCcd) {
                errors.details.push({
                    name: `sequences.${index}.smiles`,
                    message: 'Please provide either SMILES or CCD, not both'
                });
            } else if (!hasSmiles && !hasCcd) {
                errors.details.push({
                    name: `sequences.${index}.smiles`,
                    message: 'Please provide either SMILES or CCD'
                });
            }
        } else {
            if (!seq.sequence) {
                errors.details.push({
                    name: `sequences.${index}.sequence`,
                    message: `Sequence is required for entities of type ${seq.entity_type}`
                });
            } else {
                const sequence = seq.sequence.trim().toUpperCase();
                let invalidChars: string[] = [];

                if (seq.entity_type === 'protein') {
                    invalidChars = [...sequence].filter(char => !VALID_AMINO_ACIDS.has(char));
                } else if (seq.entity_type === 'dna') {
                    invalidChars = [...sequence].filter(char => !VALID_DNA.has(char));
                } else if (seq.entity_type === 'rna') {
                    invalidChars = [...sequence].filter(char => !VALID_RNA.has(char));
                }

                if (invalidChars.length > 0) {
                    errors.details.push({
                        name: `sequences.${index}.sequence`,
                        message: `Invalid ${seq.entity_type} characters found: ${Array.from(new Set(invalidChars)).join(', ')}`
                    });
                }
            }
        }
    });
}

const schemaBridge = new JSONSchemaBridge({
    schema: simpleSchema,
    validator: schemaValidator
});

/** 
 * 3) Convert a full Boltz-shaped JS object => our simpler Uniforms model.
 *    e.g. 
 *      version: 1
 *      sequences:
 *        - protein: { id: [...], sequence: ... }
 *        - ligand: { id: [...], smiles: ..., ccd: ... }
 */
function fromBoltzObjectToModel(boltzObj: any): BoltzFormModel {
    const version = boltzObj?.version ?? 1;
    const sequences: BoltzFormModel["sequences"] = [];

    // Each item in boltzObj.sequences is like:
    //   { protein: { id: [...], sequence: ... } } 
    // or { ligand: { id: [...], smiles: ... } } etc.
    (boltzObj?.sequences || []).forEach((entry: any) => {
        // We expect exactly one key in each item: "protein", "dna", "rna", or "ligand".
        const entityType = Object.keys(entry)[0] as BoltzFormModel["sequences"][number]["entity_type"];
        const data = entry[entityType] || {};

        sequences.push({
            entity_type: entityType,
            id: (data.id || []).join(", "), // turn array into comma string
            sequence: data.sequence || "",
            smiles: data.smiles || "",
            ccd: data.ccd || "",
        });
    });

    return { version, sequences };
}

/** 
 * 4) Convert our simpler Uniforms model => full Boltz-style object => YAML.
 *    i.e. 
 *      sequences: [
 *        { protein: { id: [...], sequence: "..." }},
 *        { ligand: { id: [...], smiles: "...", ccd: "..." }},
 *      ]
 */
function toBoltzYaml(model: BoltzFormModel): string {
    const boltzObj: any = {
        version: model.version,
        sequences: model.sequences.map((seq) => {
            // Turn "A, B" into ["A", "B"]
            const idArray = (seq.id || "")
                .split(",")
                .map((x) => x.trim())
                .filter(Boolean);

            if (seq.entity_type === "ligand") {
                const ligandData: any = {
                    id: idArray,
                };

                // Only include non-empty fields
                if (seq.smiles?.trim()) {
                    ligandData.smiles = seq.smiles.trim();
                }
                if (seq.ccd?.trim()) {
                    ligandData.ccd = seq.ccd.trim();
                }

                return { ligand: ligandData };
            } else if (seq.entity_type === "protein") {
                return {
                    protein: {
                        id: idArray,
                        sequence: seq.sequence || "",
                    },
                };
            } else if (seq.entity_type === "dna") {
                return {
                    dna: {
                        id: idArray,
                        // no "sequence" included in final example if you only want it for protein
                    },
                };
            } else if (seq.entity_type === "rna") {
                return {
                    rna: {
                        id: idArray,
                        // no "sequence" included for rna either
                    },
                };
            }
            // fallback
            return {};
        }),
    };

    return YAML.stringify(boltzObj);
}

/** 
 * A small helper that conditionally shows fields:
 *  - For 'protein': only show 'sequence'
 *  - For 'ligand':  only show 'smiles', 'ccd'
 *  - For 'dna'/'rna': show nothing extra
 */
const EntityTypeConditionalFields = connectField((props: { value: BoltzFormModel['sequences'][number] }) => {
    if (!props.value) return null;

    const entityType = props.value.entity_type;
    // const parent = useField([], {})[0]; // Get parent field context
    // const index = parseInt((props.name || '').split('.')[1] || '0', 10);

    if (entityType === "protein" || entityType === "dna" || entityType === "rna") {
        return <AutoField name="sequence" />;
    } else if (entityType === "ligand") {
        const hasSmiles = Boolean(props.value.smiles?.trim());
        const hasCcd = Boolean(props.value.ccd?.trim());

        return (
            <>
                <AutoField
                    name="smiles"
                    placeholder="Enter SMILES or use CCD below"
                    disabled={hasCcd}
                    value={hasCcd ? undefined : props.value.smiles}
                />
                <AutoField
                    name="ccd"
                    placeholder="Enter CCD or use SMILES above"
                    disabled={hasSmiles}
                    value={hasSmiles ? undefined : props.value.ccd}
                />
            </>
        );
    }
    return null;
});

/**
 * (A) Reusable React component to edit a Boltz config in a "simplified" Uniforms model,
 * with entity_type-based conditional fields.
 */
const BoltzYamlBuilder: React.FC<BoltzYamlBuilderProps> = ({ initialYaml, onSave }) => {
    /**
     * Parse initial YAML -> JS object -> simpler form model
     */
    let initialModel: BoltzFormModel = { version: 1, sequences: [] };
    if (initialYaml) {
        try {
            const parsed = YAML.parse(initialYaml);
            initialModel = fromBoltzObjectToModel(parsed);
        } catch (err) {
            console.warn("Failed to parse initial YAML. Starting empty.", err);
        }
    }

    const [model, setModel] = useState<BoltzFormModel>(initialModel);

    /** On submit, transform to final Boltz YAML and invoke onSave() callback. */
    function handleSubmit(submitted: BoltzFormModel) {
        try {
            const yaml = toBoltzYaml(submitted);
            onSave?.(yaml);
        } catch (error) {
            console.error('Failed to save:', error);
        }
    }

    // Uniforms `onValidate` merges custom checks with AJV checks
    function handleValidate(model: BoltzFormModel, error: any) {
        // `error.details` is an array of existing AJV-based errors
        // We'll push additional errors for protein vs ligand logic
        if (!error?.details) {
            error = { details: [] }
        }
        additionalChecks(model, error);
        console.log(`ERROR DETAILS TODO: ${error.details}`);

        // If we added new errors, return them so Uniforms displays them
        return error.details.length === 0 ? null : error;
    }

    return (
        <div style={{ margin: "1rem" }}>
            <h2>Boltz YAML Editor</h2>

            <AutoForm
                schema={schemaBridge}
                model={model}
                onChangeModel={(m: Record<string, any>) => setModel(m as BoltzFormModel)}
                onSubmit={(m: Record<string, any>) => handleSubmit(m as BoltzFormModel)}
                showInlineError
                onValidate={handleValidate}
            >
                <AutoField name="version" />

                <ListField name="sequences">
                    <ListItemField name="$">
                        {/* For each sequence item, always show `entity_type` and `id`, 
                then show conditional fields based on entity_type. 
            */}
                        <AutoField name="entity_type" />
                        <AutoField name="id" />

                        <EntityTypeConditionalFields name="" />
                    </ListItemField>
                </ListField>

                {/* Display form-level errors */}
                <ErrorsField />

                <div style={{ marginTop: "1rem" }}>
                    <SubmitField value="Save Boltz YAML" />
                </div>
            </AutoForm>
        </div>
    );
};

export default BoltzYamlBuilder;