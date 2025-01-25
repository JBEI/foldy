import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

// RJSF v3+ imports
import Form from "@rjsf/antd";
import { IChangeEvent } from "@rjsf/core";
import { ErrorSchema } from "@rjsf/utils";
import validator from "@rjsf/validator-ajv8";

// For JSON schema type definitions
import { JSONSchema7 } from "json-schema";
import './NewFoldView.scss';

// For converting JSON -> YAML
import YAML from "yaml";

// ---------------------------------------------------------------------
// 1) DEFINE TYPES / INTERFACES
// ---------------------------------------------------------------------

// This interface reflects the shape of your Boltz-style input.
export interface NewFoldFormData {
    version: number;
    sequences: Array<{
        entity_type: "protein" | "dna" | "rna" | "ligand";
        id: string;            // We'll parse commas -> array on submit
        sequence?: string;     // for protein/dna/rna
        smiles?: string;       // for ligand
        ccd?: string;          // for ligand
        msa?: string;          // for protein (path or "empty" or omitted)
        modifications?: Array<{
            position: number;    // 1-based residue index
            ccd: string;         // CCD code
        }>;
    }>;
    constraints?: Array<{
        bond?: {
            atom1: [string, string, string];
            atom2: [string, string, string];
        };
        pocket?: {
            binder: string;
            contacts: Array<[string, string]>;
        };
    }>;
    advanced?: {
        startFoldJob?: boolean;
        emailOnCompletion?: boolean;
        skipDuplicateEntries?: boolean;
        disableRelaxation?: boolean;
    };
}

export interface NewFold2Props {
    userType: string | null; // e.g. "admin" | "viewer"
}

// This is an example function you might call after forming the YAML.
async function createFold(
    yamlData: string,
    options: {
        userType: string | null;
        startFoldJob?: boolean;
        emailOnCompletion?: boolean;
        skipDuplicateEntries?: boolean;
        disableRelaxation?: boolean;
    }
): Promise<void> {
    // Hypothetical: send to your backend or do anything else
    console.log("Creating fold with YAML:\n", yamlData);
    console.log("Options:", options);
    // e.g. await postToBackend(yamlData, options);
}

// ---------------------------------------------------------------------
// 2) CONSTANTS & HELPERS
// ---------------------------------------------------------------------

const VALID_AMINO_ACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
];

// ---------------------------------------------------------------------
// 3) JSON SCHEMA (typed with JSONSchema7)
// ---------------------------------------------------------------------

const schema: JSONSchema7 = {
    title: "New Fold (Boltz)",
    type: "object",
    required: ["version", "sequences"],
    properties: {
        version: {
            type: "number",
            title: "YAML Version",
            default: 1,
        },
        sequences: {
            type: "array",
            title: "Molecules / Chains",
            description: "Add as many protein/dna/rna or ligand entries as needed.",
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
                        title: "Chain/Molecule ID(s)",
                        default: "A",
                        description:
                            "Enter single or multiple IDs (comma-separated) if identical entities."
                    },
                    sequence: {
                        type: "string",
                        title: "Sequence (for protein/dna/rna)",
                    },
                    smiles: {
                        type: "string",
                        title: "SMILES (for ligand)",
                    },
                    ccd: {
                        type: "string",
                        title: "CCD code (for ligand)",
                    },
                    msa: {
                        type: "string",
                        title: "MSA Path (for proteins)",
                        default: "empty",
                        description:
                            "Path to .a3m or CSV file, or 'empty' if single-sequence. Omit if using --use_msa_server."
                    },
                    modifications: {
                        type: "array",
                        title: "Modifications (optional)",
                        items: {
                            type: "object",
                            properties: {
                                position: {
                                    type: "number",
                                    title: "Residue Index (1-based)",
                                },
                                ccd: {
                                    type: "string",
                                    title: "CCD Code of Modified Residue",
                                },
                            },
                        },
                    },
                },
                // dependencies: {
                //     entity_type: {
                //         oneOf: [
                //             {
                //                 properties: {
                //                     entity_type: {
                //                         enum: ["protein", "dna", "rna"]
                //                     }
                //                 }
                //             },
                //             {
                //                 properties: {
                //                     entity_type: {
                //                         enum: ["ligand"]
                //                     },
                //                     smiles: {
                //                         type: "string",
                //                         title: "SMILES (for ligand)",
                //                     },
                //                     ccd: {
                //                         type: "string",
                //                         title: "CCD code (for ligand)",
                //                     }
                //                 }
                //             }
                //         ]
                //     }
                // }
            },
        },
        constraints: {
            type: "array",
            title: "Constraints (Optional)",
            items: {
                type: "object",
                properties: {
                    bond: {
                        type: "object",
                        title: "Covalent Bond Constraint",
                        properties: {
                            atom1: {
                                type: "array",
                                title: "Atom 1 [chainID, resIdx, atomName]",
                                items: { type: "string" },
                                minItems: 3,
                                maxItems: 3,
                            },
                            atom2: {
                                type: "array",
                                title: "Atom 2 [chainID, resIdx, atomName]",
                                items: { type: "string" },
                                minItems: 3,
                                maxItems: 3,
                            },
                        },
                        required: ["atom1", "atom2"],
                    },
                    pocket: {
                        type: "object",
                        title: "Pocket Constraint",
                        properties: {
                            binder: {
                                type: "string",
                                title: "Binder Chain ID",
                            },
                            contacts: {
                                type: "array",
                                title: "List of [ChainID, ResidueIdx]",
                                items: {
                                    type: "array",
                                    items: { type: "string" },
                                    minItems: 2,
                                    maxItems: 2,
                                },
                            },
                        },
                        required: ["binder", "contacts"],
                    },
                },
            },
        },
        advanced: {
            type: "object",
            title: "Advanced Settings",
            properties: {
                startFoldJob: {
                    type: "boolean",
                    title: "Start Fold Job Immediately",
                    default: true,
                },
                emailOnCompletion: {
                    type: "boolean",
                    title: "Email on Completion",
                    default: true,
                },
                skipDuplicateEntries: {
                    type: "boolean",
                    title: "Skip Duplicate Entries",
                    default: false,
                },
                disableRelaxation: {
                    type: "boolean",
                    title: "Disable AMBER Relaxation",
                    default: false,
                },
            },
        },
    },
};

// ---------------------------------------------------------------------
// 4) UI SCHEMA (OPTIONAL)
// ---------------------------------------------------------------------

const uiSchema = {
    sequences: {
        items: {
            "ui:options": {
                // This creates a more compact layout
                labelCol: { span: 8 },
                wrapperCol: { span: 16 }
            },
            entity_type: {
                "ui:widget": "select",
                // "ui:options": {
                //     labelCol: { span: 8 },
                //     wrapperCol: { span: 8 } // Make this field narrower
                // }
                "classNames": 'ant-col ant-col-3'
            },
            id: {
                "ui:options": {
                    labelCol: { span: 8 },
                    wrapperCol: { span: 8 } // Make this field narrower
                }
            },
            sequence: {
                "ui:help": "Required if Entity Type is protein/dna/rna.",
            },
            smiles: {
                "ui:help": "Required if Entity Type = ligand (optionally use CCD).",
                // "ui:widget": 'hidden'
            },
            ccd: {
                "ui:help": "Required if Entity Type = ligand (optionally use SMILES).",
            },
        },
    },
};

// ---------------------------------------------------------------------
// 5) CUSTOM VALIDATION WITH STRONG TYPES
// ---------------------------------------------------------------------

/**
 * The `customValidate` function is typed so that we receive:
 *   - formData: NewFoldFormData
 *   - errors: ErrorSchema<NewFoldFormData>
 */
function customValidate(
    formData: NewFoldFormData,
    errors: ErrorSchema<NewFoldFormData>
): ErrorSchema<NewFoldFormData> {
    // For each sequence, enforce logic:
    formData.sequences.forEach((seq, index) => {
        const { entity_type, sequence, smiles, ccd } = seq;

        // If protein/dna/rna, must have sequence
        if (entity_type === "protein" || entity_type === "dna" || entity_type === "rna") {
            if (!sequence || sequence.trim().length === 0) {
                errors.sequences?.[index]?.sequence?.__errors?.push('A sequence is required for this entity type.')
                // errors.sequences?.[index]?.sequence?.addError?.(
                //   "A sequence is required for this entity type."
                // );
            } else if (entity_type === "protein") {
                // Validate amino acids
                const cleanSeq = sequence.replace(/\s+/g, "").toUpperCase();
                for (const char of cleanSeq) {
                    if (!VALID_AMINO_ACIDS.includes(char)) {
                        errors.sequences?.[index]?.sequence?.__errors?.push(`Invalid amino acid '${char}'.`)
                        // errors.sequences?.[index]?.sequence?.addError?.(
                        //     `Invalid amino acid '${char}'.`
                        // );
                    }
                }
            }
        }

        // If ligand, must have either SMILES or CCD
        if (entity_type === "ligand") {
            if ((!smiles || smiles.trim().length === 0) && (!ccd || ccd.trim().length === 0)) {
                errors.sequences?.[index]?.smiles?.__errors?.push("Ligand must have either a SMILES or a CCD value.")
                // errors.sequences?.[index]?.addError?.(
                //     "Ligand must have either a SMILES or a CCD value."
                // );
            }
        }
    });

    return errors;
}

// ---------------------------------------------------------------------
// 6) COMPONENT IMPLEMENTATION
// ---------------------------------------------------------------------

const NewFold2: React.FC<NewFold2Props> = ({ userType }) => {
    // Pre-populate formData with minimal defaults
    const [formData, setFormData] = useState<NewFoldFormData>({
        version: 1,
        sequences: [
            {
                entity_type: "protein",
                id: "A",
                sequence: "",
                msa: "empty",
            },
        ],
        constraints: [],
        advanced: {
            startFoldJob: true,
            emailOnCompletion: true,
            skipDuplicateEntries: false,
            disableRelaxation: false,
        },
    });

    const [isSubmitting, setIsSubmitting] = useState(false);
    const navigate = useNavigate();

    // Handle final submit
    async function handleSubmit(e: IChangeEvent<NewFoldFormData>) {
        setIsSubmitting(true);

        try {
            // 1) Copy data
            const data: NewFoldFormData = JSON.parse(JSON.stringify(e.formData));

            // 2) Convert comma-separated IDs -> arrays, strip whitespace, etc.
            data.sequences = data.sequences.map((seq) => {
                const updated = { ...seq };

                // Convert CSV "A,B" to an array in final YAML
                // Boltz often expects e.g. id: [A, B] for multiple identical entities.
                if (updated.id.includes(",")) {
                    const pieces = updated.id
                        .split(",")
                        .map((p) => p.trim())
                        .filter((p) => p.length > 0);
                    // Replace 'id' with an array of strings in the final YAML
                    // but keep it typed as string in formData at runtime.
                    // We'll handle that in the final YAML conversion.
                    (updated as any).id = pieces; // We'll do a slight cast here
                } else {
                    // If user typed just one ID, we might wrap it in an array
                    (updated as any).id = [updated.id];
                }

                // Clean up sequences if protein/dna/rna
                if (updated.sequence) {
                    updated.sequence = updated.sequence.replace(/\s+/g, "").toUpperCase();
                }
                return updated;
            });

            // 3) Convert to YAML
            const yamlString = YAML.stringify(data);

            // 4) Call your hypothetical createFold function
            await createFold(yamlString, {
                userType,
                startFoldJob: data.advanced?.startFoldJob,
                emailOnCompletion: data.advanced?.emailOnCompletion,
                skipDuplicateEntries: data.advanced?.skipDuplicateEntries,
                disableRelaxation: data.advanced?.disableRelaxation,
            });

            alert("Fold successfully submitted!");
            navigate("/");
        } catch (err: unknown) {
            console.error(err);
            alert(`Failed to submit fold: ${String(err)}`);
        } finally {
            setIsSubmitting(false);
        }
    }

    return (
        <div style={{ margin: "1rem", overflowY: "auto" }}>
            <h2>New Fold (Boltz) - Fully Typed</h2>
            <hr />

            {userType === "viewer" && (
                <div style={{ color: "red", marginBottom: "1rem" }}>
                    You do not have permissions to submit folds on this instance.
                </div>
            )}

            <Form<NewFoldFormData>
                schema={schema}
                uiSchema={uiSchema}
                validator={validator}            // Provide the AJV 8 validator
                liveValidate
                formData={formData}
                onChange={(e) => setFormData(e.formData)}
                onSubmit={handleSubmit}
                customValidate={customValidate}
                disabled={userType === "viewer"} // or readOnly, etc.
            >
                <div style={{ marginTop: "1rem" }}>
                    <button
                        type="submit"
                        disabled={isSubmitting || userType === "viewer"}
                        className="uk-button uk-button-primary"
                    >
                        {isSubmitting ? "Submitting..." : "Submit Fold"}
                    </button>
                </div>
            </Form>
        </div>
    );
};

export default NewFold2;