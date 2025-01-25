import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

// Uniforms
import { AutoForm, AutoField, ErrorsField, SubmitField } from "uniforms-antd";
import { JSONSchemaBridge } from "uniforms-bridge-json-schema";

// Ant Design layout
import { Row, Col } from "antd";

import YAML from 'yaml'

// For YAML conversion

// For AJV-based validation
import Ajv, { ErrorObject } from "ajv";
import addErrors from "ajv-errors";
import { JSONSchema7 } from "json-schema";
import { FoldInput } from "../../types/types";
import { postFolds } from "../../api/foldApi";

// Your types (same as before)
export interface NewFoldFormData {
    name: string;
    version: number;
    sequences: Array<{
        entity_type: "protein" | "dna" | "rna" | "ligand";
        id: string;       // We'll parse commas -> array on submit
        sequence?: string;
        smiles?: string;
        ccd?: string;
        msa?: string;
        modifications?: Array<{
            position: number;
            ccd: string;
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

// ---------------------------------------------------------------------
// Hypothetical "createFold" function (same as your example)
// ---------------------------------------------------------------------
async function createFold(
    foldName: string,
    yamlData: string,
    options: {
        userType: string | null;
        startFoldJob?: boolean;
        emailOnCompletion?: boolean;
        skipDuplicateEntries?: boolean;
        disableRelaxation?: boolean;
    }
): Promise<void> {
    if (options.userType === "viewer") {
        throw new Error("Viewers cannot create folds");
    }

    const fold: FoldInput = {
        name: foldName, // You might want to add a name field to your form
        tags: [], // You might want to add tags to your form
        yaml_config: yamlData,
        yaml_helper: null,
        sequence: null, // Using yaml_config instead of old inputs
        af2_model_preset: "boltz", // Using yaml_config instead of old inputs
        disable_relaxation: options.disableRelaxation || false,
    };

    await postFolds([fold], {
        startJob: options.startFoldJob || false,
        emailOnCompletion: options.emailOnCompletion || false,
        skipDuplicates: options.skipDuplicateEntries || false,
    });
}
// ---------------------------------------------------------------------
// 1) BOLTZ JSON SCHEMA (same structure as your original RJSF schema)
// ---------------------------------------------------------------------
const schema: JSONSchema7 = {
    title: "New Fold (Boltz)",
    type: "object",
    required: ["name", "sequences", "version"],
    properties: {
        name: {
            type: "string",
            title: "Fold Name",
        },
        version: {
            type: "number",
            title: "YAML Version",
            default: 1,
        },
        sequences: {
            type: "array",
            minItems: 0,
            title: "Molecules / Chains",
            description: "Add as many protein/dna/rna or ligand entries as needed.",
            default: [], // Ensure default is set
            items: {
                type: "object",
                properties: {
                    protein: {
                        type: "object",
                        title: "Protein",
                        properties: {
                            id: {
                                type: "array",
                                items: {
                                    type: "string"
                                },
                                minItems: 1,
                                title: "Chain ID(s)",
                                default: ["A"]
                            },
                            sequence: {
                                type: "string",
                                title: "Sequence"
                            },
                            msa: {
                                type: "string",
                                title: "MSA Path",
                                default: "empty"
                            }
                        },
                        required: ["id", "sequence"]
                    },
                    ligand: {
                        type: "object",
                        title: "Ligand",
                        properties: {
                            id: {
                                type: "array",
                                items: {
                                    type: "string"
                                },
                                minItems: 1,
                                title: "Chain ID(s)",
                                default: ["L"]
                            },
                            smiles: {
                                type: "string",
                                title: "SMILES"
                            },
                            ccd: {
                                type: "string",
                                title: "CCD code"
                            }
                        },
                        required: ["id"]
                    }
                },
                oneOf: [
                    {
                        required: ["protein"],
                        not: { required: ["ligand"] }
                    },
                    {
                        required: ["ligand"],
                        not: { required: ["protein"] }
                    }
                ]
            }
        },
        // ... rest of existing schema ...    
        constraints: {
            type: "array",
            default: [],
            title: "Constraints (Optional)",
            items: {
                type: "object",
                properties: {
                    // bond: {
                    //     type: "object",
                    //     title: "Covalent Bond Constraint",
                    //     properties: {
                    //         atom1: {
                    //             type: "array",
                    //             default: [],
                    //             title: "Atom 1 [chainID, resIdx, atomName]",
                    //             items: { type: "string" },
                    //             // minItems: 3,
                    //             maxItems: 3,
                    //         },
                    //         atom2: {
                    //             type: "array",
                    //             default: [],
                    //             title: "Atom 2 [chainID, resIdx, atomName]",
                    //             items: { type: "string" },
                    //             // minItems: 3,
                    //             maxItems: 3,
                    //         },
                    //     },
                    //     required: ["atom1", "atom2"],
                    // },
                    // pocket: {
                    //     type: "object",
                    //     title: "Pocket Constraint",
                    //     properties: {
                    //         binder: {
                    //             type: "string",
                    //             title: "Binder Chain ID",
                    //         },
                    //         contacts: {
                    //             type: "array",
                    //             default: [],
                    //             title: "List of [ChainID, ResidueIdx]",
                    //             items: {
                    //                 type: "array",
                    //                 items: { type: "string" },
                    //                 // minItems: 2,
                    //                 maxItems: 2,
                    //             },
                    //         },
                    //     },
                    //     required: ["binder", "contacts"],
                    // },
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
// 2) CREATE AJV VALIDATOR + CUSTOM VALIDATION
// ---------------------------------------------------------------------

// 2a) Basic AJV setup
const ajv = new Ajv({ allErrors: true, allowUnionTypes: true });
addErrors(ajv); // Allows for better error messages

// 2b) Compile the schema
const validate = ajv.compile(schema);

// 2c) We'll define some constants & custom logic (like your RJSF version)
const VALID_AMINO_ACIDS = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
];

// 2d) Convert AJV errors to Uniforms errors
function transformAjvErrors(errors: ErrorObject[] | null | undefined) {
    if (!errors) return null;

    // In Uniforms, we typically return an array of { name, message } or throw
    // a ValidationError. But we can do minimal transformations for demonstration:
    return errors.map((err) => {
        // e.g. "/sequences/0/sequence" => "sequences.0.sequence"
        const field = err.instancePath.replace(/\//g, ".").replace(/^\./, "");
        const message = err.message || "Validation error";
        return { name: field, message };
    });
}

// 2e) Our final Uniforms "validator" function
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

// 2f) Additional custom logic (like your customValidate in RJSF) can be done
// in Uniforms via `onValidate` or by hooking into the same "schemaValidator."
// For demonstration, we'll do it in `onValidate`:

function additionalChecks(model: NewFoldFormData, errors: any /* Uniforms error list */) {
    model.sequences.forEach((seq, i) => {
        const { entity_type, sequence, smiles, ccd } = seq;

        // If protein/dna/rna, must have sequence
        if (entity_type !== "ligand") {
            if (!sequence) {
                errors.push({
                    name: `sequences.${i}.sequence`,
                    message: "A sequence is required for protein/dna/rna."
                });
            } else if (entity_type === "protein") {
                // Check each char in the sequence
                const cleanSeq = sequence.replace(/\s+/g, "").toUpperCase();
                for (const char of cleanSeq) {
                    if (!VALID_AMINO_ACIDS.includes(char)) {
                        errors.push({
                            name: `sequences.${i}.sequence`,
                            message: `Invalid amino acid '${char}'.`
                        });
                    }
                }
                if (cleanSeq.length === 0) {
                    errors.push({
                        name: `sequences.${i}.sequence`,
                        message: `All protein sequences must be >0 length.'.`
                    })
                }
            }
        }

        // If ligand, must have either SMILES or CCD
        if (entity_type === "ligand") {
            if ((!smiles || !smiles.trim()) && (!ccd || !ccd.trim())) {
                errors.push({
                    name: `sequences.${i}.smiles`,
                    message: "Ligand must have either a SMILES or a CCD value.",
                });
            }
        }
    });
    console.log(`ERROR DETAILS INSIDER: ${errors}`);
}

// 2g) Build a uniforms-bridge from our JSON schema + AJV-based validator
const uniformsSchema = new JSONSchemaBridge({ schema, validator: schemaValidator });

// ---------------------------------------------------------------------
// 3) THE UNIFORMS FORM COMPONENT
// ---------------------------------------------------------------------
const NewFoldView: React.FC<NewFold2Props> = ({ userType }) => {
    const navigate = useNavigate();

    // Similar to your RJSF "formData"
    const [formData, setFormData] = useState<NewFoldFormData>({
        name: "",
        version: 1,
        sequences: [],
        constraints: [],
        advanced: {
            startFoldJob: true,
            emailOnCompletion: true,
            skipDuplicateEntries: false,
            disableRelaxation: false,
        },
    });

    const [isSubmitting, setIsSubmitting] = useState(false);

    // Uniforms `onSubmit` handler (similar to handleSubmit in RJSF)
    async function handleSubmit(model: NewFoldFormData) {
        setIsSubmitting(true);
        try {
            // 1) Make a deep copy so we can safely mutate
            const data: NewFoldFormData = JSON.parse(JSON.stringify(model));

            // 2) Convert comma-separated IDs -> arrays, strip whitespace, uppercase seq, etc.
            data.sequences = data.sequences.map((seq) => {
                const updated = { ...seq };

                // ID as array
                if (updated.id.includes(",")) {
                    const pieces = updated.id
                        .split(",")
                        .map((p) => p.trim())
                        .filter((p) => p.length > 0);
                    (updated as any).id = pieces;
                } else {
                    (updated as any).id = [updated.id];
                }

                // Clean up sequence if protein/dna/rna
                if (updated.sequence) {
                    updated.sequence = updated.sequence.replace(/\s+/g, "").toUpperCase();
                }
                return updated;
            });
            // Remove advanced settings from YAML output
            const { name, advanced, ...dataWithoutAdvanced } = data;


            // 3) Convert final object to YAML
            const yamlString = YAML.stringify(dataWithoutAdvanced);

            // 4) Hypothetical backend call
            await createFold(name, yamlString, {
                userType,
                startFoldJob: data.advanced?.startFoldJob,
                emailOnCompletion: data.advanced?.emailOnCompletion,
                skipDuplicateEntries: data.advanced?.skipDuplicateEntries,
                disableRelaxation: data.advanced?.disableRelaxation,
            });

            alert("Fold successfully submitted!");
            navigate("/");
        } catch (err) {
            console.error(err);
            alert(`Failed to submit fold: ${String(err)}`);
        } finally {
            setIsSubmitting(false);
        }
    }

    // Uniforms `onValidate` merges custom checks with AJV checks
    function handleValidate(model: NewFoldFormData, error: any) {
        // `error.details` is an array of existing AJV-based errors
        // We'll push additional errors for protein vs ligand logic
        if (!error?.details) {
            error = { details: [] }
        }
        additionalChecks(model, error.details);
        console.log(`ERROR DETAILS TODO: ${error.details}`);

        // If we added new errors, return them so Uniforms displays them
        return error.details.length === 0 ? null : error;
    }

    return (
        <div style={{ margin: "1rem", overflowY: "auto" }}>
            <h2>New Fold (Boltz) - Uniforms</h2>
            <hr />

            {userType === "viewer" && (
                <div style={{ color: "red", marginBottom: "1rem" }}>
                    You do not have permissions to submit folds on this instance.
                </div>
            )}

            {/*
        We'll use <AutoForm> from uniforms-antd, bridging via uniformsSchema.
        Instead of manually listing every field, you can either:
          - Use <AutoFields /> to auto-render everything, or
          - Manually place fields in columns. 
        Here, we demonstrate a 2-column layout using Ant Design's Grid. 
      */}
            <AutoForm
                schema={uniformsSchema}
                model={formData}
                onChangeModel={(m: Record<string, any>) => setFormData(m as NewFoldFormData)}
                onSubmit={(m: Record<string, any>) => handleSubmit(m as NewFoldFormData)}
                onValidate={handleValidate}
                disabled={userType === "viewer"}
            >
                <Row gutter={24}>
                    <Col span={12}>
                        <AutoField name="name" />
                        {/* Single field example: version */}
                        {/* sequences - you can rely on the built-in ArrayField / NestField,
                or do more custom rendering. For brevity, let's just let <AutoField> handle it. */}
                        <AutoField name="sequences" />
                    </Col>
                    <Col span={12}>
                        <AutoField name="constraints" />
                        <AutoField name="version" />
                        <AutoField name="advanced" />
                    </Col>
                </Row>

                {/* Display form-level errors */}
                <ErrorsField />

                <div style={{ marginTop: "1rem" }}>
                    <SubmitField
                        disabled={isSubmitting || userType === "viewer"}
                        value={isSubmitting ? "Submitting..." : "Submit Fold"}
                    />
                </div>
            </AutoForm>
        </div>
    );
};

export default NewFoldView;