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
import { Row, Col, Button, Input } from 'antd';
import { BoltzYamlHelper } from './boltzYamlHelper';

/** Minimal shape we'll edit in Uniforms (internal model). */
interface BoltzFormModel {
    version: number;
    sequences: Array<{
        entity_type: "protein" | "dna" | "rna" | "ligand";
        id?: string;      // stored as comma-separated chain IDs
        sequence?: string;
        smiles?: string;
        ccd?: string;
        modifications?: Array<{
            position: number;
            ccd: string;
        }>;
    }>;
    constraints?: Array<{
        // bond: {
        // atom1: { // [string, number, string]; // [CHAIN_ID, RES_IDX, ATOM_NAME]
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
        // }
        // atom2: {
        //     chain_id: string;
        //     res_idx: number;
        //     atom_name: string;
        // };
        // };
        // pocket?: {
        //     binder: string;
        //     contacts: Array<[string, number]>; // Array of [CHAIN_ID, RES_IDX]
        // };
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

const { TextArea } = Input;

/** 1) JSON Schema for our simpler internal shape (Uniforms). */
const simpleSchema = {
    title: "Boltz Config Editor",
    type: "object",
    required: ["version", "sequences"],
    properties: {
        version: {
            type: "integer",
            default: 1,
            title: "YAML Version (you probably want 1)",
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
                        title: "Sequence (for protein, DNA, RNA)",
                    },
                    smiles: {
                        type: "string",
                        title: "SMILES (for ligand)",
                    },
                    ccd: {
                        type: "string",
                        title: "CCD code (for ligand)",
                    },
                    modifications: {
                        type: "array",
                        title: "Modifications",
                        items: {
                            type: "object",
                            required: ["position", "ccd"],
                            properties: {
                                position: {
                                    type: "integer",
                                    title: "Position",
                                    minimum: 1,
                                },
                                ccd: {
                                    type: "string",
                                    title: "CCD Code",
                                },
                            },
                        },
                    },
                },
            },
        },
        constraints: {
            type: "array",
            title: "Constraints",
            items: {
                type: "object",
                properties: {
                    constraint_type: {
                        type: "string",
                        title: "Constraint Type",
                        enum: ["bond", "pocket"]
                    },
                    bond_chain_id_1: {
                        type: "string",
                        title: "Chain ID 1"
                    },
                    bond_res_idx_1: {
                        type: "integer",
                        title: "Residue Index 1",
                        minimum: 1,
                    },
                    bond_atom_name_1: {
                        type: "string",
                        title: "Atom Name 1"
                    },
                    bond_chain_id_2: {
                        type: "string",
                        title: "Chain ID 2"
                    },
                    bond_res_idx_2: {
                        type: "integer",
                        title: "Residue Index 2",
                        minimum: 1,
                    },
                    bond_atom_name_2: {
                        type: "string",
                        title: "Atom Name 2"
                    },
                    binder: {
                        type: "string",
                        title: "Binder"
                    },
                    contacts: {
                        type: "array",
                        title: "Contacts",
                        items: {
                            type: "object",
                            properties: {
                                chain_id: {
                                    type: "string",
                                    title: "Chain ID"
                                },
                                res_idx: {
                                    type: "integer",
                                    title: "Residue Index",
                                    minimum: 1,
                                }
                            }
                        }
                    }
                    // bond: {
                    //     type: "object",
                    //     title: "Bond Constraint",
                    //     properties: {
                    //         atom1: {
                    //             type: "array",
                    //             title: "Atom 1",
                    //             minItems: 3,
                    //             maxItems: 3,
                    //             items: {
                    //                 type: "string",
                    //                 title: "Value"
                    //             }
                    //         },
                    //         atom2: {
                    //             type: "array",
                    //             title: "Atom 2",
                    //             minItems: 3,
                    //             maxItems: 3,
                    //             items: {
                    //                 type: "string",
                    //                 title: "Value"
                    //             }
                    //         }
                    //     }
                    // },
                    // pocket: {
                    //     type: "object",
                    //     title: "Pocket Constraint",
                    //     properties: {
                    //         binder: {
                    //             type: "string",
                    //             title: "Binder"
                    //         },
                    //         contacts: {
                    //             type: "array",
                    //             title: "Contacts",
                    //             items: {
                    //                 type: "array",
                    //                 minItems: 2,
                    //                 maxItems: 2,
                    //                 items: {
                    //                     type: "string",
                    //                     title: "Value"
                    //                 }
                    //             }
                    //         }
                    //     }
                    // }
                }
            }
        }
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
    if (!model.sequences || !Array.isArray(model.sequences) || model.sequences.length === 0) {
        errors.details.push({
            name: "sequences",
            message: "Sequences are required"
        });
        return;
    }
    model.sequences?.forEach((seq: any, index: number) => {
        if (!seq) {
            errors.details.push({
                name: `sequences.${index}`,
                message: "Sequence is invalid"
            });
            return;
        }
        // Add chain ID length validation
        if (seq.id) {
            const chainIds = seq.id.split(',').map((id: string) => id.trim());
            const invalidChainIds = chainIds.filter((id: string) => id.length > 5);
            if (invalidChainIds.length > 0) {
                errors.details.push({
                    name: `sequences.${index}.id`,
                    message: `Boltz chain IDs must be 5 or fewer characters. We do not know why. Invalid IDs: ${invalidChainIds.join(', ')}`
                });
            }
        }

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

            if (hasSmiles) {
                // Check if the SMILES string has any whitespace.
                if (/\s/.test(seq.smiles)) {
                    errors.details.push({
                        name: `sequences.${index}.smiles`,
                        message: 'SMILES cannot contain whitespace'
                    });
                }
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
 * 3) Convert a BoltzYamlHelper instance to our simpler Uniforms model.
 */
function fromBoltzObjectToModel(helper: BoltzYamlHelper): BoltzFormModel {
    const version = helper.getVersion() ?? 1;

    // Transform sequences
    const sequences = helper.getAllSequences().map(seq => {
        if (seq.entity_type === "protein" || seq.entity_type === "dna" || seq.entity_type === "rna") {
            return {
                entity_type: seq.entity_type,
                id: seq.id.join(", "), // turn array into comma string
                sequence: seq.sequence,
                modifications: seq.modifications || [],
            };
        } else if (seq.entity_type === "ligand" && seq.smiles) {
            return {
                entity_type: seq.entity_type,
                id: seq.id.join(", "), // turn array into comma string
                smiles: seq.smiles,
                modifications: seq.modifications || [],
            };
        } else if (seq.entity_type === "ligand" && seq.ccd) {
            return {
                entity_type: seq.entity_type,
                id: seq.id.join(", "), // turn array into comma string
                ccd: seq.ccd,
                modifications: seq.modifications || [],
            };
        }
        throw new Error(`Invalid entity type in sequence: ${seq.entity_type}`);
    }).filter(Boolean);

    // Transform constraints
    const constraints = helper.getNormalizedConstraints();

    return {
        version,
        sequences,
        constraints
    };
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
            if (!seq?.id || !seq?.entity_type) {
                return {};
            }
            // Turn "A, B" into ["A", "B"]
            const idArray = (seq.id || "")
                .split(",")
                .map((x) => x.trim())
                .filter(Boolean);

            const baseData: any = {
                id: idArray,
            };

            // Add modifications if present and non-empty
            if (seq.modifications?.length) {
                baseData.modifications = seq.modifications;
            }

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
                        sequence: seq.sequence || "",
                    },
                };
            } else if (seq.entity_type === "rna") {
                return {
                    rna: {
                        id: idArray,
                        sequence: seq.sequence || "",
                    },
                };
            }
            // fallback
            return {};
        }),
    };

    // Add constraints if present
    if (model.constraints?.length) {
        boltzObj.constraints = model.constraints.map(constraint => {
            if (constraint.constraint_type === "bond") {
                return {
                    bond: {
                        atom1: [
                            constraint.bond_chain_id_1,
                            constraint.bond_res_idx_1,
                            constraint.bond_atom_name_1
                        ],
                        atom2: [
                            constraint.bond_chain_id_2,
                            constraint.bond_res_idx_2,
                            constraint.bond_atom_name_2
                        ]
                    }
                };
            } else if (constraint.constraint_type === "pocket") {
                return {
                    pocket: {
                        binder: constraint.binder,
                        contacts: constraint.contacts?.map(contact => [
                            contact.chain_id,
                            contact.res_idx
                        ])
                    }
                };
            }
            return {};
        });
    }

    return YAML.stringify(boltzObj);
}

/** Custom sequence field with canonicalization */
const SequenceField = connectField((props: any) => {
    const { onChange, value } = props;

    const canonicalize = () => {
        if (!value) return;
        // Remove whitespace, trailing *, and capitalize
        const canonicalized = value
            .replace(/\s+/g, '')  // remove whitespace
            .replace(/\*$/, '')   // remove trailing *
            .toUpperCase();       // capitalize
        onChange(canonicalized);
    };

    return (
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
            <TextArea
                value={value || ''}
                onChange={(e) => onChange(e.target.value)}
                placeholder="Enter sequence"
                rows={4}
                style={{ fontFamily: 'monospace', flex: 1 }}
            />
            <Button
                onClick={canonicalize}
                title="Remove whitespace, trailing *, and capitalize"
            >
                Canonicalize
            </Button>
        </div>
    );
});

/** 
 * A small helper that conditionally shows fields based on entity type
 */
const EntityTypeConditionalFields = connectField((props: { value: BoltzFormModel['sequences'][number] }) => {
    if (!props.value) return null;

    const entityType = props.value.entity_type;

    if (entityType === "protein" || entityType === "dna" || entityType === "rna") {
        return (
            <>
                <SequenceField name="sequence" />
                <ListField name="modifications">
                    <ListItemField name="$">
                        {/* <AutoField name="position" />
                        <AutoField name="ccd" /> */}
                    </ListItemField>
                </ListField>
            </>
        );
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

const ConstraintTypeConditionalFields = connectField((props: { value: BoltzFormModel['constraints'][number] }) => {
    if (!props.value) return null;

    const constraintType = props.value.constraint_type;

    if (constraintType === "bond") {
        return <>
            <AutoField name="bond_chain_id_1" />
            <AutoField name="bond_res_idx_1" />
            <AutoField name="bond_atom_name_1" />
            <AutoField name="bond_chain_id_2" />
            <AutoField name="bond_res_idx_2" />
            <AutoField name="bond_atom_name_2" />
        </>;
    } else if (constraintType === "pocket") {
        return <>
            <AutoField name="binder" />
            <ListField name="contacts">
                <ListItemField name="$">
                    <AutoField name="chain_id" />
                    <AutoField name="res_idx" />
                </ListItemField>
            </ListField>
        </>;
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
    let initialModel: BoltzFormModel = {
        version: 1,
        sequences: [],
        constraints: [] // Initialize empty constraints array
    };

    if (initialYaml) {
        try {
            const helper = new BoltzYamlHelper(initialYaml);
            initialModel = fromBoltzObjectToModel(helper);
        } catch (err) {
            console.warn("Failed to parse initial YAML. Starting empty.", err);
        }
    }

    const [model, setModel] = useState<BoltzFormModel>(initialModel);
    const [showYamlEditor, setShowYamlEditor] = useState(false);
    const [yamlText, setYamlText] = useState(initialYaml || '');

    /** On submit, transform to final Boltz YAML and invoke onSave() callback. */
    function handleSubmit(submitted: BoltzFormModel) {
        try {
            const yaml = toBoltzYaml(submitted);
            setYamlText(yaml); // Keep YAML view in sync
            onSave?.(yaml);
        } catch (error) {
            console.error('Failed to save:', error);
        }
    }

    // Add handler for YAML text changes
    const handleYamlChange = (newYaml: string) => {
        setYamlText(newYaml);
        try {
            const helper = new BoltzYamlHelper(newYaml);
            const newModel = fromBoltzObjectToModel(helper);
            setModel(newModel);
        } catch (err) {
            console.warn("Invalid YAML", err);
            // Don't update model if YAML is invalid
        }
    };

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
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h2>Boltz YAML Editor</h2>
                <Button
                    onClick={() => setShowYamlEditor(!showYamlEditor)}
                    type={showYamlEditor ? 'primary' : 'default'}
                >
                    {showYamlEditor ? 'Show Form Editor' : 'Show YAML Editor'}
                </Button>
            </div>

            {showYamlEditor ? (
                <div>
                    <TextArea
                        value={yamlText}
                        onChange={(e) => handleYamlChange(e.target.value)}
                        rows={20}
                        style={{ fontFamily: 'monospace' }}
                    />
                    <Button
                        type="primary"
                        onClick={() => onSave?.(yamlText)}
                        style={{ marginTop: '1rem' }}
                    >
                        Save YAML
                    </Button>
                </div>
            ) : (
                <AutoForm
                    schema={schemaBridge}
                    model={model}
                    onChangeModel={(m: Record<string, any>) => {
                        const newModel = m as BoltzFormModel;
                        setModel(newModel);
                        setYamlText(toBoltzYaml(newModel)); // Keep YAML view in sync
                    }}
                    onSubmit={(m: Record<string, any>) => handleSubmit(m as BoltzFormModel)}
                    showInlineError
                    onValidate={handleValidate}
                >
                    <Row gutter={24}>
                        <Col xs={24} xl={12}>
                            {/* Main sequence editor */}
                            <AutoField name="version" />

                            <ListField name="sequences">
                                <ListItemField name="$">
                                    <div style={{ backgroundColor: '#f8f8f8', border: '1px solid #a0a0a0', padding: "6px", borderRadius: "8px", marginBottom: "1rem" }}>
                                        <AutoField name="id" />
                                        <AutoField name="entity_type" />
                                        <EntityTypeConditionalFields name="" />
                                    </div>
                                </ListItemField>
                            </ListField>
                        </Col>

                        <Col xs={24} xl={12}>
                            {/* Constraints editor */}
                            <div style={{
                                backgroundColor: "#f5f5f5",
                                padding: "1rem",
                                borderRadius: "8px",
                                marginBottom: "1rem"
                            }}>
                                <h3>Constraints</h3>
                                <ListField name="constraints">
                                    <ListItemField name="$">
                                        <AutoField name="constraint_type" />
                                        <ConstraintTypeConditionalFields name="" />
                                        {/* Bond constraint */}
                                        {/* <h4>Bond Constraint</h4>
                                        <AutoField name="bond.atom1" />
                                        <AutoField name="bond.atom2" /> */}

                                        {/* Pocket constraint */}
                                        {/* <h4>Pocket Constraint</h4>
                                        <AutoField name="pocket.binder" />
                                        <ListField name="pocket.contacts">
                                            <ListItemField name="$" />
                                        </ListField> */}
                                    </ListItemField>
                                </ListField>
                            </div>
                        </Col>
                    </Row>

                    {/* Display form-level errors */}
                    <ErrorsField />

                    <div style={{ marginTop: "1rem" }}>
                        <SubmitField value="Save Boltz YAML" />
                    </div>
                </AutoForm>
            )}
        </div>
    );
};

export default BoltzYamlBuilder;