import React, { useState } from "react";
import UIkit from "uikit";
import { Button, Select, Input, Form, Checkbox, Row, Col, Space } from 'antd';
import { postDock } from "../api/dockApi";
import { DockInput } from "../types/types";
import { notify } from "../services/NotificationService";

const { TextArea } = Input;

const getLigandNameErrorMessage = (ligandName: string | null) => {
    if (!ligandName) {
        return null;
    }
    const foldNameIsValid = ligandName.match(/^[0-9a-zA-Z]+$/);
    if (!foldNameIsValid) {
        return "Must be alphanumeric.";
    }
    return null;
};

const validateAndSetInput = (
    e: HTMLInputElement,
    validationFn: (input: string | null) => string | null,
    setFn: (input: string | null) => void,
    input: string | null
) => {
    const invalidMessage = validationFn(input);
    if (invalidMessage) {
        e.setCustomValidity(invalidMessage);
        e.reportValidity();
    } else {
        e.setCustomValidity("");
    }

    setFn(input);
};

const getBBResidueErrorMessage = (bboxResidue: string | null) => {
    if (!bboxResidue) {
        return null;
    }
    const bboxResidueIsValid = bboxResidue.match(/^[A-Z][0-9]+$/);
    if (!bboxResidueIsValid) {
        return "Must be of the form <amino acid><index>, like W81.";
    }
    return null;
};

interface newDockTextboxInterface {
    foldIds: number[];
    existingLigands: { [foldId: number]: Array<string> };
}

const checkForExistingLigands = (
    newLigandName: string,
    existingLigands: { [foldId: number]: Array<string> }
) => {
    const matchingFolds = [];

    for (const foldId in existingLigands) {
        const ligandList = existingLigands[foldId];
        if (ligandList.includes(newLigandName)) {
            matchingFolds.push(foldId);
        }
    }

    return matchingFolds.length > 0 ? matchingFolds : null;
};

export function NewDockPrompt(props: newDockTextboxInterface) {
    const [toolName, setToolName] = useState<string>("");
    const [ligandName, setLigandName] = useState<string | null>(null);
    const [ligandSmiles, setLigandSmiles] = useState<string | null>(null);
    const [boundingBoxResidue, setBoundingBoxResidue] = useState<string | null>(
        null
    );
    const [boundingBoxRadiusAngstrom, setBoundingBoxRadiusAngstrom] = useState<
        string | null
    >(null);
    const [overrideExistingLigands, setOverrideExistingLigands] =
        useState<boolean>(false);
    const [showTextbox, setShowTextbox] = useState<boolean>(false);
    const [textboxContents, setTextboxContents] = useState<string | null>(null);

    const submitTextboxDocks = () => {
        if (!textboxContents) {
            notify.warning("Must provide a smiles string and ligand name!");
            return;
        }

        const errors: string[] = [];
        const newDocks: DockInput[] = [];

        textboxContents.split("\n").forEach((lineContents, lineNumber) => {
            const lineItems = lineContents.split(",");

            var bounding_box_residue: string | null = null;
            var bounding_box_radius_angstrom: number | null = null;
            if (lineItems.length === 4) {
                bounding_box_residue = lineItems[2];
                bounding_box_radius_angstrom = parseFloat(lineItems[3]);
            } else if (lineItems.length !== 2) {
                errors.push(
                    `Lines can have either two or four arguments, got ${lineContents}`
                );
                return;
            }
            const name = lineItems[0].trim();
            const smiles = lineItems[1].trim();

            if (!name.match(/^[0-9a-zA-Z]+$/)) {
                errors.push(`All ligand names must be alphanumeric, "${name}" is not`);
                return;
            }

            if (toolName === "") {
                errors.push("Must select a docking tool.");
                return;
            }

            const foldsWithExistingLigand = checkForExistingLigands(
                name,
                props.existingLigands
            );
            if (foldsWithExistingLigand && !overrideExistingLigands) {
                errors.push(
                    `This would overwrite the "${name}" ligand for ${foldsWithExistingLigand.length} folds, including for ${foldsWithExistingLigand[0]}. Aborting.`
                );
            }

            props.foldIds.forEach((foldId) => {
                newDocks.push({
                    fold_id: foldId,
                    ligand_name: name,
                    ligand_smiles: smiles,
                    tool: toolName,
                    bounding_box_residue: bounding_box_residue,
                    bounding_box_radius_angstrom: bounding_box_radius_angstrom,
                });
            });
        });

        console.log(errors);

        if (errors.length) {
            notify.warning(errors.join("\n"));
            return;
        }

        newDocks.forEach((newDock) => {
            postDock(newDock).then(
                () => {
                    notify.success(
                        `Successfully started docking run for ${newDock.ligand_name}`
                    );
                },
                (e) => {
                    notify.error(`Docking ${newDock.ligand_name} failed: ${e}`);
                }
            );
        });
    };

    const submitFormDocks = () => {
        if (!ligandName || !ligandSmiles) {
            notify.warning("Must provide a ligand name and SMILES string!");
            return;
        }

        if (!ligandName.match(/^[0-9a-zA-Z]+$/)) {
            notify.warning(
                `All ligand names must be alphanumeric, "${ligandName}" is not`
            );
            return;
        }

        if (toolName === "") {
            notify.warning("Must select a docking tool.");
            return;
        }

        const foldsWithExistingLigand = checkForExistingLigands(
            ligandName,
            props.existingLigands
        );
        if (foldsWithExistingLigand && !overrideExistingLigands) {
            notify.warning(
                `This would overwrite the "${ligandName}" ligand for ${foldsWithExistingLigand.length} folds, including for ${foldsWithExistingLigand[0]}. Aborting.`
            );
            return;
        }

        const newDocks: DockInput[] = [];
        props.foldIds.forEach((foldId) => {
            newDocks.push({
                fold_id: foldId,
                ligand_name: ligandName,
                ligand_smiles: ligandSmiles,
                tool: toolName,
                bounding_box_residue: boundingBoxResidue,
                bounding_box_radius_angstrom: boundingBoxRadiusAngstrom
                    ? parseFloat(boundingBoxRadiusAngstrom)
                    : null,
            });
        });

        newDocks.forEach((newDock) => {
            postDock(newDock).then(
                () => {
                    notify.success(
                        `Successfully started docking run for ${newDock.ligand_name}`
                    );
                },
                (e) => {
                    notify.error(`Docking ${newDock.ligand_name} failed: ${e}`);
                }
            );
        });
    };

    const runDocks = () => {
        if (showTextbox) {
            submitTextboxDocks();
        } else {
            submitFormDocks();
        }
    };

    return (
        <div>
            <div style={{ marginBottom: "16px" }}>
                <Select
                    placeholder="Select a docking program..."
                    style={{ width: "100%" }}
                    onChange={(value) => setToolName(value)}
                    value={toolName || undefined}
                    options={[
                        { value: "vina", label: "Docking with Autodock Vina" },
                        { value: "diffdock", label: "Docking with Diffdock" },
                    ]}
                />
            </div>

            {showTextbox ? (
                <TextArea
                    rows={5}
                    placeholder={
                        "ligand1_name,ligand1_smiles[,bbox_residue,bbox_radius]\nligand2_name,ligand2_smiles[,bbox_residue,bbox_radius]"
                    }
                    style={{ fontFamily: 'consolas,"Liberation Mono",courier,monospace', marginBottom: "16px" }}
                    value={textboxContents || ""}
                    onChange={(e) => setTextboxContents(e.target.value)}
                />
            ) : (
                <div style={{ marginBottom: "16px" }}>
                    <Row gutter={[16, 16]}>
                        <Col span={24}>
                            <Input
                                placeholder="Ligand Name"
                                value={ligandName || ""}
                                status={getLigandNameErrorMessage(ligandName) ? "error" : ""}
                                onChange={(e) => {
                                    const value = e.target.value;
                                    const errorMessage = getLigandNameErrorMessage(value);
                                    setLigandName(value);
                                }}
                            />
                            {getLigandNameErrorMessage(ligandName) && (
                                <div style={{ color: "#ff4d4f", fontSize: "12px", marginTop: "4px" }}>
                                    {getLigandNameErrorMessage(ligandName)}
                                </div>
                            )}
                        </Col>
                        <Col span={24}>
                            <Input
                                placeholder="Ligand SMILES"
                                value={ligandSmiles || ""}
                                onChange={(e) => setLigandSmiles(e.target.value)}
                            />
                        </Col>
                        <Col span={12}>
                            <Input
                                placeholder="[Bounding Box Residue Center]"
                                value={boundingBoxResidue || ""}
                                status={getBBResidueErrorMessage(boundingBoxResidue) ? "error" : ""}
                                disabled={toolName === "diffdock"}
                                onChange={(e) => {
                                    const value = e.target.value;
                                    const errorMessage = getBBResidueErrorMessage(value);
                                    setBoundingBoxResidue(value);
                                }}
                            />
                            {getBBResidueErrorMessage(boundingBoxResidue) && (
                                <div style={{ color: "#ff4d4f", fontSize: "12px", marginTop: "4px" }}>
                                    {getBBResidueErrorMessage(boundingBoxResidue)}
                                </div>
                            )}
                        </Col>
                        <Col span={12}>
                            <Input
                                type="number"
                                placeholder="[Bounding Box Radius (Angstroms)]"
                                value={boundingBoxRadiusAngstrom || ""}
                                disabled={toolName === "diffdock"}
                                onChange={(e) => setBoundingBoxRadiusAngstrom(e.target.value)}
                            />
                        </Col>
                        <Col span={24}>
                            <Checkbox
                                checked={overrideExistingLigands || false}
                                onChange={(e) => setOverrideExistingLigands(e.target.checked)}
                            >
                                Override Existing Docking Runs
                            </Checkbox>
                        </Col>
                    </Row>
                </div>
            )}
            <Space>
                <Button
                    onClick={() => setShowTextbox(!showTextbox)}
                >
                    {showTextbox ? "Hide" : "Show"} Bulk Input
                </Button>
                <Button
                    type="primary"
                    onClick={runDocks}
                >
                    Dock
                </Button>
            </Space>
        </div>
    );
}
