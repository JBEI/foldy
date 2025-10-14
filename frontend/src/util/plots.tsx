import * as d3 from "d3";
import React from "react";
import Plot from "react-plotly.js";
import { RdYlBu } from "./color";
// const NGL = require("./../../node_modules/ngl/dist/ngl");

// // Option A: Import from the package directly
// import NGL from 'ngl';

// // or Option B: if you must point to a dist build (and it supports ESM)
// import NGL from 'ngl/dist/ngl.js';

// or if it doesn't have a default export, you may need
// import * as NGL from 'ngl/dist/ngl.js';
import { Annotations } from "../types/types";
import { Selection } from "../components/FoldView/StructurePane";
import { BoltzYamlHelper } from "./boltzYamlHelper";

export const matricesAreEqual = (
    mat1: number[][],
    mat2: number[][]
): boolean => {
    if (mat1.length !== mat2.length) {
        return false;
    }
    if (!mat1.every((row, ii) => row.length === mat2[ii].length)) {
        return false;
    }
    return mat1.every((row, ii) => row.every((val, jj) => mat2[ii][jj] === val));
};

export const getResidueHeatmap = (
    sequence: string,
    residueMatrix: number[][],
    colorscale: string,
    minOrMax: string,
    zmin: number | undefined,
    zmax: number | undefined,
    setSelectedSubsequence?: (selection: Selection | null) => void,
    yamlConfig?: string | undefined
) => {
    const isMonomer = !sequence.includes(";");

    // Per chain metrics.
    const chainNames = Array<string>();
    const chainLengths = Array<number>();

    // Per residue metrics.
    const residueChainIdx = Array<number>();
    const residueNames = Array<string>();
    const residueIsFringe = Array<boolean>();

    if (isMonomer) {
        chainNames.push("A");
        chainLengths.push(sequence.length);
        for (var ii = 0; ii < sequence.length + 1; ii++) {
            residueChainIdx.push(0);
            residueNames.push(`${ii}`);
            residueIsFringe.push(ii < 10 || ii > sequence.length - 10);
        }
    } else {
        sequence.split(";").forEach((chain) => {
            const nameAndSeq = chain.split(":");

            // If it's a monomer! Don't worry about chain names, etc...
            if (nameAndSeq.length !== 2) {
                console.error(
                    `This should not happen, the sequence is malformed: ${chain}`
                );
                return;
            }
            chainNames.push(nameAndSeq[0]);
            chainLengths.push(nameAndSeq[1].length);

            for (var ii = 1; ii < nameAndSeq[1].length + 1; ii++) {
                residueChainIdx.push(chainNames.length - 1);
                residueNames.push(`${nameAndSeq[0]}/${ii}`);
                residueIsFringe.push(ii < 10 || ii > nameAndSeq[1].length - 10);
            }
        });
    }

    const totalSequenceLength = chainLengths.reduce((partial, a) => partial + a);
    if (totalSequenceLength !== residueMatrix.length) {
        return (
            <div className="uk-alert-danger">
                Somehow, the sequence length ({totalSequenceLength}) doesn't match the
                residue matrix size ({residueMatrix.length}) for sequence {sequence}.
            </div>
        );
    }

    const blockVals: number[][] = new Array(chainNames.length);
    for (var i = 0; i < chainNames.length; i++) {
        blockVals[i] = new Array(chainNames.length).fill(minOrMax === "max" ? -Infinity : Infinity);
    }
    // const blockVals = Array.from(
    //   Array(chainNames.length).fill(null),
    //   () => (new Array(chainNames.length)).fill(null)
    // );
    residueMatrix.forEach((row, rowIdx) => {
        row.forEach((val, colIdx) => {
            const rowChainIdx = residueChainIdx[rowIdx];
            const colChainIdx = residueChainIdx[colIdx];

            if (residueIsFringe[rowIdx] || residueIsFringe[colIdx]) {
                return;
            }

            const priorValue = blockVals[rowChainIdx][colChainIdx];
            if (minOrMax === "min") {
                blockVals[rowChainIdx][colChainIdx] = Math.min(priorValue, val);
            } else if (minOrMax === "max") {
                blockVals[rowChainIdx][colChainIdx] = Math.max(priorValue, val);
            } else {
                console.log(`Invalid minOrMax ${minOrMax}`);
            }
        });
    });
    console.log("blockVals", blockVals);
    console.log("residueMatrix", residueMatrix);
    console.log("residueChainIdx", residueChainIdx);
    console.log("residueNames", residueNames);
    console.log("residueIsFringe", residueIsFringe);

    const boundaryAxis1 = Array<string | null>();
    const boundaryAxis2 = Array<string | null>();
    if (!isMonomer) {
        for (var resi = 1; resi < chainNames.length; ++resi) {
            boundaryAxis1.push(`${chainNames[resi]}/1`);
            boundaryAxis1.push(`${chainNames[resi]}/1`);
            boundaryAxis1.push(null);

            boundaryAxis2.push(`${chainNames[0]}/1`);
            boundaryAxis2.push(
                `${chainNames[chainNames.length - 1]}/${chainLengths[chainNames.length - 1]
                }`
            );
            boundaryAxis2.push(null);
        }
    }

    const blockAnnotations: object[] = [];
    blockVals.forEach((row, rowIdx) => {
        row.forEach((val, colIdx) => {
            blockAnnotations.push({
                xref: "x1",
                yref: "y1",
                x: colIdx,
                y: rowIdx,
                text: val.toFixed(2),
                font: {
                    family: "Arial",
                    size: 12,
                    color: "white", // 'rgb(50, 171, 96)'
                },
                showarrow: false,
            });
        });
    });

    return (
        <span>
            {isMonomer ? null : (
                <Plot
                    data={[
                        {
                            x: chainNames,
                            y: chainNames,
                            z: blockVals,
                            type: "heatmap",
                            colorscale: colorscale,
                            zmin: zmin,
                            zmax: zmax,
                        },
                    ]}
                    layout={{
                        title: `${minOrMax} value between chains`,
                        // width: 300,
                        // height: 300,
                        yaxis: {
                            autorange: "reversed",
                            scaleanchor: "x",
                        },
                        annotations: blockAnnotations,
                        margin: {
                            l: 100,
                            r: 40,
                            b: 100,
                            t: 40,
                            pad: 5,
                        },
                    }}
                    useResizeHandler={true}
                    style={{ width: "100%", height: "100%" }}
                />
            )}

            <br></br>
            <Plot
                data={[
                    {
                        x: residueNames,
                        y: residueNames,
                        z: residueMatrix,
                        type: "heatmap",
                        // hovertemplate: 'x:%{x}<br>y:%{y}<br>z:%{z}<br>%{customdata[0]}',
                        colorscale: colorscale,
                        zmin: zmin,
                        zmax: zmax,
                        // labels: {x: 'residue i', y: 'residue j', z: 'amplitude'}
                    },
                    {
                        x: boundaryAxis1.concat(boundaryAxis2), // ['A/1', 'B/401'],
                        y: boundaryAxis2.concat(boundaryAxis1), // ['B/1', 'B/1'],
                        type: "scatter",
                        mode: "lines",
                        line: {
                            color: "white",
                            width: 0.5,
                        },
                    },
                ]}
                layout={{
                    // width: 600,
                    // height: 600,
                    yaxis: {
                        autorange: "reversed",
                        scaleanchor: "x",
                    },
                    margin: {
                        l: 100,
                        r: 40,
                        b: 100,
                        t: 40,
                        pad: 5,
                    },
                }}
                useResizeHandler={true}
                style={{ width: "100%", height: "100%" }}
                onHover={(event) => {
                    if (event.points && event.points.length > 0 && setSelectedSubsequence && yamlConfig) {
                        const configHelper = new BoltzYamlHelper(yamlConfig);

                        const point = event.points[0];
                        // Convert fractional coordinates to discrete indices
                        const xIndex = Math.ceil(point.x as number);
                        const yIndex = Math.ceil(point.y as number);

                        // Get the actual residue names from the arrays
                        if (xIndex >= 0 && xIndex < residueNames.length && yIndex >= 0 && yIndex < residueNames.length) {
                            const xiLabel = residueNames[xIndex];
                            const yiLabel = residueNames[yIndex];

                            // Parse residue numbers from labels (e.g., "A/123" -> 123 or "123" -> 123)
                            const parseResidue = (label: string): { chainId: string, residueNumber: number } | null => {
                                if (isMonomer) {
                                    const chainId = configHelper.getProteinSequences()[0][0];
                                    return { chainId, residueNumber: parseInt(label) };
                                } else {
                                    const match = label.match(/(.*)\/(\d+)$/);
                                    return match ? { chainId: match[1], residueNumber: parseInt(match[2]) } : null;
                                }
                            };

                            const iResidue = parseResidue(xiLabel);
                            const jResidue = parseResidue(yiLabel);

                            if (iResidue && jResidue) {
                                // Get chain ID from yaml config
                                if (configHelper.getProteinSequences().length > 1) {
                                    return; // Don't highlight on multimers for now
                                }

                                setSelectedSubsequence({
                                    data: [
                                        {
                                            struct_asym_id: iResidue.chainId,
                                            start_residue_number: iResidue.residueNumber,
                                            end_residue_number: iResidue.residueNumber,
                                            color: "#FFD700", // Gold color for hover
                                        },
                                        {
                                            struct_asym_id: jResidue.chainId,
                                            start_residue_number: jResidue.residueNumber,
                                            end_residue_number: jResidue.residueNumber,
                                            color: "#FFD700", // Gold color for hover
                                        }
                                    ],
                                    nonSelectedColor: "white",
                                });
                            }
                        }
                    }
                }}
                onUnhover={() => {
                    if (setSelectedSubsequence) {
                        setSelectedSubsequence(null);
                    }
                }}
            />
        </span>
    );
};

export interface SequenceAnnotation {
    start: number;
    end: number;
    bgcolor: string;
    tooltip: string;
}
