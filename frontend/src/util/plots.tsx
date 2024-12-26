import * as d3 from "d3";
import React from "react";
import Plot from "react-plotly.js";
import { Annotations } from "../services/backend.service";
import { RdYlBu } from "./color";
// const NGL = require("./../../node_modules/ngl/dist/ngl");

// // Option A: Import from the package directly
// import NGL from 'ngl';

// // or Option B: if you must point to a dist build (and it supports ESM)
// import NGL from 'ngl/dist/ngl.js';

// or if it doesn’t have a default export, you may need
import * as NGL from 'ngl/dist/ngl.js';

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
  zmax: number | undefined
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
        residue matrix size ({residueMatrix.length}).
      </div>
    );
  }

  const blockVals: number[][] = new Array(chainNames.length);
  for (var i = 0; i < chainNames.length; i++) {
    blockVals[i] = new Array(chainNames.length).fill(null);
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

      if (minOrMax === "min") {
        const priorValue =
          blockVals[rowChainIdx][colChainIdx] === undefined
            ? Infinity
            : blockVals[rowChainIdx][colChainIdx];
        blockVals[rowChainIdx][colChainIdx] = Math.min(priorValue, val);
      } else if (minOrMax === "max") {
        const priorValue =
          blockVals[rowChainIdx][colChainIdx] === undefined
            ? -Infinity
            : blockVals[rowChainIdx][colChainIdx];
        blockVals[rowChainIdx][colChainIdx] = Math.max(priorValue, val);
      } else {
        console.log(`Invalid minOrMax ${minOrMax}`);
      }
    });
  });

  const boundaryAxis1 = Array<string | null>();
  const boundaryAxis2 = Array<string | null>();
  if (!isMonomer) {
    for (var resi = 1; resi < chainNames.length; ++resi) {
      boundaryAxis1.push(`${chainNames[resi]}/1`);
      boundaryAxis1.push(`${chainNames[resi]}/1`);
      boundaryAxis1.push(null);

      boundaryAxis2.push(`${chainNames[0]}/1`);
      boundaryAxis2.push(
        `${chainNames[chainNames.length - 1]}/${
          chainLengths[chainNames.length - 1]
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
        x: rowIdx,
        y: colIdx,
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
      />
    </span>
  );
};

export class VariousColorSchemes {
  nglColorscheme: string;
  sVCoverage: object[][];
  sVLegend: object[][];
  constructor(
    nglColorscheme: string,
    sVCoverage: object[][],
    sVLegend: object[][]
  ) {
    this.nglColorscheme = nglColorscheme;
    this.sVCoverage = sVCoverage;
    this.sVLegend = sVLegend;
  }
}

export const getColorsForAnnotations = (
  sequence: string,
  annotations: Annotations
): VariousColorSchemes => {
  const chainIDs: string[] = [];
  const domains: {
    start: number;
    end: number;
    chainID: string;
    type: string;
  }[] = [];
  sequence.split(";").forEach((rawChain, idx) => {
    var chain;
    if (sequence.includes(";")) {
      chain = rawChain.split(":")[0];
    } else {
      const chainNamesInAnnotation = Object.keys(annotations);
      if (chainNamesInAnnotation.length !== 1) {
        console.log(
          `The annotation file should have one chain name, found ${chainNamesInAnnotation}!`
        );
        chain = "INVALID CHAIN NAME";
      } else {
        chain = chainNamesInAnnotation[0];
      }
    }
    const chainID = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[idx];
    chainIDs.push(chainID);
    if (!annotations[chain]) {
      console.log(`chain does not have annotations ${chain}`);
      return;
    }
    annotations[chain].forEach((domain) => {
      domains.push({
        start: domain.start,
        end: domain.end,
        chainID: chainID,
        type: domain.type,
      });
    });
  });

  const colors = RdYlBu.get(domains.length < 11 ? domains.length : 11) || [];

  const nglViewerColors: string[][] = [];
  const sequenceViewerCoverage: object[][] = [];
  const sequenceViewerLegend: object[][] = [];

  const whiten = (c: d3.RGBColor, alpha: number): d3.RGBColor => {
    return d3.rgb(
      (1.0 - alpha) * c.r + alpha * 255.0,
      (1.0 - alpha) * c.g + alpha * 255.0,
      (1.0 - alpha) * c.b + alpha * 255.0
    );
  };

  domains.forEach((domain, idx) => {
    const color = whiten(d3.rgb(colors[idx % colors.length]), 0.3).toString();
    nglViewerColors.push([
      color,
      `${domain.start}-${domain.end}:${domain.chainID}`,
    ]);
  });
  chainIDs.forEach((chainID, chainIdx) => {
    const chainCoverage: object[] = [];
    const chainLegend: object[] = [];
    domains.forEach((domain, domainIdx) => {
      if (domain.chainID !== chainID) {
        return;
      }
      const color = colors[domainIdx % colors.length];
      const bgColor = whiten(d3.rgb(color), 0.5); //.copy({opacity: 0.5});
      chainCoverage.push({
        start: domain.start,
        end: domain.end,
        bgcolor: bgColor.toString(),
        tooltip: domain.type,
      });
      chainLegend.push({
        name: domain.type,
        color: bgColor.toString(),
      });
    });
    sequenceViewerCoverage.push(chainCoverage);
    sequenceViewerLegend.push(chainLegend);
  });

  console.log(NGL);
  const nglColorscheme = NGL.ColormakerRegistry.addSelectionScheme(
    nglViewerColors,
    "pfam_colors"
  );

  return new VariousColorSchemes(
    nglColorscheme,
    sequenceViewerCoverage,
    sequenceViewerLegend
  );
};
