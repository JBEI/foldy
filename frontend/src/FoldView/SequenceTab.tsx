import React from "react";
import {
  EditableTagList,
} from "../Util";
import {
  VariousColorSchemes,
} from "../helpers/plots";
const ReactSequenceViewer = require("react-sequence-viewer");

interface SequenceTabProps {
  foldId: number;
  foldName: string;
  foldTags: string[];
  foldOwner: string;
  foldModelPreset: string | null;
  foldDisableRelaxation: boolean | null;
  sequence: string;
  colorScheme: string;
  antismashColors: VariousColorSchemes | null;
  pfamColors: VariousColorSchemes | null;

  addTag: (tagToAdd: string) => void;
  deleteTag: (tagToDelete: string) => void;
  handleTagClick: (tagToOpen: string) => void;
}

const SequenceTab = React.memo(
  (props: SequenceTabProps) => {
    const getSequenceViewerCoverage = (chainIdx: number) => {
      if (props.colorScheme === "antismash") {
        return props.antismashColors?.sVCoverage
          ? props.antismashColors.sVCoverage[chainIdx]
          : [];
      } else if (props.colorScheme === "pfam") {
        return props.pfamColors?.sVCoverage
          ? props.pfamColors.sVCoverage[chainIdx]
          : [];
      } else {
        return [];
      }
    };

    const getSequenceViewerLegend = (chainIdx: number) => {
      if (props.colorScheme === "antismash") {
        return props.antismashColors?.sVLegend
          ? props.antismashColors.sVLegend[chainIdx]
          : [];
      } else if (props.colorScheme === "pfam") {
        return props.pfamColors?.sVLegend
          ? props.pfamColors.sVLegend[chainIdx]
          : [];
      } else {
        return [];
      }
    };

    return (
      <div>
        {props.sequence.split(";").map((ss: string, idx: number) => {
          var subseq, subseqName;
          if (ss.includes(":")) {
            subseqName = ss.split(":")[0];
            subseq = ss.split(":")[1];
          } else {
            subseqName = props.foldName;
            subseq = ss;
          }
          return (
            <ReactSequenceViewer.default
              key={idx}
              sequence={subseq}
              title={subseqName}
              badge={false}
              id={idx.toString() + "rsv"}
              charsPerLine={50}
              wrapAminoAcids={true}
              coverage={getSequenceViewerCoverage(idx)}
              legend={getSequenceViewerLegend(idx)}
            />
          );
        })}
        <form className="uk-form-horizontal uk-margin-large">
          <div className="uk-margin">
            <label className="uk-form-label" htmlFor="form-horizontal-text">
              Owner
            </label>
            <div className="uk-form-controls">
              <input
                className="uk-input uk-form-width-large"
                id="form-horizontal-text"
                type="text"
                value={props.foldOwner}
                disabled
              />
            </div>
          </div>
          <div className="uk-margin">
            <label className="uk-form-label" htmlFor="form-horizontal-text">
              Tags
            </label>
            <div className="uk-form-controls">
              <EditableTagList
                tags={props.foldTags || []}
                addTag={props.addTag}
                deleteTag={props.deleteTag}
                handleTagClick={props.handleTagClick}
              />
            </div>
          </div>
          <div className="uk-margin">
            <label className="uk-form-label" htmlFor="form-horizontal-text">
              Model Preset
            </label>
            <div className="uk-form-controls">
              <input
                className="uk-input uk-form-width-large"
                id="form-horizontal-text"
                type="text"
                value={props.foldModelPreset || "unset"}
                disabled
              />
            </div>
          </div>
          <div className="uk-margin">
            <label className="uk-form-label" htmlFor="form-horizontal-text">
              Disable Relaxation
            </label>
            <div className="uk-form-controls">
              <input
                className="uk-input uk-form-width-large"
                id="form-horizontal-text"
                type="text"
                value={`${props.foldDisableRelaxation}`}
                disabled
              />
            </div>
          </div>
        </form>
      </div>
    );
  },
  (prevProps: SequenceTabProps, nextProps: SequenceTabProps) => {
    return (
      prevProps.foldName === nextProps.foldName &&
      prevProps.foldTags.length === nextProps.foldTags.length &&
      prevProps.foldTags.every((ee, ii) => nextProps.foldTags[ii] === ee) &&
      prevProps.foldDisableRelaxation === nextProps.foldDisableRelaxation &&
      prevProps.foldOwner === nextProps.foldOwner &&
      prevProps.foldModelPreset === nextProps.foldModelPreset &&
      prevProps.sequence === nextProps.sequence &&
      prevProps.colorScheme === nextProps.colorScheme &&
      prevProps.antismashColors === nextProps.antismashColors &&
      prevProps.pfamColors === nextProps.pfamColors
    );
  }
);

export default SequenceTab;
