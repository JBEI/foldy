import React from "react";
import { EditableTagList } from "../../util/editableTagList";
import { VariousColorSchemes } from "../../util/plots";
import { AiFillEdit } from "react-icons/ai";
const ReactSequenceViewer = require("react-sequence-viewer");

export interface SubsequenceSelection {
  chainIdx: number;
  startResidue: number;
  endResidue: number;
  subsequence: string;
}

interface SequenceTabProps {
  foldId: number;
  foldName: string;
  foldTags: string[];
  foldOwner: string;
  foldCreateDate: string;
  foldPublic: boolean | null;
  foldModelPreset: string | null;
  foldDisableRelaxation: boolean | null;
  sequence: string;
  colorScheme: string;
  pfamColors: VariousColorSchemes | null;

  setPublic: (is_public: boolean) => void;
  setFoldName: () => void;
  addTag: (tagToAdd: string) => void;
  deleteTag: (tagToDelete: string) => void;
  handleTagClick: (tagToOpen: string) => void;

  setSelectedSubsequence: (sele: SubsequenceSelection) => void;

  userType: string | null;
}

const SequenceTab = React.memo(
  (props: SequenceTabProps) => {
    const getSequenceViewerCoverage = (chainIdx: number) => {
      if (props.colorScheme === "pfam") {
        return props.pfamColors?.sVCoverage
          ? props.pfamColors.sVCoverage[chainIdx]
          : [];
      } else {
        return [];
      }
    };

    const getSequenceViewerLegend = (chainIdx: number) => {
      if (props.colorScheme === "pfam") {
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
          var chainSeq;
          var chainName: string;
          if (ss.includes(":")) {
            chainName = ss.split(":")[0];
            chainSeq = ss.split(":")[1];
          } else {
            chainName = props.foldName;
            chainSeq = ss;
          }
          return (
            <ReactSequenceViewer.default
              key={idx}
              sequence={chainSeq}
              title={chainName}
              badge={false}
              id={idx.toString() + "rsv"}
              charsPerLine={50}
              wrapAminoAcids={true}
              coverage={getSequenceViewerCoverage(idx)}
              legend={getSequenceViewerLegend(idx)}
              onMouseSelection={(sele: {
                detail: {
                  start: number;
                  end: number;
                  selection: string;
                };
              }) => {
                console.log(`on mouse selection ${sele}`);
                console.log(sele);
                // have access to .detail.{selection, start, end}
                props.setSelectedSubsequence({
                  chainIdx: idx,
                  startResidue: sele.detail.start,
                  endResidue: sele.detail.end,
                  subsequence: sele.detail.selection,
                });
              }}
              // onSubpartSelected={(sele: string) => {
              //   console.log(`on subpart selected ${sele}`);
              // }}
            />
          );
        })}
        <form className="uk-form-horizontal uk-margin-large">
          <div className="uk-margin">
            <label className="uk-form-label" htmlFor="form-horizontal-text">
              Name
            </label>
            <div className="uk-form-controls">
              <input
                className="uk-input uk-width-3-4"
                value={props.foldName}
                disabled
              ></input>
              <span className="uk-width-1-4">
                <button
                  className="uk-button uk-button-default uk-width-auto uk-margin-small-left"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    props.setFoldName();
                  }}
                  disabled={props.userType === "viewer"}
                >
                  <AiFillEdit />
                </button>
              </span>
            </div>
          </div>
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
              Created
            </label>
            <div className="uk-form-controls">
              <input
                className="uk-input uk-form-width-large"
                id="form-horizontal-text"
                type="text"
                value={props.foldCreateDate}
                disabled
              />
            </div>
          </div>
          <div className="uk-margin">
            <label className="uk-form-label" htmlFor="form-horizontal-text">
              Public
            </label>
            <div
              className="uk-form-controls"
              onClick={(e) => e.stopPropagation()}
            >
              <input
                className="uk-input uk-form-width-large uk-checkbox"
                type="checkbox"
                checked={props.foldPublic || false}
                onChange={(e) => {
                  e.stopPropagation();
                  props.setPublic(!props.foldPublic);
                }}
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
      prevProps.foldCreateDate === nextProps.foldCreateDate &&
      prevProps.foldPublic === nextProps.foldPublic &&
      prevProps.foldModelPreset === nextProps.foldModelPreset &&
      prevProps.sequence === nextProps.sequence &&
      prevProps.colorScheme === nextProps.colorScheme &&
      prevProps.pfamColors === nextProps.pfamColors
    );
  }
);

export default SequenceTab;
