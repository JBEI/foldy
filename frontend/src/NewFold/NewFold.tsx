import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { EditableTagList } from "./../util/editableTagList";
import UIkit from "uikit";
import { FoldInput, postFolds } from "../services/backend.service";

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
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "Y",
];

const LARGE_SIZE_WARNING_THRESHOLD = 2000;

class Chain {
  chainId: string | null;
  chainSequence: string | null;
  constructor(chainId: string | null, chainSequence: string | null) {
    this.chainId = chainId;
    this.chainSequence = chainSequence;
  }
}

function NewFold(props: { setErrorText: (a: string) => void }) {
  const [name, setName] = useState<string | null>(null);
  const [isAMonomer, setIsAMonomer] = useState<boolean>(true);
  const [textboxContents, setTextboxContents] = useState<string | null>(null);
  const [chains, setChains] = useState<Chain[]>([new Chain(null, null)]);
  const [tags, setTags] = useState<string[]>([]);

  const [showAdvancedSettings, setShowAdvancedSettings] =
    useState<boolean>(false);
  const [isBatchEntry, setIsBatchEntry] = useState<boolean>(false);
  const [startFoldJob, setStartFoldJob] = useState<boolean>(true);
  const [emailOnCompletion, setEmailOnCompletion] = useState<boolean>(true);
  const [skipDuplicateEntries, setSkipDuplicateEntries] =
    useState<boolean>(false);
  const [af2ModelPresetOverride, setAf2ModelPresetOverride] = useState<
    string | null
  >(null);
  const [disableRelaxation, setDisableRelaxation] = useState<boolean>(false);

  const [isActivelySubmitting, setIsActivelySubmitting] =
    useState<boolean>(false);

  const navigate = useNavigate();

  const numAminoAcidsInInput = (() => {
    if (isBatchEntry) {
      return null;
    }
    if (isAMonomer) {
      return (textboxContents || "").length;
    } else {
      return chains
        .map((c) => (c.chainSequence || "").length)
        .reduce((a, b) => a + b);
    }
  })();

  const handleSingleSubmit = () => {
    if (!name) {
      UIkit.modal.alert("Name must be set.");
      return;
    }

    var sequence;
    if (isAMonomer) {
      if (!textboxContents) {
        UIkit.modal.alert("Sequence must be set.");
        return;
      }
      sequence = textboxContents;
    } else {
      const sequences: string[] = [];
      var exampleErrorMessage: string | null = null;
      chains.forEach((chain: Chain, ii: number) => {
        const idError = getChainIdErrorMessage(chain.chainId, ii);
        const sequenceError = getChainSequenceErrorMessage(chain.chainSequence);

        exampleErrorMessage = exampleErrorMessage || idError || sequenceError;

        sequences.push(`${chain.chainId}:${chain.chainSequence}`);
      });
      if (exampleErrorMessage) {
        UIkit.modal.alert(`Failed validation: ${exampleErrorMessage}`);
        return;
      }
      sequence = sequences.join(";");
    }

    var af2ModelPreset = isAMonomer ? "monomer_ptm" : "multimer";
    if (af2ModelPresetOverride) {
      af2ModelPreset = af2ModelPresetOverride;
    }

    const newFold = {
      name: name,
      sequence: sequence,
      tags: tags,
      af2_model_preset: af2ModelPreset,
      disable_relaxation: disableRelaxation,
    };
    postFolds(
      [newFold],
      startFoldJob,
      emailOnCompletion,
      skipDuplicateEntries
    ).then(
      (e) => {
        setIsActivelySubmitting(false);
        UIkit.notification("Fold successfully submitted.");
        navigate({
          pathname: "/",
        });
      },
      (e) => {
        setIsActivelySubmitting(false);
        props.setErrorText(e.toString());
      }
    );
  };

  const handleBatchSubmit = () => {
    if (!textboxContents) {
      UIkit.modal.alert("Sequence must be set.");
      return;
    }

    const newFolds = textboxContents
      .split("\n")
      .map((line: string): FoldInput => {
        const elems = line.split(",");
        if (elems.length !== 2) {
          UIkit.modal.alert("Every line must have a single comma.");
        }
        const name = elems[0].trim();
        const sequence = elems[1].trim();
        const newFold = {
          name: name,
          sequence: sequence,
          tags: tags,
          af2_model_preset: af2ModelPresetOverride,
          disable_relaxation: disableRelaxation,
        };
        return newFold;
      });
    postFolds(
      newFolds,
      startFoldJob,
      emailOnCompletion,
      skipDuplicateEntries
    ).then(
      (e) => {
        setIsActivelySubmitting(false);
        UIkit.notification("Folds successfully submitted.");
        navigate({
          pathname: "/",
        });
      },
      (e) => {
        setIsActivelySubmitting(false);
        props.setErrorText(e.toString());
      }
    );
  };

  const handleSubmit = () => {
    setIsActivelySubmitting(true);
    if (isBatchEntry) {
      handleBatchSubmit();
    } else {
      handleSingleSubmit();
    }
  };

  const canonicalizeSequence = () => {
    if (isBatchEntry) {
      return;
    }
    if (isAMonomer) {
      if (textboxContents) {
        setTextboxContents(textboxContents.replace(/\s/g, "").toUpperCase());
      }
    } else {
      const newChains = chains.map(
        (chain) =>
          new Chain(
            chain.chainId,
            chain.chainSequence
              ? chain.chainSequence.replace(/\s/g, "").toUpperCase()
              : null
          )
      );
      setChains(newChains);
    }
  };

  const handleTagDelete = (tagToDelete: string) => {
    setTags(tags.filter((tag, index) => tag !== tagToDelete));
  };

  const handleTagAddition = (newTag: string) => {
    setTags([...tags, newTag]);
  };

  const getFoldNameErrorMessage = (foldName: string | null) => {
    if (!foldName) {
      return null;
    }
    const foldNameIsValid = foldName.match(/^[0-9a-zA-Z_\- ]+$/);
    if (!foldNameIsValid) {
      return "Can only contain letters, numbers, -, _, or spaces.";
    }
    return null;
  };

  const localSetName = (e: HTMLInputElement, foldName: string | null) => {
    const invalidMessage = getFoldNameErrorMessage(foldName);
    if (invalidMessage) {
      e.setCustomValidity(invalidMessage);
      e.reportValidity();
    } else {
      e.setCustomValidity("");
    }

    setName(foldName);
  };

  const getChainIdErrorMessage = (chainId: string | null, ii: number) => {
    if (!chainId) {
      return null;
    }

    const chainIdIsAlphanumeric = chainId.match(/^[0-9a-zA-Z_-]+$/);
    if (!chainIdIsAlphanumeric) {
      return "Chain IDs must be alphanumeric, dash, or underscore.";
    }

    const thereIsADupe = chains.some((otherChain, jj) => {
      return jj !== ii && otherChain.chainId === chainId;
    });
    if (thereIsADupe) {
      return "Chain IDs must be unique.";
    }
    return null;
  };

  const getChainSequenceErrorMessage = (chainSequence: string | null) => {
    if (!chainSequence) {
      return null;
    }
    const hasInvalidChar = chainSequence.split("").some((aa) => {
      return !VALID_AMINO_ACIDS.includes(aa);
    });
    if (hasInvalidChar) {
      return "Detected non-amino acid character. Try canonicalizing.";
    }
    return null;
  };

  const localSetTextboxContents = (
    e: HTMLInputElement | HTMLTextAreaElement,
    newSequence: string
  ) => {
    if (!isBatchEntry) {
      const invalidMessage = getChainSequenceErrorMessage(newSequence);
      if (invalidMessage) {
        e.setCustomValidity(invalidMessage);
        e.reportValidity();
      } else {
        e.setCustomValidity("");
      }
    }

    setTextboxContents(newSequence);
  };

  const removeChain = (ii: number) => {
    if (chains.length <= 1) {
      return;
    }
    const newChains = chains.filter((c: Chain, index: number) => index !== ii);
    setChains(newChains);
  };

  const addChain = () => {
    const newChains = [...chains];
    newChains.push(new Chain(null, null));
    setChains(newChains);
  };

  const setChainId = (e: HTMLInputElement, ii: number, newId: string) => {
    const invalidMessage = getChainIdErrorMessage(newId, ii);
    if (invalidMessage) {
      e.setCustomValidity(invalidMessage);
      e.reportValidity();
    } else {
      e.setCustomValidity("");
    }
    const newChains = [...chains];
    newChains[ii].chainId = newId;
    setChains(newChains);
  };

  const setChainSequence = (
    e: HTMLInputElement,
    ii: number,
    newSequence: string
  ) => {
    const invalidMessage = getChainSequenceErrorMessage(newSequence);
    if (invalidMessage) {
      e.setCustomValidity(invalidMessage);
      e.reportValidity();
    } else {
      e.setCustomValidity("");
    }
    const newChains = [...chains];
    newChains[ii].chainSequence = newSequence;
    setChains(newChains);
  };

  return (
    <div data-testid="NewFold">
      <fieldset className="uk-fieldset">
        <form className="uk-form-horizontal uk-margin-left uk-margin-right">
          <legend className="uk-legend">
            New {isBatchEntry ? "Folds" : "Fold"}
          </legend>
          <div
            className="uk-button-group uk-margin-small-top"
            style={{ borderRadius: "5px" }}
          >
            <button
              type="button"
              className={
                "uk-button uk-button-default uk-margin-small-right " +
                (isAMonomer ? "uk-button-primary" : "uk-button-default")
              }
              onClick={() => setIsAMonomer(true)}
            >
              monomer
            </button>
            <button
              type="button"
              className={
                "uk-button uk-button-default " +
                (!isAMonomer ? "uk-button-primary" : "uk-button-default")
              }
              onClick={() => setIsAMonomer(false)}
            >
              multimer
            </button>
          </div>

          {isBatchEntry ? null : (
            <div className="uk-margin-small-top">
              <input
                className={
                  "uk-input " +
                  (getFoldNameErrorMessage(name) ? "uk-form-danger" : null)
                }
                type="text"
                placeholder="Fold Name"
                id="name"
                uk-tooltip="Choose any name for this protein folding job, preferably something short and convenient. Must be unique, but we will tell you if it's not."
                value={name || ""}
                style={{ borderRadius: "500px" }}
                onChange={(e) => localSetName(e.target, e.target.value)}
              />
            </div>
          )}

          {isAMonomer || isBatchEntry ? (
            <textarea
              className={
                "uk-textarea uk-margin-small-top " +
                (!isBatchEntry && getChainSequenceErrorMessage(textboxContents)
                  ? "uk-form-danger"
                  : null)
              }
              rows={5}
              placeholder={
                isBatchEntry
                  ? "foldId,chainAID:chainASequence;chainBID:chainBSequence"
                  : "Amino acid sequence"
              }
              style={{
                fontFamily: 'consolas,"Liberation Mono",courier,monospace',
                borderRadius: "20px",
              }}
              value={textboxContents || ""}
              onChange={(e) =>
                localSetTextboxContents(e.target, e.target.value)
              }
            ></textarea>
          ) : (
            <div>
              {chains.map((chain: Chain, ii: number) => {
                return (
                  <div
                    className="uk-margin-small-top uk-flex uk-flex-middle"
                    key={`chain_${ii}`}
                  >
                    <div className="uk-flex-1" style={{ flex: "1 1 0" }}>
                      <input
                        className={
                          "uk-input uk-padding-right " +
                          (getChainIdErrorMessage(chain.chainId, ii)
                            ? "uk-form-danger"
                            : null)
                        }
                        placeholder={`Chain ${ii + 1} name`}
                        style={{
                          fontFamily:
                            'consolas,"Liberation Mono",courier,monospace',
                          borderRadius: "100px",
                        }}
                        value={chain.chainId || ""}
                        onChange={(e) =>
                          setChainId(e.target, ii, e.target.value)
                        }
                      ></input>
                    </div>
                    <div
                      className="uk-flex-1"
                      style={{ flex: "3 1 0", marginLeft: "10px" }}
                    >
                      <input
                        className={
                          "uk-input " +
                          (getChainSequenceErrorMessage(chain.chainSequence)
                            ? "uk-form-danger"
                            : null)
                        }
                        placeholder={`Chain ${ii + 1} sequence`}
                        style={{
                          fontFamily:
                            'consolas,"Liberation Mono",courier,monospace',
                          borderRadius: "100px",
                        }}
                        value={chain.chainSequence || ""}
                        onChange={(e) =>
                          setChainSequence(e.target, ii, e.target.value)
                        }
                      ></input>
                    </div>
                    <div
                      className="uk-flex-1"
                      style={{ flex: "0 1 auto", marginLeft: "10px" }}
                    >
                      <button
                        className="uk-button uk-button-danger"
                        onClick={(e) => {
                          e.preventDefault();
                          removeChain(ii);
                        }}
                      >
                        X
                      </button>
                    </div>
                  </div>
                );
              })}
              <button
                className="uk-button uk-margin-small-top"
                onClick={(e) => {
                  e.preventDefault();
                  addChain();
                }}
              >
                Add Chain
              </button>
            </div>
          )}
          <div className="uk-margin-small">
            <EditableTagList
              tags={tags}
              addTag={handleTagAddition}
              deleteTag={handleTagDelete}
              handleTagClick={(v) => null}
            />
          </div>

          {numAminoAcidsInInput != null &&
          numAminoAcidsInInput > LARGE_SIZE_WARNING_THRESHOLD ? (
            <div className="uk-alert-warning" uk-alert={1}>
              <p>
                Folds large than than {LARGE_SIZE_WARNING_THRESHOLD} amino acids
                are likely to fail during AMBER relaxation, a post-processing
                step which better aligns residues. It can be disabled in
                Advanced Settings.
              </p>
            </div>
          ) : null}

          {isBatchEntry ? (
            <div className="uk-alert-warning" uk-alert={1}>
              <p>
                Make sure to set the AlphaFold mode preset from the Advanced
                Settings.
              </p>
            </div>
          ) : null}

          {showAdvancedSettings ? (
            <div>
              <div className="uk-margin">
                <label
                  className="uk-form-label"
                  htmlFor="name"
                  uk-tooltip="Have lots of folds to submit? Try doing it in bulk!"
                >
                  Bulk CSV Entry
                </label>
                <div className="uk-form-controls">
                  <input
                    className="uk-checkbox"
                    type="checkbox"
                    checked={isBatchEntry}
                    onChange={(e) => setIsBatchEntry(e.target.checked)}
                  />
                </div>
              </div>
              <div className="uk-margin">
                <label className="uk-form-label" htmlFor="name">
                  Start Fold Job Immediately
                </label>
                <div className="uk-form-controls">
                  <input
                    className="uk-checkbox"
                    type="checkbox"
                    checked={startFoldJob}
                    onChange={(e) => setStartFoldJob(e.target.checked)}
                  />
                </div>
              </div>
              <div className="uk-margin">
                <label className="uk-form-label" htmlFor="name">
                  Send Completion Email
                </label>
                <div className="uk-form-controls">
                  <input
                    className="uk-checkbox"
                    type="checkbox"
                    checked={emailOnCompletion}
                    onChange={(e) => setEmailOnCompletion(e.target.checked)}
                  />
                </div>
              </div>
              <div className="uk-margin">
                <label
                  className="uk-form-label"
                  htmlFor="skiptaken"
                  uk-tooltip="Skip folds with taken names, rather than crash (CSV mode)"
                >
                  Silence Name Check
                </label>
                <div className="uk-form-controls" id="skiptaken">
                  <input
                    className="uk-checkbox"
                    type="checkbox"
                    checked={skipDuplicateEntries}
                    onChange={(e) => setSkipDuplicateEntries(e.target.checked)}
                  />
                </div>
              </div>
              <div className="uk-margin">
                <label
                  className="uk-form-label"
                  htmlFor="name"
                  uk-tooltip="Disabling relaxation makes folding faster but degrades residue placement."
                >
                  Disable AMBER Relaxation
                </label>
                <div className="uk-form-controls" id="disableRelax">
                  <input
                    className="uk-checkbox"
                    type="checkbox"
                    checked={disableRelaxation}
                    onChange={(e) => setDisableRelaxation(e.target.checked)}
                  />
                </div>
              </div>
              <div className="uk-margin">
                <label
                  className="uk-form-label"
                  htmlFor="af2modelpreset"
                  uk-tooltip="Override the default alphafold model preset, which is either monomer_ptm or multimer."
                >
                  Override AF Model Preset
                </label>
                <div className="uk-form-controls" id="af2modelpreset">
                  <select
                    className="uk-select"
                    onChange={(e) => {
                      setAf2ModelPresetOverride(e.target.value || null);
                    }}
                  >
                    <option></option>
                    <option>monomer_ptm</option>
                    <option>multimer</option>
                    <option>monomer</option>
                  </select>
                </div>
              </div>
            </div>
          ) : null}

          <button
            type="button"
            className="uk-button uk-button-default uk-margin-small-right"
            onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
          >
            Advanced Settings
          </button>

          {isBatchEntry ? null : (
            <button
              type="button"
              className="uk-button uk-button-default uk-margin-small-right"
              onClick={canonicalizeSequence}
              uk-tooltip="Remove whitespace and capitalizes all letters."
            >
              Canonicalize Sequence{isAMonomer ? "" : "s"}
            </button>
          )}

          <button
            type="button"
            className="uk-button uk-button-primary"
            onClick={handleSubmit}
            disabled={isActivelySubmitting}
          >
            Submit {numAminoAcidsInInput} AA Fold
          </button>
        </form>
      </fieldset>
    </div>
  );
}

export default NewFold;
