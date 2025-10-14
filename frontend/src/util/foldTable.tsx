import React, { useMemo, useState } from "react";
import { FaCheckCircle } from "react-icons/fa";
import { Spin } from "antd";
import { describeFoldState, getFoldAffinityPrediction, updateFold } from "../api/foldApi";
import { AffinityPrediction, Fold } from "../types/types";
import { BoltzYamlHelper } from "./boltzYamlHelper";
import { EditableTagList } from "./editableTagList";
import { notify } from "../services/NotificationService";
import { Link } from "react-router-dom";


interface FoldTagsProps {
    fold: Fold;
    onTagsChange?: () => void;
    userType?: string | null;
    editable?: boolean;
}

function FoldTags({ fold, onTagsChange, userType, editable = false }: FoldTagsProps) {
    const [isUpdating, setIsUpdating] = useState(false);

    const addTag = async (tagToAdd: string) => {
        if (!fold.tags.includes(tagToAdd) && fold.id !== null) {
            setIsUpdating(true);
            try {
                const newTags = [...fold.tags, tagToAdd];
                await updateFold(fold.id, { tags: newTags });
                onTagsChange?.();
                notify.success("Tag added successfully");
            } catch (error) {
                notify.error(`Failed to add tag: ${error}`);
            } finally {
                setIsUpdating(false);
            }
        }
    };

    const deleteTag = async (tagToDelete: string) => {
        if (fold.id !== null) {
            setIsUpdating(true);
            try {
                const newTags = fold.tags.filter(tag => tag !== tagToDelete);
                await updateFold(fold.id, { tags: newTags });
                onTagsChange?.();
                notify.success("Tag removed successfully");
            } catch (error) {
                notify.error(`Failed to remove tag: ${error}`);
            } finally {
                setIsUpdating(false);
            }
        }
    };

    const handleTagClick = (tag: string) => {
        window.open(`/tag/${tag}`, "_self");
    };

    // Always use EditableTagList, but with viewOnly when not editable
    const isViewOnly = !editable || userType === "viewer";

    return (
        <div style={{ opacity: isUpdating ? 0.6 : 1, pointerEvents: isUpdating ? 'none' : 'auto' }}>
            <EditableTagList
                tags={fold.tags}
                addTag={addTag}
                deleteTag={deleteTag}
                handleTagClick={handleTagClick}
                viewOnly={isViewOnly}
            />
        </div>
    );
}

interface AffinityResult {
    binder_id: string;
    affinity: AffinityPrediction | null;
}

type AffinityState = 'loading' | 'loaded' | 'failed' | 'no-affinity';

function FoldAffinity({ foldId, foldYamlHelper }: { foldId: number | null, foldYamlHelper: BoltzYamlHelper | null }) {
    const [affinityResult, setAffinityResult] = useState<AffinityResult | null>(null);
    const [affinityState, setAffinityState] = useState<AffinityState>('loading');

    useMemo(() => {
        if (foldId === null || !foldYamlHelper) {
            setAffinityState('no-affinity');
            return null;
        }

        const properties = foldYamlHelper.getProperties();
        const affinityProperty = properties?.find(p => 'affinity' in p);
        const affinityBinderId = affinityProperty?.affinity?.binder;
        if (!affinityBinderId) {
            setAffinityState('no-affinity');
            return null;
        }

        setAffinityState('loading');

        // Try querying the affinity file from the backend.
        getFoldAffinityPrediction(foldId).then(
            (predictedAffinity: AffinityPrediction) => {
                setAffinityResult({
                    binder_id: affinityBinderId,
                    affinity: predictedAffinity,
                });
                setAffinityState('loaded');
            },
            (e: any) => {
                console.log(e);
                setAffinityResult({
                    binder_id: affinityBinderId,
                    affinity: null,
                });
                setAffinityState('failed');
                // notify.error(e.toString());
            }
        );
    }, [foldId, foldYamlHelper]);

    const binderId = affinityResult?.binder_id || '';

    if (affinityState === 'no-affinity') {
        return <><td>{binderId}</td><td /></>;
    }

    if (affinityState === 'loading') {
        return (
            <>
                <td>{binderId}</td>
                <td><Spin size="small" /></td>
            </>
        );
    }

    if (!affinityResult || affinityState === 'failed' || affinityResult.affinity === null) {
        return <><td>{binderId}</td><td><i>-</i></td></>;
    }

    return (
        <>
            <td>{binderId}</td>
            <td>{Math.pow(10, affinityResult.affinity.affinity_pred_value).toPrecision(2)} μM</td>
        </>
    );
}

interface FoldTableOptions {
    editable?: boolean;
    userType?: string | null;
    onTagsChange?: () => void;
}

export function makeFoldTable(folds: Fold[], options: FoldTableOptions = {}) {
    const { editable = false, userType = null, onTagsChange } = options;
    return (
        <div className="uk-overflow-auto">
            <table
                className="uk-table uk-table-hover uk-table-small"
                style={{ tableLayout: "fixed" }}
            >
                <thead>
                    <tr>
                        <th className="uk-width-medium">Name</th>
                        <th
                            className="uk-width-small"
                            style={{ width: "70px" }}
                        >
                            Length
                        </th>
                        <th
                            className="uk-width-small"
                            style={{ width: "70px" }}
                        >
                            State
                        </th>
                        <th
                            className="uk-width-small"
                            style={{ width: "120px" }}
                        >
                            Owner
                        </th>
                        <th
                            className="uk-width-small"
                            style={{ width: "100px" }}
                        >
                            Date Created
                        </th>
                        <th
                            className="uk-width-small"
                            uk-tooltip={"Whether a fold is visible to the public."}
                            style={{ width: "50px" }}
                        >
                            Public
                        </th>
                        {/* 250228: Disabled for now because it's not populated. */}
                        {/* <th className="uk-width-small">Docked Ligands</th> */}
                        <th className="uk-width-small" style={{ width: "100px" }}>Tags</th>
                        <th className="uk-width-small" style={{ width: "50px" }}>Affinity<br />Target</th>
                        <th className="uk-width-small" style={{ width: "100px" }}>Affinity<br />Prediction</th>
                    </tr>
                </thead>
                <tbody>
                    {[...folds].map((fold) => {
                        return (
                            <tr key={fold.name}>
                                <td
                                    className="uk-text-truncate"
                                    uk-tooltip={`title: ${fold.name}`}
                                >
                                    <Link
                                        to={"/fold/" + fold.id}
                                        style={{
                                            fontSize: '16px',
                                            fontWeight: 600,
                                            textDecoration: 'none',
                                            transition: 'all 0.2s',
                                        }}
                                        onMouseEnter={(e) => {
                                            e.currentTarget.style.textDecoration = 'underline';
                                        }}
                                        onMouseLeave={(e) => {
                                            e.currentTarget.style.textDecoration = 'none';
                                        }}
                                    >
                                        {fold.name}
                                    </Link>
                                </td>
                                <td>{fold.sequence?.length || fold.yaml_helper?.getProteinSequences().reduce((accumulator, cs) => accumulator + cs[1].length, 0)}</td>
                                <td
                                    className="uk-text-truncate"
                                    uk-tooltip={`title: ${describeFoldState(fold)}`}
                                >
                                    {/* {foldIsFinished(fold) ? null : <Spin size="small" />}&nbsp;  */}
                                    {describeFoldState(fold)}
                                </td>
                                <td className="uk-text-truncate" uk-tooltip={fold.owner}>{fold.owner}</td>
                                <td className="uk-text-truncate">
                                    {(() => {
                                        try {
                                            const date = new Date(fold.create_date);
                                            if (isNaN(date.getTime())) {
                                                console.warn(`Invalid date value for fold ${fold.id}: ${fold.create_date}`);
                                                return "Invalid date";
                                            }
                                            return new Intl.DateTimeFormat('en-US', {
                                                timeStyle: "short",
                                                dateStyle: "short",
                                                timeZone: "America/Los_Angeles"
                                            }).format(date);
                                        } catch (error) {
                                            console.error(`Error formatting date for fold ${fold.id}:`, error);
                                            return "Error";
                                        }
                                    })()}
                                </td>
                                <td className="uk-text-truncate">
                                    {fold.public ? (
                                        <FaCheckCircle
                                            uk-tooltip={"This fold is visible to the public."}
                                        />
                                    ) : null}
                                </td>
                                {/* 250228: Disabled for now because it's not populated. */}
                                {/* <td
                                    className="uk-text-truncate "
                                    uk-tooltip={`title: ${(fold.docks || [])
                                        .map((d) => d.ligand_name)
                                        .slice(0, 5)
                                        .join(", ")}`}
                                >
                                    {(fold.docks || []).length}
                                </td> */}
                                <td style={{ overflowX: 'scroll' }}>
                                    <FoldTags
                                        fold={fold}
                                        onTagsChange={onTagsChange}
                                        userType={userType}
                                        editable={editable}
                                    />
                                </td>

                                <FoldAffinity foldId={fold.id} foldYamlHelper={fold.yaml_helper} />
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}
