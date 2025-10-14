import React, { useState, useEffect, useRef } from "react";
import { getResidueHeatmap } from "../../util/plots";
import { getFoldPae } from "../../api/foldApi";
import { FoldPae } from "../../types/types";
import { TabContainer, DescriptionSection, SectionCard } from "../../util/tabComponents";
import { Spin } from "antd";
import { notify } from "src/services/NotificationService";
import { BoltzYamlHelper } from "../../util/boltzYamlHelper";
import { Selection } from "./StructurePane";

interface PaeTabProps {
    foldId: number;
    foldSequence: string | undefined;
    yamlConfig: string | undefined;
    setSelectedSubsequence: (selection: Selection | null) => void;
}

const PaeTab = React.memo(
    (props: PaeTabProps) => {
        const [pae, setPae] = useState<FoldPae | null>(null);
        const [errorMessage, setErrorMessage] = useState<string | null>(null);
        const [isLoading, setIsLoading] = useState<boolean>(true);
        const mountedRef = useRef(true);

        useEffect(() => {
            setIsLoading(true);
            setErrorMessage(null);

            console.log(`Fetching PAE data for fold ID: ${props.foldId}`);
            getFoldPae(props.foldId, 0).then(
                (newPae) => {
                    if (!mountedRef.current) return null;
                    console.log("PAE data loaded successfully", newPae);
                    setPae(newPae);
                    setIsLoading(false);
                },
                (e) => {
                    if (!mountedRef.current) return null;
                    console.error("Failed to load PAE data:", e);
                    setErrorMessage(`Failed to load PAE: ${e.message || "Unknown error"}`);
                    setIsLoading(false);
                }
            );
        }, [props.foldId]);

        useEffect(() => {
            return () => {
                mountedRef.current = false;
            };
        }, []);

        const getPaeProbHeatmap = () => {
            if (isLoading) {
                return (
                    <div style={{ textAlign: 'center', padding: '60px 0' }}>
                        <Spin size="large" />
                        <p>Loading PAE data...</p>
                    </div>
                );
            }

            if (errorMessage) {
                return (
                    <div className="uk-alert-danger" uk-alert>
                        <p>{errorMessage}</p>
                        <p>This may happen if the PAE data format is not compatible or if the file is missing.</p>
                    </div>
                );
            }

            if (!pae) {
                return (
                    <div className="uk-alert-warning" uk-alert>
                        <p>No PAE data available.</p>
                    </div>
                );
            }

            let sequence: string;
            if (props.foldSequence) {
                sequence = props.foldSequence;
            }
            else if (props.yamlConfig) {
                const configHelper = new BoltzYamlHelper(props.yamlConfig);
                if (configHelper.getAllSequences().length > 1) {
                    sequence = configHelper.getAllSequences().map(seq => {
                        if (seq.sequence) {
                            return `${seq.id}:${seq.sequence}`
                        }
                        else if (seq.ccd) {
                            return `${seq.id}:${seq.ccd}`
                        }
                        else {
                            // TODO: Use Rdkit or openchemlib to count heavy atoms.
                            const heavyAtomString = (seq.smiles?.match(/[A-Z]/g) || [])
                                .filter(letter => letter !== 'H')
                                .join('');
                            return `${seq.id}:${heavyAtomString}`
                        }
                    }).join(';');
                }
                else {
                    if (configHelper.getAllSequences()[0].entity_type !== 'protein') {
                        return <div>If there is only one entity, it must be protein.</div>;
                    }
                    sequence = configHelper.getAllSequences()[0].sequence || '';
                }
            }
            else {
                return (
                    <div className="uk-alert-warning" uk-alert>
                        <p>Fold sequence is required to display PAE heatmap.</p>
                    </div>
                );
            }

            try {
                return getResidueHeatmap(
                    sequence,
                    pae.pae,
                    "Jet",
                    "min",
                    0,
                    undefined,
                    props.setSelectedSubsequence,
                    props.yamlConfig
                );
            } catch (error) {
                console.error("Error rendering PAE heatmap:", error);
                return (
                    <div className="uk-alert-danger">
                        <p>Error rendering PAE heatmap: {error instanceof Error ? error.message : "Unknown error"}</p>
                    </div>
                );
            }
        };

        return (
            <TabContainer key="Pae">
                <DescriptionSection title="PAE">
                    <p>
                        Predicted alignment error (PAE) may indicate whether two domains are
                        rigid with respect to one another (
                        <a href="https://alphafold.ebi.ac.uk/faq">AlphaFold FAQ</a>).
                    </p>
                </DescriptionSection>

                <SectionCard style={{ textAlign: 'center' }}>
                    {getPaeProbHeatmap()}
                </SectionCard>
            </TabContainer>
        );
    },
    (prevProps: PaeTabProps, nextProps: PaeTabProps) => {
        // Implements AreEqual.
        const sequenceIsUnchanged =
            prevProps.foldSequence === nextProps.foldSequence;
        const foldIdIsUnchanged = prevProps.foldId === nextProps.foldId;
        return sequenceIsUnchanged && foldIdIsUnchanged;
    }
);

export default PaeTab;
