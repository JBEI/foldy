import React, { useState, useEffect, useRef } from "react";
import { getResidueHeatmap } from "../../util/plots";
import { getFoldPae } from "../../services/backend.service";
import { FoldPae } from "../../types/types";

interface PaeTabProps {
    foldId: number;
    foldSequence: string | undefined;
}

const PaeTab = React.memo(
    (props: PaeTabProps) => {
        const [pae, setPae] = useState<FoldPae | null>(null);
        const [errorMessage, setErrorMessage] = useState<string | null>(null);
        const mountedRef = useRef(true);

        useEffect(() => {
            getFoldPae(props.foldId, 0).then(
                (newPae) => {
                    if (!mountedRef.current) return null;
                    setPae(newPae);
                },
                (e) => {
                    if (!mountedRef.current) return null;
                    setErrorMessage("Failed to load PAE.");
                }
            );
        }, [props.foldId]);

        useEffect(() => {
            return () => {
                mountedRef.current = false;
            };
        }, []);

        const getPaeProbHeatmap = () => {
            if (errorMessage) {
                return <div className="uk-alert-danger">{errorMessage}</div>;
            }
            if (!pae || !props.foldSequence) {
                return (
                    <div className="uk-text-center">
                        <div uk-spinner="ratio: 4"></div>
                    </div>
                );
            }
            return getResidueHeatmap(
                props.foldSequence,
                pae.pae,
                "Jet",
                "min",
                0,
                undefined
            );
        };

        return (
            <div className="uk-text-center" key="Pae">
                <h3>PAE</h3>
                Predicted alignment error (PAE) may indicate whether two domains are
                rigid with respect to one another (
                <a href="https://alphafold.ebi.ac.uk/faq">AlphaFold FAQ</a>).
                <br></br>
                {getPaeProbHeatmap()}
            </div>
        );
    },
    (prevProps: PaeTabProps, nextProps: PaeTabProps) => {
        // Implements AreEqual.
        const sequenceIsUnchanged =
            prevProps.foldSequence === nextProps.foldSequence;
        // var dataIsUnchanged: boolean;
        // if (prevProps.pae === nextProps.pae) {
        //   dataIsUnchanged = true;
        // } else if (prevProps.pae && nextProps.pae) {
        //   dataIsUnchanged = matricesAreEqual(prevProps.pae.pae, nextProps.pae.pae);
        // } else {
        //   dataIsUnchanged = false;
        // }
        return Boolean(sequenceIsUnchanged);
    }
);

export default PaeTab;
