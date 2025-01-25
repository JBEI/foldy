import React, { useEffect, useRef, useState } from "react";
import {
    getFoldContactProb,
} from "../../services/backend.service";
import { getResidueHeatmap } from "../../util/plots";
import { FoldContactProb } from "../../types/types";

interface ContactTabProps {
    foldId: number;
    foldSequence: string | undefined;
}

const ContactTab = React.memo(
    (props: ContactTabProps) => {
        const [contactProb, setContactProb] = useState<FoldContactProb | null>(
            null
        );
        const [errorMessage, setErrorMessage] = useState<string | null>(null);
        const mountedRef = useRef(true);

        useEffect(() => {
            getFoldContactProb(props.foldId, 0).then(
                (contact_prob) => {
                    if (!mountedRef.current) return null;
                    setContactProb(contact_prob);
                },
                (e) => {
                    if (!mountedRef.current) return null;
                    setErrorMessage("Failed to load contact probability map.");
                }
            );
        }, [props.foldId]);

        useEffect(() => {
            return () => {
                mountedRef.current = false;
            };
        }, []);

        const getContactProbHeatmap = () => {
            if (errorMessage) {
                return <div className="uk-alert-danger">{errorMessage}</div>;
            }
            if (!contactProb || !props.foldSequence) {
                return (
                    <div className="uk-text-center">
                        <div uk-spinner="ratio: 4"></div>
                    </div>
                );
            }
            return getResidueHeatmap(
                props.foldSequence,
                contactProb.contact_prob,
                "Viridis",
                "max",
                0,
                1
            );
        };

        return (
            <div className="uk-text-center" key="contact">
                <h3>Contact Probabilities</h3>
                If two domains or modules have a high relative contact probability, they
                are likely to interact (
                <a href="https://science.org/doi/full/10.1126/science.abm4805">
                    Humphreys et al, 2021
                </a>
                ).
                <br></br>
                {getContactProbHeatmap()}
            </div>
        );
    },
    (prevProps: ContactTabProps, nextProps: ContactTabProps) => {
        // Implements AreEqual.
        const sequenceIsUnchanged =
            prevProps.foldSequence === nextProps.foldSequence;
        return Boolean(sequenceIsUnchanged);
    }
);

export default ContactTab;
