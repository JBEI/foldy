import React, { useEffect, useRef, useState } from "react";
import { getFoldContactProb } from "../../api/foldApi";
import { getResidueHeatmap } from "../../util/plots";
import { FoldContactProb } from "../../types/types";
import { TabContainer, DescriptionSection, SectionCard } from "../../util/tabComponents";
import { Spin } from "antd";

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
                () => {
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
                    <div style={{ textAlign: 'center', padding: '60px 0' }}>
                        <Spin size="large" />
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
            <TabContainer key="contact">
                <DescriptionSection title="Contact Probabilities">
                    <p>
                        If two domains or modules have a high relative contact probability, they
                        are likely to interact (
                        <a href="https://science.org/doi/full/10.1126/science.abm4805">
                            Humphreys et al, 2021
                        </a>
                        ).
                    </p>
                </DescriptionSection>

                <SectionCard style={{ textAlign: 'center' }}>
                    {getContactProbHeatmap()}
                </SectionCard>
            </TabContainer>
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
