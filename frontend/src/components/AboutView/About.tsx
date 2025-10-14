import React, { useState } from "react";
import { getDescriptionOfUserType } from "../../services/authentication.service";
import CitationModal from "../shared/CitationModal";

const About = (props: { userType: string | null }) => {
    const [showCitationModal, setShowCitationModal] = useState(false);

    return (
        <div
            data-testid="About"
            style={{
                flexGrow: 1,
                overflowY: "scroll",
                padding: "24px",
            }}
        >
            <h1>About {import.meta.env.VITE_INSTITUTION} Foldy</h1>

            {/* Citation Info - Prominent placement */}
            <div style={{
                backgroundColor: "#f6ffed",
                border: "1px solid #b7eb8f",
                borderRadius: "6px",
                padding: "16px",
                marginBottom: "24px"
            }}>
                <h3 style={{ color: "#389e0d", marginTop: 0 }}>📚 Citations & Attribution</h3>
                <p style={{ marginBottom: "12px" }}>
                    If you publish research using this platform, please consider citing the relevant papers
                    to support the developers and help others discover these tools. This includes both the
                    Foldy platform and the underlying computational methods.
                </p>
                <button
                    onClick={() => setShowCitationModal(true)}
                    style={{
                        backgroundColor: "#1890ff",
                        color: "white",
                        border: "none",
                        padding: "8px 16px",
                        borderRadius: "4px",
                        cursor: "pointer"
                    }}
                >
                    View Citation Information
                </button>
            </div>

            <p>
                <b>{import.meta.env.VITE_INSTITUTION} Foldy</b> is a democratized protein folding platform
                developed by the Keasling Lab. We use state-of-the-art AI models to predict protein structures
                and provide advanced analysis tools for structural biology research.
            </p>

            <h2>What Can You Do Here?</h2>
            <ul>
                <li><strong>Predict protein structures</strong> using Boltz-2x, a cutting-edge AI model with exceptional accuracy for multimers, small molecule docking, and nucleic acid interactions</li>
                <li><strong>Analyze protein sequences</strong> with automated annotations and structural insights</li>
                <li><strong>Visualize 3D structures</strong> interactively in your browser</li>
                <li><strong>Download results</strong> in standard formats (PDB, CIF) for further analysis</li>
                <li><strong>Dock small molecules</strong> to predicted structures (advanced feature)</li>
                <li><strong>Explore evolutionary relationships</strong> through sequence analysis tools</li>
            </ul>

            <h2>Your Access Level</h2>
            <p>
                Users at {import.meta.env.VITE_INSTITUTION} have <b>editor</b> permission
                to this Foldy instance, all others have <b>viewer</b> permissions.{" "}
                {props.userType ? getDescriptionOfUserType(props.userType) : null}
            </p>

            {props.userType === "editor" || props.userType === "admin" ? (
                <p>
                    <strong>As an editor,</strong> you can create new structure predictions by clicking the "NEW" button
                    on the dashboard. Choose meaningful names and ensure your sequences contain only canonical amino acids.
                </p>
            ) : (
                <p>
                    <strong>As a viewer,</strong> you can browse and analyze all public structures but cannot create new predictions.
                </p>
            )}

            <h2>Getting Started</h2>
            <ol>
                <li><strong>Browse existing structures</strong> from the dashboard (home page)</li>
                <li><strong>Create new predictions</strong> by clicking "NEW" and following the guided workflow</li>
                <li><strong>Monitor progress</strong> - most predictions complete within 24 hours</li>
                <li><strong>Analyze results</strong> using the comprehensive tools in each fold's detail page</li>
                <li><strong>Download data</strong> in formats compatible with your research workflow</li>
            </ol>

            <h2>Current Limitations</h2>
            <ul>
                <li>Sequences over 3,000 amino acids may fail due to memory constraints</li>
                <li>Please contact administrators before submitting more than 100 sequences per day</li>
                <li>Very large output files may occasionally be corrupted during download</li>
            </ul>

            <h2>Frequently Asked Questions</h2>

            <h4>When will my structure prediction finish?</h4>
            <p>We aim for less than 24-hour turnaround. If your prediction hasn't completed after 48 hours,
                try restarting the failed steps from the "Actions" tab. Contact administrators if problems persist.</p>

            <h4>Why did my prediction fail?</h4>
            <p>Common causes include memory limitations (very long sequences) or temporary resource constraints.
                Check the "Logs" tab for detailed error information, and feel free to restart failed steps from the "Actions" tab.</p>

            <h4>How do I access all the output files?</h4>
            <p>All AlphaFold and Boltz-2x output files are available through the "Files" tab in each fold's detail page.
                This includes structure files, confidence scores, and intermediate results.</p>

            <h4>What's the difference between this and other folding services?</h4>
            <p>Foldy emphasizes accessibility and provides integrated analysis tools. We use Boltz-2x, which excels
                at complex scenarios like protein multimers and small molecule interactions that other tools struggle with.</p>

            <h4>How do I report bugs or request features?</h4>
            <p>Please file issues on the <a href="https://github.com/JBEI/foldy">Foldy GitHub repository</a>.
                We welcome bug reports, feature requests, and general feedback.</p>

            <h4>Who can I contact for help?</h4>
            <p>Contact your Foldy administrators for technical support, account issues, or questions about using the platform for your research.</p>

            <CitationModal
                open={showCitationModal}
                onClose={() => setShowCitationModal(false)}
            />
        </div>
    );
};

export default About;
