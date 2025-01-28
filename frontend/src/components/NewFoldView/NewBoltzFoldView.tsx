import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import BoltzYamlBuilder from "../../util/boltzYamlBuilder";
import { Row, Col, Form, Input, Switch, Alert } from "antd";
import { postFolds } from "../../api/foldApi";
import { FoldInput } from "../../types/types";
import UIkit from "uikit";

interface NewBoltzFoldViewProps {
    userType: string | null;
    setErrorText: (error: string) => void;
}

interface AdvancedSettings {
    startFoldJob: boolean;
    emailOnCompletion: boolean;
    skipDuplicateEntries: boolean;
}

async function createFold(
    foldName: string,
    yamlData: string,
    options: {
        userType: string | null;
        startFoldJob?: boolean;
        emailOnCompletion?: boolean;
        skipDuplicateEntries?: boolean;
    }
): Promise<void> {
    if (options.userType === "viewer") {
        throw new Error("Viewers cannot create folds");
    }

    const fold: FoldInput = {
        name: foldName,
        tags: [],
        yaml_config: yamlData,
        yaml_helper: null,
        sequence: null,
        af2_model_preset: "boltz",
        disable_relaxation: false,
    };

    await postFolds([fold], {
        startJob: options.startFoldJob || false,
        emailOnCompletion: options.emailOnCompletion || false,
        skipDuplicates: options.skipDuplicateEntries || false,
    });
}

const NewBoltzFoldView: React.FC<NewBoltzFoldViewProps> = ({ userType, setErrorText }) => {
    const navigate = useNavigate();
    const [foldName, setFoldName] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [advancedSettings, setAdvancedSettings] = useState<AdvancedSettings>({
        startFoldJob: true,
        emailOnCompletion: true,
        skipDuplicateEntries: false,
    });

    // Example partial YAML (you can keep or remove this)
    const partialYaml = `
version: 1
sequences:
`;

    async function handleSave(yamlString: string) {
        if (!foldName.trim()) {
            setErrorText("Please enter a fold name");
            return;
        }

        setIsSubmitting(true);
        try {
            await createFold(foldName, yamlString, {
                userType,
                ...advancedSettings,
            });

            UIkit.alert("Fold successfully created!");
            navigate("/");
        } catch (err) {
            console.error(err);
            setErrorText(`Failed to create fold: ${String(err)}`);
        } finally {
            setIsSubmitting(false);
        }
    }

    return (
        <div
            data-testid="About"
            style={{
                flexGrow: 1,
                overflowY: "scroll",
                paddingTop: "10px",
                paddingBottom: "10px",
            }}>
            {/* Fixed Header */}
            <div style={{ padding: "1rem", borderBottom: "1px solid #f0f0f0" }}>
                <h1>New Boltz Fold</h1>

                {userType === "viewer" && (
                    <Alert
                        message="You do not have permissions to submit folds on this instance."
                        type="error"
                        style={{ marginBottom: "1rem" }}
                    />
                )}
            </div>

            {/* Scrollable Content */}
            <div style={{ flex: 1, overflow: "auto", padding: "1rem" }}>
                <Row gutter={24}>
                    <Col span={18}>
                        <Form.Item
                            label="Fold Name"
                            required
                            style={{ marginBottom: "2rem" }}
                        >
                            <Input
                                value={foldName}
                                onChange={(e) => setFoldName(e.target.value)}
                                placeholder="Enter fold name"
                                disabled={userType === "viewer"}
                            />
                        </Form.Item>

                        <BoltzYamlBuilder
                            initialYaml={partialYaml}
                            onSave={handleSave}
                        />
                    </Col>

                    {/* Advanced settings column - will scroll with content */}
                    <Col span={6} style={{ position: "sticky", top: "1rem" }}>
                        <div style={{
                            backgroundColor: "#f5f5f5",
                            padding: "1rem",
                            borderRadius: "8px"
                        }}>
                            <h3>Advanced Settings</h3>
                            <Form layout="vertical">
                                <Form.Item label="Start Fold Job Immediately">
                                    <Switch
                                        checked={advancedSettings.startFoldJob}
                                        onChange={(checked) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                startFoldJob: checked,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>

                                <Form.Item label="Email on Completion">
                                    <Switch
                                        checked={advancedSettings.emailOnCompletion}
                                        onChange={(checked) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                emailOnCompletion: checked,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>

                                <Form.Item label="Skip Duplicate Entries">
                                    <Switch
                                        checked={advancedSettings.skipDuplicateEntries}
                                        onChange={(checked) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                skipDuplicateEntries: checked,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>

                            </Form>
                        </div>
                    </Col>
                </Row>
            </div>
        </div>
    );
};

export default NewBoltzFoldView;