import React, { useState } from "react";
import {
    addInvokationToAllJobs,
    backfillDateCreated,
    backfillInputActivityFpath,
    bulkAddTag,
    createDbs,
    killFoldsInRange,
    killWorker,
    populateOutputFpathFields,
    queueTestJob,
    removeFailedJobs,
    runUnrunStages,
    sendTestEmail,
    setAllUnsetModelPresets,
    stampDbs,
    upgradeDbs,
} from "../../api/adminApi";
import { notify } from '../../services/NotificationService';
import {
    Card,
    Button,
    Input,
    Typography,
    Row,
    Col,
    Space,
    Alert,
    Divider,
    Form
} from 'antd';
import {
    DatabaseOutlined,
    PlayCircleOutlined,
    StopOutlined,
    MailOutlined,
    SettingOutlined,
    DeleteOutlined,
    TagOutlined,
    FileTextOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;

function SudoPage() {
    const [revision, setRevision] = useState<string>("");
    const [newJobQueue, setNewJobQueue] = useState<string>("");
    const [queueToClear, setQueueToClear] = useState<string>("");
    const [workerToKill, setWorkerToKill] = useState<string>("");
    const [newInvokationType, setNewInvokationType] = useState<string | null>(
        null
    );
    const [newInvokationState, setNewInvokationState] = useState<string | null>(
        null
    );
    const [stageToRun, setStageToRun] = useState<string | null>(null);
    const [foldsToKill, setFoldsToKill] = useState<string | null>(null);
    const [foldsToBulkAddTag, setFoldsToBulkAddTag] = useState<string | null>(
        null
    );
    const [tagToBulkAdd, setTagToBulkAdd] = useState<string | null>(null);

    const localCreateDbs = () => {
        createDbs().then(
            () => {
                notify.info("Create successul.");
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localUpgradeDbs = () => {
        upgradeDbs().then(
            () => {
                notify.info("Upgrade successul.");
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localStampDbs = () => {
        stampDbs(revision).then(
            () => {
                notify.info("Upgrade successul.");
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localQueueJob = () => {
        queueTestJob(newJobQueue).then(
            () => {
                notify.info("Successfully queued.");
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localRemoveFailedJobs = () => {
        removeFailedJobs(queueToClear).then(
            () => {
                notify.success(`Successfully removed failed jobs from ${queueToClear}.`);
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localKillWorker = () => {
        killWorker(workerToKill).then(
            () => {
                notify.info(`Successfully killed worker ${workerToKill}.`);
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localSendEmail = () => {
        sendTestEmail().then(
            () => {
                notify.info("Sent email.");
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localAddInvokationToAllJobs = () => {
        if (!newInvokationType || !newInvokationState) {
            return;
        }
        addInvokationToAllJobs(newInvokationType, newInvokationState).then(
            () => {
                notify.info("Successfully added job type.");
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localRunUnrunStages = () => {
        if (!stageToRun) {
            return;
        }
        runUnrunStages(stageToRun).then(
            () => {
                notify.info("Successfully started all stages.");
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localSetAllUnsetModelPresets = () => {
        setAllUnsetModelPresets().then(
            () => {
                notify.info("Successfully set all model presets.");
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localKillFoldsInRange = () => {
        if (!foldsToKill) {
            return;
        }
        killFoldsInRange(foldsToKill).then(
            () => {
                notify.info("Successfully killed folds.");
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localBulkAddTag = () => {
        if (!foldsToBulkAddTag || !tagToBulkAdd) {
            return;
        }
        bulkAddTag(foldsToBulkAddTag, tagToBulkAdd).then(
            () => {
                notify.info("Successfully added tags.");
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localPopulateOutputFpathFields = () => {
        populateOutputFpathFields().then(
            (result) => {
                notify.success(
                    `Successfully populated output_fpath fields! Updated ${result.total_updated} records: ` +
                    `${result.naturalness_updated} naturalness, ${result.embedding_updated} embeddings, ` +
                    `${result.few_shot_updated} few shots.`
                );
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localBackfillDateCreated = () => {
        backfillDateCreated().then(
            (result) => {
                notify.success(
                    `Successfully backfilled date_created fields! Updated ${result.total_updated} records: ` +
                    `${result.naturalness_updated} naturalness, ${result.embedding_updated} embeddings, ` +
                    `${result.few_shot_updated} few shots.`
                );
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    const localBackfillInputActivityFpath = () => {
        backfillInputActivityFpath().then(
            (result) => {
                notify.success(
                    `Successfully backfilled input_activity_fpath fields! Updated ${result.total_updated} FewShot records.`
                );
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    return (
        <div
            data-testid="Workers"
            style={{
                flexGrow: 1,
                overflowY: "auto",
                padding: "16px",
                backgroundColor: "#f5f5f5"
            }}
        >
            <div style={{ maxWidth: "1400px", margin: "0 auto" }}>
                <Title level={3} style={{ textAlign: "center", marginBottom: "16px" }}>
                    <SettingOutlined /> Admin Control Panel
                </Title>

                <Alert
                    message="Use these controls carefully - all actions affect production data"
                    type="warning"
                    showIcon
                    style={{ marginBottom: "16px" }}
                    banner
                />

                <Row gutter={[16, 16]}>
                    {/* Database Management */}
                    <Col xs={24} lg={8}>
                        <Card
                            title={<><DatabaseOutlined /> Database</>}
                            size="small"
                        >
                            <Space direction="vertical" style={{ width: "100%" }} size="small">
                                <Button
                                    type="primary"
                                    onClick={localCreateDbs}
                                    icon={<DatabaseOutlined />}
                                    block
                                    size="small"
                                >
                                    Create DBs
                                </Button>

                                <Button
                                    type="primary"
                                    onClick={localUpgradeDbs}
                                    icon={<DatabaseOutlined />}
                                    block
                                    size="small"
                                >
                                    Upgrade DBs
                                </Button>

                                <Input
                                    placeholder="Revision number"
                                    value={revision}
                                    onChange={(e) => setRevision(e.target.value)}
                                    size="small"
                                />
                                <Button
                                    type="primary"
                                    onClick={localStampDbs}
                                    disabled={!revision}
                                    icon={<DatabaseOutlined />}
                                    block
                                    size="small"
                                >
                                    Stamp Revision
                                </Button>
                            </Space>
                        </Card>
                    </Col>

                    {/* Job Management */}
                    <Col xs={24} lg={8}>
                        <Card
                            title={<><PlayCircleOutlined /> Jobs</>}
                            size="small"
                        >
                            <Space direction="vertical" style={{ width: "100%" }} size="small">
                                <Input
                                    placeholder="Queue name"
                                    value={newJobQueue}
                                    onChange={(e) => setNewJobQueue(e.target.value)}
                                    size="small"
                                />
                                <Button
                                    type="primary"
                                    onClick={localQueueJob}
                                    disabled={!newJobQueue}
                                    icon={<PlayCircleOutlined />}
                                    block
                                    size="small"
                                >
                                    Queue Job
                                </Button>

                                <Input
                                    placeholder="Queue to clear"
                                    value={queueToClear}
                                    onChange={(e) => setQueueToClear(e.target.value)}
                                    size="small"
                                />
                                <Button
                                    type="primary"
                                    danger
                                    onClick={localRemoveFailedJobs}
                                    disabled={!queueToClear}
                                    icon={<DeleteOutlined />}
                                    block
                                    size="small"
                                >
                                    Remove Failed Jobs
                                </Button>

                                <Input
                                    placeholder="Stage name"
                                    value={stageToRun || ""}
                                    onChange={(e) => setStageToRun(e.target.value)}
                                    size="small"
                                />
                                <Button
                                    type="primary"
                                    onClick={localRunUnrunStages}
                                    disabled={!stageToRun}
                                    icon={<PlayCircleOutlined />}
                                    block
                                    size="small"
                                >
                                    Run Stages
                                </Button>
                            </Space>
                        </Card>
                    </Col>

                    {/* Worker Management */}
                    <Col xs={24} lg={8}>
                        <Card
                            title={<><StopOutlined /> Workers</>}
                            size="small"
                        >
                            <Space direction="vertical" style={{ width: "100%" }} size="small">
                                <Input
                                    placeholder="Worker name"
                                    value={workerToKill}
                                    onChange={(e) => setWorkerToKill(e.target.value)}
                                    size="small"
                                />
                                <Button
                                    type="primary"
                                    danger
                                    onClick={localKillWorker}
                                    disabled={!workerToKill}
                                    icon={<StopOutlined />}
                                    block
                                    size="small"
                                >
                                    Kill Worker
                                </Button>

                                <Input
                                    placeholder="Job type"
                                    value={newInvokationType || ""}
                                    onChange={(e) => setNewInvokationType(e.target.value)}
                                    size="small"
                                />
                                <Input
                                    placeholder="Job state"
                                    value={newInvokationState || ""}
                                    onChange={(e) => setNewInvokationState(e.target.value)}
                                    size="small"
                                />
                                <Button
                                    type="primary"
                                    onClick={localAddInvokationToAllJobs}
                                    disabled={!newInvokationType || !newInvokationState}
                                    icon={<PlayCircleOutlined />}
                                    block
                                    size="small"
                                >
                                    Add Jobs to All Folds
                                </Button>
                            </Space>
                        </Card>
                    </Col>

                    {/* Fold Management */}
                    <Col xs={24} lg={8}>
                        <Card
                            title={<><TagOutlined /> Folds</>}
                            size="small"
                        >
                            <Space direction="vertical" style={{ width: "100%" }} size="small">
                                <Button
                                    type="primary"
                                    onClick={localSetAllUnsetModelPresets}
                                    icon={<SettingOutlined />}
                                    block
                                    size="small"
                                >
                                    Set Model Presets
                                </Button>

                                <Input
                                    placeholder="Range (e.g., '10-60')"
                                    value={foldsToKill || ""}
                                    onChange={(e) => setFoldsToKill(e.target.value)}
                                    size="small"
                                />
                                <Button
                                    type="primary"
                                    danger
                                    onClick={localKillFoldsInRange}
                                    disabled={!foldsToKill}
                                    icon={<DeleteOutlined />}
                                    block
                                    size="small"
                                >
                                    Kill Folds
                                </Button>

                                <Input
                                    placeholder="Fold range (e.g., '10-60')"
                                    value={foldsToBulkAddTag || ""}
                                    onChange={(e) => setFoldsToBulkAddTag(e.target.value)}
                                    size="small"
                                />
                                <Input
                                    placeholder="Tag name"
                                    value={tagToBulkAdd || ""}
                                    onChange={(e) => setTagToBulkAdd(e.target.value)}
                                    size="small"
                                />
                                <Button
                                    type="primary"
                                    onClick={localBulkAddTag}
                                    disabled={!foldsToBulkAddTag || !tagToBulkAdd}
                                    icon={<TagOutlined />}
                                    block
                                    size="small"
                                >
                                    Bulk Add Tags
                                </Button>
                            </Space>
                        </Card>
                    </Col>

                    {/* System Management */}
                    <Col xs={24} lg={8}>
                        <Card title={<><MailOutlined /> System</>} size="small">
                            <Space direction="vertical" style={{ width: "100%" }} size="small">
                                <Button
                                    type="primary"
                                    onClick={localSendEmail}
                                    icon={<MailOutlined />}
                                    block
                                    size="small"
                                >
                                    Send Test Email
                                </Button>

                                <Button
                                    type="primary"
                                    onClick={localPopulateOutputFpathFields}
                                    icon={<FileTextOutlined />}
                                    block
                                    size="small"
                                >
                                    Populate Output File Paths
                                </Button>

                                <Button
                                    type="primary"
                                    onClick={localBackfillDateCreated}
                                    icon={<FileTextOutlined />}
                                    block
                                    size="small"
                                >
                                    Backfill Date Created Fields
                                </Button>

                                <Button
                                    type="primary"
                                    onClick={localBackfillInputActivityFpath}
                                    icon={<FileTextOutlined />}
                                    block
                                    size="small"
                                >
                                    Backfill Input Activity Paths
                                </Button>
                            </Space>
                        </Card>
                    </Col>
                </Row>
            </div>
        </div>
    );
}

export default SudoPage;
