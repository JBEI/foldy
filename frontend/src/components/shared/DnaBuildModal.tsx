import React, { useState, useEffect } from 'react';
import { Modal, Alert, Button as AntButton, Typography, Upload, Switch, InputNumber } from 'antd';
import { QuestionCircleOutlined, UploadOutlined } from '@ant-design/icons';
import { useJwt } from 'react-jwt';
import { CSVLink } from 'react-csv';
import { createDnaBuild, DnaBuildRequest, DnaBuildResponse } from '../../api/dnaBuildApi';
import { notify } from '../../services/NotificationService';
import { FormRow, FormField } from '../../util/tabComponents';
import { TextInputControl, TextAreaControl } from '../../util/controlComponents';
import { authenticationService, isFullDecodedJwt, DecodedJwt } from '../../services/authentication.service';

const { Text, Paragraph, Title } = Typography;

interface DnaBuildModalProps {
    open: boolean;
    onClose: () => void;
    foldId: string;
    defaultSeqIds?: string[];
    title?: string;
}

interface GenBankFile {
    name: string;
    content: string;
}

export const DnaBuildModal: React.FC<DnaBuildModalProps> = ({
    open,
    onClose,
    foldId,
    defaultSeqIds = [],
    title = "Send DNA Build to Teselagen"
}) => {
    // Get current user info from JWT
    const [currentJwtString, setCurrentJwtString] = useState<string>('');
    const { decodedToken } = useJwt(currentJwtString);

    // Check if Teselagen integration is enabled
    const teselagenBackendUrl = import.meta.env.VITE_TESELAGEN_BACKEND_URL;
    const isTeselagenEnabled = !!teselagenBackendUrl;

    const [designId, setDesignId] = useState<string>('');
    const [seqIds, setSeqIds] = useState<string>(defaultSeqIds.join('\n'));
    const [numberOfMutations, setNumberOfMutations] = useState<number>(1);
    const [genBankFiles, setGenBankFiles] = useState<GenBankFile[]>([]);
    const [dryRun, setDryRun] = useState<boolean>(true);
    const [username, setUsername] = useState<string>('');
    const [otp, setOtp] = useState<string>('');
    const [projectId, setProjectId] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [showHelpModal, setShowHelpModal] = useState<boolean>(false);
    const [showSuccessModal, setShowSuccessModal] = useState<boolean>(false);
    const [lastResult, setLastResult] = useState<DnaBuildResponse | null>(null);

    // Subscribe to JWT changes
    useEffect(() => {
        const subscription = authenticationService.currentJwtString.subscribe(setCurrentJwtString);
        return () => subscription.unsubscribe();
    }, []);

    // Set default username to current user's email
    useEffect(() => {
        if (decodedToken && isFullDecodedJwt(decodedToken)) {
            setUsername(decodedToken.user_claims.email);
        }
    }, [decodedToken]);

    // Auto-download results when success modal opens
    useEffect(() => {
        if (showSuccessModal && lastResult) {
            // Trigger download after a short delay to ensure modal is rendered
            const timer = setTimeout(() => {
                const csvData = Object.entries(lastResult.seq_id_results).map(([seq_id, details]) => ({
                    seq_id,
                    success: details.success,
                    error_msg: details.error_msg || '',
                    template_used: details.template_used || '',
                    teselagen_seq_id: details.teselagen_seq_id || ''
                }));

                const csvContent = 'data:text/csv;charset=utf-8,' +
                    'seq_id,success,error_msg,template_used,teselagen_seq_id\n' +
                    csvData.map(row => Object.values(row).join(',')).join('\n');

                const encodedUri = encodeURI(csvContent);
                const link = document.createElement('a');
                link.setAttribute('href', encodedUri);
                link.setAttribute('download', `${lastResult.design_name}_build_results.csv`);
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }, 500);

            return () => clearTimeout(timer);
        }
    }, [showSuccessModal, lastResult]);

    // Reset form when modal opens with new default values
    useEffect(() => {
        if (open && defaultSeqIds.length > 0) {
            setSeqIds(defaultSeqIds.join('\n'));
        }
    }, [open, defaultSeqIds]);

    const handleGenBankFileChange = (fileList: any[]) => {
        Promise.all(
            fileList.map((file) => {
                return new Promise<GenBankFile>((resolve) => {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        const content = e.target?.result as string;
                        resolve({
                            name: file.name,
                            content: content
                        });
                    };
                    reader.readAsText(file.originFileObj || file);
                });
            })
        ).then(setGenBankFiles);
    };

    const handleCreateDnaBuild = async () => {
        if (!designId.trim()) {
            notify.error('Design ID is required.');
            return;
        }

        if (!seqIds.trim()) {
            notify.error('At least one sequence ID is required.');
            return;
        }

        if (genBankFiles.length === 0) {
            notify.error('At least one GenBank file must be uploaded.');
            return;
        }

        if (!dryRun && (!username.trim() || !otp.trim() || !projectId.trim())) {
            notify.error('Username, OTP, and Project ID are required when not in dry run mode.');
            return;
        }

        const seqIdsArray = seqIds
            .split('\n')
            .map(line => line.trim())
            .filter(line => line !== '');

        const genBankFilesMap = genBankFiles.reduce((acc, file) => {
            acc[file.name] = file.content;
            return acc;
        }, {} as { [filename: string]: string });

        const request: DnaBuildRequest = {
            design_id: designId,
            fold_id: foldId,
            genbank_files: genBankFilesMap,
            seq_ids: seqIdsArray,
            number_of_mutations: numberOfMutations,
            dry_run: dryRun
        };

        if (!dryRun) {
            request.username = username;
            request.otp = otp;
            request.project_id = projectId;
        }

        setIsLoading(true);

        try {
            const result = await createDnaBuild(request);
            setLastResult(result);

            // Count successful builds from new structure
            const successfulBuilds = Object.values(result.seq_id_results).filter(r => r.success);

            if (!result.teselagen_id) {
                notify.success(`Dry run completed successfully! Found ${successfulBuilds.length} buildable constructs.`);
            } else {
                // Show success modal instead of notification
                setShowSuccessModal(true);
                // Reset form on successful real build
                setDesignId('');
                setSeqIds(defaultSeqIds.join('\n'));
                setGenBankFiles([]);
                setUsername('');
                setOtp('');
                setProjectId('');
            }
        } catch (error: any) {
            notify.error(`Failed to create DNA build: ${error.response?.data?.message || error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const canSubmit = designId.trim() && seqIds.trim() && genBankFiles.length > 0 &&
        (dryRun || (username.trim() && otp.trim() && projectId.trim()));

    return (
        <>
            <Modal
                title={title}
                open={open}
                onCancel={onClose}
                footer={[
                    <AntButton key="cancel" onClick={onClose}>
                        Cancel
                    </AntButton>,
                    <AntButton
                        key="submit"
                        type="primary"
                        onClick={handleCreateDnaBuild}
                        disabled={!canSubmit}
                        loading={isLoading}
                    >
                        {dryRun ? 'Test Build' : 'Send to Teselagen'}
                    </AntButton>
                ]}
                width={800}
            >
                {/* Help Alert */}
                <Alert
                    message="DNA Build & Teselagen Integration"
                    description={
                        <div>
                            <Paragraph>
                                Create multi-site mutagenesis designs from GenBank templates and post them to Teselagen
                                for primer design and automated assembly.
                            </Paragraph>
                            <AntButton
                                type="link"
                                icon={<QuestionCircleOutlined />}
                                onClick={() => setShowHelpModal(true)}
                                style={{ padding: 0 }}
                            >
                                View detailed DNA build guide
                            </AntButton>
                        </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: '20px' }}
                />

                <TextInputControl
                    label="Design ID"
                    value={designId}
                    onChange={setDesignId}
                    placeholder="e.g., MyProtein_Round2"
                    required
                />

                <div style={{ marginBottom: '16px' }}>
                    <Text strong style={{ display: 'block', marginBottom: '8px' }}>
                        Number of Mutations per Construct
                    </Text>
                    <InputNumber
                        min={1}
                        max={10}
                        value={numberOfMutations}
                        onChange={(value) => setNumberOfMutations(value || 1)}
                        style={{ width: '100px' }}
                    />
                    <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                        Distance from template sequences (1 = add 1 mutation, 2 = add 2 mutations, etc.)
                    </div>
                </div>

                <TextAreaControl
                    label="Target Sequence IDs"
                    value={seqIds}
                    onChange={setSeqIds}
                    placeholder="Enter one sequence ID per line, e.g.&#10;D104G&#10;G429R&#10;D104G_G429R"
                    rows={6}
                    required
                />

                <div style={{ marginBottom: '16px' }}>
                    <Text strong style={{ display: 'block', marginBottom: '8px' }}>
                        GenBank Template Files
                    </Text>
                    <Upload
                        multiple
                        beforeUpload={() => false} // Prevent auto upload
                        accept=".gb,.gbk"
                        onChange={({ fileList }) => handleGenBankFileChange(fileList)}
                        fileList={genBankFiles.map((file, index) => ({
                            uid: index.toString(),
                            name: file.name,
                            status: 'done' as const
                        }))}
                    >
                        <AntButton icon={<UploadOutlined />}>
                            Upload GenBank Files (.gb, .gbk)
                        </AntButton>
                    </Upload>
                    <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                        Upload GenBank files for template plasmids. Each should contain exactly one CDS feature.
                    </div>
                </div>

                <FormRow>
                    <FormField>
                        <div style={{ marginBottom: '16px' }}>
                            <Text strong style={{ display: 'block', marginBottom: '8px' }}>
                                Dry Run Mode
                            </Text>
                            <Switch
                                checked={dryRun}
                                onChange={setDryRun}
                                checkedChildren="Test Only"
                                unCheckedChildren="Send to Teselagen"
                            />
                            <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                                {dryRun ? 'Validate inputs without posting to Teselagen' : 'Actually send design to Teselagen'}
                            </div>
                        </div>
                    </FormField>
                </FormRow>

                {!dryRun && (
                    <div style={{ padding: '16px', backgroundColor: '#f6ffed', border: '1px solid #b7eb8f', borderRadius: '6px', marginBottom: '16px' }}>
                        <Text strong style={{ color: '#52c41a', display: 'block', marginBottom: '12px' }}>
                            Teselagen Credentials (Required for Live Send)
                        </Text>

                        <TextInputControl
                            label="Username"
                            value={username}
                            onChange={setUsername}
                            placeholder="your.email@domain.com"
                            required
                        />

                        <TextInputControl
                            label="One-Time Password (OTP)"
                            value={otp}
                            onChange={setOtp}
                            placeholder="Get from Settings → API Password in Teselagen"
                            type="password"
                            required
                        />

                        <TextInputControl
                            label="Project ID"
                            value={projectId}
                            onChange={setProjectId}
                            placeholder="e.g., 631fdf1d-5b6d-4e03-8f0c-a0f23f5066ed"
                            required
                        />
                    </div>
                )}

                {/* Results Display */}
                {lastResult && (
                    <div style={{ marginTop: '16px', padding: '12px', backgroundColor: '#f5f5f5', borderRadius: '6px' }}>
                        <Text strong>Last Build Result:</Text>
                        <div style={{ marginTop: '8px' }}>
                            <div>✅ Successful builds: {Object.values(lastResult.seq_id_results).filter(r => r.success).length}</div>
                            {Object.values(lastResult.seq_id_results).filter(r => !r.success).length > 0 && (
                                <div>❌ Failed builds: {Object.values(lastResult.seq_id_results).filter(r => !r.success).length}</div>
                            )}
                            {lastResult.teselagen_id && (
                                <div>🔬 Teselagen Design ID: <Text code>{lastResult.teselagen_id}</Text></div>
                            )}
                        </div>
                        <div style={{ marginTop: '12px' }}>
                            <CSVLink
                                data={Object.entries(lastResult.seq_id_results).map(([seq_id, details]) => ({
                                    seq_id,
                                    success: details.success,
                                    error_msg: details.error_msg || '',
                                    template_used: details.template_used || '',
                                    teselagen_seq_id: details.teselagen_seq_id || ''
                                }))}
                                filename={`${lastResult.design_name}_build_results.csv`}
                                style={{ textDecoration: "none" }}
                            >
                                <AntButton size="small" type="primary">
                                    Download Design Results
                                </AntButton>
                            </CSVLink>
                        </div>
                    </div>
                )}
            </Modal>

            {/* Help Modal */}
            <Modal
                title="DNA Build Guide"
                open={showHelpModal}
                onCancel={() => setShowHelpModal(false)}
                footer={[
                    <AntButton key="close" onClick={() => setShowHelpModal(false)}>
                        Close
                    </AntButton>
                ]}
                width={700}
            >
                <div>
                    <Title level={4}>What is DNA Build?</Title>
                    <Paragraph>
                        This tool creates multi-site mutagenesis designs from your GenBank template files and posts them
                        to Teselagen for automated primer design and Gibson assembly protocols.
                    </Paragraph>

                    <Title level={4}>How It Works</Title>
                    <ol>
                        <li><Text strong>Template Discovery:</Text> Analyzes your GenBank files to determine what mutations they contain</li>
                        <li><Text strong>Build Planning:</Text> Finds optimal template sequences that are the right "distance" away from your targets</li>
                        <li><Text strong>Construct Design:</Text> Creates Gibson assembly designs with alternating template segments and synthetic mutation codons</li>
                        <li><Text strong>Teselagen Integration:</Text> Posts the complete design for primer design and lab automation</li>
                    </ol>

                    <Title level={4}>Input Requirements</Title>
                    <ul>
                        <li><Text strong>GenBank Files:</Text> Template plasmids you have in stock
                            <ul style={{ marginTop: '8px', marginLeft: '16px' }}>
                                <li>Must contain exactly one CDS feature on the forward strand</li>
                                <li>CDS should include start and stop codons</li>
                                <li>Files should be indexed at position 0 within the antibiotic marker</li>
                            </ul>
                        </li>
                        <li><Text strong>Sequence IDs:</Text> Target mutants you want to build (e.g., "D104G", "D104G_G429R")</li>
                        <li><Text strong>Number of Mutations:</Text> How many new mutations to add per construct</li>
                    </ul>

                    <Title level={4}>Mutation Distance</Title>
                    <Paragraph>
                        The "Number of Mutations" parameter controls the build strategy:
                    </Paragraph>
                    <ul>
                        <li><Text strong>Distance 1:</Text> Add 1 mutation to existing templates (e.g., WT → D104G, G429R → D104G_G429R)</li>
                        <li><Text strong>Distance 2:</Text> Add 2 mutations to existing templates (e.g., WT → D104G_G429R)</li>
                        <li>The tool automatically finds the best template that's exactly that distance away</li>
                    </ul>

                    <Title level={4}>Dry Run vs Live Mode</Title>
                    <ul>
                        <li><Text strong>Dry Run (Default):</Text> Tests your inputs and shows what would be built without posting to Teselagen</li>
                        <li><Text strong>Live Mode:</Text> Actually creates the design in Teselagen - requires your credentials</li>
                    </ul>

                    <Alert
                        message="Teselagen Credentials"
                        description={
                            <div>
                                <Text strong>Username:</Text> Your Teselagen email<br />
                                <Text strong>OTP:</Text> Get from Settings → API Password (these expire, so get fresh ones)<br />
                                <Text strong>Project ID:</Text> Go to Settings → Projects, show the ID column, copy the project UUID
                            </div>
                        }
                        type="info"
                        showIcon
                        style={{ marginTop: '12px' }}
                    />

                    <Title level={4}>Next Steps</Title>
                    <Paragraph>
                        After posting to Teselagen, follow the{' '}
                        <Text strong>FolDE SDM Protocol</Text> in your Teselagen project to:
                    </Paragraph>
                    <ol>
                        <li>Review and approve the assembly design</li>
                        <li>Order primers and DNA synthesis</li>
                        <li>Use lab robots to complete the builds</li>
                    </ol>
                </div>
            </Modal>

            {/* Success Modal */}
            <Modal
                title="DNA Build Created Successfully! 🎉"
                open={showSuccessModal}
                onCancel={() => {
                    setShowSuccessModal(false);
                    onClose();
                }}
                footer={[
                    lastResult && (
                        <CSVLink
                            key="download"
                            data={Object.entries(lastResult.seq_id_results).map(([seq_id, details]) => ({
                                seq_id,
                                success: details.success,
                                error_msg: details.error_msg || '',
                                template_used: details.template_used || '',
                                teselagen_seq_id: details.teselagen_seq_id || ''
                            }))}
                            filename={`${lastResult.design_name}_build_results.csv`}
                            style={{ textDecoration: "none" }}
                        >
                            <AntButton>
                                Download Design Results
                            </AntButton>
                        </CSVLink>
                    ),
                    <AntButton
                        key="close"
                        onClick={() => {
                            setShowSuccessModal(false);
                            onClose();
                        }}
                    >
                        Close
                    </AntButton>,
                    lastResult?.teselagen_id && isTeselagenEnabled && (
                        <AntButton
                            key="open-teselagen"
                            type="primary"
                            onClick={() => {
                                const teselagenUrl = `${teselagenBackendUrl}/client/designs/${lastResult.teselagen_id}`;
                                window.open(teselagenUrl, '_blank');
                            }}
                        >
                            Open in Teselagen
                        </AntButton>
                    )
                ].filter(Boolean)}
                width={600}
            >
                <div style={{ padding: '20px 0' }}>
                    <div style={{ marginBottom: '24px', textAlign: 'center' }}>
                        <div style={{ fontSize: '48px', marginBottom: '16px' }}>🧬</div>
                        <Typography.Title level={4} style={{ color: '#52c41a', margin: 0 }}>
                            Your DNA build has been successfully created!
                        </Typography.Title>
                    </div>

                    {lastResult && (
                        <div style={{ marginBottom: '24px' }}>
                            <div style={{ marginBottom: '16px' }}>
                                <Typography.Text strong>Build Summary:</Typography.Text>
                            </div>
                            <div style={{ marginLeft: '16px' }}>
                                <div>✅ Successful builds: {Object.values(lastResult.seq_id_results).filter(r => r.success).length}</div>
                                {Object.values(lastResult.seq_id_results).filter(r => !r.success).length > 0 && (
                                    <div style={{ color: '#ff4d4f' }}>❌ Failed builds: {Object.values(lastResult.seq_id_results).filter(r => !r.success).length}</div>
                                )}
                            </div>
                        </div>
                    )}

                    {lastResult?.teselagen_id && isTeselagenEnabled && (
                        <div style={{
                            padding: '16px',
                            backgroundColor: '#f6ffed',
                            borderRadius: '6px',
                            border: '1px solid #b7eb8f'
                        }}>
                            <Typography.Text strong style={{ color: '#52c41a' }}>
                                Teselagen Design ID:
                            </Typography.Text>
                            <Typography.Text code style={{ marginLeft: '8px' }}>
                                {lastResult.teselagen_id}
                            </Typography.Text>
                            <div style={{ marginTop: '12px', fontSize: '14px', color: '#666' }}>
                                Click "Open in Teselagen" to view your design and begin primer ordering and automated assembly.
                            </div>
                        </div>
                    )}
                </div>
            </Modal>
        </>
    );
};
