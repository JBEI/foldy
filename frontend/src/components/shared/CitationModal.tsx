import React from 'react';
import { Modal, Button as AntButton, Typography, Collapse, Card, Alert } from 'antd';

const { Text, Paragraph } = Typography;
const { Panel } = Collapse;

interface CitationModalProps {
    open: boolean;
    onClose: () => void;
}

const CitationModal: React.FC<CitationModalProps> = ({ open, onClose }) => {
    return (
        <Modal
            title="Citations & References"
            open={open}
            onCancel={onClose}
            footer={[
                <AntButton key="close" onClick={onClose}>
                    Close
                </AntButton>
            ]}
            width={800}
        >
            <div>
                <Paragraph>
                    <Text strong>Please cite the appropriate papers when publishing research that uses this platform:</Text>
                </Paragraph>

                <Collapse ghost>
                    <Panel header={<Text strong>🧪 Foldy Platform (Required for any use of this tool)</Text>} key="foldy">
                        <Card size="small" style={{ backgroundColor: '#f9f9f9' }}>
                            <Text>Cite this paper if you use the Foldy platform for structure prediction, analysis, or any other purpose:</Text>
                            <pre style={{
                                marginTop: '8px',
                                padding: '12px',
                                backgroundColor: '#fff',
                                border: '1px solid #d9d9d9',
                                borderRadius: '4px',
                                fontSize: '12px',
                                lineHeight: '1.4'
                            }}>
                                {`@article{roberts2023foldy,
  title = {Foldy: A democratized protein folding platform},
  author = {Roberts, Jacob B. and Nava, Alberto A. and Pearson, Allison N. and
            Incha, Matthew R. and Valencia, Luis E. and Ma, Melody and
            Rao, Abhay and Keasling, Jay D.},
  year = {2023},
  doi = {10.1101/2023.05.11.540333},
  journal = {bioRxiv}
}`}
                            </pre>
                        </Card>
                    </Panel>

                    <Panel header={<Text strong>📝 Boltz: Structure Prediction Engine (Required)</Text>} key="boltz">
                        <Card size="small" style={{ backgroundColor: '#f9f9f9' }}>
                            <Text>Use this citation if you use Boltz-2x structure predictions from this platform:</Text>
                            <pre style={{
                                marginTop: '8px',
                                padding: '12px',
                                backgroundColor: '#fff',
                                border: '1px solid #d9d9d9',
                                borderRadius: '4px',
                                fontSize: '12px',
                                lineHeight: '1.4'
                            }}>
                                {`@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and
            Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and
            Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and
            Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}`}
                            </pre>
                        </Card>
                    </Panel>

                    <Panel header={<Text strong>🧬 ColabFold: MSA Generation (If Applicable)</Text>} key="colabfold">
                        <Card size="small" style={{ backgroundColor: '#f9f9f9' }}>
                            <Text>Include this citation if Foldy generated Multiple Sequence Alignments (MSAs) for your fold:</Text>
                            <pre style={{
                                marginTop: '8px',
                                padding: '12px',
                                backgroundColor: '#fff',
                                border: '1px solid #d9d9d9',
                                borderRadius: '4px',
                                fontSize: '12px',
                                lineHeight: '1.4'
                            }}>
                                {`@article{mirdita2022colabfold,
  title = {ColabFold: making protein folding accessible to all},
  author = {Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and
            Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal = {Nature Methods},
  volume = {19},
  number = {6},
  pages = {679--682},
  year = {2022},
  publisher = {Nature Publishing Group}
}`}
                            </pre>
                        </Card>
                    </Panel>

                    <Panel header={<Text strong>🔬 Additional Tools (If Used)</Text>} key="additional">
                        <Card size="small" style={{ backgroundColor: '#f9f9f9' }}>
                            <Text>If you used additional features from this platform, also cite:</Text>
                            <ul style={{ marginTop: '8px' }}>
                                <li><Text strong>AutoDock Vina:</Text> For molecular docking results</li>
                                <li><Text strong>Pfam:</Text> For protein family annotations</li>
                                <li><Text strong>DiffDock:</Text> For AI-powered molecular docking</li>
                            </ul>
                            <Text type="secondary" style={{ fontSize: '12px' }}>
                                Specific citations for these tools can be found in their respective documentation.
                            </Text>
                        </Card>
                    </Panel>
                </Collapse>

                <Alert
                    message="Citation Best Practices"
                    description="Most reference managers can import these citations directly from DOI. Always cite both the Foldy platform and the underlying computational methods you used."
                    type="info"
                    showIcon
                    style={{ marginTop: '16px' }}
                />
            </div>
        </Modal>
    );
};

export default CitationModal;
