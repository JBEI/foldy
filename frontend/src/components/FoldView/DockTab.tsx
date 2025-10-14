import fileDownload from "js-file-download";
import React, { useMemo, useState } from "react";
import { Button } from 'antd';
import { LeftOutlined, RightOutlined, ClockCircleOutlined, DownloadOutlined, EyeOutlined, FrownOutlined, RedoOutlined, DeleteOutlined, PlusOutlined } from '@ant-design/icons';
import { getDockSdf, postDock } from "../../api/dockApi";
import { Dock, Invokation, DockInput } from "../../types/types";
import { notify } from "../../services/NotificationService";
import { TabContainer, DescriptionSection, TableSection } from "../../util/tabComponents";
import { AntTable, createActionButtons, defaultExpandableContent } from "../../util/AntTable";
import { DockModal } from "../shared/DockModal";

interface DockTabProps {
    foldId: number;
    foldName: string | null;
    foldSequence: string | undefined;
    docks: Dock[] | null;
    jobs: Invokation[] | null;

    // UI Commands managed by the FoldView.
    displayedLigandNames: string[];
    ranks: { [ligandname: string]: number };
    displayLigandPose: (ligandName: string) => void;
    shiftFrame: (ligandName: string, shift: number) => void;
    deleteLigandPose: (ligandId: number, ligandName: string) => void;
}


const DockTab = React.memo((props: DockTabProps) => {
    const [showDockModal, setShowDockModal] = useState(false);

    const getDockState = (dock: Dock, jobs: Invokation[] | null) => {
        if (!jobs) return "queued";
        const job = jobs.find((invokation) => invokation.id === dock.invokation_id);
        return job?.state || "failed";
    };

    const downloadLigandPose = (ligandName: string) => {
        notify.info(`Downloading SDF file for ${ligandName}`);
        getDockSdf(props.foldId, ligandName).then(
            (sdf: Blob) => {
                if (!props.foldName) return;
                fileDownload(sdf, `${props.foldName}_${ligandName}.sdf`);
            },
            (error) => {
                notify.error(error.toString());
            }
        );
    };

    const rerunDock = (dock: Dock) => {
        const dockCopy: DockInput = { ...dock, fold_id: props.foldId };
        postDock(dockCopy).then(
            () => notify.success(`Successfully restarted docking for ${dock.ligand_name}`),
            (error) => notify.error(`Docking ${dock.ligand_name} failed: ${error}`)
        );
    };

    const getFit = (dock: Dock) => {
        if (dock.tool === "diffdock") {
            const confidenceStr =
                dock.pose_confidences?.split(",")[
                (props.ranks[dock.ligand_name] || 1) - 1
                ];
            return confidenceStr ? parseFloat(confidenceStr) : null;
        }
        return (props.ranks[dock.ligand_name] || 1) === 1 ? dock.pose_energy : null;
    };

    const compareValues = (
        key: keyof Dock | "fit",
        direction: "ascending" | "descending"
    ) => {
        return (a: Dock, b: Dock) => {
            let aValue, bValue;

            if (key === "fit") {
                if (a.tool !== b.tool) {
                    aValue = a.tool;
                    bValue = b.tool;
                } else {
                    aValue = Number(getFit(a));
                    bValue = Number(getFit(b));
                }
            } else {
                aValue = a[key];
                bValue = b[key];
            }

            if (aValue === bValue) return 0;
            if (aValue === null) return direction === "ascending" ? -1 : 1;
            if (bValue === null) return direction === "ascending" ? 1 : -1;
            return aValue < bValue
                ? direction === "ascending"
                    ? -1
                    : 1
                : direction === "ascending"
                    ? 1
                    : -1;
        };
    };

    const sortedDocks = useMemo(() => {
        if (!props.docks) return null;
        return [...props.docks].sort(
            compareValues("ligand_name", "ascending")
        );
    }, [props.docks]);


    return (
        <TabContainer>
            {/* Description Section */}
            <DescriptionSection title="Small Molecule Docking">
                <p>
                    Use <a href="https://onlinelibrary.wiley.com/doi/pdf/10.1002/jcc.21334">Autodock Vina</a> or DiffDock to predict ligand poses. Sort and manage docking results or dock new ligands below.
                </p>
            </DescriptionSection>

            {/* Docking Results Table */}
            <TableSection
                title="Docking Runs"
                extra={
                    <Button
                        type="primary"
                        icon={<PlusOutlined />}
                        onClick={() => setShowDockModal(true)}
                    >
                        New
                    </Button>
                }
            >
                <AntTable<Dock>
                    dataSource={sortedDocks || []}
                    rowKey="id"
                    expandableContent={defaultExpandableContent}
                    columns={[
                        {
                            key: 'ligand_name',
                            title: 'Name',
                            dataIndex: 'ligand_name',
                            sortable: true,
                            sorter: (a, b) => a.ligand_name.localeCompare(b.ligand_name),
                        },
                        {
                            key: 'fit',
                            title: 'Fit',
                            sortable: true,
                            sorter: (a, b) => {
                                const aFit = getFit(a) || 0;
                                const bFit = getFit(b) || 0;
                                return aFit - bFit;
                            },
                            render: (_, dock) => getFit(dock),
                        },
                        {
                            key: 'rank',
                            title: 'Rank',
                            render: (_, dock) => props.ranks[dock.ligand_name],
                        },
                        {
                            key: 'tool',
                            title: 'Tool',
                            dataIndex: 'tool',
                            sortable: true,
                            sorter: (a, b) => (a.tool || '').localeCompare(b.tool || ''),
                        },
                        {
                            key: 'bounding_box',
                            title: 'Bounding Box',
                            render: (_, dock) => (
                                dock.bounding_box_residue && dock.bounding_box_radius_angstrom
                                    ? `${dock.bounding_box_residue} (${dock.bounding_box_radius_angstrom} Å)`
                                    : "N/A"
                            ),
                        },
                        {
                            key: 'ligand_smiles',
                            title: 'SMILES',
                            dataIndex: 'ligand_smiles',
                            ellipsis: true,
                            width: 200,
                            sortable: true,
                            sorter: (a, b) => a.ligand_smiles.localeCompare(b.ligand_smiles),
                            render: (smiles) => (
                                <span title={smiles}>{smiles}</span>
                            ),
                        },
                        {
                            key: 'actions',
                            title: 'Actions',
                            width: 250,
                            render: (_, dock) => {
                                const dockState = getDockState(dock, props.jobs);
                                const buttons = [];

                                // Status button
                                if (dockState === "queued" || dockState === "running") {
                                    buttons.push({
                                        icon: <ClockCircleOutlined />,
                                        onClick: () => {},
                                        tooltip: `Docking is currently ${dockState}`,
                                        disabled: true,
                                    });
                                } else if (dockState === "failed") {
                                    buttons.push({
                                        icon: <FrownOutlined />,
                                        onClick: () => {},
                                        tooltip: 'Docking failed. Consider rerunning this docking job.',
                                        disabled: true,
                                    });
                                } else {
                                    buttons.push({
                                        icon: <EyeOutlined />,
                                        onClick: () => props.displayLigandPose(dock.ligand_name),
                                        tooltip: 'View this ligand\'s pose in the visualization pane.',
                                    });
                                }

                                // Navigation buttons (if ligand is displayed)
                                if (props.displayedLigandNames.includes(dock.ligand_name)) {
                                    buttons.push({
                                        icon: <LeftOutlined />,
                                        onClick: () => props.shiftFrame(dock.ligand_name, -1),
                                        tooltip: 'View the previous pose prediction for this ligand.',
                                    });
                                    buttons.push({
                                        icon: <RightOutlined />,
                                        onClick: () => props.shiftFrame(dock.ligand_name, 1),
                                        tooltip: 'View the next pose prediction for this ligand.',
                                    });
                                }

                                // Action buttons
                                buttons.push({
                                    icon: <DeleteOutlined />,
                                    onClick: () => props.deleteLigandPose(dock.id, dock.ligand_name),
                                    tooltip: 'Delete this docking result.',
                                    danger: true,
                                });

                                buttons.push({
                                    icon: <RedoOutlined />,
                                    onClick: () => rerunDock(dock),
                                    tooltip: 'Rerun this docking job.',
                                });

                                buttons.push({
                                    icon: <DownloadOutlined />,
                                    onClick: () => downloadLigandPose(dock.ligand_name),
                                    tooltip: 'Download the SDF file for this ligand pose.',
                                });

                                return createActionButtons(buttons);
                            },
                        },
                    ]}
                />
            </TableSection>

            {/* Dock Modal */}
            <DockModal
                open={showDockModal}
                onClose={() => setShowDockModal(false)}
                foldIds={[props.foldId]}
                existingLigands={{
                    [props.foldId]: (props.docks || []).map((dock) => dock.ligand_name),
                }}
                title="Dock New Ligands"
            />
        </TabContainer>
    );
});

export default DockTab;
