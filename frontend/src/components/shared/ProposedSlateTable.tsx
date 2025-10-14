import React, { useState, useEffect, useMemo } from 'react';
import ReactDataGrid from 'react-data-grid';
import { Selection } from '../FoldView/StructurePane';
import { BoltzYamlHelper } from '../../util/boltzYamlHelper';
import { DataTableContainer } from '../../util/plotComponents';
import { ButtonGroup } from '../../util/tabComponents';
import { notify } from '../../services/NotificationService';
import { Button } from 'antd';

export interface ProposedSlateTableColumn {
    key: string;
    name: string;
    sortable?: boolean;
    resizable?: boolean;
    width?: number;
    sortDescendingFirst?: boolean;
    formatter?: ({ row }: { row: any }) => React.ReactNode;
}

export interface ProposedSlateTableRow {
    seqId: string;
    [key: string]: any;
}

interface ProposedSlateTableProps {
    description?: string;
    data: ProposedSlateTableRow[];
    columns: ProposedSlateTableColumn[];
    yamlConfig?: string | null;
    setSelectedSubsequence?: (selection: Selection | null) => void;
    enableRowSelection?: boolean; // deprecated
    rowSelection?: boolean;
    enableRowClick?: boolean;
    showCopyButton?: boolean;
    showHighlightButton?: boolean;
    showHighlightOnModelButton?: boolean;
    showSlateBuilderButton?: boolean;
    disableSlateBuilderButton?: boolean;
    disableRowSelection?: boolean; // Override to disable all row selection functionality
    onSelectedRowsChange?: (selectedSeqIds: string[]) => void;
    onBuildSlate?: (seqIds: string[]) => void;
    minHeight?: number;
}

const ProposedSlateTable: React.FC<ProposedSlateTableProps> = ({
    description,
    data,
    columns,
    yamlConfig,
    setSelectedSubsequence,
    enableRowSelection = false, // deprecated
    rowSelection = false,
    enableRowClick = false,
    showCopyButton = true,
    showHighlightButton = true,
    showHighlightOnModelButton = true,
    showSlateBuilderButton = false,
    disableSlateBuilderButton = false,
    disableRowSelection = false,
    onSelectedRowsChange,
    onBuildSlate,
    minHeight = 400,
}) => {
    // Handle deprecated prop with warning and respect disableRowSelection override
    const isRowSelectionEnabled = disableRowSelection ? false : (rowSelection !== undefined ? rowSelection : enableRowSelection);
    if (enableRowSelection !== undefined && rowSelection === undefined) {
        console.warn('enableRowSelection has been deprecated and will be removed in a future version. Please use rowSelection instead.');
    }

    const [sortColumn, setSortColumn] = useState<string | null>(null);
    const [sortDirection, setSortDirection] = useState<'ASC' | 'DESC'>('DESC');
    const [selectedSeqIds, setSelectedSeqIds] = useState<string[]>([]);

    const sortedData = useMemo(() => {
        if (!sortColumn) return data;

        return [...data].sort((a, b) => {
            const aValue = a[sortColumn];
            const bValue = b[sortColumn];
            if (aValue === null || aValue === undefined) return 1;
            if (bValue === null || bValue === undefined) return -1;
            return sortDirection === 'ASC'
                ? (aValue < bValue ? -1 : 1)
                : (aValue > bValue ? -1 : 1);
        });
    }, [data, sortColumn, sortDirection]);

    const seqIdListToLociList = (seqIdList: string[]): number[] => {
        const lociToHighlightList: number[] = [];
        seqIdList.forEach(seqId => {
            // Check if seqId is defined and is a string
            if (!seqId || typeof seqId !== 'string') {
                console.warn(`Invalid seqId: ${seqId}`);
                return;
            }

            // Handle all mutations (single or multi) by splitting on underscore
            const alleleIds = seqId.split('_');
            alleleIds.forEach(alleleId => {
                try {
                    // Use full match regex to extract locus from each allele (e.g., "K2715A" -> 2715)
                    const match = alleleId.match(/^[A-Z](\d+)[A-Z]$/);
                    if (match) {
                        const locus = parseInt(match[1]);
                        if (!isNaN(locus)) {
                            lociToHighlightList.push(locus);
                        }
                    } else {
                        console.warn(`Invalid allele format: ${alleleId} in seqId: ${seqId}`);
                    }
                } catch (error) {
                    console.error(`Error parsing allele ${alleleId} in seqId ${seqId}:`, error);
                }
            });
        });

        return Array.from(new Set(lociToHighlightList));
    };

    const copyMutationsToClipboard = () => {
        if (!data) return;
        const mutations = data
            .map(row => row.seqId)
            .join('\n');

        navigator.clipboard.writeText(mutations);
        notify.success('Seq IDs copied to clipboard!');
    };

    const highlightResiduesOnModel = () => {
        if (!data || !setSelectedSubsequence) return;
        if (!yamlConfig) {
            console.log('No yaml config, cannot highlight residues on model.');
            return;
        }
        const configHelper = new BoltzYamlHelper(yamlConfig);
        if (configHelper.getProteinSequences().length > 1) {
            notify.error('Cannot currently highlight residues on multimers.');
            return;
        }
        let chainId = configHelper.getProteinSequences()[0][0];

        const uniqueLociToHighlight = seqIdListToLociList(data.map(row => row.seqId));
        const specialSelectedLoci = seqIdListToLociList(selectedSeqIds);

        const selection = uniqueLociToHighlight.map(locus => {
            const color = specialSelectedLoci.includes(locus) ? "#FFD700" : "#39f";
            return {
                struct_asym_id: chainId,
                start_residue_number: locus,
                end_residue_number: locus,
                color: color,
            };
        });

        setSelectedSubsequence({
            data: selection,
            nonSelectedColor: "white",
        });
    };

    useEffect(() => {
        if (!disableRowSelection && (isRowSelectionEnabled || enableRowClick)) {
            highlightResiduesOnModel();
        }
    }, [selectedSeqIds, data, disableRowSelection]);

    useEffect(() => {
        if (onSelectedRowsChange) {
            onSelectedRowsChange(selectedSeqIds);
        }
    }, [selectedSeqIds, onSelectedRowsChange]);

    const handleRowSelect = (rows: any[]) => {
        if (isRowSelectionEnabled) {
            const newSelectedSeqIds = rows.map(row => row.seqId);
            setSelectedSeqIds(newSelectedSeqIds);
        }
    };

    const handleRowClick = (_: any, row: any) => {
        if (enableRowClick) {
            setSelectedSeqIds([row.seqId]);
        }
    };

    return (
        <DataTableContainer>
            {description && <p>{description}</p>}

            <ReactDataGrid
                columns={columns as any}
                rowGetter={i => sortedData[i]}
                rowsCount={sortedData.length}
                onGridSort={(sortCol, direction) => {
                    setSortColumn(sortCol);
                    setSortDirection(direction.toUpperCase() as 'ASC' | 'DESC');
                }}
                rowSelection={isRowSelectionEnabled ? {
                    showCheckbox: true,
                    onRowsSelected: handleRowSelect,
                    onRowsDeselected: handleRowSelect,
                    selectBy: {
                        keys: {
                            rowKey: 'seqId',
                            values: selectedSeqIds
                        }
                    }
                } : undefined}
                onRowClick={(enableRowClick && !disableRowSelection) ? handleRowClick : undefined}
                minHeight={minHeight}
            />

            {(showCopyButton || showHighlightButton || showSlateBuilderButton) && (
                <ButtonGroup>
                    {showCopyButton && (
                        <Button
                            onClick={copyMutationsToClipboard}
                        >
                            Copy mutations to clipboard
                        </Button>
                    )}
                    {showHighlightButton && showHighlightOnModelButton && setSelectedSubsequence && (
                        <Button
                            type="primary"
                            onClick={highlightResiduesOnModel}
                        >
                            Highlight residues on model
                        </Button>
                    )}
                    {showSlateBuilderButton && onBuildSlate && (
                        <Button
                            type="primary"
                            disabled={disableSlateBuilderButton}
                            onClick={() => onBuildSlate(selectedSeqIds.length > 0 ? selectedSeqIds : data.map(row => row.seqId))}
                        >
                            Send mutants to slate builder
                        </Button>
                    )}
                </ButtonGroup>
            )}
        </DataTableContainer>
    );
};

export default ProposedSlateTable;
