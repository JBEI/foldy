import React from 'react';
import { Modal } from 'antd';
import { NewDockPrompt } from '../../util/newDockPrompt';

interface DockModalProps {
    open: boolean;
    onClose: () => void;
    foldIds: number[];
    existingLigands: { [foldId: number]: Array<string> };
    title?: string;
}

export const DockModal: React.FC<DockModalProps> = ({
    open,
    onClose,
    foldIds,
    existingLigands,
    title = "Dock New Ligands"
}) => {
    return (
        <Modal
            title={title}
            open={open}
            onCancel={onClose}
            footer={null}
            width={800}
        >
            <NewDockPrompt
                foldIds={foldIds}
                existingLigands={existingLigands}
            />
        </Modal>
    );
};
