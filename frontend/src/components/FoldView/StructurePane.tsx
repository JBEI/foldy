import React, { useEffect, useRef } from 'react';
// import { PDBeMolstarPlugin } from 'pdbe-molstar/build/pdbe-molstar-plugin.js';
import 'pdbe-molstar/build/pdbe-molstar-plugin.js';
import 'pdbe-molstar/build/pdbe-molstar.css';
// import 'pdbe-molstar-plugin';   // JavaScript bundle
// import 'pdbe-molstar.css';         // Viewer styling
import { Spin } from 'antd';
import { FoldyMascot } from '../../util/foldyMascot';
// If you prefer light theme, use this instead:

import './StructurePane.scss';
// import 'pdbe-molstar/pdbe-molstar-light.css';

declare global {
    interface Window {
        PDBeMolstarPlugin: any;
    }
}

export interface SelectionData {
    struct_asym_id: string;
    start_residue_number: number;
    end_residue_number: number;
    color: string;
}

export interface Selection {
    data: SelectionData[];
    nonSelectedColor?: string;
}

interface StructurePaneProps {
    cifString: string | null;
    pdbString: string | null;
    structureFailedToLoad: boolean;
    selection: Selection | null;
}

const StructurePane: React.FC<StructurePaneProps> = ({ cifString, pdbString, structureFailedToLoad, selection }) => {
    const viewerRef = useRef<HTMLDivElement>(null);
    const pluginRef = useRef<any | null>(null);

    useEffect(() => {
        if (!viewerRef.current) return;

        if (!window.PDBeMolstarPlugin) {
            console.error('PDBeMolstarPlugin not found on window object');
            return;
        }

        let fileData;
        if (cifString) {
            fileData = {
                url: URL.createObjectURL(new Blob([cifString], { type: 'text/plain' })),
                format: 'cif',
                binary: false
            };
        } else if (pdbString) {
            fileData = {
                url: URL.createObjectURL(new Blob([pdbString], { type: 'text/plain' })),
                format: 'pdb',
                binary: false
            };
        } else {
            return;
        }

        const viewer = new window.PDBeMolstarPlugin();
        pluginRef.current = viewer;

        // API for PDBe-Molstar:
        // https://github.com/molstar/pdbe-molstar/wiki/1.-PDBe-Molstar-as-JS-plugin
        const options = {
            customData: fileData,
            bgColor: 'white',
            defaultPreset: 'default',
            visualStylesSpec: {
                polymer: {
                    type: 'cartoon',
                    params: {
                        aspectRatio: 5,
                        quality: 'high',
                        alpha: 1,
                    }
                },
                ligand: {
                    type: 'ball-and-stick',
                    params: {
                        scale: 1.25
                    }
                }
            },
            alphafoldView: true,
            hideCanvasControls: ['expand'],
            domainAnnotation: true,
            hideControls: true,
            loadingOverlay: false
        };

        viewer.render(viewerRef.current, options);

        return () => {
            if (pluginRef.current) {
                // Double check that destroy is a function
                if (typeof pluginRef.current.destroy === 'function') {
                    pluginRef.current.destroy();
                }
            }
        };
    }, [cifString, pdbString]);

    // Separate effect for handling selection changes
    useEffect(() => {
        if (!pluginRef.current) return;

        console.log("SELECTION: ", selection?.data);
        pluginRef.current.visual.clearSelection();

        if (!selection) {
            // Clear selection when selection is null
            return;
        }

        pluginRef.current.visual.select({
            data: selection.data,
            nonSelectedColor: selection.nonSelectedColor
        });
    }, [selection]);

    if (structureFailedToLoad) {
        return (
            <div style={{ textAlign: 'center' }}>
                <FoldyMascot text={"Looks like your structure failed to load."} moveTextAbove={false} isCartwheeling={false} isKanKaning={false} />
            </div>
        );
    }

    if (!cifString && !pdbString) {
        return (
            <div style={{ textAlign: 'center', padding: '60px 0' }}>
                <Spin size="large" />
            </div>
        );
    }

    return (
        <div
            ref={viewerRef}
            style={{ width: '100%', height: '100%' }}
        />
    );
};

export default StructurePane;
