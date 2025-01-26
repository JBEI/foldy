import React, { useEffect, useRef } from 'react';
import 'pdbe-molstar/build/pdbe-molstar-plugin';
// import { PDBeMolstarPlugin } from 'pdbe-molstar/build/pdbe-molstar-plugin.js';
import 'pdbe-molstar/build/pdbe-molstar.css';
import { FoldyMascot } from '../../util/foldyMascot';
// If you prefer light theme, use this instead:

import './StructurePane.scss';
// import 'pdbe-molstar/pdbe-molstar-light.css';

declare global {
    interface Window {
        PDBeMolstarPlugin: any;
    }
}

interface StructurePaneProps {
    pdbString: string | null;
    pdbFailedToLoad: boolean;
}

const StructurePane: React.FC<StructurePaneProps> = ({ pdbString, pdbFailedToLoad }) => {
    const viewerRef = useRef<HTMLDivElement>(null);
    const pluginRef = useRef<any | null>(null);

    useEffect(() => {
        if (!viewerRef.current || !pdbString) return;

        // Check if PDBeMolstarPlugin is available on window
        if (!window.PDBeMolstarPlugin) {
            console.error('PDBeMolstarPlugin not found on window object');
            return;
        }

        const viewer = new window.PDBeMolstarPlugin();

        pluginRef.current = viewer;

        const options = {
            customData: {
                url: URL.createObjectURL(new Blob([pdbString], { type: 'text/plain' })),
                format: 'pdb',
                binary: false
            },
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
            hideCanvasControls: ['expand'],
            domainAnnotation: true
        };

        viewer.render(viewerRef.current, options);

        return () => {
            if (pluginRef.current) {
                pluginRef.current.destroy();
            }
        };
    }, [pdbString]);

    if (pdbFailedToLoad) {
        return (
            <div className="uk-text-center">
                <FoldyMascot text={"Looks like your structure isn't ready."} moveTextAbove={false} />
            </div>
        );
    }

    if (!pdbString) {
        return (
            <div className="uk-text-center">
                <div uk-spinner="ratio: 4"></div>
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