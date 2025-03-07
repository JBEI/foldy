import React, { useState } from "react";
import UIkit from "uikit";
import {
    FileBrowser,
    FileNavbar,
    FileToolbar,
    FileList,
    ChonkyActions,
    ChonkyFileActionData,
} from "chonky";
import { FileInfo } from "../../types/types";
import { downloadFileStraightToFilesystem } from "../../api/fileApi";
import { removeLeadingSlash } from "../../api/commonApi";

interface FileTabProps {
    foldId: number;
    foldName: string | null;
    pdbString: string | null;
    maybeDownloadPdb: () => void;
    files: FileInfo[];
    setErrorText: (a: string) => void;
}

const FileTab = React.memo((props: FileTabProps) => {
    // Start at root folder "/"
    const [currentFolderPath, setCurrentFolderPath] = useState<string>("/");

    // Normalize the raw files coming from the backend.
    // For each FileInfo, treat the key as its full path.
    // We mark an entry as a file (isDir=false) and then add synthetic directories for all parent folders.
    const normalizeFiles = () => {
        if (!props.files) return [];

        // First, map each file entry to a normalized object.
        // For backend files, assume they never end with "/" (they are files),
        // even if their key contains "/" indicating a subdirectory.
        const normalized = props.files.map(file => {
            let key = file.key;
            if (!key.startsWith("/")) {
                key = "/" + key;
            }
            return {
                id: key,
                name: key.split("/").filter(Boolean).pop() || file.key,
                path: key,
                isDir: false, // by default, assume a backend entry is a file
                modDate: new Date(file.modified),
                size: file.size,
            }
        });

        // Next, build a set of all parent directory paths.
        // For each file with a path like "boltz/input.yml", we want to include "boltz/".
        const parentDirs = new Set<string>();
        normalized.forEach(entry => {
            const parts = entry.path.split("/").filter(Boolean);
            let cumulative = "/";
            // For each level except the file itself, add a parent folder.
            for (let i = 0; i < parts.length - 1; i++) {
                cumulative += parts[i] + "/";
                parentDirs.add(cumulative);
            }
        });

        // Create synthetic directory objects for each parent directory if not already present.
        parentDirs.forEach(dir => {
            // Check if any entry already has this exact path.
            const exists = normalized.some(e => e.path === dir && e.isDir === true);
            if (!exists) {
                normalized.push({
                    id: dir,
                    name: dir.split("/").filter(Boolean).pop() || dir,
                    path: dir,
                    isDir: true,
                    modDate: new Date(),
                    size: 0,
                });
            }
        });

        return normalized;
    };

    const allFiles = normalizeFiles();

    // Return only the immediate children of the current folder.
    // For example, if currentFolderPath = "boltz/", then "boltz/input.yml" and "boltz/boltz_results_input/" are immediate children,
    // but "boltz/boltz_results_input/msa/input_0.csv" is not.
    const getImmediateChildren = () => {
        return allFiles.filter(file => {
            // The file's path must start with the current folder path.
            if (!file.path.startsWith(currentFolderPath)) return false;

            // Remove the current folder prefix.
            let remainder = file.path.slice(currentFolderPath.length);
            if (remainder.startsWith("/")) remainder = remainder.slice(1);
            if (!remainder) return false; // Skip the folder itself

            // Immediate children will have exactly one segment.
            const parts = remainder.split("/").filter(Boolean);
            return parts.length === 1;
        }).map(file => ({
            id: file.path,
            name: file.name,
            path: file.path,
            isDir: file.isDir,
            modDate: file.modDate,
            size: file.size,
        }));
    };

    // Build the folder chain (breadcrumb) from root to current folder.
    // E.g. "/" => Root, "boltz/" => Root > boltz.
    const getFolderChain = () => {
        console.log("Getting folder chain for:", currentFolderPath);
        if (currentFolderPath === "/") {
            return [{ id: "/", name: "Root", isDir: true }];
        }
        const parts = currentFolderPath.split("/").filter(Boolean);
        const chain = [{ id: "/", name: "Root", isDir: true }];
        let cumulative = "/";
        parts.forEach(part => {
            cumulative += part + "/";
            chain.push({ id: cumulative, name: part, isDir: true });
        });
        return chain;
    };

    // Handle file actions: if a directory is clicked, navigate into it.
    // If a file is clicked, trigger a download.
    const handleFileAction = (action: ChonkyFileActionData) => {
        console.log("Handling file action:", action);
        if (action.id === ChonkyActions.OpenFiles.id) {
            const { targetFile, files } = action.payload;
            const fileToOpen = targetFile || files?.[0];
            if (!fileToOpen) {
                console.error("No file found to open", action.payload);
                return;
            }
            // Use the id property (which holds the file key) for navigation or download.
            if (fileToOpen.isDir) {
                console.log("Navigating to folder:", fileToOpen.id);
                setCurrentFolderPath(fileToOpen.id);
            } else {
                console.log("Downloading file:", fileToOpen.id);
                downloadFile([fileToOpen.id]);
            }
        }
    };

    // Trigger a download for the given file keys.
    const downloadFile = (keys: string[]) => {
        console.log("Downloading files:", keys);
        keys.forEach(key => {
            UIkit.notification(`Getting ${key} from server...`);
            downloadFileStraightToFilesystem(
                props.foldId,
                removeLeadingSlash(key),
                (progress: number) => {
                    console.log(`Downloading ${key}: ${progress}%`);
                }
            );
        });
    };

    return (
        <div>
            <h3>Quick Access</h3>
            <form className="uk-margin-bottom">
                <fieldset className="uk-fieldset">
                    <div>
                        <button
                            type="button"
                            className="uk-button uk-button-primary uk-margin-left uk-form-small"
                            onClick={props.maybeDownloadPdb}
                            disabled={!(props.foldName && props.pdbString)}
                        >
                            Download Best PDB
                        </button>
                    </div>
                </fieldset>
            </form>

            <h3>Files</h3>
            <div style={{ height: "400px" }}>
                <FileBrowser
                    files={getImmediateChildren()}
                    folderChain={getFolderChain()}
                    defaultFileViewActionId={ChonkyActions.EnableListView.id}
                    fileActions={[ChonkyActions.DownloadFiles, ChonkyActions.OpenFiles]}
                    onFileAction={handleFileAction}
                >
                    <FileNavbar />
                    <FileToolbar />
                    <FileList />
                </FileBrowser>
            </div>
        </div>
    );
});

export default FileTab;