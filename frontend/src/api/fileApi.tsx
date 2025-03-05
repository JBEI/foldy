import { FileInfo } from '../types/types';
import axiosInstance from '../services/axiosInstance';
import { authHeader } from '../util/authHeader';
import axios from 'axios';
import streamSaver from 'streamsaver';
import { authenticationService } from '../services/authentication.service';


export const getFileList = async (fold_id: number): Promise<FileInfo[]> => {
    const response = await axiosInstance.get(`/api/file/list/${fold_id}`);
    return response.data;
}

export const getFile = async (fold_id: number, filePath: string): Promise<Blob> => {
    try {
        const response = await axiosInstance.get(`/api/file/download/${fold_id}/${filePath}`, {
            headers: authHeader(),
            responseType: 'blob',
            timeout: 60000 * 5,  // 5 minutes
            onDownloadProgress: (progressEvent) => {
                // Optional: Add progress tracking
                console.log('Download Progress:', progressEvent);
            },
        });
        return response.data;
    } catch (error) {
        if (axios.isAxiosError(error)) {
            if (error.code === 'ECONNABORTED') {
                throw new Error('Download timed out');
            }
            console.error('Download failed:', error.message);
        }
        throw error;
    }
}


export const downloadFileStraightToFilesystem = async (
    fold_id: number,
    filePath: string,
    onProgress?: (progressPercent: number) => void
): Promise<void> => {
    const fileName = filePath.split('/').pop() || 'downloaded_file';
    console.log(`Downloading ${fileName}`);

    // Build headers, including Authorization if we have a token
    const token = authenticationService.currentJwtStringValue;
    const headers: Record<string, string> = {};

    if (token) {
        headers.Authorization = `Bearer ${token}`;
    }

    // Note: We typically do NOT need Content-Type for a GET download
    // If your server specifically requires it, you can add: headers['Content-Type'] = 'application/json';

    const response = await fetch(
        // If you were using a base URL in Axios, prepend it here:
        `${import.meta.env.VITE_BACKEND_URL}/api/file/download/${fold_id}/${filePath}`,
        // `/api/file/download/${fold_id}/${filePath}`,
        {
            method: 'GET',
            headers,
            // You can optionally control timeouts in fetch with an AbortController
        }
    );

    if (!response.ok) {
        throw new Error(`Failed to download. Status: ${response.status}`);
    }

    const contentLength = parseInt(response.headers.get('Content-Length') || '0', 10);
    let bytesDownloaded = 0;

    // Create the StreamSaver write stream
    const fileStream = streamSaver.createWriteStream(fileName);

    // If the browser supports ReadableStream from fetch
    if (response.body) {
        console.log('Piping through a TransformStream to track progress');
        // We'll pipe through a TransformStream to track progress
        const progressStream = response.body.pipeThrough(
            new TransformStream({
                transform(chunk, controller) {
                    bytesDownloaded += chunk.byteLength;
                    if (onProgress && contentLength) {
                        onProgress(Math.round((bytesDownloaded / contentLength) * 100));
                    }
                    controller.enqueue(chunk);
                },
            })
        );

        await progressStream.pipeTo(fileStream);
    } else {
        console.log('Falling back to old method');
        // Fallback for older browsers: get the whole blob, then pipe
        const blob = await response.blob();
        const readableStream = new Response(blob).body;
        if (!readableStream) {
            throw new Error('ReadableStream not supported in this browser');
        }
        await readableStream.pipeTo(fileStream);
    }
};