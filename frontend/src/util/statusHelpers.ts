import { Invokation, FewShot, Naturalness, Embedding } from '../types/types';

/**
 * Helper functions for determining the status of various job types
 * based on their associated invocation job state.
 */

/**
 * Get the status of a FewShot run by looking up its associated job
 */
export const getFewShotStatus = (fewShot: FewShot, jobs: Invokation[] | null): string => {
    const job = jobs?.find(job => job.id === fewShot.invokation_id);
    return job?.state || 'Unknown';
};

/**
 * Get the status of a Naturalness run by looking up its associated job
 */
export const getNaturalnessStatus = (naturalness: Naturalness, jobs: Invokation[] | null): string => {
    const job = jobs?.find(job => job.id === naturalness.invokation_id);
    return job?.state || 'Unknown';
};

/**
 * Get the status of an Embedding run by looking up its associated job
 */
export const getEmbeddingStatus = (embedding: Embedding, jobs: Invokation[] | null): string => {
    const job = jobs?.find(job => job.id === embedding.invokation_id);
    return job?.state || 'Unknown';
};

/**
 * Check if a run is complete based on the job state
 */
export const isRunComplete = (status: string): boolean => {
    return status === 'finished';
};

/**
 * Check if a run is running based on the job state
 */
export const isRunRunning = (status: string): boolean => {
    return status === 'started' || status === 'queued' || status === 'running';
};

/**
 * Check if a run has failed based on the job state
 */
export const isRunFailed = (status: string): boolean => {
    return status === 'failed';
};

/**
 * Get a human-readable status with appropriate styling
 */
export const getStatusDisplay = (status: string): { text: string, color: string, icon: string } => {
    if (isRunComplete(status)) {
        return { text: 'Complete', color: '#52c41a', icon: '✓' };
    } else if (isRunRunning(status)) {
        return { text: 'Running', color: '#faad14', icon: '⏳' };
    } else if (isRunFailed(status)) {
        return { text: 'Failed', color: '#ff4d4f', icon: '✗' };
    } else {
        return { text: status || 'Unknown', color: '#8c8c8c', icon: '?' };
    }
};

/**
 * Legacy output_fpath-based status check (for backward compatibility)
 * Some components still check output_fpath directly instead of job state
 */
export const isOutputComplete = (outputPath: string | null): boolean => {
    return outputPath !== null && outputPath !== '';
};
