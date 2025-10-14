import axiosInstance from '../services/axiosInstance';

/**
 * Creates database tables
 */
export const createDbs = async (): Promise<any> => {
  const response = await axiosInstance.post('/api/createdbs', {});
  return response.data;
};

/**
 * Upgrades database tables
 */
export const upgradeDbs = async (): Promise<any> => {
  const response = await axiosInstance.post('/api/upgradedbs', {});
  return response.data;
};

/**
 * Stamps database with revision
 */
export const stampDbs = async (revision: string): Promise<any> => {
  const response = await axiosInstance.post('/api/stampdbs', { revision });
  return response.data;
};

/**
 * Queues a test job
 */
export const queueTestJob = async (queue: string): Promise<any> => {
  const response = await axiosInstance.post('/api/queuetestjob', { queue });
  return response.data;
};

/**
 * Removes failed jobs from a queue
 */
export const removeFailedJobs = async (queue: string): Promise<any> => {
  const response = await axiosInstance.post('/api/remove_failed_jobs', { queue });
  return response.data;
};

/**
 * Kills a worker
 */
export const killWorker = async (workerToKill: string): Promise<any> => {
  const response = await axiosInstance.post(
    '/api/kill_worker',
    { worker_id: workerToKill }
  );
  return response.data;
};

/**
 * Sends a test email
 */
export const sendTestEmail = async (): Promise<any> => {
  const response = await axiosInstance.post('/api/sendtestemail', {});
  return response.data;
};

/**
 * Adds invokation to all jobs of a specific type and state
 */
export const addInvokationToAllJobs = async (
  jobType: string,
  jobState: string
): Promise<any> => {
  const response = await axiosInstance.post(
    `/api/addInvokationToAllJobs/${jobType}/${jobState}`,
    {}
  );
  return response.data;
};

/**
 * Runs unrun stages of a specific type
 */
export const runUnrunStages = async (stageToRun: string): Promise<any> => {
  const response = await axiosInstance.post(
    `/api/runUnrunStages/${stageToRun}`,
    {}
  );
  return response.data;
};

/**
 * Sets all unset model presets
 */
export const setAllUnsetModelPresets = async (): Promise<boolean> => {
  const response = await axiosInstance.post(
    '/api/set_all_unset_model_presets',
    {}
  );
  return response.data;
};

/**
 * Kills folds in a range
 */
export const killFoldsInRange = async (foldRange: string): Promise<any> => {
  const response = await axiosInstance.post(
    `/api/killFolds/${foldRange}`,
    {}
  );
  return response.data;
};

/**
 * Bulk adds a tag to folds in a range
 */
export const bulkAddTag = async (foldRange: string, newTag: string): Promise<any> => {
  const response = await axiosInstance.post(
    `/api/bulkAddTag/${foldRange}/${newTag}`,
    {}
  );
  return response.data;
};

/**
 * Populates output_fpath fields for existing naturalness, embedding, and few shot records
 */
export const populateOutputFpathFields = async (): Promise<{
  naturalness_updated: number;
  embedding_updated: number;
  few_shot_updated: number;
  total_updated: number;
}> => {
  const response = await axiosInstance.post('/api/populate_output_fpath_fields', {});
  return response.data;
};

/**
 * Backfill date_created fields for existing naturalness, embedding, and few shot records
 */
export const backfillDateCreated = async (): Promise<{
  naturalness_updated: number;
  embedding_updated: number;
  few_shot_updated: number;
  total_updated: number;
}> => {
  const response = await axiosInstance.post('/api/backfill_date_created', {});
  return response.data;
};

/**
 * Backfills input_activity_fpath for FewShot records
 */
export const backfillInputActivityFpath = async (): Promise<{
  few_shot_updated: number;
  total_updated: number;
}> => {
  const response = await axiosInstance.post('/api/backfill_input_activity_fpath', {});
  return response.data;
};

/**
 * Gets system messages to display to users
 */
export const getMessages = async (): Promise<Array<{ message: string; type: string }>> => {
  const response = await axiosInstance.get('/api/messages');
  return response.data;
};
