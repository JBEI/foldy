import axiosInstance from '../services/axiosInstance';
import {
    getFold,
    getFolds,
    getJobStatus,
    describeFoldState,
    foldIsFinished,
    getFoldsWithPagination
} from './foldApi';

// Mock axiosInstance
jest.mock('../services/axiosInstance', () => ({
    __esModule: true,
    default: {
        get: jest.fn(),
        post: jest.fn(),
        put: jest.fn(),
        delete: jest.fn(),
    }
}));

// Get the mocked version of axiosInstance
const mockedAxiosInstance = axiosInstance as jest.Mocked<typeof axiosInstance>;

describe('foldApi', () => {
    afterEach(() => {
        jest.clearAllMocks();
    });

    describe('getFold', () => {
        it('should fetch a fold by ID', async () => {
            // Mock data
            const mockFold = {
                id: 1,
                name: 'Test Fold',
                sequence: 'ACDEFGHIKLMNPQRSTVWY',
                jobs: [{ type: 'models', state: 'finished' }]
            };

            // Setup mock
            mockedAxiosInstance.get.mockResolvedValueOnce({ data: mockFold });

            // Call the function
            const result = await getFold(1);

            // Assert the result
            expect(result).toEqual(mockFold);
            expect(mockedAxiosInstance.get).toHaveBeenCalledWith('/api/fold/1');
        });

        it('should handle errors', async () => {
            // Setup mock to reject
            mockedAxiosInstance.get.mockRejectedValueOnce(new Error('Network error'));

            // Call and expect rejection
            await expect(getFold(1)).rejects.toThrow('Network error');
        });
    });

    describe('getFolds', () => {
        it('should fetch folds with filters', async () => {
            // Mock data
            const mockFolds = [
                { id: 1, name: 'Test Fold 1' },
                { id: 2, name: 'Test Fold 2' }
            ];

            // Setup mock
            mockedAxiosInstance.get.mockResolvedValueOnce({ data: mockFolds });

            // Call the function with filters
            const result = await getFoldsWithPagination('test', 'tag1', 1, 10);

            // Assert the result
            expect(result).toEqual(mockFolds);
            expect(mockedAxiosInstance.get).toHaveBeenCalledWith('/api/fold', {
                params: {
                    filter: 'test',
                    tag: 'tag1',
                    page: 1,
                    per_page: 10
                }
            });
        });
    });

    describe('helper functions', () => {
        describe('getJobStatus', () => {
            it('should return job status for a specific job type', () => {
                const fold = {
                    id: 1,
                    name: 'Test Fold',
                    jobs: [
                        { type: 'features', state: 'finished' },
                        { type: 'models', state: 'running' }
                    ]
                };

                expect(getJobStatus(fold, 'features')).toBe('finished');
                expect(getJobStatus(fold, 'models')).toBe('running');
                expect(getJobStatus(fold, 'nonexistent')).toBeNull();
            });

            it('should handle fold without jobs', () => {
                const fold = { id: 1, name: 'Test Fold' };
                expect(getJobStatus(fold, 'features')).toBeNull();
            });
        });

        describe('describeFoldState', () => {
            it('should correctly describe the fold state based on job states', () => {
                // Test case: Boltz job present
                const foldWithBoltz = {
                    id: 1,
                    jobs: [{ type: 'boltz', state: 'running' }]
                };
                expect(describeFoldState(foldWithBoltz)).toBe('running');

                // Test case: All states finished
                const foldAllFinished = {
                    id: 2,
                    jobs: [
                        { type: 'features', state: 'finished' },
                        { type: 'models', state: 'finished' },
                    ]
                };
                expect(describeFoldState(foldAllFinished)).toBe('finished');

                // Test case: Features running
                const foldFeaturesRunning = {
                    id: 3,
                    jobs: [
                        { type: 'features', state: 'running' },
                        { type: 'models', state: 'queued' },
                    ]
                };
                expect(describeFoldState(foldFeaturesRunning)).toBe('features running');
            });
        });

        describe('foldIsFinished', () => {
            it('should return true if models job is finished', () => {
                const fold = {
                    id: 1,
                    jobs: [{ type: 'models', state: 'finished' }]
                };
                expect(foldIsFinished(fold)).toBe(true);
            });

            it('should return false if models job is not finished', () => {
                const fold = {
                    id: 1,
                    jobs: [{ type: 'models', state: 'running' }]
                };
                expect(foldIsFinished(fold)).toBe(false);
            });
        });
    });
});
