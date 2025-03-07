import axios from 'axios';
import { authenticationService } from './authentication.service';

// Create an Axios instance with default configurations
const axiosInstance = axios.create({
    baseURL: import.meta.env.VITE_BACKEND_URL,
    timeout: 10000, // Timeout after 10 seconds
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add Authorization header dynamically
axiosInstance.interceptors.request.use((config) => {
    // Ensure headers object exists
    config.headers = config.headers || {};

    // Only set Content-Type to application/json if:
    // 1. Content-Type is not already set AND
    // 2. We're not sending FormData
    if (!config.headers['Content-Type'] && !(config.data instanceof FormData)) {
        config.headers['Content-Type'] = 'application/json';
    }

    // Add authorization if token exists
    const token = authenticationService.currentJwtStringValue;
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
});

// Global error handling
axiosInstance.interceptors.response.use(
    (response) => response,
    async (error) => {  // Make async to handle Blob reading
        if (error.response?.status === 401) {
            console.log(`IN AXIOS, interceptor, calling logout`);
            authenticationService.logout();
        }

        if (error.response) {
            console.error('Response error:', error.response);
            const data = error.response.data;

            // Handle Blob error responses
            if (data instanceof Blob) {
                try {
                    const text = await data.text();
                    try {
                        // Try to parse as JSON
                        const jsonData = JSON.parse(text);
                        throw new Error(jsonData.message || text);
                    } catch {
                        // If not JSON, use text directly
                        throw new Error(text);
                    }
                } catch (blobError) {
                    throw new Error(`Failed to read error response: ${blobError}`);
                }
            }

            // Handle regular JSON responses
            throw new Error(data?.message || data || error.message);
        } else if (error.request) {
            console.error('Request error:', error.request);
            throw new Error("Network error: no response received");
        } else {
            console.error('Setup error:', error.message);
            throw new Error(error.message);
        }
    }
);

export default axiosInstance;