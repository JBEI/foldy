import axios from 'axios';
import { authenticationService } from './authenticationService';

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
    (error) => {
        if (error.response?.status === 401) {
            authenticationService.logout();
        }

        if (error.response) {
            // The request was made and the server responded with a status code
            console.error('Response error:', error.response);
            throw new Error(error.response.data?.message || error.response.data || error.message);
        } else if (error.request) {
            // The request was made but no response was received
            console.error('Request error:', error.request);
            throw new Error("Network error: no response received");
        } else {
            // Something happened in setting up the request
            console.error('Setup error:', error.message);
            throw new Error(error.message);
        }
    }
);

export default axiosInstance;