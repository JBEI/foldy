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
    return Promise.reject(error);
  }
);

export default axiosInstance;