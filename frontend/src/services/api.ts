import axios from 'axios'

const baseURL = import.meta.env.VITE_API_URL || '/api'

export const api = axios.create({
  baseURL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('authToken')
      window.location.href = '/login'
    }
    return Promise.reject(error.response?.data || error.message)
  }
)

// API endpoints
export const apiEndpoints = {
  // API Specifications
  getApiSpecs: () => api.get('/specs'),
  getApiSpec: (id: number) => api.get(`/specs/${id}`),
  createApiSpec: (data: any) => api.post('/specs', data),
  updateApiSpec: (id: number, data: any) => api.put(`/specs/${id}`, data),
  deleteApiSpec: (id: number) => api.delete(`/specs/${id}`),

  // Test Cases
  getTestCases: (apiSpecId?: number) => 
    api.get('/test-cases', { params: { api_spec_id: apiSpecId } }),
  getTestCase: (id: number) => api.get(`/test-cases/${id}`),
  createTestCase: (data: any) => api.post('/test-cases', data),
  updateTestCase: (id: number, data: any) => api.put(`/test-cases/${id}`, data),
  deleteTestCase: (id: number) => api.delete(`/test-cases/${id}`),
  generateTestCases: (apiSpecId: number, options: any) => 
    api.post(`/test-generation/generate/${apiSpecId}`, options),

  // Test Execution
  executeTestSuite: (apiSpecId: number, options: any) =>
    api.post(`/test-execution/execute/${apiSpecId}`, options),
  getExecutionSessions: (apiSpecId?: number) =>
    api.get('/test-execution/sessions', { params: { api_spec_id: apiSpecId } }),
  getExecutionSession: (id: number) => api.get(`/test-execution/sessions/${id}`),
  cancelExecution: (sessionId: number) => api.post(`/test-execution/cancel/${sessionId}`),

  // Coverage
  getCoverageReport: (apiSpecId: number) => api.get(`/coverage/coverage-report/${apiSpecId}`),
  getCoverageTrends: (apiSpecId: number, days: number = 30) =>
    api.get(`/coverage/coverage-trends/${apiSpecId}`, { params: { days } }),
  getAnalyticsReport: (apiSpecId: number, days: number = 30) =>
    api.get(`/coverage/analytics/${apiSpecId}`, { params: { days } }),

  // Healing
  getFailedTestCases: (apiSpecId: number) => 
    api.get(`/test-healing/failed-tests/${apiSpecId}`),
  healTests: (apiSpecId: number, options: any) =>
    api.post(`/test-healing/heal/${apiSpecId}`, options),
  getHealingRecommendations: (apiSpecId: number) =>
    api.get(`/test-healing/recommendations/${apiSpecId}`),
  getHealingHistory: (apiSpecId?: number) =>
    api.get('/test-healing/history', { params: { api_spec_id: apiSpecId } }),

  // RL Optimization
  optimizeTestSelection: (apiSpecId: number, options: any) =>
    api.post(`/rl-optimization/optimize/${apiSpecId}`, options),
  getOptimizationRecommendations: (apiSpecId: number) =>
    api.get(`/rl-optimization/recommendations/${apiSpecId}`),
  getRLModelPerformance: (apiSpecId: number) =>
    api.get(`/rl-optimization/models/${apiSpecId}`),
  getTrainingHistory: (apiSpecId: number) =>
    api.get(`/rl-optimization/training-history/${apiSpecId}`),

  // Dashboard
  getDashboardStats: () => api.get('/dashboard/stats'),
  getSystemHealth: () => api.get('/dashboard/health'),
}
