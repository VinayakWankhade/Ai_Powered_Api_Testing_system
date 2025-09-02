import { useQuery } from '@tanstack/react-query'
import { 
  ChartBarIcon, 
  BeakerIcon, 
  PlayIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  BugAntIcon
} from '@heroicons/react/24/outline'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts'
import { api } from '../services/api'

const COLORS = ['#10b981', '#ef4444', '#f59e0b', '#6b7280']

interface DashboardStats {
  totalApiSpecs: number
  totalTestCases: number
  totalExecutions: number
  coveragePercentage: number
  successRate: number
  pendingHealing: number
  recentActivity: Array<{
    id: number
    type: string
    description: string
    timestamp: string
    status: string
  }>
  executionTrends: Array<{
    date: string
    success: number
    failed: number
    total: number
  }>
  coverageTrends: Array<{
    date: string
    coverage: number
  }>
  testTypeDistribution: Array<{
    name: string
    value: number
    color: string
  }>
}

export default function Dashboard() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: async (): Promise<DashboardStats> => {
      // For now, return mock data. In production, this would call the API
      await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate loading
      
      return {
        totalApiSpecs: 12,
        totalTestCases: 847,
        totalExecutions: 2341,
        coveragePercentage: 78.5,
        successRate: 89.2,
        pendingHealing: 23,
        recentActivity: [
          {
            id: 1,
            type: 'execution',
            description: 'Test execution completed for UserAPI v2.1',
            timestamp: '2024-01-15T10:30:00Z',
            status: 'success'
          },
          {
            id: 2,
            type: 'healing',
            description: 'Auto-healed 3 failing tests in PaymentAPI',
            timestamp: '2024-01-15T10:15:00Z',
            status: 'success'
          },
          {
            id: 3,
            type: 'generation',
            description: 'Generated 15 new test cases for OrderAPI',
            timestamp: '2024-01-15T09:45:00Z',
            status: 'success'
          },
          {
            id: 4,
            type: 'execution',
            description: 'Test execution failed for AuthAPI v1.3',
            timestamp: '2024-01-15T09:30:00Z',
            status: 'error'
          },
        ],
        executionTrends: [
          { date: '2024-01-10', success: 145, failed: 12, total: 157 },
          { date: '2024-01-11', success: 132, failed: 18, total: 150 },
          { date: '2024-01-12', success: 167, failed: 9, total: 176 },
          { date: '2024-01-13', success: 143, failed: 15, total: 158 },
          { date: '2024-01-14', success: 189, failed: 7, total: 196 },
          { date: '2024-01-15', success: 156, failed: 11, total: 167 },
        ],
        coverageTrends: [
          { date: '2024-01-10', coverage: 72.3 },
          { date: '2024-01-11', coverage: 74.1 },
          { date: '2024-01-12', coverage: 75.8 },
          { date: '2024-01-13', coverage: 76.9 },
          { date: '2024-01-14', coverage: 77.2 },
          { date: '2024-01-15', coverage: 78.5 },
        ],
        testTypeDistribution: [
          { name: 'Functional', value: 45, color: '#10b981' },
          { name: 'Edge Cases', value: 25, color: '#3b82f6' },
          { name: 'Security', value: 20, color: '#f59e0b' },
          { name: 'Performance', value: 10, color: '#8b5cf6' },
        ]
      }
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded w-1/4"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-24 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="h-64 bg-gray-200 rounded"></div>
            <div className="h-64 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600">Overview of your API testing platform</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">API Specifications</p>
              <p className="text-2xl font-bold text-gray-900">{stats?.totalApiSpecs || 0}</p>
            </div>
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
              <ChartBarIcon className="w-6 h-6 text-primary-600" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Test Cases</p>
              <p className="text-2xl font-bold text-gray-900">{stats?.totalTestCases || 0}</p>
            </div>
            <div className="w-12 h-12 bg-success-100 rounded-lg flex items-center justify-center">
              <BeakerIcon className="w-6 h-6 text-success-600" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Executions</p>
              <p className="text-2xl font-bold text-gray-900">{stats?.totalExecutions || 0}</p>
            </div>
            <div className="w-12 h-12 bg-warning-100 rounded-lg flex items-center justify-center">
              <PlayIcon className="w-6 h-6 text-warning-600" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Coverage</p>
              <p className="text-2xl font-bold text-gray-900">{stats?.coveragePercentage || 0}%</p>
            </div>
            <div className="w-12 h-12 bg-error-100 rounded-lg flex items-center justify-center">
              <ChartBarIcon className="w-6 h-6 text-error-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Success Rate & Pending Healing */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Success Rate</h3>
          <div className="flex items-center">
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Overall Success Rate</span>
                <span className="text-sm font-medium text-gray-900">{stats?.successRate || 0}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-success-500 h-2 rounded-full" 
                  style={{ width: `${stats?.successRate || 0}%` }}
                ></div>
              </div>
            </div>
            <CheckCircleIcon className="w-8 h-8 text-success-500 ml-4" />
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Pending Healing</h3>
          <div className="flex items-center">
            <div className="flex-1">
              <p className="text-2xl font-bold text-error-600">{stats?.pendingHealing || 0}</p>
              <p className="text-sm text-gray-600">Tests require healing</p>
            </div>
            <BugAntIcon className="w-8 h-8 text-error-500 ml-4" />
          </div>
          {(stats?.pendingHealing || 0) > 0 && (
            <div className="mt-4">
              <button className="btn btn-error btn-sm">
                View Healing Queue
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Execution Trends */}
        <div className="card">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Execution Trends</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={stats?.executionTrends || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                tickFormatter={(value) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
              />
              <YAxis />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value, name) => [value, name === 'success' ? 'Success' : name === 'failed' ? 'Failed' : 'Total']}
              />
              <Bar dataKey="success" fill="#10b981" name="success" />
              <Bar dataKey="failed" fill="#ef4444" name="failed" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Test Type Distribution */}
        <div className="card">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Test Type Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={stats?.testTypeDistribution || []}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="value"
              >
                {(stats?.testTypeDistribution || []).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-4 grid grid-cols-2 gap-2">
            {(stats?.testTypeDistribution || []).map((item, index) => (
              <div key={index} className="flex items-center">
                <div 
                  className="w-3 h-3 rounded-full mr-2" 
                  style={{ backgroundColor: item.color }}
                ></div>
                <span className="text-sm text-gray-600">{item.name} ({item.value}%)</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Coverage Trends */}
      <div className="card">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Coverage Trends</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={stats?.coverageTrends || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="date" 
              tickFormatter={(value) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
            />
            <YAxis domain={[0, 100]} />
            <Tooltip 
              labelFormatter={(value) => new Date(value).toLocaleDateString()}
              formatter={(value) => [`${value}%`, 'Coverage']}
            />
            <Line 
              type="monotone" 
              dataKey="coverage" 
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Recent Activity */}
      <div className="card">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h3>
        <div className="space-y-4">
          {(stats?.recentActivity || []).map((activity) => (
            <div key={activity.id} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                activity.status === 'success' ? 'bg-success-100' : 'bg-error-100'
              }`}>
                {activity.status === 'success' ? (
                  <CheckCircleIcon className="w-5 h-5 text-success-600" />
                ) : (
                  <ExclamationTriangleIcon className="w-5 h-5 text-error-600" />
                )}
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-900">{activity.description}</p>
                <p className="text-xs text-gray-500">
                  <ClockIcon className="w-3 h-3 inline mr-1" />
                  {new Date(activity.timestamp).toLocaleString()}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
