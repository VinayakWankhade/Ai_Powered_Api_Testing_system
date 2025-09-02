import { useState } from 'react'
import { Link } from 'react-router-dom'
import { PlusIcon, DocumentTextIcon, EyeIcon, PencilIcon, TrashIcon } from '@heroicons/react/24/outline'

const mockApiSpecs = [
  {
    id: 1,
    name: 'User Management API',
    version: 'v2.1',
    baseUrl: 'https://api.example.com/users',
    description: 'Complete user management system API',
    testCases: 45,
    coverage: 78.5,
    lastExecution: '2024-01-15T10:30:00Z',
    status: 'active'
  },
  {
    id: 2,
    name: 'Payment Processing API',
    version: 'v1.3',
    baseUrl: 'https://api.example.com/payments',
    description: 'Secure payment processing and transaction management',
    testCases: 32,
    coverage: 85.2,
    lastExecution: '2024-01-15T09:15:00Z',
    status: 'active'
  },
  {
    id: 3,
    name: 'Order Management API',
    version: 'v3.0',
    baseUrl: 'https://api.example.com/orders',
    description: 'Order lifecycle management and tracking',
    testCases: 67,
    coverage: 72.8,
    lastExecution: '2024-01-14T16:45:00Z',
    status: 'active'
  },
  {
    id: 4,
    name: 'Authentication API',
    version: 'v1.5',
    baseUrl: 'https://api.example.com/auth',
    description: 'User authentication and authorization services',
    testCases: 28,
    coverage: 91.3,
    lastExecution: '2024-01-15T08:20:00Z',
    status: 'deprecated'
  }
]

export default function ApiSpecs() {
  const [searchTerm, setSearchTerm] = useState('')
  const [filterStatus, setFilterStatus] = useState('all')

  const filteredSpecs = mockApiSpecs.filter(spec => {
    const matchesSearch = spec.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         spec.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesFilter = filterStatus === 'all' || spec.status === filterStatus
    return matchesSearch && matchesFilter
  })

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return <span className="badge badge-success">Active</span>
      case 'deprecated':
        return <span className="badge badge-warning">Deprecated</span>
      default:
        return <span className="badge badge-gray">Unknown</span>
    }
  }

  const getCoverageColor = (coverage: number) => {
    if (coverage >= 80) return 'text-success-600'
    if (coverage >= 60) return 'text-warning-600'
    return 'text-error-600'
  }

  return (
    <div className="p-6">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">API Specifications</h1>
            <p className="text-gray-600">Manage and monitor your API specifications</p>
          </div>
          <button className="btn btn-primary">
            <PlusIcon className="w-4 h-4 mr-2" />
            Add API Spec
          </button>
        </div>

        {/* Filters */}
        <div className="card">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <input
                type="text"
                placeholder="Search API specifications..."
                className="input"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <select
              className="input sm:w-40"
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
            >
              <option value="all">All Status</option>
              <option value="active">Active</option>
              <option value="deprecated">Deprecated</option>
            </select>
          </div>
        </div>

        {/* API Specs Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredSpecs.map((spec) => (
            <div key={spec.id} className="card hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center">
                  <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center mr-3">
                    <DocumentTextIcon className="w-5 h-5 text-primary-600" />
                  </div>
                  <div>
                    <h3 className="font-medium text-gray-900">{spec.name}</h3>
                    <p className="text-sm text-gray-500">{spec.version}</p>
                  </div>
                </div>
                {getStatusBadge(spec.status)}
              </div>

              <p className="text-sm text-gray-600 mb-4">{spec.description}</p>

              <div className="space-y-2 mb-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Base URL:</span>
                  <span className="text-gray-900 font-mono text-xs truncate ml-2" title={spec.baseUrl}>
                    {spec.baseUrl}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Test Cases:</span>
                  <span className="font-medium text-gray-900">{spec.testCases}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Coverage:</span>
                  <span className={`font-medium ${getCoverageColor(spec.coverage)}`}>
                    {spec.coverage}%
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Last Execution:</span>
                  <span className="text-gray-900">
                    {new Date(spec.lastExecution).toLocaleDateString()}
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between pt-4 border-t border-gray-200">
                <Link
                  to={`/api-specs/${spec.id}/test-cases`}
                  className="text-primary-600 hover:text-primary-700 font-medium text-sm"
                >
                  View Test Cases
                </Link>
                <div className="flex items-center space-x-2">
                  <button className="p-1 text-gray-400 hover:text-gray-600">
                    <EyeIcon className="w-4 h-4" />
                  </button>
                  <button className="p-1 text-gray-400 hover:text-gray-600">
                    <PencilIcon className="w-4 h-4" />
                  </button>
                  <button className="p-1 text-gray-400 hover:text-error-600">
                    <TrashIcon className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {filteredSpecs.length === 0 && (
          <div className="text-center py-12">
            <DocumentTextIcon className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No API specifications found</h3>
            <p className="text-gray-600 mb-4">
              {searchTerm || filterStatus !== 'all' 
                ? 'Try adjusting your search or filter criteria.'
                : 'Get started by adding your first API specification.'
              }
            </p>
            {!searchTerm && filterStatus === 'all' && (
              <button className="btn btn-primary">
                <PlusIcon className="w-4 h-4 mr-2" />
                Add API Spec
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
