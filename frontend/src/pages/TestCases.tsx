import { BeakerIcon } from '@heroicons/react/24/outline'

export default function TestCases() {
  return (
    <div className="p-6">
      <div className="text-center py-12">
        <BeakerIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Test Cases</h1>
        <p className="text-gray-600">
          Manage and organize your API test cases. This page will show detailed test case management interface.
        </p>
      </div>
    </div>
  )
}
