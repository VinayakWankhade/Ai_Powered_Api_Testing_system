import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ApiSpecs from './pages/ApiSpecs'
import TestCases from './pages/TestCases'
import Executions from './pages/Executions'
import Coverage from './pages/Coverage'
import Healing from './pages/Healing'
import Optimization from './pages/Optimization'
import NotFound from './pages/NotFound'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/api-specs" element={<ApiSpecs />} />
        <Route path="/api-specs/:id/test-cases" element={<TestCases />} />
        <Route path="/executions" element={<Executions />} />
        <Route path="/executions/:id" element={<Executions />} />
        <Route path="/coverage" element={<Coverage />} />
        <Route path="/coverage/:apiSpecId" element={<Coverage />} />
        <Route path="/healing" element={<Healing />} />
        <Route path="/healing/:apiSpecId" element={<Healing />} />
        <Route path="/optimization" element={<Optimization />} />
        <Route path="/optimization/:apiSpecId" element={<Optimization />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Layout>
  )
}

export default App
