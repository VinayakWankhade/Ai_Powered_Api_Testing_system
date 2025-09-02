"""
Defect Detection and Clustering System for failure analysis and reproducer synthesis.
Implements fingerprinting, clustering, deduplication, and minimal test case generation.
"""

import json
import re
import hashlib
import base64
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

try:
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..database.connection import get_db_session
from ..database.models import (
    TestExecution, TestCase, APISpecification, ExecutionSession, TestStatus
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DefectSeverity(Enum):
    """Defect severity levels."""
    CRITICAL = "critical"  # 5xx errors, security issues
    HIGH = "high"         # Contract breaks, invariant violations
    MEDIUM = "medium"     # 4xx unexpected, schema issues
    LOW = "low"           # Performance, minor inconsistencies

class DefectCategory(Enum):
    """Defect categorization."""
    SERVER_ERROR = "server_error"          # 5xx responses
    CLIENT_ERROR = "client_error"          # 4xx responses
    CONTRACT_VIOLATION = "contract_violation"  # Schema/spec violations
    INVARIANT_VIOLATION = "invariant_violation"  # API invariant breaks
    SECURITY_ISSUE = "security_issue"      # Security-related problems
    PERFORMANCE_ISSUE = "performance_issue"  # Response time issues
    DATA_INTEGRITY = "data_integrity"      # Data consistency issues

@dataclass
class DefectFingerprint:
    """Stable fingerprint for defect deduplication."""
    fingerprint_id: str
    endpoint: str
    method: str
    status_code: int
    error_pattern: str
    normalized_message: str
    schema_path: Optional[str]

@dataclass
class DefectCluster:
    """Group of similar defects."""
    cluster_id: str
    fingerprints: List[DefectFingerprint]
    representative_execution: int  # ID of representative test execution
    category: DefectCategory
    severity: DefectSeverity
    affected_endpoints: List[str]
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int
    minimal_reproducer: Optional[Dict[str, Any]]

@dataclass
class MinimalReproducer:
    """Minimal reproducible test case for a defect."""
    reproducer_id: str
    curl_command: str
    har_file: Dict[str, Any]
    test_case_json: Dict[str, Any]
    environment_snapshot: Dict[str, Any]
    reproduction_steps: List[str]
    expected_outcome: str
    actual_outcome: str

class DefectDetectionSystem:
    """
    Advanced defect detection and clustering system.
    
    Features:
    - Stable fingerprinting for deduplication
    - ML-based clustering of similar failures
    - Minimal reproducer synthesis
    - Defect severity and impact analysis
    - Automated triage and ticket generation
    """
    
    def __init__(self):
        self.db = get_db_session()
        self.fingerprint_cache = {}
        self.cluster_cache = {}
        
    async def detect_and_cluster_defects(
        self,
        execution_session_id: int,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Detect defects in a test session and cluster with historical failures.
        
        Args:
            execution_session_id: Session to analyze
            lookback_days: How far back to look for similar failures
            
        Returns:
            Defect analysis results with clusters and reproducers
        """
        try:
            # Get failed executions from this session
            failed_executions = self.db.query(TestExecution).filter(
                TestExecution.session_id == execution_session_id,
                TestExecution.status.in_([TestStatus.FAILED, TestStatus.ERROR])
            ).all()
            
            if not failed_executions:
                return {
                    "message": "No failed executions found in session",
                    "clusters": [],
                    "new_defects": 0,
                    "known_defects": 0
                }
            
            # Generate fingerprints for failures
            fingerprints = []
            for execution in failed_executions:
                fingerprint = await self._generate_defect_fingerprint(execution)
                fingerprints.append((execution, fingerprint))
            
            # Get historical failures for clustering
            cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
            historical_executions = self.db.query(TestExecution).filter(
                TestExecution.status.in_([TestStatus.FAILED, TestStatus.ERROR]),
                TestExecution.created_at >= cutoff_date,
                TestExecution.session_id != execution_session_id
            ).limit(1000).all()
            
            # Generate fingerprints for historical failures
            historical_fingerprints = []
            for execution in historical_executions:
                fingerprint = await self._generate_defect_fingerprint(execution)
                historical_fingerprints.append((execution, fingerprint))
            
            # Combine current and historical for clustering
            all_fingerprints = fingerprints + historical_fingerprints
            
            # Perform clustering
            clusters = await self._cluster_defects(all_fingerprints)
            
            # Identify new vs known defects
            new_defects = 0
            known_defects = 0
            
            for execution, fingerprint in fingerprints:
                is_known = any(
                    fingerprint.fingerprint_id == hist_fp.fingerprint_id 
                    for _, hist_fp in historical_fingerprints
                )
                if is_known:
                    known_defects += 1
                else:
                    new_defects += 1
            
            # Generate minimal reproducers for new defects
            reproducers = []
            for cluster in clusters:
                if any(fp.fingerprint_id in [fp2.fingerprint_id for _, fp2 in fingerprints] for fp in cluster.fingerprints):
                    reproducer = await self._generate_minimal_reproducer(cluster)
                    if reproducer:
                        reproducers.append(reproducer)
            
            return {
                "session_id": execution_session_id,
                "analysis_summary": {
                    "total_failures": len(failed_executions),
                    "new_defects": new_defects,
                    "known_defects": known_defects,
                    "clusters_found": len(clusters),
                    "reproducers_generated": len(reproducers)
                },
                "clusters": [self._serialize_cluster(cluster) for cluster in clusters],
                "reproducers": [self._serialize_reproducer(reproducer) for reproducer in reproducers],
                "defect_severity_distribution": self._calculate_severity_distribution(clusters),
                "recommendations": self._generate_defect_recommendations(clusters)
            }
            
        except Exception as e:
            logger.error(f"Defect detection and clustering failed: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_defect_fingerprint(self, test_execution: TestExecution) -> DefectFingerprint:
        """Generate a stable fingerprint for defect deduplication."""
        
        try:
            test_case = test_execution.test_case
            endpoint = test_case.endpoint if test_case else "unknown"
            method = test_case.method if test_case else "unknown"
            status_code = test_execution.response_code or 0
            error_message = test_execution.error_message or ""
            
            # Normalize error message for stable fingerprinting
            normalized_message = self._normalize_error_message(error_message)
            
            # Extract error pattern
            error_pattern = self._extract_error_pattern(error_message, test_execution.response_body)
            
            # Extract schema path if available
            schema_path = self._extract_schema_path(test_execution.response_body)
            
            # Generate stable fingerprint ID
            fingerprint_data = f"{endpoint}|{method}|{status_code}|{error_pattern}|{normalized_message}"
            fingerprint_id = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
            
            return DefectFingerprint(
                fingerprint_id=fingerprint_id,
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                error_pattern=error_pattern,
                normalized_message=normalized_message,
                schema_path=schema_path
            )
            
        except Exception as e:
            logger.error(f"Failed to generate defect fingerprint: {str(e)}")
            return DefectFingerprint(
                fingerprint_id="unknown",
                endpoint="unknown",
                method="unknown", 
                status_code=0,
                error_pattern="unknown",
                normalized_message="",
                schema_path=None
            )
    
    def _normalize_error_message(self, error_message: str) -> str:
        """Normalize error message for consistent fingerprinting."""
        
        if not error_message:
            return ""
        
        # Remove timestamps
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[.\d]*[Z]?', '', error_message)
        
        # Remove UUIDs
        normalized = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', 'UUID', normalized)
        
        # Remove numeric IDs (but preserve error codes)
        normalized = re.sub(r'\b(?<!error\s)(?<!code\s)\d{3,}\b', 'ID', normalized)
        
        # Remove request/trace IDs
        normalized = re.sub(r'(request|trace|correlation|session)[-_]?id[:\s]*\S+', '', normalized, flags=re.IGNORECASE)
        
        # Remove file paths and line numbers
        normalized = re.sub(r'(/[a-zA-Z0-9_.-]+)+\.[a-zA-Z]{2,4}:\d+', 'FILE:LINE', normalized)
        
        # Remove memory addresses
        normalized = re.sub(r'0x[a-fA-F0-9]{8,}', 'MEMORY_ADDR', normalized)
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized.lower().strip()
    
    def _extract_error_pattern(self, error_message: str, response_body: Any) -> str:
        """Extract high-level error pattern for clustering."""
        
        patterns = []
        
        # HTTP status-based patterns
        if "400" in str(error_message) or "Bad Request" in str(error_message):
            patterns.append("bad_request")
        elif "401" in str(error_message) or "Unauthorized" in str(error_message):
            patterns.append("unauthorized")
        elif "403" in str(error_message) or "Forbidden" in str(error_message):
            patterns.append("forbidden")
        elif "404" in str(error_message) or "Not Found" in str(error_message):
            patterns.append("not_found")
        elif "422" in str(error_message) or "Unprocessable" in str(error_message):
            patterns.append("validation_error")
        elif "500" in str(error_message) or "Internal Server" in str(error_message):
            patterns.append("server_error")
        elif "502" in str(error_message) or "Bad Gateway" in str(error_message):
            patterns.append("gateway_error")
        elif "503" in str(error_message) or "Service Unavailable" in str(error_message):
            patterns.append("service_unavailable")
        elif "timeout" in str(error_message).lower():
            patterns.append("timeout")
        
        # Error type patterns
        if isinstance(response_body, dict):
            error_type = response_body.get("error_type") or response_body.get("type")
            if error_type:
                patterns.append(f"type_{error_type}")
            
            error_code = response_body.get("error_code") or response_body.get("code")
            if error_code:
                patterns.append(f"code_{error_code}")
        
        # Exception patterns
        if "Exception" in str(error_message):
            exception_match = re.search(r'([A-Z][a-zA-Z]+Exception)', str(error_message))
            if exception_match:
                patterns.append(f"exception_{exception_match.group(1).lower()}")
        
        # Connection/network patterns
        if any(term in str(error_message).lower() for term in ["connection", "network", "dns", "socket"]):
            patterns.append("network_issue")
        
        # Database patterns
        if any(term in str(error_message).lower() for term in ["database", "sql", "connection", "deadlock"]):
            patterns.append("database_issue")
        
        return "_".join(patterns) if patterns else "unknown_error"
    
    def _extract_schema_path(self, response_body: Any) -> Optional[str]:
        """Extract the first schema path where an error occurred."""
        
        if not isinstance(response_body, dict):
            return None
        
        # Look for validation error paths
        for field in ["field", "path", "property", "attribute"]:
            if field in response_body:
                return str(response_body[field])
        
        # Look for nested error details
        if "errors" in response_body and isinstance(response_body["errors"], list):
            for error in response_body["errors"]:
                if isinstance(error, dict) and "field" in error:
                    return str(error["field"])
        
        return None
    
    async def _cluster_defects(
        self, 
        fingerprints_with_executions: List[Tuple[TestExecution, DefectFingerprint]]
    ) -> List[DefectCluster]:
        """Cluster similar defects using fingerprints and ML clustering."""
        
        try:
            if not fingerprints_with_executions:
                return []
            
            # Group by fingerprint ID (exact matches)
            fingerprint_groups = defaultdict(list)
            for execution, fingerprint in fingerprints_with_executions:
                fingerprint_groups[fingerprint.fingerprint_id].append((execution, fingerprint))
            
            clusters = []
            
            # Create clusters from fingerprint groups
            for fingerprint_id, group in fingerprint_groups.items():
                if len(group) == 0:
                    continue
                
                executions, fingerprints = zip(*group)
                representative_execution = executions[0]  # Use first as representative
                
                # Determine category and severity
                category = self._categorize_defect(representative_execution)
                severity = self._assess_defect_severity(representative_execution, len(group))
                
                # Get affected endpoints
                affected_endpoints = list(set(
                    fp.endpoint for fp in fingerprints if fp.endpoint != "unknown"
                ))
                
                # Get time range
                timestamps = [ex.created_at for ex in executions if ex.created_at]
                first_seen = min(timestamps) if timestamps else datetime.utcnow()
                last_seen = max(timestamps) if timestamps else datetime.utcnow()
                
                cluster = DefectCluster(
                    cluster_id=fingerprint_id,
                    fingerprints=list(fingerprints),
                    representative_execution=representative_execution.id,
                    category=category,
                    severity=severity,
                    affected_endpoints=affected_endpoints,
                    first_seen=first_seen,
                    last_seen=last_seen,
                    occurrence_count=len(group),
                    minimal_reproducer=None  # Will be generated separately
                )
                
                clusters.append(cluster)
            
            # Additional ML-based clustering for edge cases (if sklearn available)
            if SKLEARN_AVAILABLE and len(clusters) > 5:
                clusters = await self._refine_clusters_with_ml(clusters)
            
            # Sort clusters by severity and occurrence count
            clusters.sort(key=lambda c: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[c.severity.value],
                c.occurrence_count
            ), reverse=True)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Defect clustering failed: {str(e)}")
            return []
    
    def _categorize_defect(self, execution: TestExecution) -> DefectCategory:
        """Categorize a defect based on execution results."""
        
        status_code = execution.response_code or 0
        error_message = execution.error_message or ""
        
        # Server errors (5xx)
        if 500 <= status_code < 600:
            return DefectCategory.SERVER_ERROR
        
        # Client errors (4xx)
        elif 400 <= status_code < 500:
            # Special cases
            if status_code == 422:
                return DefectCategory.CONTRACT_VIOLATION
            elif status_code in [401, 403]:
                return DefectCategory.SECURITY_ISSUE
            else:
                return DefectCategory.CLIENT_ERROR
        
        # Check error message for specific patterns
        error_lower = error_message.lower()
        
        if any(term in error_lower for term in ["schema", "validation", "contract"]):
            return DefectCategory.CONTRACT_VIOLATION
        elif any(term in error_lower for term in ["auth", "token", "permission", "forbidden"]):
            return DefectCategory.SECURITY_ISSUE
        elif any(term in error_lower for term in ["timeout", "slow", "performance"]):
            return DefectCategory.PERFORMANCE_ISSUE
        elif any(term in error_lower for term in ["integrity", "consistency", "invariant"]):
            return DefectCategory.INVARIANT_VIOLATION
        elif any(term in error_lower for term in ["data", "format", "type"]):
            return DefectCategory.DATA_INTEGRITY
        else:
            return DefectCategory.SERVER_ERROR
    
    def _assess_defect_severity(self, execution: TestExecution, occurrence_count: int) -> DefectSeverity:
        """Assess the severity of a defect."""
        
        status_code = execution.response_code or 0
        error_message = execution.error_message or ""
        response_time = execution.response_time_ms or 0
        
        # Critical: 5xx errors, security issues, high frequency
        if status_code >= 500:
            return DefectSeverity.CRITICAL
        
        if any(term in error_message.lower() for term in ["security", "token", "password", "auth"]):
            return DefectSeverity.CRITICAL
        
        if occurrence_count > 10:  # High frequency failures
            return DefectSeverity.HIGH
        
        # High: Contract violations, auth issues
        if status_code in [401, 403, 422]:
            return DefectSeverity.HIGH
        
        if any(term in error_message.lower() for term in ["contract", "schema", "validation"]):
            return DefectSeverity.HIGH
        
        # Medium: Other 4xx, performance issues
        if 400 <= status_code < 500:
            return DefectSeverity.MEDIUM
        
        if response_time > 10000:  # 10+ seconds
            return DefectSeverity.MEDIUM
        
        # Low: Minor issues
        return DefectSeverity.LOW
    
    async def _refine_clusters_with_ml(self, initial_clusters: List[DefectCluster]) -> List[DefectCluster]:
        """Refine clustering using ML techniques for better grouping."""
        
        try:
            if not SKLEARN_AVAILABLE:
                return initial_clusters
            
            # Extract text features for clustering
            texts = []
            cluster_mapping = []
            
            for i, cluster in enumerate(initial_clusters):
                for fingerprint in cluster.fingerprints:
                    combined_text = f"{fingerprint.error_pattern} {fingerprint.normalized_message}"
                    texts.append(combined_text)
                    cluster_mapping.append(i)
            
            if len(texts) < 3:
                return initial_clusters
            
            # Vectorize text features
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            features = vectorizer.fit_transform(texts)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
            cluster_labels = clustering.fit_predict(features.toarray())
            
            # Reorganize clusters based on ML results
            ml_clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Not noise
                    original_cluster_idx = cluster_mapping[i]
                    ml_clusters[label].append((i, initial_clusters[original_cluster_idx]))
            
            # Merge similar clusters
            refined_clusters = []
            processed_clusters = set()
            
            for ml_cluster_id, items in ml_clusters.items():
                if ml_cluster_id in processed_clusters:
                    continue
                
                # Merge clusters in this ML group
                merged_fingerprints = []
                merged_executions = []
                affected_endpoints = set()
                timestamps = []
                
                for _, cluster in items:
                    merged_fingerprints.extend(cluster.fingerprints)
                    affected_endpoints.update(cluster.affected_endpoints)
                    timestamps.extend([cluster.first_seen, cluster.last_seen])
                
                if merged_fingerprints:
                    # Create merged cluster
                    representative_fp = merged_fingerprints[0]
                    merged_cluster = DefectCluster(
                        cluster_id=f"ml_{ml_cluster_id}_{representative_fp.fingerprint_id[:8]}",
                        fingerprints=merged_fingerprints,
                        representative_execution=items[0][1].representative_execution,
                        category=items[0][1].category,
                        severity=max(item[1].severity for _, item in items),  # Take highest severity
                        affected_endpoints=list(affected_endpoints),
                        first_seen=min(timestamps),
                        last_seen=max(timestamps),
                        occurrence_count=len(merged_fingerprints),
                        minimal_reproducer=None
                    )
                    
                    refined_clusters.append(merged_cluster)
                    processed_clusters.add(ml_cluster_id)
            
            # Add any clusters that weren't merged
            for i, cluster in enumerate(initial_clusters):
                if not any(i in [cluster_mapping[j] for j, _ in items] for items in ml_clusters.values()):
                    refined_clusters.append(cluster)
            
            return refined_clusters
            
        except Exception as e:
            logger.error(f"ML clustering refinement failed: {str(e)}")
            return initial_clusters
    
    async def _generate_minimal_reproducer(self, cluster: DefectCluster) -> Optional[MinimalReproducer]:
        """Generate a minimal reproducer for a defect cluster."""
        
        try:
            # Get the representative execution
            representative_execution = self.db.query(TestExecution).filter(
                TestExecution.id == cluster.representative_execution
            ).first()
            
            if not representative_execution or not representative_execution.test_case:
                return None
            
            test_case = representative_execution.test_case
            
            # Create minimal test case by removing unnecessary data
            minimal_test_data = await self._minimize_test_case(test_case, representative_execution)
            
            # Generate cURL command
            curl_command = self._generate_curl_command(test_case, minimal_test_data)
            
            # Generate HAR file
            har_file = self._generate_har_file(test_case, representative_execution, minimal_test_data)
            
            # Create environment snapshot
            environment_snapshot = await self._create_environment_snapshot(representative_execution)
            
            # Generate reproduction steps
            reproduction_steps = self._generate_reproduction_steps(test_case, minimal_test_data)
            
            reproducer_id = f"{cluster.cluster_id}_reproducer"
            
            return MinimalReproducer(
                reproducer_id=reproducer_id,
                curl_command=curl_command,
                har_file=har_file,
                test_case_json=minimal_test_data,
                environment_snapshot=environment_snapshot,
                reproduction_steps=reproduction_steps,
                expected_outcome=f"Expected: {test_case.expected_response}" if test_case.expected_response else "Success",
                actual_outcome=f"Actual: {representative_execution.response_code} - {representative_execution.error_message}"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate minimal reproducer: {str(e)}")
            return None
    
    async def _minimize_test_case(
        self, 
        test_case: TestCase, 
        execution: TestExecution
    ) -> Dict[str, Any]:
        """Create a minimal test case that still reproduces the defect."""
        
        # Start with full test data
        test_data = test_case.test_data.copy() if test_case.test_data else {}
        
        # Create minimal version
        minimal_data = {
            "endpoint": test_case.endpoint,
            "method": test_case.method,
            "description": f"Minimal reproducer for {execution.error_message or 'defect'}",
            "test_data": {
                "headers": self._minimize_headers(test_data.get("headers", {})),
                "query_params": self._minimize_query_params(test_data.get("query_params", {})),
                "path_params": test_data.get("path_params", {}),  # Usually required
                "body": self._minimize_body(test_data.get("body", {}))
            },
            "expected_outcome": {
                "should_fail": True,
                "expected_status": execution.response_code,
                "error_pattern": execution.error_message
            }
        }
        
        return minimal_data
    
    def _minimize_headers(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only essential headers for reproduction."""
        
        essential_headers = {}
        
        # Always keep content-type and authorization
        for key in ["Content-Type", "Authorization", "Accept"]:
            if key in headers:
                essential_headers[key] = headers[key]
            # Check case-insensitive
            for header_key, header_value in headers.items():
                if header_key.lower() == key.lower():
                    essential_headers[key] = header_value
                    break
        
        return essential_headers
    
    def _minimize_query_params(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only essential query parameters."""
        
        # For minimal reproduction, try to keep only required params
        # This is a heuristic - in practice, you'd want to test which params are actually needed
        essential_params = {}
        
        for key, value in query_params.items():
            # Keep params that look required
            if key.lower() in ["id", "key", "token", "api_key", "required"]:
                essential_params[key] = value
            # Keep params that are likely to cause the error
            elif isinstance(value, str) and (value == "" or value == "null" or value == "invalid"):
                essential_params[key] = value
        
        return essential_params
    
    def _minimize_body(self, body: Any) -> Any:
        """Minimize request body while preserving error-causing elements."""
        
        if not isinstance(body, dict):
            return body
        
        minimal_body = {}
        
        for key, value in body.items():
            # Keep fields that are likely to cause validation errors
            if isinstance(value, str) and value in ["", "null", "invalid"]:
                minimal_body[key] = value
            elif value is None:
                minimal_body[key] = value
            elif isinstance(value, (int, float)) and value < 0:
                minimal_body[key] = value
            elif key.lower() in ["id", "type", "required", "mandatory"]:
                minimal_body[key] = value
        
        return minimal_body if minimal_body else {"minimal": True}
    
    def _generate_curl_command(self, test_case: TestCase, minimal_data: Dict[str, Any]) -> str:
        """Generate a cURL command for reproducing the defect."""
        
        try:
            # Get API spec for base URL
            api_spec = self.db.query(APISpecification).filter(
                APISpecification.id == test_case.api_spec_id
            ).first()
            
            base_url = api_spec.base_url if api_spec else "https://api.example.com"
            endpoint = test_case.endpoint
            method = test_case.method.upper()
            
            # Build URL with path parameters
            url = f"{base_url.rstrip('/')}{endpoint}"
            path_params = minimal_data.get("test_data", {}).get("path_params", {})
            for param_name, param_value in path_params.items():
                url = url.replace(f"{{{param_name}}}", str(param_value))
            
            # Build cURL command
            curl_parts = [f"curl -X {method}"]
            
            # Add headers
            headers = minimal_data.get("test_data", {}).get("headers", {})
            for header_name, header_value in headers.items():
                curl_parts.append(f"-H '{header_name}: {header_value}'")
            
            # Add query parameters
            query_params = minimal_data.get("test_data", {}).get("query_params", {})
            if query_params:
                query_string = "&".join(f"{k}={v}" for k, v in query_params.items())
                url += f"?{query_string}"
            
            # Add body data
            body = minimal_data.get("test_data", {}).get("body")
            if body and method in ["POST", "PUT", "PATCH"]:
                if isinstance(body, dict):
                    curl_parts.append(f"-d '{json.dumps(body)}'")
                else:
                    curl_parts.append(f"-d '{body}'")
            
            # Add URL
            curl_parts.append(f"'{url}'")
            
            return " \\\n  ".join(curl_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate cURL command: {str(e)}")
            return f"# Error generating cURL: {str(e)}"
    
    def _generate_har_file(
        self, 
        test_case: TestCase, 
        execution: TestExecution,
        minimal_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a HAR (HTTP Archive) file for the reproducer."""
        
        try:
            # Get API spec for base URL
            api_spec = self.db.query(APISpecification).filter(
                APISpecification.id == test_case.api_spec_id
            ).first()
            
            base_url = api_spec.base_url if api_spec else "https://api.example.com"
            
            # Create HAR structure
            har = {
                "log": {
                    "version": "1.2",
                    "creator": {
                        "name": "AI API Testing Framework",
                        "version": "1.0"
                    },
                    "entries": [{
                        "startedDateTime": execution.started_at.isoformat() if execution.started_at else datetime.utcnow().isoformat(),
                        "time": execution.response_time_ms or 0,
                        "request": {
                            "method": test_case.method.upper(),
                            "url": f"{base_url.rstrip('/')}{test_case.endpoint}",
                            "httpVersion": "HTTP/1.1",
                            "headers": [
                                {"name": k, "value": str(v)} 
                                for k, v in minimal_data.get("test_data", {}).get("headers", {}).items()
                            ],
                            "queryString": [
                                {"name": k, "value": str(v)}
                                for k, v in minimal_data.get("test_data", {}).get("query_params", {}).items()
                            ],
                            "postData": {
                                "mimeType": "application/json",
                                "text": json.dumps(minimal_data.get("test_data", {}).get("body", {}))
                            } if minimal_data.get("test_data", {}).get("body") else {},
                            "headersSize": -1,
                            "bodySize": len(json.dumps(minimal_data.get("test_data", {}).get("body", {}))) if minimal_data.get("test_data", {}).get("body") else 0
                        },
                        "response": {
                            "status": execution.response_code or 0,
                            "statusText": self._get_status_text(execution.response_code),
                            "httpVersion": "HTTP/1.1",
                            "headers": [
                                {"name": k, "value": str(v)}
                                for k, v in (execution.response_headers or {}).items()
                            ],
                            "content": {
                                "size": len(str(execution.response_body)) if execution.response_body else 0,
                                "mimeType": execution.response_headers.get("Content-Type", "application/json") if execution.response_headers else "application/json",
                                "text": json.dumps(execution.response_body) if execution.response_body else ""
                            },
                            "redirectURL": "",
                            "headersSize": -1,
                            "bodySize": len(str(execution.response_body)) if execution.response_body else 0
                        },
                        "cache": {},
                        "timings": {
                            "blocked": 0,
                            "dns": 0,
                            "connect": 0,
                            "send": 0,
                            "wait": execution.response_time_ms or 0,
                            "receive": 0,
                            "ssl": 0
                        }
                    }]
                }
            }
            
            return har
            
        except Exception as e:
            logger.error(f"Failed to generate HAR file: {str(e)}")
            return {"error": str(e)}
    
    def _get_status_text(self, status_code: Optional[int]) -> str:
        """Get status text for HTTP status code."""
        
        status_texts = {
            200: "OK", 201: "Created", 204: "No Content",
            400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 
            404: "Not Found", 422: "Unprocessable Entity",
            500: "Internal Server Error", 502: "Bad Gateway", 
            503: "Service Unavailable", 504: "Gateway Timeout"
        }
        
        return status_texts.get(status_code or 0, "Unknown")
    
    async def _create_environment_snapshot(self, execution: TestExecution) -> Dict[str, Any]:
        """Create a snapshot of the environment for reproduction."""
        
        return {
            "timestamp": execution.created_at.isoformat() if execution.created_at else datetime.utcnow().isoformat(),
            "test_environment": {
                "execution_id": execution.id,
                "session_id": execution.session_id,
                "test_case_id": execution.test_case_id
            },
            "system_info": {
                "framework_version": "1.0",
                "execution_engine": "AsyncHTTP",
                "timeout_seconds": 300  # Default timeout
            },
            "api_context": {
                "api_spec_id": execution.test_case.api_spec_id if execution.test_case else None,
                "endpoint": execution.test_case.endpoint if execution.test_case else None,
                "method": execution.test_case.method if execution.test_case else None
            }
        }
    
    def _generate_reproduction_steps(
        self, 
        test_case: TestCase, 
        minimal_data: Dict[str, Any]
    ) -> List[str]:
        """Generate human-readable reproduction steps."""
        
        steps = []
        
        # Basic setup
        steps.append("## Reproduction Steps")
        steps.append("1. Set up API testing environment")
        
        # Authentication if needed
        headers = minimal_data.get("test_data", {}).get("headers", {})
        if "Authorization" in headers:
            steps.append("2. Configure authentication token")
        
        # Request preparation
        method = test_case.method.upper()
        endpoint = test_case.endpoint
        
        steps.append(f"3. Prepare {method} request to {endpoint}")
        
        # Add specific parameters
        query_params = minimal_data.get("test_data", {}).get("query_params", {})
        if query_params:
            steps.append(f"4. Add query parameters: {json.dumps(query_params)}")
        
        path_params = minimal_data.get("test_data", {}).get("path_params", {})
        if path_params:
            steps.append(f"5. Set path parameters: {json.dumps(path_params)}")
        
        body = minimal_data.get("test_data", {}).get("body")
        if body:
            steps.append(f"6. Set request body: {json.dumps(body)}")
        
        # Execution
        steps.append("7. Execute the request")
        steps.append("8. Observe the failure/error response")
        
        return steps
    
    def _calculate_severity_distribution(self, clusters: List[DefectCluster]) -> Dict[str, Any]:
        """Calculate the distribution of defect severities."""
        
        distribution = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        category_distribution = defaultdict(int)
        
        for cluster in clusters:
            distribution[cluster.severity.value] += cluster.occurrence_count
            category_distribution[cluster.category.value] += cluster.occurrence_count
        
        total_defects = sum(distribution.values())
        
        return {
            "by_severity": distribution,
            "by_category": dict(category_distribution),
            "total_defects": total_defects,
            "severity_percentages": {
                severity: (count / total_defects * 100) if total_defects > 0 else 0
                for severity, count in distribution.items()
            }
        }
    
    def _generate_defect_recommendations(self, clusters: List[DefectCluster]) -> List[str]:
        """Generate recommendations based on defect analysis."""
        
        recommendations = []
        
        # Count by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for cluster in clusters:
            severity_counts[cluster.severity.value] += 1
        
        # Generate severity-based recommendations
        if severity_counts["critical"] > 0:
            recommendations.append(f"URGENT: {severity_counts['critical']} critical defects require immediate attention")
        
        if severity_counts["high"] > 3:
            recommendations.append(f"{severity_counts['high']} high-severity defects should be prioritized in next sprint")
        
        # Category-based recommendations
        category_counts = defaultdict(int)
        for cluster in clusters:
            category_counts[cluster.category.value] += 1
        
        if category_counts["server_error"] > 2:
            recommendations.append("Multiple server errors detected - investigate backend stability")
        
        if category_counts["security_issue"] > 0:
            recommendations.append("Security issues found - conduct security audit")
        
        if category_counts["contract_violation"] > 1:
            recommendations.append("Contract violations detected - validate API specification accuracy")
        
        # Frequency-based recommendations
        high_frequency_clusters = [c for c in clusters if c.occurrence_count > 5]
        if high_frequency_clusters:
            recommendations.append(f"{len(high_frequency_clusters)} defects are occurring frequently - investigate root causes")
        
        # If no issues
        if not clusters:
            recommendations.append("No defects detected - excellent API quality!")
        
        return recommendations
    
    def _serialize_cluster(self, cluster: DefectCluster) -> Dict[str, Any]:
        """Serialize defect cluster for JSON response."""
        
        return {
            "cluster_id": cluster.cluster_id,
            "category": cluster.category.value,
            "severity": cluster.severity.value,
            "occurrence_count": cluster.occurrence_count,
            "affected_endpoints": cluster.affected_endpoints,
            "first_seen": cluster.first_seen.isoformat(),
            "last_seen": cluster.last_seen.isoformat(),
            "representative_execution_id": cluster.representative_execution,
            "fingerprint_count": len(cluster.fingerprints),
            "sample_error_patterns": list(set(fp.error_pattern for fp in cluster.fingerprints[:3]))
        }
    
    def _serialize_reproducer(self, reproducer: MinimalReproducer) -> Dict[str, Any]:
        """Serialize minimal reproducer for JSON response."""
        
        return {
            "reproducer_id": reproducer.reproducer_id,
            "curl_command": reproducer.curl_command,
            "test_case": reproducer.test_case_json,
            "reproduction_steps": reproducer.reproduction_steps,
            "expected_outcome": reproducer.expected_outcome,
            "actual_outcome": reproducer.actual_outcome,
            "har_available": bool(reproducer.har_file.get("log")),
            "environment_context": reproducer.environment_snapshot
        }
    
    async def get_defect_trends(
        self, 
        api_spec_id: int, 
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Analyze defect trends over time."""
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Get failed executions in time range
            failed_executions = self.db.query(TestExecution).join(ExecutionSession).filter(
                ExecutionSession.api_spec_id == api_spec_id,
                TestExecution.status.in_([TestStatus.FAILED, TestStatus.ERROR]),
                TestExecution.created_at >= cutoff_date
            ).all()
            
            # Group by day
            daily_failures = defaultdict(int)
            daily_categories = defaultdict(lambda: defaultdict(int))
            
            for execution in failed_executions:
                day = execution.created_at.date().isoformat()
                daily_failures[day] += 1
                
                category = self._categorize_defect(execution)
                daily_categories[day][category.value] += 1
            
            # Calculate trend metrics
            days = sorted(daily_failures.keys())
            failure_counts = [daily_failures[day] for day in days]
            
            trend_direction = "stable"
            if len(failure_counts) >= 7:  # Need at least a week of data
                recent_avg = sum(failure_counts[-7:]) / 7
                older_avg = sum(failure_counts[-14:-7]) / 7 if len(failure_counts) >= 14 else recent_avg
                
                if recent_avg > older_avg * 1.2:
                    trend_direction = "increasing"
                elif recent_avg < older_avg * 0.8:
                    trend_direction = "decreasing"
            
            return {
                "time_range": {
                    "start_date": cutoff_date.isoformat(),
                    "end_date": datetime.utcnow().isoformat(),
                    "days_analyzed": days_back
                },
                "trend_analysis": {
                    "direction": trend_direction,
                    "total_failures": sum(failure_counts),
                    "avg_failures_per_day": sum(failure_counts) / len(days) if days else 0,
                    "peak_failure_day": max(days, key=lambda d: daily_failures[d]) if days else None
                },
                "daily_breakdown": [
                    {
                        "date": day,
                        "total_failures": daily_failures[day],
                        "by_category": dict(daily_categories[day])
                    }
                    for day in days
                ],
                "category_trends": self._analyze_category_trends(daily_categories)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze defect trends: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_category_trends(self, daily_categories: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """Analyze trends by defect category."""
        
        category_trends = {}
        all_categories = set()
        
        for day_categories in daily_categories.values():
            all_categories.update(day_categories.keys())
        
        for category in all_categories:
            daily_counts = []
            for day in sorted(daily_categories.keys()):
                daily_counts.append(daily_categories[day].get(category, 0))
            
            if len(daily_counts) >= 3:
                # Simple trend calculation
                recent_sum = sum(daily_counts[-3:])
                older_sum = sum(daily_counts[-6:-3]) if len(daily_counts) >= 6 else recent_sum
                
                if recent_sum > older_sum:
                    trend = "increasing"
                elif recent_sum < older_sum:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            category_trends[category] = {
                "trend": trend,
                "total_occurrences": sum(daily_counts),
                "avg_per_day": sum(daily_counts) / len(daily_counts) if daily_counts else 0
            }
        
        return category_trends
    
    async def deduplicate_defects(
        self, 
        execution_ids: List[int]
    ) -> Dict[str, Any]:
        """Deduplicate defects based on fingerprinting."""
        
        try:
            executions = self.db.query(TestExecution).filter(
                TestExecution.id.in_(execution_ids)
            ).all()
            
            # Generate fingerprints
            fingerprint_groups = defaultdict(list)
            
            for execution in executions:
                fingerprint = await self._generate_defect_fingerprint(execution)
                fingerprint_groups[fingerprint.fingerprint_id].append(execution)
            
            # Analyze deduplication results
            unique_defects = len(fingerprint_groups)
            total_executions = len(executions)
            duplicate_count = total_executions - unique_defects
            
            # Create deduplication report
            duplicate_groups = []
            for fingerprint_id, group_executions in fingerprint_groups.items():
                if len(group_executions) > 1:
                    duplicate_groups.append({
                        "fingerprint_id": fingerprint_id,
                        "duplicate_count": len(group_executions),
                        "execution_ids": [ex.id for ex in group_executions],
                        "first_occurrence": min(ex.created_at for ex in group_executions if ex.created_at).isoformat(),
                        "last_occurrence": max(ex.created_at for ex in group_executions if ex.created_at).isoformat()
                    })
            
            return {
                "deduplication_summary": {
                    "total_executions": total_executions,
                    "unique_defects": unique_defects,
                    "duplicates_removed": duplicate_count,
                    "deduplication_ratio": duplicate_count / total_executions if total_executions > 0 else 0
                },
                "duplicate_groups": duplicate_groups,
                "unique_fingerprints": list(fingerprint_groups.keys())
            }
            
        except Exception as e:
            logger.error(f"Defect deduplication failed: {str(e)}")
            return {"error": str(e)}
    
    async def create_jira_ticket_data(self, cluster: DefectCluster) -> Dict[str, Any]:
        """Generate JIRA ticket data for a defect cluster."""
        
        try:
            # Get representative execution details
            execution = self.db.query(TestExecution).filter(
                TestExecution.id == cluster.representative_execution
            ).first()
            
            if not execution or not execution.test_case:
                return {"error": "Representative execution not found"}
            
            # Generate ticket title
            title = f"[API Defect] {cluster.severity.value.upper()}: {cluster.category.value.replace('_', ' ').title()} in {execution.test_case.endpoint}"
            
            # Generate description
            description_parts = [
                f"## Defect Summary",
                f"**Severity:** {cluster.severity.value.upper()}",
                f"**Category:** {cluster.category.value.replace('_', ' ').title()}",
                f"**Occurrences:** {cluster.occurrence_count}",
                f"**First Seen:** {cluster.first_seen.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Last Seen:** {cluster.last_seen.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"## Affected Endpoints",
                "\n".join(f"- {endpoint}" for endpoint in cluster.affected_endpoints),
                "",
                f"## Error Details",
                f"**Status Code:** {execution.response_code}",
                f"**Error Message:** {execution.error_message or 'No error message'}",
                "",
                f"## Test Case Details",
                f"**Endpoint:** {execution.test_case.endpoint}",
                f"**Method:** {execution.test_case.method.upper()}",
                f"**Test Type:** {execution.test_case.test_type.value}",
                ""
            ]
            
            # Add reproducer if available
            if cluster.minimal_reproducer:
                description_parts.extend([
                    "## Reproduction",
                    "```bash",
                    cluster.minimal_reproducer.curl_command if hasattr(cluster.minimal_reproducer, 'curl_command') else "# Reproducer not available",
                    "```"
                ])
            
            description = "\n".join(description_parts)
            
            # Determine priority and labels
            priority_map = {
                DefectSeverity.CRITICAL: "Highest",
                DefectSeverity.HIGH: "High", 
                DefectSeverity.MEDIUM: "Medium",
                DefectSeverity.LOW: "Low"
            }
            
            labels = [
                "api-testing",
                f"severity-{cluster.severity.value}",
                f"category-{cluster.category.value}",
                "automated-detection"
            ]
            
            if cluster.occurrence_count > 5:
                labels.append("high-frequency")
            
            return {
                "title": title,
                "description": description,
                "priority": priority_map[cluster.severity],
                "labels": labels,
                "issue_type": "Bug",
                "assignee": None,  # To be determined by team
                "components": ["API"],
                "custom_fields": {
                    "cluster_id": cluster.cluster_id,
                    "occurrence_count": cluster.occurrence_count,
                    "execution_id": cluster.representative_execution,
                    "fingerprint_ids": [fp.fingerprint_id for fp in cluster.fingerprints[:5]]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to create JIRA ticket data: {str(e)}")
            return {"error": str(e)}
    
    async def analyze_defect_patterns(
        self,
        api_spec_id: int,
        time_window_days: int = 7
    ) -> Dict[str, Any]:
        """Analyze patterns in defects to identify systemic issues."""
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            
            # Get recent failures
            failed_executions = self.db.query(TestExecution).join(ExecutionSession).filter(
                ExecutionSession.api_spec_id == api_spec_id,
                TestExecution.status.in_([TestStatus.FAILED, TestStatus.ERROR]),
                TestExecution.created_at >= cutoff_date
            ).all()
            
            if not failed_executions:
                return {
                    "message": "No failures in time window",
                    "patterns": []
                }
            
            # Analyze patterns
            patterns = {}
            
            # Endpoint patterns
            endpoint_failures = defaultdict(int)
            for execution in failed_executions:
                if execution.test_case:
                    endpoint_failures[execution.test_case.endpoint] += 1
            
            patterns["endpoint_hotspots"] = [
                {"endpoint": endpoint, "failure_count": count}
                for endpoint, count in sorted(endpoint_failures.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
            
            # Time patterns
            hourly_failures = defaultdict(int)
            for execution in failed_executions:
                if execution.created_at:
                    hour = execution.created_at.hour
                    hourly_failures[hour] += 1
            
            patterns["time_patterns"] = {
                "hourly_distribution": dict(hourly_failures),
                "peak_failure_hour": max(hourly_failures.items(), key=lambda x: x[1])[0] if hourly_failures else None
            }
            
            # Error message patterns
            error_patterns = defaultdict(int)
            for execution in failed_executions:
                if execution.error_message:
                    # Extract key error terms
                    error_terms = re.findall(r'\b[A-Z][a-z]+(?:Error|Exception)\b', execution.error_message)
                    for term in error_terms:
                        error_patterns[term] += 1
            
            patterns["error_patterns"] = dict(sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Response code patterns
            status_code_failures = defaultdict(int)
            for execution in failed_executions:
                if execution.response_code:
                    status_code_failures[execution.response_code] += 1
            
            patterns["status_code_distribution"] = dict(status_code_failures)
            
            # Generate insights
            insights = []
            
            # Endpoint insights
            top_failing_endpoint = max(endpoint_failures.items(), key=lambda x: x[1]) if endpoint_failures else None
            if top_failing_endpoint and top_failing_endpoint[1] > len(failed_executions) * 0.3:
                insights.append(f"Endpoint {top_failing_endpoint[0]} accounts for {top_failing_endpoint[1]} failures ({top_failing_endpoint[1]/len(failed_executions)*100:.1f}%)")
            
            # Time insights
            if hourly_failures:
                peak_hour = max(hourly_failures.items(), key=lambda x: x[1])
                if peak_hour[1] > len(failed_executions) * 0.2:
                    insights.append(f"Failures peak at hour {peak_hour[0]} with {peak_hour[1]} failures")
            
            # Severity insights
            critical_failures = sum(1 for ex in failed_executions if ex.response_code and ex.response_code >= 500)
            if critical_failures > 0:
                insights.append(f"{critical_failures} critical server errors detected ({critical_failures/len(failed_executions)*100:.1f}%)")
            
            return {
                "analysis_period": {
                    "start_date": cutoff_date.isoformat(),
                    "end_date": datetime.utcnow().isoformat(),
                    "total_failures": len(failed_executions)
                },
                "patterns": patterns,
                "insights": insights,
                "recommendations": self._generate_pattern_recommendations(patterns, insights)
            }
            
        except Exception as e:
            logger.error(f"Defect pattern analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _generate_pattern_recommendations(
        self, 
        patterns: Dict[str, Any], 
        insights: List[str]
    ) -> List[str]:
        """Generate recommendations based on defect patterns."""
        
        recommendations = []
        
        # Endpoint-based recommendations
        endpoint_hotspots = patterns.get("endpoint_hotspots", [])
        if endpoint_hotspots and endpoint_hotspots[0]["failure_count"] > 10:
            recommendations.append(f"Focus testing efforts on {endpoint_hotspots[0]['endpoint']} - highest failure rate")
        
        # Time-based recommendations
        time_patterns = patterns.get("time_patterns", {})
        peak_hour = time_patterns.get("peak_failure_hour")
        if peak_hour is not None:
            recommendations.append(f"Investigate system load at hour {peak_hour} - peak failure time")
        
        # Error pattern recommendations
        error_patterns = patterns.get("error_patterns", {})
        if error_patterns:
            top_error = max(error_patterns.items(), key=lambda x: x[1])
            recommendations.append(f"Address {top_error[0]} - most common error pattern ({top_error[1]} occurrences)")
        
        # Status code recommendations
        status_dist = patterns.get("status_code_distribution", {})
        if 500 in status_dist and status_dist[500] > 5:
            recommendations.append("High number of 500 errors - investigate server stability")
        if 422 in status_dist and status_dist[422] > 3:
            recommendations.append("Multiple validation errors - review input validation logic")
        
        return recommendations
    
    def __del__(self):
        """Clean up database session."""
        if hasattr(self, 'db'):
            self.db.close()
