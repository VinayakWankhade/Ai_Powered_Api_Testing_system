"""
API specification ingestion and parsing system.
Supports OpenAPI/Swagger specs and raw API logs.
"""

import json
import yaml
import re
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse, urljoin
from datetime import datetime

import requests
from jsonschema import validate, ValidationError
from swagger_spec_validator import validate_spec_url
from openapi_spec_validator import validate_spec as validate_openapi_spec

from ..database.models import APISpecification, SpecType, DocumentationStore
from ..database.connection import get_db_session
from ..utils.logger import get_logger

logger = get_logger(__name__)

class SpecIngestionError(Exception):
    """Custom exception for specification ingestion errors."""
    pass

class APISpecIngester:
    """
    Main class for ingesting and parsing API specifications.
    """

    def __init__(self):
        self.db = get_db_session()

    def ingest_spec(
        self,
        spec_content: Union[str, Dict, bytes],
        spec_type: SpecType,
        name: str,
        version: Optional[str] = None,
        base_url: Optional[str] = None,
        description: Optional[str] = None
    ) -> APISpecification:
        """
        Ingest an API specification and store it in the database.
        
        Args:
            spec_content: The specification content (JSON, YAML, or raw logs)
            spec_type: Type of specification (OpenAPI, Swagger, Raw Logs)
            name: Name identifier for the API
            version: Version of the API
            base_url: Base URL for the API
            description: Description of the API
            
        Returns:
            APISpecification: The created specification record
        """
        try:
            # Parse the specification content
            parsed_spec = self._parse_spec_content(spec_content, spec_type)
            
            # Extract metadata from the specification
            extracted_metadata = self._extract_metadata(parsed_spec, spec_type)
            
            # Merge provided metadata with extracted metadata
            final_metadata = {
                "name": name,
                "version": version or extracted_metadata.get("version", "1.0.0"),
                "description": description or extracted_metadata.get("description", ""),
                "base_url": base_url or extracted_metadata.get("base_url", ""),
            }
            
            # Parse endpoints
            parsed_endpoints = self._parse_endpoints(parsed_spec, spec_type)
            
            # Validate the specification
            validation_result = self._validate_spec(parsed_spec, spec_type)
            if not validation_result["valid"]:
                logger.warning(f"Specification validation failed: {validation_result['errors']}")
            
            # Create database record
            api_spec = APISpecification(
                name=final_metadata["name"],
                version=final_metadata["version"],
                description=final_metadata["description"],
                spec_type=spec_type,
                base_url=final_metadata["base_url"],
                raw_content=parsed_spec,
                parsed_endpoints=parsed_endpoints
            )
            
            # Check for existing specification with same name/version
            existing_spec = self.db.query(APISpecification).filter(
                APISpecification.name == final_metadata["name"],
                APISpecification.version == final_metadata["version"]
            ).first()
            
            if existing_spec:
                # Update existing specification
                existing_spec.description = final_metadata["description"]
                existing_spec.base_url = final_metadata["base_url"]
                existing_spec.raw_content = parsed_spec
                existing_spec.parsed_endpoints = parsed_endpoints
                existing_spec.updated_at = datetime.utcnow()
                api_spec = existing_spec
            else:
                # Add new specification
                self.db.add(api_spec)
            
            self.db.commit()
            self.db.refresh(api_spec)
            
            # Extract and store documentation
            self._extract_and_store_documentation(api_spec, parsed_spec, spec_type)
            
            logger.info(f"Successfully ingested API specification: {final_metadata['name']} v{final_metadata['version']}")
            return api_spec
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to ingest API specification: {str(e)}")
            raise SpecIngestionError(f"Ingestion failed: {str(e)}")

    def ingest_from_url(
        self,
        url: str,
        spec_type: SpecType,
        name: str,
        version: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> APISpecification:
        """
        Ingest API specification from a URL.
        
        Args:
            url: URL to fetch the specification from
            spec_type: Type of specification expected
            name: Name identifier for the API
            version: Version of the API
            headers: Optional HTTP headers for the request
            
        Returns:
            APISpecification: The created specification record
        """
        try:
            response = requests.get(url, headers=headers or {}, timeout=30)
            response.raise_for_status()
            
            # Try to determine base URL from the specification URL
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            return self.ingest_spec(
                spec_content=response.text,
                spec_type=spec_type,
                name=name,
                version=version,
                base_url=base_url
            )
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch specification from URL {url}: {str(e)}")
            raise SpecIngestionError(f"Failed to fetch from URL: {str(e)}")

    def _parse_spec_content(
        self,
        content: Union[str, Dict, bytes],
        spec_type: SpecType
    ) -> Dict[str, Any]:
        """Parse specification content based on type."""
        
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        if isinstance(content, dict):
            return content
        
        if spec_type in [SpecType.OPENAPI, SpecType.SWAGGER]:
            try:
                # Try JSON first
                return json.loads(content)
            except json.JSONDecodeError:
                try:
                    # Try YAML
                    return yaml.safe_load(content)
                except yaml.YAMLError as e:
                    raise SpecIngestionError(f"Failed to parse YAML content: {str(e)}")
        
        elif spec_type == SpecType.RAW_LOGS:
            return self._parse_raw_logs(content)
        
        else:
            raise SpecIngestionError(f"Unsupported specification type: {spec_type}")

    def _parse_raw_logs(self, log_content: str) -> Dict[str, Any]:
        """
        Parse raw API logs to extract endpoint information.
        This is a simplified implementation - can be enhanced for specific log formats.
        """
        endpoints = {}
        lines = log_content.strip().split('\n')
        
        # Common log patterns
        patterns = [
            # Apache/Nginx access log format
            r'\"([A-Z]+)\s+([^\s]+)\s+HTTP/[\d.]+\"\s+(\d+)',
            # Simple format: METHOD /path STATUS
            r'([A-Z]+)\s+([^\s]+)\s+(\d+)',
            # JSON log format
            r'\{.*\"method\":\s*\"([A-Z]+)\".*\"path\":\s*\"([^\"]+)\".*\"status\":\s*(\d+)',
        ]
        
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    method, path, status = match.groups()
                    
                    if path not in endpoints:
                        endpoints[path] = {}
                    
                    if method not in endpoints[path]:
                        endpoints[path][method] = {
                            "responses": {},
                            "examples": []
                        }
                    
                    if status not in endpoints[path][method]["responses"]:
                        endpoints[path][method]["responses"][status] = {"count": 0}
                    
                    endpoints[path][method]["responses"][status]["count"] += 1
                    break
        
        return {
            "info": {
                "title": "API from Raw Logs",
                "version": "1.0.0",
                "description": f"API specification extracted from {len(lines)} log entries"
            },
            "paths": endpoints,
            "extracted_from_logs": True,
            "log_lines_processed": len(lines)
        }

    def _extract_metadata(self, spec: Dict[str, Any], spec_type: SpecType) -> Dict[str, Any]:
        """Extract metadata from the parsed specification."""
        
        metadata = {}
        
        if spec_type in [SpecType.OPENAPI, SpecType.SWAGGER]:
            info = spec.get("info", {})
            metadata["version"] = info.get("version", "1.0.0")
            metadata["description"] = info.get("description", "")
            metadata["title"] = info.get("title", "")
            
            # Extract base URL
            if "servers" in spec:  # OpenAPI 3.x
                servers = spec["servers"]
                if servers:
                    metadata["base_url"] = servers[0].get("url", "")
            elif "host" in spec:  # Swagger 2.x
                scheme = spec.get("schemes", ["http"])[0]
                host = spec["host"]
                base_path = spec.get("basePath", "")
                metadata["base_url"] = f"{scheme}://{host}{base_path}"
        
        elif spec_type == SpecType.RAW_LOGS:
            info = spec.get("info", {})
            metadata["version"] = info.get("version", "1.0.0")
            metadata["description"] = info.get("description", "")
            metadata["title"] = info.get("title", "")
        
        return metadata

    def _parse_endpoints(self, spec: Dict[str, Any], spec_type: SpecType) -> Dict[str, Any]:
        """Parse endpoints from the specification."""
        
        endpoints = {}
        
        if spec_type in [SpecType.OPENAPI, SpecType.SWAGGER]:
            paths = spec.get("paths", {})
            
            for path, path_item in paths.items():
                endpoints[path] = {}
                
                for method, operation in path_item.items():
                    if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]:
                        endpoints[path][method.upper()] = {
                            "summary": operation.get("summary", ""),
                            "description": operation.get("description", ""),
                            "parameters": operation.get("parameters", []),
                            "responses": operation.get("responses", {}),
                            "tags": operation.get("tags", []),
                            "operationId": operation.get("operationId", ""),
                            "requestBody": operation.get("requestBody", {}),
                            "security": operation.get("security", [])
                        }
        
        elif spec_type == SpecType.RAW_LOGS:
            endpoints = spec.get("paths", {})
        
        return endpoints

    def _validate_spec(self, spec: Dict[str, Any], spec_type: SpecType) -> Dict[str, Any]:
        """Validate the specification according to its type."""
        
        result = {"valid": True, "errors": []}
        
        try:
            if spec_type == SpecType.OPENAPI:
                validate_openapi_spec(spec)
            elif spec_type == SpecType.SWAGGER:
                # Note: validate_spec_url requires a URL, we'll skip swagger validation for now
                pass
            # Raw logs don't have standard validation
        
        except Exception as e:
            result["valid"] = False
            result["errors"].append(str(e))
        
        return result

    def _extract_and_store_documentation(
        self,
        api_spec: APISpecification,
        spec: Dict[str, Any],
        spec_type: SpecType
    ):
        """Extract and store documentation for RAG system."""
        
        try:
            # Extract general API documentation
            if "info" in spec:
                info = spec["info"]
                if info.get("description"):
                    doc = DocumentationStore(
                        api_spec_id=api_spec.id,
                        title=f"{info.get('title', 'API')} - General Description",
                        doc_type="api_description",
                        source="specification",
                        content=info["description"]
                    )
                    self.db.add(doc)
            
            # Extract endpoint documentation
            if spec_type in [SpecType.OPENAPI, SpecType.SWAGGER]:
                paths = spec.get("paths", {})
                
                for path, path_item in paths.items():
                    for method, operation in path_item.items():
                        if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]:
                            
                            # Create documentation for each endpoint
                            doc_content = self._format_endpoint_documentation(
                                path, method.upper(), operation
                            )
                            
                            doc = DocumentationStore(
                                api_spec_id=api_spec.id,
                                title=f"{method.upper()} {path}",
                                doc_type="endpoint_doc",
                                source="specification",
                                content=doc_content,
                                structured_content={
                                    "path": path,
                                    "method": method.upper(),
                                    "operation": operation
                                }
                            )
                            self.db.add(doc)
            
            self.db.commit()
            logger.info(f"Extracted documentation for API specification: {api_spec.name}")
        
        except Exception as e:
            logger.error(f"Failed to extract documentation: {str(e)}")

    def _format_endpoint_documentation(
        self,
        path: str,
        method: str,
        operation: Dict[str, Any]
    ) -> str:
        """Format endpoint information into readable documentation."""
        
        doc_parts = [
            f"Endpoint: {method} {path}",
            f"Summary: {operation.get('summary', 'No summary provided')}",
            f"Description: {operation.get('description', 'No description provided')}"
        ]
        
        # Add parameters information
        parameters = operation.get("parameters", [])
        if parameters:
            doc_parts.append("Parameters:")
            for param in parameters:
                param_info = f"- {param.get('name', 'unknown')} ({param.get('in', 'unknown')})"
                if param.get('required'):
                    param_info += " [REQUIRED]"
                if param.get('description'):
                    param_info += f": {param['description']}"
                doc_parts.append(param_info)
        
        # Add response information
        responses = operation.get("responses", {})
        if responses:
            doc_parts.append("Responses:")
            for status_code, response in responses.items():
                response_info = f"- {status_code}: {response.get('description', 'No description')}"
                doc_parts.append(response_info)
        
        return "\n".join(doc_parts)

    def get_spec_by_id(self, spec_id: int) -> Optional[APISpecification]:
        """Get API specification by ID."""
        return self.db.query(APISpecification).filter(APISpecification.id == spec_id).first()

    def list_specs(self, active_only: bool = True) -> List[APISpecification]:
        """List all API specifications."""
        query = self.db.query(APISpecification)
        if active_only:
            query = query.filter(APISpecification.is_active == True)
        return query.all()

    def delete_spec(self, spec_id: int) -> bool:
        """Soft delete an API specification."""
        spec = self.get_spec_by_id(spec_id)
        if spec:
            spec.is_active = False
            self.db.commit()
            return True
        return False

    def __del__(self):
        """Clean up database session."""
        if hasattr(self, 'db'):
            self.db.close()
