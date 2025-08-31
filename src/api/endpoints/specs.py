"""
API endpoints for managing API specifications.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...database.connection import get_db
from ...database.models import APISpecification, SpecType
from ...core.spec_ingestion import APISpecIngester
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

class SpecUploadRequest(BaseModel):
    """Request model for uploading API specification."""
    name: str = Field(..., description="Name of the API")
    version: Optional[str] = Field("1.0.0", description="Version of the API")
    description: Optional[str] = Field(None, description="Description of the API")
    spec_type: SpecType = Field(..., description="Type of specification")
    spec_content: str = Field(..., description="Specification content (JSON/YAML)")
    base_url: Optional[str] = Field(None, description="Base URL for the API")

class SpecUploadFromURLRequest(BaseModel):
    """Request model for uploading API specification from URL."""
    name: str = Field(..., description="Name of the API")
    version: Optional[str] = Field("1.0.0", description="Version of the API")
    spec_type: SpecType = Field(..., description="Type of specification")
    url: str = Field(..., description="URL to fetch specification from")
    headers: Optional[Dict[str, str]] = Field(None, description="Optional HTTP headers")

class SpecResponse(BaseModel):
    """Response model for API specification."""
    id: int
    name: str
    version: str
    description: Optional[str]
    spec_type: SpecType
    base_url: Optional[str]
    created_at: str
    updated_at: str
    is_active: bool
    endpoint_count: int

@router.post("/upload-spec", response_model=SpecResponse)
async def upload_spec(
    request: SpecUploadRequest,
    db: Session = Depends(get_db)
):
    """
    Upload an API specification.
    
    Supports OpenAPI/Swagger specifications and raw API logs.
    """
    try:
        ingester = APISpecIngester()
        
        api_spec = ingester.ingest_spec(
            spec_content=request.spec_content,
            spec_type=request.spec_type,
            name=request.name,
            version=request.version,
            base_url=request.base_url,
            description=request.description
        )
        
        endpoint_count = len(api_spec.parsed_endpoints) if api_spec.parsed_endpoints else 0
        
        return SpecResponse(
            id=api_spec.id,
            name=api_spec.name,
            version=api_spec.version,
            description=api_spec.description,
            spec_type=api_spec.spec_type,
            base_url=api_spec.base_url,
            created_at=api_spec.created_at.isoformat(),
            updated_at=api_spec.updated_at.isoformat(),
            is_active=api_spec.is_active,
            endpoint_count=endpoint_count
        )
        
    except Exception as e:
        logger.error(f"Failed to upload specification: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to upload specification: {str(e)}"
        )

@router.post("/upload-spec-from-url", response_model=SpecResponse)
async def upload_spec_from_url(
    request: SpecUploadFromURLRequest,
    db: Session = Depends(get_db)
):
    """
    Upload an API specification from a URL.
    
    Fetches the specification from the provided URL and processes it.
    """
    try:
        ingester = APISpecIngester()
        
        api_spec = ingester.ingest_from_url(
            url=request.url,
            spec_type=request.spec_type,
            name=request.name,
            version=request.version,
            headers=request.headers
        )
        
        endpoint_count = len(api_spec.parsed_endpoints) if api_spec.parsed_endpoints else 0
        
        return SpecResponse(
            id=api_spec.id,
            name=api_spec.name,
            version=api_spec.version,
            description=api_spec.description,
            spec_type=api_spec.spec_type,
            base_url=api_spec.base_url,
            created_at=api_spec.created_at.isoformat(),
            updated_at=api_spec.updated_at.isoformat(),
            is_active=api_spec.is_active,
            endpoint_count=endpoint_count
        )
        
    except Exception as e:
        logger.error(f"Failed to upload specification from URL: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to upload specification from URL: {str(e)}"
        )

@router.post("/upload-spec-file", response_model=SpecResponse)
async def upload_spec_file(
    name: str,
    spec_type: SpecType,
    file: UploadFile = File(...),
    version: Optional[str] = "1.0.0",
    description: Optional[str] = None,
    base_url: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Upload an API specification from a file.
    
    Accepts JSON, YAML, or text files containing API specifications.
    """
    try:
        content = await file.read()
        
        ingester = APISpecIngester()
        
        api_spec = ingester.ingest_spec(
            spec_content=content,
            spec_type=spec_type,
            name=name,
            version=version,
            base_url=base_url,
            description=description
        )
        
        endpoint_count = len(api_spec.parsed_endpoints) if api_spec.parsed_endpoints else 0
        
        return SpecResponse(
            id=api_spec.id,
            name=api_spec.name,
            version=api_spec.version,
            description=api_spec.description,
            spec_type=api_spec.spec_type,
            base_url=api_spec.base_url,
            created_at=api_spec.created_at.isoformat(),
            updated_at=api_spec.updated_at.isoformat(),
            is_active=api_spec.is_active,
            endpoint_count=endpoint_count
        )
        
    except Exception as e:
        logger.error(f"Failed to upload specification file: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to upload specification file: {str(e)}"
        )

@router.get("/specs", response_model=List[SpecResponse])
async def list_specs(
    active_only: bool = True,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List all API specifications.
    
    Returns a paginated list of API specifications.
    """
    try:
        query = db.query(APISpecification)
        
        if active_only:
            query = query.filter(APISpecification.is_active == True)
        
        specs = query.offset(offset).limit(limit).all()
        
        return [
            SpecResponse(
                id=spec.id,
                name=spec.name,
                version=spec.version,
                description=spec.description,
                spec_type=spec.spec_type,
                base_url=spec.base_url,
                created_at=spec.created_at.isoformat(),
                updated_at=spec.updated_at.isoformat(),
                is_active=spec.is_active,
                endpoint_count=len(spec.parsed_endpoints) if spec.parsed_endpoints else 0
            )
            for spec in specs
        ]
        
    except Exception as e:
        logger.error(f"Failed to list specifications: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list specifications: {str(e)}"
        )

@router.get("/specs/{spec_id}", response_model=SpecResponse)
async def get_spec(
    spec_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific API specification by ID.
    """
    try:
        spec = db.query(APISpecification).filter(APISpecification.id == spec_id).first()
        
        if not spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {spec_id} not found"
            )
        
        return SpecResponse(
            id=spec.id,
            name=spec.name,
            version=spec.version,
            description=spec.description,
            spec_type=spec.spec_type,
            base_url=spec.base_url,
            created_at=spec.created_at.isoformat(),
            updated_at=spec.updated_at.isoformat(),
            is_active=spec.is_active,
            endpoint_count=len(spec.parsed_endpoints) if spec.parsed_endpoints else 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get specification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get specification: {str(e)}"
        )

@router.get("/specs/{spec_id}/endpoints")
async def get_spec_endpoints(
    spec_id: int,
    db: Session = Depends(get_db)
):
    """
    Get endpoints for a specific API specification.
    """
    try:
        spec = db.query(APISpecification).filter(APISpecification.id == spec_id).first()
        
        if not spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {spec_id} not found"
            )
        
        return {
            "api_spec_id": spec.id,
            "api_name": spec.name,
            "endpoints": spec.parsed_endpoints or {},
            "total_endpoints": len(spec.parsed_endpoints) if spec.parsed_endpoints else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get specification endpoints: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get specification endpoints: {str(e)}"
        )

@router.delete("/specs/{spec_id}")
async def delete_spec(
    spec_id: int,
    db: Session = Depends(get_db)
):
    """
    Soft delete an API specification.
    
    Marks the specification as inactive rather than permanently deleting it.
    """
    try:
        ingester = APISpecIngester()
        success = ingester.delete_spec(spec_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {spec_id} not found"
            )
        
        return {"message": f"API specification {spec_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete specification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete specification: {str(e)}"
        )

@router.get("/specs/{spec_id}/summary")
async def get_spec_summary(
    spec_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a comprehensive summary of an API specification.
    
    Includes endpoints, test cases, execution history, and other metrics.
    """
    try:
        from ...database.models import TestCase, ExecutionSession
        
        spec = db.query(APISpecification).filter(APISpecification.id == spec_id).first()
        
        if not spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {spec_id} not found"
            )
        
        # Get additional statistics
        test_case_count = db.query(TestCase).filter(TestCase.api_spec_id == spec_id).count()
        active_test_cases = db.query(TestCase).filter(
            TestCase.api_spec_id == spec_id,
            TestCase.is_active == True
        ).count()
        execution_sessions = db.query(ExecutionSession).filter(
            ExecutionSession.api_spec_id == spec_id
        ).count()
        
        return {
            "specification": {
                "id": spec.id,
                "name": spec.name,
                "version": spec.version,
                "description": spec.description,
                "spec_type": spec.spec_type.value,
                "base_url": spec.base_url,
                "created_at": spec.created_at.isoformat(),
                "updated_at": spec.updated_at.isoformat(),
                "is_active": spec.is_active
            },
            "endpoints": {
                "total": len(spec.parsed_endpoints) if spec.parsed_endpoints else 0,
                "methods": list(set(
                    method.upper() 
                    for path_data in (spec.parsed_endpoints or {}).values()
                    for method in path_data.keys()
                ))
            },
            "test_cases": {
                "total": test_case_count,
                "active": active_test_cases
            },
            "execution_history": {
                "total_sessions": execution_sessions
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get specification summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get specification summary: {str(e)}"
        )
