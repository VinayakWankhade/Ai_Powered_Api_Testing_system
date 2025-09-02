"""
AI-powered test case generation using OpenAI GPT models.
Simplified version for MVP without complex dependencies.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..database.connection import get_db_session
from ..database.models import (
    APISpecification, TestCase, TestType, 
    AIGenerationLog
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

class AITestGenerator:
    """AI-powered test case generator using OpenAI."""
    
    def __init__(self):
        self.db = get_db_session()
        self.openai_client = None
        
        # Initialize OpenAI client if available and API key is configured
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key != "your_openai_api_key_here":
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
            else:
                logger.warning("OpenAI API key not configured. Using template-based generation.")
        else:
            logger.warning("OpenAI library not available. Using template-based generation.")
    
    async def generate_test_cases(
        self,
        api_spec_id: int,
        endpoint_path: str,
        method: str,
        test_types: Optional[List[TestType]] = None,
        count: int = 3,
        include_edge_cases: bool = True,
        custom_context: Optional[str] = None
    ) -> List[TestCase]:
        """Generate test cases for a specific endpoint."""
        
        # Get API specification
        api_spec = self.db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise ValueError(f"API specification with ID {api_spec_id} not found")
        
        # Get endpoint information
        endpoint_info = self._get_endpoint_info(api_spec, endpoint_path, method)
        
        # Generate test cases
        test_cases = []
        test_types = test_types or [TestType.FUNCTIONAL, TestType.EDGE_CASE]
        
        for test_type in test_types:
            type_count = count // len(test_types)
            if test_type == test_types[-1]:  # Add remainder to last type
                type_count += count % len(test_types)
            
            if self.openai_client:
                # Use AI generation
                try:
                    ai_test_cases = await self._generate_with_openai(
                        api_spec, endpoint_path, method, test_type, 
                        type_count, endpoint_info, custom_context
                    )
                    test_cases.extend(ai_test_cases)
                except Exception as e:
                    logger.error(f"AI generation failed: {str(e)}, falling back to templates")
                    template_test_cases = self._generate_with_templates(
                        api_spec, endpoint_path, method, test_type, type_count, endpoint_info
                    )
                    test_cases.extend(template_test_cases)
            else:
                # Use template-based generation
                template_test_cases = self._generate_with_templates(
                    api_spec, endpoint_path, method, test_type, type_count, endpoint_info
                )
                test_cases.extend(template_test_cases)
        
        # Save test cases to database
        saved_test_cases = []
        for tc_data in test_cases:
            test_case = TestCase(
                api_spec_id=api_spec_id,
                name=tc_data["name"],
                description=tc_data["description"],
                test_type=tc_data["test_type"],
                endpoint=endpoint_path,
                method=method.upper(),
                test_data=tc_data["test_data"],
                expected_response=tc_data.get("expected_response"),
                assertions=tc_data.get("assertions", []),
                generated_by_llm=tc_data.get("generated_by_llm", False),
                generation_prompt=tc_data.get("generation_prompt"),
                generation_context=tc_data.get("generation_context")
            )
            
            self.db.add(test_case)
            self.db.flush()  # Get the ID
            saved_test_cases.append(test_case)
        
        self.db.commit()
        logger.info(f"Generated and saved {len(saved_test_cases)} test cases for {method} {endpoint_path}")
        return saved_test_cases
    
    async def generate_test_suite(
        self,
        api_spec_id: int,
        include_all_endpoints: bool = True,
        endpoint_filter: Optional[List[str]] = None,
        test_types: Optional[List[TestType]] = None
    ) -> Dict[str, Any]:
        """Generate a complete test suite for an API specification."""
        
        api_spec = self.db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise ValueError(f"API specification with ID {api_spec_id} not found")
        
        endpoints = api_spec.parsed_endpoints or {}
        all_test_cases = []
        
        for endpoint_path, methods in endpoints.items():
            if not include_all_endpoints and endpoint_filter:
                if endpoint_path not in endpoint_filter:
                    continue
            
            for method in methods.keys():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    try:
                        test_cases = await self.generate_test_cases(
                            api_spec_id=api_spec_id,
                            endpoint_path=endpoint_path,
                            method=method,
                            test_types=test_types,
                            count=3,
                            include_edge_cases=True
                        )
                        all_test_cases.extend(test_cases)
                    except Exception as e:
                        logger.error(f"Failed to generate tests for {method} {endpoint_path}: {str(e)}")
        
        return {
            "test_cases": all_test_cases,
            "statistics": {
                "total_test_cases": len(all_test_cases),
                "endpoints_covered": len(endpoints),
                "test_types_generated": list(set(tc.test_type for tc in all_test_cases))
            }
        }
    
    async def _generate_with_openai(
        self,
        api_spec: APISpecification,
        endpoint_path: str,
        method: str,
        test_type: TestType,
        count: int,
        endpoint_info: Dict[str, Any],
        custom_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate test cases using OpenAI."""
        
        prompt = self._build_generation_prompt(
            api_spec, endpoint_path, method, test_type, count, endpoint_info, custom_context
        )
        
        start_time = datetime.utcnow()
        
        response = self.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "You are an expert API testing engineer. Generate comprehensive test cases in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Parse the response
        content = response.choices[0].message.content
        test_cases_data = self._parse_openai_response(content, test_type)
        
        # Log the generation
        self._log_ai_generation(
            prompt, content, test_cases_data, generation_time, 
            response.usage._asdict() if response.usage else None
        )
        
        return test_cases_data[:count]  # Limit to requested count
    
    def _generate_with_templates(
        self,
        api_spec: APISpecification,
        endpoint_path: str,
        method: str,
        test_type: TestType,
        count: int,
        endpoint_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate test cases using templates (fallback)."""
        
        test_cases = []
        
        for i in range(count):
            if test_type == TestType.FUNCTIONAL:
                test_case = {
                    "name": f"Test {method} {endpoint_path} - Functional {i+1}",
                    "description": f"Functional test for {method} {endpoint_path}",
                    "test_type": test_type,
                    "test_data": self._generate_functional_test_data(endpoint_info, method),
                    "expected_response": {"status_code": 200},
                    "assertions": [
                        {"type": "status_code", "expected": 200},
                        {"type": "response_time", "max_ms": 5000}
                    ],
                    "generated_by_llm": False
                }
            elif test_type == TestType.EDGE_CASE:
                test_case = {
                    "name": f"Test {method} {endpoint_path} - Edge Case {i+1}",
                    "description": f"Edge case test for {method} {endpoint_path}",
                    "test_type": test_type,
                    "test_data": self._generate_edge_case_test_data(endpoint_info, method),
                    "expected_response": {"status_code": [400, 404, 422]},
                    "assertions": [
                        {"type": "status_code_in", "expected": [400, 404, 422]},
                        {"type": "response_time", "max_ms": 5000}
                    ],
                    "generated_by_llm": False
                }
            else:
                test_case = {
                    "name": f"Test {method} {endpoint_path} - {test_type.value.title()} {i+1}",
                    "description": f"{test_type.value.title()} test for {method} {endpoint_path}",
                    "test_type": test_type,
                    "test_data": self._generate_basic_test_data(endpoint_info, method),
                    "expected_response": {"status_code": 200},
                    "assertions": [{"type": "status_code", "expected": 200}],
                    "generated_by_llm": False
                }
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _build_generation_prompt(
        self,
        api_spec: APISpecification,
        endpoint_path: str,
        method: str,
        test_type: TestType,
        count: int,
        endpoint_info: Dict[str, Any],
        custom_context: Optional[str] = None
    ) -> str:
        """Build the prompt for OpenAI generation."""
        
        base_url = api_spec.base_url or "https://api.example.com"
        
        prompt = f"""
Generate {count} comprehensive {test_type.value} test cases for the following API endpoint:

API: {api_spec.name} (v{api_spec.version})
Base URL: {base_url}
Endpoint: {method.upper()} {endpoint_path}
Description: {api_spec.description or 'No description provided'}

Endpoint Details:
{json.dumps(endpoint_info, indent=2)}

Requirements:
- Generate {test_type.value} test cases
- Include realistic test data
- Specify expected responses
- Add appropriate assertions
- Consider authentication if needed
- Include edge cases for error scenarios

{f"Additional Context: {custom_context}" if custom_context else ""}

Return the test cases as a JSON array with this structure:
[
  {{
    "name": "Test case name",
    "description": "Detailed description",
    "test_data": {{
      "headers": {{}},
      "query_params": {{}},
      "path_params": {{}},
      "body": {{}}
    }},
    "expected_response": {{
      "status_code": 200,
      "body_contains": ["key", "value"]
    }},
    "assertions": [
      {{"type": "status_code", "expected": 200}},
      {{"type": "response_time", "max_ms": 5000}}
    ]
  }}
]
"""
        return prompt.strip()
    
    def _parse_openai_response(self, content: str, test_type: TestType) -> List[Dict[str, Any]]:
        """Parse OpenAI response into test case data."""
        
        try:
            # Try to extract JSON from the response
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                test_cases_raw = json.loads(json_str)
                
                # Process and validate test cases
                test_cases = []
                for tc in test_cases_raw:
                    processed_tc = {
                        "name": tc.get("name", f"AI Generated {test_type.value} Test"),
                        "description": tc.get("description", "AI generated test case"),
                        "test_type": test_type,
                        "test_data": tc.get("test_data", {}),
                        "expected_response": tc.get("expected_response", {}),
                        "assertions": tc.get("assertions", []),
                        "generated_by_llm": True,
                        "generation_context": {"model": "openai", "test_type": test_type.value}
                    }
                    test_cases.append(processed_tc)
                
                return test_cases
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse OpenAI response: {str(e)}")
        
        # Return empty list if parsing fails
        return []
    
    def _get_endpoint_info(self, api_spec: APISpecification, endpoint_path: str, method: str) -> Dict[str, Any]:
        """Extract endpoint information from API specification."""
        
        endpoints = api_spec.parsed_endpoints or {}
        path_info = endpoints.get(endpoint_path, {})
        method_info = path_info.get(method.lower(), {})
        
        return {
            "path": endpoint_path,
            "method": method.upper(),
            "summary": method_info.get("summary", ""),
            "description": method_info.get("description", ""),
            "parameters": method_info.get("parameters", []),
            "responses": method_info.get("responses", {}),
            "tags": method_info.get("tags", []),
            "security": method_info.get("security", [])
        }
    
    def _generate_functional_test_data(self, endpoint_info: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Generate functional test data."""
        
        test_data = {
            "headers": {"Content-Type": "application/json"},
            "query_params": {},
            "path_params": {},
            "body": {}
        }
        
        # Add basic parameters based on endpoint info
        for param in endpoint_info.get("parameters", []):
            param_name = param.get("name", "param")
            param_in = param.get("in", "query")
            
            if param_in == "query":
                test_data["query_params"][param_name] = "test_value"
            elif param_in == "path":
                test_data["path_params"][param_name] = "123"
            elif param_in == "header":
                test_data["headers"][param_name] = "test_value"
        
        # Add body for POST/PUT/PATCH requests
        if method.upper() in ["POST", "PUT", "PATCH"]:
            test_data["body"] = {"data": "test_value"}
        
        return test_data
    
    def _generate_edge_case_test_data(self, endpoint_info: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Generate edge case test data."""
        
        test_data = {
            "headers": {"Content-Type": "application/json"},
            "query_params": {},
            "path_params": {},
            "body": {}
        }
        
        # Add invalid/edge case parameters
        for param in endpoint_info.get("parameters", []):
            param_name = param.get("name", "param")
            param_in = param.get("in", "query")
            
            if param_in == "query":
                test_data["query_params"][param_name] = ""  # Empty value
            elif param_in == "path":
                test_data["path_params"][param_name] = "invalid_id"
            elif param_in == "header":
                test_data["headers"][param_name] = ""
        
        # Add invalid body for POST/PUT/PATCH requests
        if method.upper() in ["POST", "PUT", "PATCH"]:
            test_data["body"] = {}  # Empty body
        
        return test_data
    
    def _generate_basic_test_data(self, endpoint_info: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Generate basic test data."""
        return self._generate_functional_test_data(endpoint_info, method)
    
    def _log_ai_generation(
        self,
        prompt: str,
        response: str,
        parsed_response: List[Dict[str, Any]],
        generation_time_ms: float,
        token_usage: Optional[Dict[str, Any]] = None
    ):
        """Log AI generation for debugging and optimization."""
        
        try:
            log_entry = AIGenerationLog(
                prompt_template="openai_test_generation",
                final_prompt=prompt,
                ai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                raw_response=response,
                parsed_response=parsed_response,
                generation_time_ms=generation_time_ms,
                token_usage=token_usage,
                validation_passed=len(parsed_response) > 0
            )
            
            self.db.add(log_entry)
            self.db.commit()
            
        except Exception as e:
            logger.warning(f"Failed to log AI generation: {str(e)}")
    
    def get_generation_history(self, api_spec_id: Optional[int] = None, limit: int = 50) -> List[AIGenerationLog]:
        """Get AI generation history."""
        
        query = self.db.query(AIGenerationLog)
        
        if api_spec_id:
            # Filter by API spec through test cases
            query = query.join(TestCase).filter(TestCase.api_spec_id == api_spec_id)
        
        return query.order_by(AIGenerationLog.created_at.desc()).limit(limit).all()
    
    def __del__(self):
        """Clean up database session."""
        if hasattr(self, 'db'):
            self.db.close()
