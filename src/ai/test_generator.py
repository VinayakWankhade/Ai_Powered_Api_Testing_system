"""
AI-powered test case generator using LLM and RAG.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback

from ..database.models import (
    APISpecification, TestCase, TestType, AIGenerationLog
)
from ..database.connection import get_db_session
from .rag_system import RAGSystem
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TestGenerationError(Exception):
    """Custom exception for test generation errors."""
    pass

class AITestGenerator:
    """
    AI-powered test case generator that uses LLM with RAG for context-aware test generation.
    """

    def __init__(self):
        self.db = get_db_session()
        self.rag_system = RAGSystem()
        
        # Initialize OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise TestGenerationError("OPENAI_API_KEY environment variable is required")
        
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.chat_model = ChatOpenAI(
            model_name=self.model_name,
            temperature=0.7,
            max_tokens=2000
        )
        
        logger.info(f"AI Test Generator initialized with model: {self.model_name}")

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
        """
        Generate test cases for a specific API endpoint.
        
        Args:
            api_spec_id: API specification ID
            endpoint_path: API endpoint path
            method: HTTP method
            test_types: Types of tests to generate
            count: Number of test cases to generate
            include_edge_cases: Whether to include edge cases
            custom_context: Additional context for generation
            
        Returns:
            List of generated test cases
        """
        try:
            # Get API specification
            api_spec = self.db.query(APISpecification).filter(
                APISpecification.id == api_spec_id
            ).first()
            
            if not api_spec:
                raise TestGenerationError(f"API specification with ID {api_spec_id} not found")
            
            # Get endpoint information
            endpoint_info = self._extract_endpoint_info(api_spec, endpoint_path, method)
            
            # Retrieve relevant documentation using RAG
            relevant_docs = self.rag_system.retrieve_endpoint_examples(
                endpoint_path, method, api_spec_id
            )
            
            # Generate test cases
            test_cases = []
            test_types = test_types or [TestType.FUNCTIONAL, TestType.EDGE_CASE]
            
            for test_type in test_types:
                cases_for_type = await self._generate_tests_for_type(
                    api_spec=api_spec,
                    endpoint_path=endpoint_path,
                    method=method,
                    endpoint_info=endpoint_info,
                    test_type=test_type,
                    relevant_docs=relevant_docs,
                    count=count // len(test_types) + (1 if test_type == test_types[0] else 0),
                    custom_context=custom_context
                )
                test_cases.extend(cases_for_type)
            
            logger.info(f"Generated {len(test_cases)} test cases for {method} {endpoint_path}")
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to generate test cases: {str(e)}")
            raise TestGenerationError(f"Test generation failed: {str(e)}")

    async def _generate_tests_for_type(
        self,
        api_spec: APISpecification,
        endpoint_path: str,
        method: str,
        endpoint_info: Dict[str, Any],
        test_type: TestType,
        relevant_docs: List[Dict[str, Any]],
        count: int,
        custom_context: Optional[str] = None
    ) -> List[TestCase]:
        """Generate test cases for a specific test type."""
        
        # Prepare context for the LLM
        context = self._prepare_generation_context(
            api_spec, endpoint_info, relevant_docs, test_type, custom_context
        )
        
        # Create prompt based on test type
        prompt = self._create_prompt_for_test_type(
            test_type, endpoint_path, method, context, count
        )
        
        # Generate tests using LLM
        with get_openai_callback() as cb:
            response = await self._call_llm(prompt)
            
            # Log the generation request
            log_entry = AIGenerationLog(
                prompt_template=prompt.template if hasattr(prompt, 'template') else str(prompt),
                prompt_variables=context,
                final_prompt=str(prompt),
                ai_model=self.model_name,
                raw_response=response,
                generation_time_ms=cb.total_time * 1000 if hasattr(cb, 'total_time') else None,
                token_usage={
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens,
                    "total_cost": cb.total_cost
                } if cb else None,
                temperature=0.7
            )
        
        try:
            # Parse the LLM response
            parsed_tests = self._parse_llm_response(response)
            
            # Create TestCase objects
            test_cases = []
            for test_data in parsed_tests:
                test_case = self._create_test_case_object(
                    api_spec.id,
                    endpoint_path,
                    method,
                    test_type,
                    test_data,
                    context
                )
                
                if test_case:
                    test_cases.append(test_case)
            
            # Mark generation as successful
            log_entry.validation_passed = True
            log_entry.parsed_response = parsed_tests
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            log_entry.validation_passed = False
            log_entry.validation_errors = [str(e)]
            test_cases = []
        
        # Save generation log
        self.db.add(log_entry)
        self.db.commit()
        
        return test_cases

    def _prepare_generation_context(
        self,
        api_spec: APISpecification,
        endpoint_info: Dict[str, Any],
        relevant_docs: List[Dict[str, Any]],
        test_type: TestType,
        custom_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Prepare context for test generation."""
        
        context = {
            "api_name": api_spec.name,
            "api_version": api_spec.version,
            "api_description": api_spec.description,
            "base_url": api_spec.base_url,
            "endpoint_info": endpoint_info,
            "test_type": test_type.value,
            "relevant_examples": [doc["content"] for doc in relevant_docs[:3]],
            "custom_context": custom_context or ""
        }
        
        return context

    def _create_prompt_for_test_type(
        self,
        test_type: TestType,
        endpoint_path: str,
        method: str,
        context: Dict[str, Any],
        count: int
    ) -> ChatPromptTemplate:
        """Create a prompt template for the specific test type."""
        
        base_system_message = """You are an expert API testing engineer. Your task is to generate comprehensive test cases for API endpoints. 

Generate test cases as JSON objects with the following structure:
{
    "name": "descriptive test name",
    "description": "what the test validates",
    "test_data": {
        "headers": {},
        "query_params": {},
        "body": {},
        "path_params": {}
    },
    "expected_response": {
        "status_code": 200,
        "body_schema": {},
        "headers": {}
    },
    "assertions": [
        "response.status_code == 200",
        "response.json()['field'] == 'expected_value'"
    ]
}

API Context:
- API Name: {api_name}
- Version: {api_version}
- Description: {api_description}
- Base URL: {base_url}

Endpoint Information:
{endpoint_info}

Relevant Examples:
{relevant_examples}

Additional Context:
{custom_context}"""

        type_specific_prompts = {
            TestType.FUNCTIONAL: """
Generate {count} functional test cases for {method} {endpoint_path}.
Focus on:
- Valid request scenarios
- Different input combinations
- Success path validation
- Response structure verification
""",
            TestType.EDGE_CASE: """
Generate {count} edge case test cases for {method} {endpoint_path}.
Focus on:
- Boundary value testing
- Invalid input scenarios
- Missing required parameters
- Malformed requests
- Unusual but valid inputs
""",
            TestType.SECURITY: """
Generate {count} security test cases for {method} {endpoint_path}.
Focus on:
- Authentication bypass attempts
- Authorization testing
- Input injection (SQL, XSS, etc.)
- Rate limiting
- Data exposure risks
""",
            TestType.PERFORMANCE: """
Generate {count} performance test cases for {method} {endpoint_path}.
Focus on:
- Load testing scenarios
- Response time validation
- Resource usage monitoring
- Concurrent request handling
"""
        }
        
        human_message = type_specific_prompts.get(test_type, type_specific_prompts[TestType.FUNCTIONAL])
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=base_system_message),
            HumanMessage(content=human_message)
        ]).partial(**context, count=count, method=method, endpoint_path=endpoint_path)

    async def _call_llm(self, prompt: ChatPromptTemplate) -> str:
        """Call the LLM with the prepared prompt."""
        try:
            messages = await prompt.aformat_messages()
            response = await self.chat_model.agenerate([messages])
            return response.generations[0][0].text
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise TestGenerationError(f"LLM call failed: {str(e)}")

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the LLM response to extract test case data."""
        try:
            # Try to parse as JSON directly
            if response.strip().startswith('['):
                return json.loads(response)
            
            # Look for JSON blocks in the response
            import re
            json_blocks = re.findall(r'\{[^{}]*\}|\[[^\[\]]*\]', response, re.DOTALL)
            
            parsed_tests = []
            for block in json_blocks:
                try:
                    test_data = json.loads(block)
                    if isinstance(test_data, list):
                        parsed_tests.extend(test_data)
                    else:
                        parsed_tests.append(test_data)
                except json.JSONDecodeError:
                    continue
            
            if not parsed_tests:
                # Fallback: try to extract key information manually
                parsed_tests = self._manual_parse_response(response)
            
            return parsed_tests
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return []

    def _manual_parse_response(self, response: str) -> List[Dict[str, Any]]:
        """Manually parse response when JSON parsing fails."""
        # This is a simplified fallback parser
        # In a production system, you'd want more robust parsing
        tests = []
        
        lines = response.split('\n')
        current_test = {}
        
        for line in lines:
            line = line.strip()
            
            if 'name:' in line.lower():
                if current_test:
                    tests.append(current_test)
                current_test = {"name": line.split(':', 1)[1].strip()}
            elif 'description:' in line.lower() and current_test:
                current_test["description"] = line.split(':', 1)[1].strip()
            elif line and not current_test.get("test_data"):
                current_test["test_data"] = {"headers": {}, "query_params": {}, "body": {}}
                current_test["expected_response"] = {"status_code": 200}
                current_test["assertions"] = []
        
        if current_test:
            tests.append(current_test)
        
        return tests

    def _create_test_case_object(
        self,
        api_spec_id: int,
        endpoint_path: str,
        method: str,
        test_type: TestType,
        test_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[TestCase]:
        """Create a TestCase object from parsed test data."""
        try:
            test_case = TestCase(
                api_spec_id=api_spec_id,
                name=test_data.get("name", f"{test_type.value}_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                description=test_data.get("description", ""),
                test_type=test_type,
                endpoint=endpoint_path,
                method=method.upper(),
                test_data=test_data.get("test_data", {}),
                expected_response=test_data.get("expected_response", {}),
                assertions=test_data.get("assertions", []),
                generated_by_llm=True,
                generation_context=context
            )
            
            self.db.add(test_case)
            self.db.commit()
            self.db.refresh(test_case)
            
            return test_case
            
        except Exception as e:
            logger.error(f"Failed to create test case object: {str(e)}")
            self.db.rollback()
            return None

    def _extract_endpoint_info(
        self,
        api_spec: APISpecification,
        endpoint_path: str,
        method: str
    ) -> Dict[str, Any]:
        """Extract endpoint information from API specification."""
        
        try:
            parsed_endpoints = api_spec.parsed_endpoints or {}
            endpoint_info = parsed_endpoints.get(endpoint_path, {})
            method_info = endpoint_info.get(method.upper(), {})
            
            return {
                "path": endpoint_path,
                "method": method.upper(),
                "summary": method_info.get("summary", ""),
                "description": method_info.get("description", ""),
                "parameters": method_info.get("parameters", []),
                "responses": method_info.get("responses", {}),
                "requestBody": method_info.get("requestBody", {}),
                "security": method_info.get("security", []),
                "tags": method_info.get("tags", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to extract endpoint info: {str(e)}")
            return {
                "path": endpoint_path,
                "method": method.upper(),
                "summary": "",
                "description": "",
                "parameters": [],
                "responses": {},
                "requestBody": {},
                "security": [],
                "tags": []
            }

    async def generate_test_suite(
        self,
        api_spec_id: int,
        include_all_endpoints: bool = True,
        endpoint_filter: Optional[List[str]] = None,
        test_types: Optional[List[TestType]] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete test suite for an API specification.
        
        Args:
            api_spec_id: API specification ID
            include_all_endpoints: Whether to include all endpoints
            endpoint_filter: Specific endpoints to include
            test_types: Types of tests to generate
            
        Returns:
            Dictionary with generation results
        """
        try:
            api_spec = self.db.query(APISpecification).filter(
                APISpecification.id == api_spec_id
            ).first()
            
            if not api_spec:
                raise TestGenerationError(f"API specification with ID {api_spec_id} not found")
            
            # Get endpoints to process
            endpoints_to_process = []
            parsed_endpoints = api_spec.parsed_endpoints or {}
            
            for path, path_info in parsed_endpoints.items():
                if endpoint_filter and path not in endpoint_filter:
                    continue
                
                for method in path_info.keys():
                    if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                        endpoints_to_process.append((path, method.upper()))
            
            # Generate test cases for each endpoint
            all_test_cases = []
            generation_stats = {
                "total_endpoints": len(endpoints_to_process),
                "successful_generations": 0,
                "failed_generations": 0,
                "total_test_cases": 0
            }
            
            for path, method in endpoints_to_process:
                try:
                    test_cases = await self.generate_test_cases(
                        api_spec_id=api_spec_id,
                        endpoint_path=path,
                        method=method,
                        test_types=test_types,
                        count=3
                    )
                    
                    all_test_cases.extend(test_cases)
                    generation_stats["successful_generations"] += 1
                    generation_stats["total_test_cases"] += len(test_cases)
                    
                except Exception as e:
                    logger.error(f"Failed to generate tests for {method} {path}: {str(e)}")
                    generation_stats["failed_generations"] += 1
            
            return {
                "test_cases": all_test_cases,
                "statistics": generation_stats,
                "api_spec_id": api_spec_id
            }
            
        except Exception as e:
            logger.error(f"Failed to generate test suite: {str(e)}")
            raise TestGenerationError(f"Test suite generation failed: {str(e)}")

    def get_generation_history(
        self,
        api_spec_id: Optional[int] = None,
        limit: int = 50
    ) -> List[AIGenerationLog]:
        """Get AI generation history."""
        try:
            query = self.db.query(AIGenerationLog)
            if api_spec_id:
                # Join with TestCase to filter by API spec
                query = query.join(TestCase).filter(TestCase.api_spec_id == api_spec_id)
            
            return query.order_by(AIGenerationLog.created_at.desc()).limit(limit).all()
            
        except Exception as e:
            logger.error(f"Failed to get generation history: {str(e)}")
            return []

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'db'):
            self.db.close()
