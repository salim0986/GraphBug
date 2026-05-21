"""
Code Analyzer - Advanced code analysis for PR reviews
Combines AST parsing, graph queries, and vector search for comprehensive analysis
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from tree_sitter import Node
from .graph_builder import GraphBuilder
from .vector_builder import VectorBuilder
from .parser import UniversalParser
from .logger import setup_logger
import re

logger = setup_logger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class FileChange(BaseModel):
    """Represents a changed file in a PR"""
    filename: str
    status: str  # 'added', 'removed', 'modified', 'renamed'
    additions: int
    deletions: int
    patch: Optional[str] = None
    language: Optional[str] = None


class PRAnalysisRequest(BaseModel):
    """Request for analyzing an entire PR"""
    repo_id: str
    pr_number: int
    files: List[FileChange]
    base_ref: str
    head_ref: str


class FileAnalysisRequest(BaseModel):
    """Request for analyzing a single file"""
    repo_id: str
    filename: str
    content: str
    language: str


class DiffAnalysisRequest(BaseModel):
    """Request for analyzing a diff/patch"""
    repo_id: str
    filename: str
    patch: str
    language: str


class CodeIssue(BaseModel):
    """A single code issue found during analysis"""
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    category: str  # 'security', 'performance', 'bug', 'code_quality', etc.
    title: str
    description: str
    line_number: Optional[int] = None
    line_range: Optional[Tuple[int, int]] = None
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


class SimilarCode(BaseModel):
    """Similar code found via vector search"""
    file: str  # Changed from filename for consistency
    name: str  # Changed from function_name
    similarity_score: float
    code_snippet: Optional[str] = None
    line: int  # Changed from line_number
    reason: Optional[str] = None  # Add reason field


class RelatedCode(BaseModel):
    """Related code found via graph traversal"""
    file: str  # Changed from filename for consistency
    type: str  # Changed from node_type
    name: str
    reason: str  # Changed from relationship for better clarity
    line: Optional[int] = None  # Changed from line_number, made optional


class FileAnalysisResult(BaseModel):
    """Result of analyzing a single file"""
    filename: str
    language: str
    issues: List[CodeIssue]
    similar_code: List[SimilarCode]
    related_code: List[RelatedCode]
    metrics: Dict[str, Any]


class PRAnalysisResult(BaseModel):
    """Complete PR analysis result"""
    pr_number: int
    files_analyzed: int
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues_by_category: Dict[str, int]
    file_results: List[FileAnalysisResult]
    overall_metrics: Dict[str, Any]


# ============================================================================
# CODE ANALYZER CLASS
# ============================================================================

class CodeAnalyzer:
    """
    Main code analysis engine
    Combines multiple analysis techniques for comprehensive code review
    """
    
    def __init__(
        self,
        graph_db: GraphBuilder,
        vector_db: VectorBuilder,
        parser: UniversalParser
    ):
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.parser = parser
        
    # ------------------------------------------------------------------------
    # PR ANALYSIS
    # ------------------------------------------------------------------------
    
    async def analyze_pr(self, request: PRAnalysisRequest) -> PRAnalysisResult:
        """
        Analyze an entire PR
        Returns comprehensive analysis with issues, patterns, and metrics
        """
        logger.info(f"Analyzing PR #{request.pr_number} for repo {request.repo_id}")
        
        file_results: List[FileAnalysisResult] = []
        total_issues = 0
        issues_by_severity: Dict[str, int] = {}
        issues_by_category: Dict[str, int] = {}
        
        # Analyze each changed file
        for file_change in request.files:
            # Skip removed files
            if file_change.status == "removed":
                continue
                
            # Skip non-reviewable files
            if not self._is_reviewable_file(file_change.filename):
                continue
            
            try:
                # Analyze the file
                if file_change.patch:
                    result = await self.analyze_diff(DiffAnalysisRequest(
                        repo_id=request.repo_id,
                        filename=file_change.filename,
                        patch=file_change.patch,
                        language=file_change.language or "unknown"
                    ))
                else:
                    # If no patch, do basic file analysis
                    result = FileAnalysisResult(
                        filename=file_change.filename,
                        language=file_change.language or "unknown",
                        issues=[],
                        similar_code=[],
                        related_code=[],
                        metrics={}
                    )
                
                file_results.append(result)
                total_issues += len(result.issues)
                
                # Aggregate by severity and category
                for issue in result.issues:
                    issues_by_severity[issue.severity] = issues_by_severity.get(issue.severity, 0) + 1
                    issues_by_category[issue.category] = issues_by_category.get(issue.category, 0) + 1
                    
            except Exception as e:
                logger.error(f"Error analyzing {file_change.filename}: {e}")
                continue
        
        # Calculate overall metrics
        overall_metrics = {
            "total_files": len(request.files),
            "files_analyzed": len(file_results),
            "total_additions": sum(f.additions for f in request.files),
            "total_deletions": sum(f.deletions for f in request.files),
        }
        
        # Add impact analysis across the PR
        try:
            impact_analysis = self._analyze_pr_impact(request.repo_id, request.files)
            overall_metrics.update(impact_analysis)
        except Exception as e:
            logger.error(f"Error calculating PR impact: {e}")
        
        return PRAnalysisResult(
            pr_number=request.pr_number,
            files_analyzed=len(file_results),
            total_issues=total_issues,
            issues_by_severity=issues_by_severity,
            issues_by_category=issues_by_category,
            file_results=file_results,
            overall_metrics=overall_metrics
        )
    
    # ------------------------------------------------------------------------
    # FILE ANALYSIS
    # ------------------------------------------------------------------------
    
    async def analyze_file(self, request: FileAnalysisRequest) -> FileAnalysisResult:
        """
        Analyze a complete file
        Returns issues, similar code, and related code
        """
        logger.info(f"Analyzing file {request.filename} for repo {request.repo_id}")
        
        issues: List[CodeIssue] = []
        similar_code: List[SimilarCode] = []
        related_code: List[RelatedCode] = []
        metrics: Dict[str, Any] = {}
        
        # 1. Detect code smells and anti-patterns
        issues.extend(self._detect_code_smells(request.content, request.language))
        
        # 2. Find similar code via vector search
        similar_code.extend(await self._find_similar_code(
            request.repo_id,
            request.filename,
            request.content
        ))
        
        # 3. Find related code via graph queries
        related_code.extend(await self._find_related_code(
            request.repo_id,
            request.filename
        ))
        
        # 4. Calculate metrics
        metrics = self._calculate_file_metrics(request.content, request.language)
        
        return FileAnalysisResult(
            filename=request.filename,
            language=request.language,
            issues=issues,
            similar_code=similar_code,
            related_code=related_code,
            metrics=metrics
        )
    
    # ------------------------------------------------------------------------
    # DIFF ANALYSIS
    # ------------------------------------------------------------------------
    
    async def analyze_diff(self, request: DiffAnalysisRequest) -> FileAnalysisResult:
        """
        Analyze a diff/patch
        Focus on changed lines and their context
        """
        logger.info(f"Analyzing diff for {request.filename} in repo {request.repo_id}")
        
        issues: List[CodeIssue] = []
        similar_code: List[SimilarCode] = []
        related_code: List[RelatedCode] = []
        metrics: Dict[str, Any] = {}
        
        # Parse the patch to extract changed lines
        changed_lines = self._parse_patch(request.patch)
        
        # Extract code from added lines
        added_code = "\n".join([line['content'] for line in changed_lines if line['type'] == 'add'])
        
        if added_code:
            # 1. Detect issues in added code
            issues.extend(self._detect_code_smells(added_code, request.language))
            
            # 2. Find similar code
            similar_code.extend(await self._find_similar_code(
                request.repo_id,
                request.filename,
                added_code
            ))
        
        # 3. Find related code for the entire file
        related_code.extend(await self._find_related_code(
            request.repo_id,
            request.filename
        ))
        
        # 4. Calculate metrics for the diff
        metrics = {
            "lines_added": len([l for l in changed_lines if l['type'] == 'add']),
            "lines_removed": len([l for l in changed_lines if l['type'] == 'delete']),
            "hunks": len(self._extract_hunks(request.patch))
        }
        
        return FileAnalysisResult(
            filename=request.filename,
            language=request.language,
            issues=issues,
            similar_code=similar_code,
            related_code=related_code,
            metrics=metrics
        )
    
    # ------------------------------------------------------------------------
    # PRIVATE HELPER METHODS
    # ------------------------------------------------------------------------
    
    def _is_reviewable_file(self, filename: str) -> bool:
        """Check if file should be reviewed"""
        # Skip lock files
        if any(lock in filename for lock in ['package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'Gemfile.lock']):
            return False
        
        # Skip build artifacts
        if any(pattern in filename for pattern in ['dist/', 'build/', '.next/', 'target/', 'bin/', 'obj/']):
            return False
        
        # Skip generated files
        if any(pattern in filename for pattern in ['.generated.', '.min.js', '.bundle.js']):
            return False
        
        return True
    
    def _detect_code_smells(self, code: str, language: str) -> List[CodeIssue]:
        """
        Detect code smells and anti-patterns using both regex and AST analysis
        Enhanced with cyclomatic complexity and language-specific patterns
        """
        issues: List[CodeIssue] = []
        lines = code.split('\n')
        
        # 1. Basic metrics
        if len(lines) > 50:
            issues.append(CodeIssue(
                severity="medium",
                category="code_quality",
                title="Long function detected",
                description=f"This code block is {len(lines)} lines long. Consider breaking it into smaller functions.",
                suggestion="Refactor into smaller, focused functions for better maintainability."
            ))
        
        # 2. Deep nesting detection
        for i, line in enumerate(lines, 1):
            indent_level = (len(line) - len(line.lstrip())) // 4
            if indent_level > 4:
                issues.append(CodeIssue(
                    severity="low",
                    category="code_quality",
                    title="Deep nesting detected",
                    description=f"Line {i} has {indent_level} levels of indentation.",
                    line_number=i,
                    suggestion="Consider extracting nested logic into separate functions."
                ))
                break
        
        # 3. Cyclomatic complexity (AST-based)
        try:
            complexity = self._calculate_cyclomatic_complexity(code, language)
            if complexity > 10:
                issues.append(CodeIssue(
                    severity="high" if complexity > 20 else "medium",
                    category="code_quality",
                    title=f"High cyclomatic complexity: {complexity}",
                    description=f"This code has a cyclomatic complexity of {complexity}. High complexity makes code harder to test and maintain.",
                    suggestion="Refactor to reduce branching logic. Consider extracting conditional blocks into separate functions."
                ))
        except Exception as e:
            logger.debug(f"Could not calculate complexity: {e}")
        
        # 4. Enhanced security patterns
        security_patterns = [
            # Code injection
            (r'eval\s*\(', "Use of eval() is a security risk", "critical", "Never use eval(). Use JSON.parse() or safe alternatives."),
            (r'exec\s*\(', "Use of exec() is a security risk", "critical", "Avoid exec(). Use subprocess with proper input validation."),
            (r'__import__\s*\(', "Dynamic imports can be dangerous", "high", "Avoid dynamic imports with user input."),
            
            # Hardcoded secrets
            (r'password\s*=\s*["\'][^"\']["\']', "Hardcoded password detected", "critical", "Use environment variables or secret management systems."),
            (r'api[_-]?key\s*=\s*["\'][^"\']["\']', "Hardcoded API key detected", "critical", "Store API keys in environment variables or secure vaults."),
            (r'secret\s*=\s*["\'][^"\']["\']', "Hardcoded secret detected", "critical", "Use secure secret management."),
            (r'token\s*=\s*["\'][A-Za-z0-9]{20,}["\']', "Hardcoded token detected", "critical", "Store tokens securely."),
            
            # XSS vulnerabilities
            (r'\.innerHTML\s*=', "Direct innerHTML assignment can lead to XSS", "high", "Use textContent or DOMPurify for sanitization."),
            (r'dangerouslySetInnerHTML', "Dangerous HTML injection risk", "high", "Sanitize user input before rendering."),
            (r'document\.write\s*\(', "document.write is unsafe and deprecated", "high", "Use modern DOM manipulation methods."),
            
            # SQL injection
            (r'execute\s*\(\s*["\'].*%s.*["\']', "Potential SQL injection", "critical", "Use parameterized queries or ORM."),
            (r'query\s*\(\s*f["\']', "f-string in SQL query", "critical", "Use parameterized queries to prevent SQL injection."),
            (r'\+\s*request\.|\+\s*req\.', "String concatenation with request data", "high", "Validate and sanitize user input."),
            
            # Path traversal
            (r'open\s*\(.*\+.*\)', "Potential path traversal", "high", "Validate file paths and use os.path.join() safely."),
            (r'readFile\s*\(.*\+', "File operation with concatenation", "high", "Validate file paths to prevent traversal attacks."),
            
            # Insecure crypto
            (r'md5\s*\(', "MD5 is cryptographically broken", "high", "Use SHA-256 or bcrypt for hashing."),
            (r'sha1\s*\(', "SHA-1 is weak", "medium", "Use SHA-256 or stronger algorithms."),
            
            # Unsafe deserialization
            (r'pickle\.loads?\s*\(', "Pickle deserialization is unsafe", "high", "Use JSON or validate pickle sources."),
            (r'yaml\.load\s*\([^,)]*\)', "yaml.load without Loader is unsafe", "high", "Use yaml.safe_load() instead."),
        ]
        
        for pattern, description, severity, suggestion in security_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeIssue(
                        severity=severity,
                        category="security",
                        title="Security vulnerability detected",
                        description=description,
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion=suggestion
                    ))
        
        # 5. Performance anti-patterns
        performance_patterns = [
            (r'for\s+.*\s+in\s+range\s*\(\s*len\s*\(', "Use enumerate() instead of range(len())", "low", "for i, item in enumerate(items):"),
            (r'\[.*for.*in.*if.*\].*\[.*for.*in.*\]', "Nested list comprehensions", "low", "Consider breaking into multiple steps for readability."),
            (r'\.append\s*\(.*\)\s*$', "Repeated append in loop", "info", "Consider list comprehension for better performance."),
        ]
        
        for pattern, description, severity, suggestion in performance_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        severity=severity,
                        category="performance",
                        title="Performance concern",
                        description=description,
                        line_number=i,
                        suggestion=suggestion
                    ))
        
        # 6. Language-specific patterns
        issues.extend(self._detect_language_specific_issues(code, language, lines))
        
        # 7. Code quality patterns
        quality_patterns = [
            (r'except:\s*$', "Bare except clause", "medium", "Catch specific exceptions instead of using bare except."),
            (r'except\s+Exception:\s*pass', "Silent exception swallowing", "high", "Log exceptions or handle them properly."),
            (r'print\s*\(', "Debug print statement", "info", "Use proper logging instead of print statements."),
            (r'console\.log\s*\(', "Debug console.log", "info", "Remove debug logs before production."),
            (r'debugger;?', "Debugger statement", "medium", "Remove debugger statements before committing."),
            (r'var\s+\w+\s*=', "Using 'var' instead of 'let'/'const'", "low", "Use 'let' or 'const' for better scoping."),
            (r'==(?!=)', "Using '==' instead of '==='", "low", "Use strict equality '===' to avoid type coercion."),
        ]
        
        for pattern, description, severity, suggestion in quality_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        severity=severity,
                        category="code_quality",
                        title=description,
                        description=description,
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion=suggestion
                    ))
        
        # 8. TODO/FIXME comments
        for i, line in enumerate(lines, 1):
            if re.search(r'TODO|FIXME|HACK|XXX', line, re.IGNORECASE):
                issues.append(CodeIssue(
                    severity="info",
                    category="maintainability",
                    title="TODO comment found",
                    description="Code contains TODO/FIXME comment indicating incomplete work.",
                    line_number=i,
                    code_snippet=line.strip()
                ))
        
        return issues
    
    def _calculate_cyclomatic_complexity(self, code: str, language: str) -> int:
        """
        Calculate cyclomatic complexity using AST analysis
        Counts decision points: if, for, while, case, catch, &&, ||
        Base complexity = 1
        """
        complexity = 1
        
        try:
            # Try to parse with tree-sitter
            from tree_sitter_languages import get_language, get_parser
            
            # Map our language names to tree-sitter language names
            lang_map = {
                'javascript': 'javascript',
                'typescript': 'typescript',
                'python': 'python',
                'java': 'java',
                'go': 'go',
                'rust': 'rust',
                'cpp': 'cpp',
                'c': 'c',
                'ruby': 'ruby',
                'php': 'php'
            }
            
            tree_sitter_lang = lang_map.get(language.lower())
            if not tree_sitter_lang:
                # Fallback to regex-based counting
                return self._calculate_complexity_regex(code, language)
            
            parser = get_parser(tree_sitter_lang)
            tree = parser.parse(bytes(code, 'utf8'))
            
            # Count decision points in AST
            complexity += self._count_decision_points(tree.root_node)
            
        except Exception as e:
            logger.debug(f"AST parsing failed, using regex fallback: {e}")
            complexity = self._calculate_complexity_regex(code, language)
        
        return complexity
    
    def _count_decision_points(self, node) -> int:
        """Recursively count decision points in AST"""
        count = 0
        
        # Decision point node types across languages
        decision_nodes = {
            'if_statement', 'elif_clause', 'else_clause',
            'for_statement', 'while_statement', 'do_statement',
            'case_statement', 'switch_statement',
            'catch_clause', 'except_clause',
            'conditional_expression', 'ternary_expression',
            'boolean_operator', 'binary_expression'
        }
        
        if node.type in decision_nodes:
            count += 1
        
        # Check for boolean operators (&&, ||)
        if node.type == 'binary_expression':
            if node.text and (b'&&' in node.text or b'||' in node.text or b'and' in node.text or b'or' in node.text):
                count += 1
        
        # Recurse through children
        for child in node.children:
            count += self._count_decision_points(child)
        
        return count
    
    def _calculate_complexity_regex(self, code: str, language: str) -> int:
        """Fallback complexity calculation using regex"""
        complexity = 1
        
        # Common patterns across languages
        patterns = [
            r'\bif\b', r'\belif\b', r'\belse\s+if\b',
            r'\bfor\b', r'\bwhile\b', r'\bdo\b',
            r'\bcase\b', r'\bswitch\b',
            r'\bcatch\b', r'\bexcept\b',
            r'\?.*:', r'&&', r'\|\|',
            r'\band\b', r'\bor\b'
        ]
        
        for pattern in patterns:
            complexity += len(re.findall(pattern, code, re.IGNORECASE))
        
        return complexity
    
    def _detect_language_specific_issues(self, code: str, language: str, lines: List[str]) -> List[CodeIssue]:
        """
        Detect language-specific code smells and anti-patterns
        """
        issues = []
        
        if language.lower() in ['python', 'py']:
            issues.extend(self._detect_python_issues(lines))
        elif language.lower() in ['javascript', 'typescript', 'js', 'ts']:
            issues.extend(self._detect_javascript_issues(lines))
        elif language.lower() in ['java']:
            issues.extend(self._detect_java_issues(lines))
        elif language.lower() in ['go']:
            issues.extend(self._detect_go_issues(lines))
        
        return issues
    
    def _detect_python_issues(self, lines: List[str]) -> List[CodeIssue]:
        """Python-specific code smell detection"""
        issues = []
        
        patterns = [
            (r'from\s+\*\s+import', "Wildcard import", "low", "Import specific names instead of using wildcard imports."),
            (r'\btype\s*\(\s*\w+\s*\)\s*==', "Using type() for type checking", "low", "Use isinstance() instead of type() for type checking."),
            (r'len\s*\(\s*\w+\s*\)\s*==\s*0', "Using len() to check empty", "info", "Use 'if not my_list:' instead of 'if len(my_list) == 0'."),
            (r'\.has_key\s*\(', "Deprecated has_key()", "medium", "Use 'key in dict' instead of dict.has_key(key)."),
            (r'except.*,', "Old-style exception syntax", "medium", "Use 'except Exception as e:' instead of 'except Exception, e:'."),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description, severity, suggestion in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        severity=severity,
                        category="code_quality",
                        title=f"Python: {description}",
                        description=description,
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion=suggestion
                    ))
        
        return issues
    
    def _detect_javascript_issues(self, lines: List[str]) -> List[CodeIssue]:
        """JavaScript/TypeScript-specific code smell detection"""
        issues = []
        
        patterns = [
            (r'new\s+Array\s*\(\)', "Use array literal []", "info", "Use [] instead of new Array()."),
            (r'new\s+Object\s*\(\)', "Use object literal {}", "info", "Use {} instead of new Object()."),
            (r'\.innerHTML\s*\+=', "Inefficient innerHTML concatenation", "medium", "Use DocumentFragment or insertAdjacentHTML."),
            (r'for\s*\(\s*var\s+\w+\s+in\s+', "for-in loop on array", "medium", "Use for-of or forEach for arrays."),
            (r'setTimeout\s*\([^,]*,\s*0\s*\)', "setTimeout(..., 0) hack", "low", "Consider using Promise.resolve().then() or proper async handling."),
            (r'\$\(.*\)\.\w+\s*\([^)]*\)\.\w+\s*\([^)]*\)\.\w+', "jQuery chain too long", "low", "Long jQuery chains are hard to debug."),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description, severity, suggestion in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        severity=severity,
                        category="code_quality",
                        title=f"JavaScript: {description}",
                        description=description,
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion=suggestion
                    ))
        
        return issues
    
    def _detect_java_issues(self, lines: List[str]) -> List[CodeIssue]:
        """Java-specific code smell detection"""
        issues = []
        
        patterns = [
            (r'System\.out\.print', "Using System.out.print", "info", "Use proper logging framework instead of System.out."),
            (r'\.printStackTrace\s*\(\)', "Printing stack trace", "medium", "Use proper logging instead of printStackTrace()."),
            (r'new\s+String\s*\(', "Unnecessary String construction", "low", "String literals are automatically interned."),
            (r'\+\s*""\s*\+', "String concatenation with empty string", "low", "Use String.valueOf() for type conversion."),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description, severity, suggestion in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        severity=severity,
                        category="code_quality",
                        title=f"Java: {description}",
                        description=description,
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion=suggestion
                    ))
        
        return issues
    
    def _detect_go_issues(self, lines: List[str]) -> List[CodeIssue]:
        """Go-specific code smell detection"""
        issues = []
        
        patterns = [
            (r'panic\s*\(', "Using panic()", "high", "Return errors instead of using panic() for recoverable errors."),
            (r'_\s*=.*err', "Ignoring error", "critical", "Always handle errors properly in Go."),
            (r'defer.*Close\(\).*\n.*err\s*:=', "Defer after error check", "medium", "Check errors before deferring Close()."),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description, severity, suggestion in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        severity=severity,
                        category="code_quality",
                        title=f"Go: {description}",
                        description=description,
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion=suggestion
                    ))
        
        return issues
    
    async def _find_similar_code(
        self,
        repo_id: str,
        filename: str,
        code: str
    ) -> List[SimilarCode]:
        """
        Find similar code using vector search
        Detects potential duplicates, similar patterns, and related implementations
        """
        similar: List[SimilarCode] = []
        
        try:
            # Find similar code snippets (70% similarity threshold)
            similar_results = self.vector_db.search_similar_code(
                repo_id=repo_id,
                code_snippet=code,
                limit=5,
                min_score=0.7
            )
            
            logger.info(f"[SIMILAR_CODE] Vector search returned {len(similar_results)} results for {filename}")
            
            for result in similar_results:
                # Skip if it's the same file and line (exact match)
                if result.get("file") == filename and abs(result.get("line", 0) - 1) < 5:
                    continue
                
                similar.append(SimilarCode(
                    file=result.get("file", "unknown"),
                    name=result.get("name", "unknown"),
                    similarity_score=result.get("similarity", 0.0),
                    code_snippet=result.get("code", "")[:200] if result.get("code") else None,
                    line=result.get("line", 0),
                    reason=self._get_similarity_reason(result.get("similarity", 0.0))
                ))
            
            # Also check for high-confidence duplicates (90%+ similarity)
            if code.strip():  # Only check non-empty code
                duplicates = self.vector_db.find_duplicate_code(
                    repo_id=repo_id,
                    code_snippet=code,
                    limit=3,
                    threshold=0.9
                )
                
                for dup in duplicates:
                    # Skip if already added or same location
                    if dup.get("file") == filename:
                        continue
                    
                    if not any(s.file == dup.get("file") and s.line == dup.get("line") for s in similar):
                        similar.append(SimilarCode(
                            file=dup.get("file", "unknown"),
                            name=dup.get("name", "unknown"),
                            similarity_score=dup.get("similarity", 0.0),
                            code_snippet=dup.get("code", "")[:200] if dup.get("code") else None,
                            line=dup.get("line", 0),
                            reason="Potential duplicate code - consider refactoring"
                        ))
            
            if similar:
                logger.info(f"[SIMILAR_CODE] Returning {len(similar)} similar code items for {filename}")
            else:
                logger.warning(f"[SIMILAR_CODE] No similar code found for {filename}")
            
        except Exception as e:
            logger.error(f"Error finding similar code: {e}")
        
        return similar[:10]  # Limit to top 10 results
    
    def _get_similarity_reason(self, score: float) -> str:
        """Generate human-readable reason for similarity"""
        if score >= 0.95:
            return "Nearly identical code - strong duplicate candidate"
        elif score >= 0.85:
            return "Very similar implementation - consider refactoring"
        elif score >= 0.75:
            return "Similar code pattern - review for consistency"
        else:
            return "Related implementation"
    
    def _calculate_cyclomatic_complexity(self, code: str, language: str) -> int:
        """
        Calculate cyclomatic complexity using AST analysis
        Counts decision points: if, for, while, case, catch, &&, ||
        Base complexity = 1
        """
        complexity = 1
        
        try:
            # Try to parse with tree-sitter
            from tree_sitter_languages import get_language, get_parser
            
            # Map our language names to tree-sitter language names
            lang_map = {
                'javascript': 'javascript',
                'typescript': 'typescript',
                'python': 'python',
                'java': 'java',
                'go': 'go',
                'rust': 'rust',
                'cpp': 'cpp',
                'c': 'c',
                'ruby': 'ruby',
                'php': 'php'
            }
            
            tree_sitter_lang = lang_map.get(language.lower())
            if not tree_sitter_lang:
                # Fallback to regex-based counting
                return self._calculate_complexity_regex(code, language)
            
            parser = get_parser(tree_sitter_lang)
            tree = parser.parse(bytes(code, 'utf8'))
            
            # Count decision points in AST
            complexity += self._count_decision_points(tree.root_node)
            
        except Exception as e:
            logger.debug(f"AST parsing failed, using regex fallback: {e}")
            complexity = self._calculate_complexity_regex(code, language)
        
        return complexity
    
    def _count_decision_points(self, node: Node) -> int:
        """Recursively count decision points in AST"""
        count = 0
        
        # Decision point node types across languages
        decision_nodes = {
            'if_statement', 'elif_clause', 'else_clause',
            'for_statement', 'while_statement', 'do_statement',
            'case_statement', 'switch_statement',
            'catch_clause', 'except_clause',
            'conditional_expression', 'ternary_expression',
            'boolean_operator', 'binary_expression'
        }
        
        if node.type in decision_nodes:
            count += 1
        
        # Check for boolean operators (&&, ||)
        if node.type == 'binary_expression':
            if node.text and (b'&&' in node.text or b'||' in node.text or b'and' in node.text or b'or' in node.text):
                count += 1
        
        # Recurse through children
        for child in node.children:
            count += self._count_decision_points(child)
        
        return count
    
    def _calculate_complexity_regex(self, code: str, language: str) -> int:
        """Fallback complexity calculation using regex"""
        complexity = 1
        
        # Common patterns across languages
        patterns = [
            r'\bif\b', r'\belif\b', r'\belse\s+if\b',
            r'\bfor\b', r'\bwhile\b', r'\bdo\b',
            r'\bcase\b', r'\bswitch\b',
            r'\bcatch\b', r'\bexcept\b',
            r'\?.*:', r'&&', r'\|\|',
            r'\band\b', r'\bor\b'
        ]
        
        for pattern in patterns:
            complexity += len(re.findall(pattern, code, re.IGNORECASE))
        
        return complexity
    
    def _detect_language_specific_issues(self, code: str, language: str, lines: List[str]) -> List[CodeIssue]:
        """
        Detect language-specific code smells and anti-patterns
        """
        issues = []
        
        if language.lower() in ['python', 'py']:
            issues.extend(self._detect_python_issues(lines))
        elif language.lower() in ['javascript', 'typescript', 'js', 'ts']:
            issues.extend(self._detect_javascript_issues(lines))
        elif language.lower() in ['java']:
            issues.extend(self._detect_java_issues(lines))
        elif language.lower() in ['go']:
            issues.extend(self._detect_go_issues(lines))
        
        return issues
    
    def _detect_python_issues(self, lines: List[str]) -> List[CodeIssue]:
        """Python-specific code smell detection"""
        issues = []
        
        patterns = [
            (r'from\s+\*\s+import', "Wildcard import", "low", "Import specific names instead of using wildcard imports."),
            (r'\btype\s*\(\s*\w+\s*\)\s*==', "Using type() for type checking", "low", "Use isinstance() instead of type() for type checking."),
            (r'len\s*\(\s*\w+\s*\)\s*==\s*0', "Using len() to check empty", "info", "Use 'if not my_list:' instead of 'if len(my_list) == 0'."),
            (r'\.has_key\s*\(', "Deprecated has_key()", "medium", "Use 'key in dict' instead of dict.has_key(key)."),
            (r'except.*,', "Old-style exception syntax", "medium", "Use 'except Exception as e:' instead of 'except Exception, e:'."),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description, severity, suggestion in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        severity=severity,
                        category="code_quality",
                        title=f"Python: {description}",
                        description=description,
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion=suggestion
                    ))
        
        return issues
    
    def _detect_javascript_issues(self, lines: List[str]) -> List[CodeIssue]:
        """JavaScript/TypeScript-specific code smell detection"""
        issues = []
        
        patterns = [
            (r'new\s+Array\s*\(\)', "Use array literal []", "info", "Use [] instead of new Array()."),
            (r'new\s+Object\s*\(\)', "Use object literal {}", "info", "Use {} instead of new Object()."),
            (r'\.innerHTML\s*\+=', "Inefficient innerHTML concatenation", "medium", "Use DocumentFragment or insertAdjacentHTML."),
            (r'for\s*\(\s*var\s+\w+\s+in\s+', "for-in loop on array", "medium", "Use for-of or forEach for arrays."),
            (r'setTimeout\s*\([^,]*,\s*0\s*\)', "setTimeout(..., 0) hack", "low", "Consider using Promise.resolve().then() or proper async handling."),
            (r'\$\(.*\)\.\w+\s*\([^)]*\)\.\w+\s*\([^)]*\)\.\w+', "jQuery chain too long", "low", "Long jQuery chains are hard to debug."),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description, severity, suggestion in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        severity=severity,
                        category="code_quality",
                        title=f"JavaScript: {description}",
                        description=description,
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion=suggestion
                    ))
        
        return issues
    
    def _detect_java_issues(self, lines: List[str]) -> List[CodeIssue]:
        """Java-specific code smell detection"""
        issues = []
        
        patterns = [
            (r'System\.out\.print', "Using System.out.print", "info", "Use proper logging framework instead of System.out."),
            (r'\.printStackTrace\s*\(\)', "Printing stack trace", "medium", "Use proper logging instead of printStackTrace()."),
            (r'new\s+String\s*\(', "Unnecessary String construction", "low", "String literals are automatically interned."),
            (r'\+\s*""\s*\+', "String concatenation with empty string", "low", "Use String.valueOf() for type conversion."),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description, severity, suggestion in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        severity=severity,
                        category="code_quality",
                        title=f"Java: {description}",
                        description=description,
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion=suggestion
                    ))
        
        return issues
    
    def _detect_go_issues(self, lines: List[str]) -> List[CodeIssue]:
        """Go-specific code smell detection"""
        issues = []
        
        patterns = [
            (r'panic\s*\(', "Using panic()", "high", "Return errors instead of using panic() for recoverable errors."),
            (r'_\s*=.*err', "Ignoring error", "critical", "Always handle errors properly in Go."),
            (r'defer.*Close\(\).*\n.*err\s*:=', "Defer after error check", "medium", "Check errors before deferring Close()."),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description, severity, suggestion in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        severity=severity,
                        category="code_quality",
                        title=f"Go: {description}",
                        description=description,
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion=suggestion
                    ))
        
        return issues
    
    async def _find_related_code(
        self,
        repo_id: str,
        filename: str
    ) -> List[RelatedCode]:
        """
        Find related code using graph queries
        Includes: callers, callees, file dependencies, complexity hotspots
        """
        related: List[RelatedCode] = []
        
        try:
            # Find entities in the file
            file_entities = self.graph_db.find_related_by_file(repo_id, filename, limit=20)
            logger.info(f"[RELATED_CODE] Found {len(file_entities)} entities in {filename}")
            for entity in file_entities:
                related.append(RelatedCode(
                    type="file_entity",
                    file=filename,
                    name=entity["name"],
                    reason=f"{entity['type']} defined in this file",
                    line=entity["line"]
                ))
            
            # Extract function names from entities for deeper analysis
            function_names = [e["name"] for e in file_entities if e["type"] == "Function"]
            
            # For each function, find callers (impact analysis)
            for func_name in function_names[:5]:  # Limit to top 5 functions
                callers = self.graph_db.find_callers(repo_id, func_name, limit=5)
                for caller in callers:
                    related.append(RelatedCode(
                        type="caller",
                        file=caller.get("file", "unknown"),
                        name=caller.get("name", "unknown"),
                        reason=f"Calls function '{func_name}'",
                        line=caller.get("line")
                    ))
            
            # Find file-level dependencies
            file_deps = self.graph_db.find_file_dependencies(repo_id, filename)
            for dep in file_deps[:5]:  # Top 5 most coupled files
                related.append(RelatedCode(
                    type="file_dependency",
                    file=dep.get("file", "unknown"),
                    name=f"{dep.get('call_count', 0)} calls",
                    reason=f"File dependency ({dep.get('call_count', 0)} function calls)",
                    line=None
                ))
            
            # Check for complexity hotspots in this file
            complexity_issues = self.graph_db.get_complexity_hotspots(repo_id, min_calls=5, limit=10)
            for hotspot in complexity_issues:
                if hotspot.get("file") == filename:
                    related.append(RelatedCode(
                        type="complexity_hotspot",
                        file=hotspot.get("file", "unknown"),
                        name=hotspot.get("name", "unknown"),
                        reason=f"Complexity hotspot: {hotspot.get('call_count', 0)} outgoing calls",
                        line=hotspot.get("line")
                    ))
            
            logger.info(f"[RELATED_CODE] Returning {len(related)} related code items for {filename}")
            
        except Exception as e:
            logger.error(f"Error finding related code: {e}")
        
        return related[:30]  # Limit total results
    
    def _calculate_file_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate basic file metrics"""
        lines = code.split('\n')
        
        return {
            "total_lines": len(lines),
            "code_lines": len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
            "comment_lines": len([l for l in lines if l.strip().startswith('#')]),
            "blank_lines": len([l for l in lines if not l.strip()]),
            "language": language
        }
    
    def _parse_patch(self, patch: str) -> List[Dict[str, Any]]:
        """
        Parse unified diff patch
        Returns list of changed lines with metadata
        """
        changed_lines: List[Dict[str, Any]] = []
        
        for line in patch.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                changed_lines.append({
                    'type': 'add',
                    'content': line[1:],
                    'line': line
                })
            elif line.startswith('-') and not line.startswith('---'):
                changed_lines.append({
                    'type': 'delete',
                    'content': line[1:],
                    'line': line
                })
        
        return changed_lines
    
    def _extract_hunks(self, patch: str) -> List[str]:
        """Extract hunks from patch"""
        hunks = []
        current_hunk = []
        
        for line in patch.split('\n'):
            if line.startswith('@@'):
                if current_hunk:
                    hunks.append('\n'.join(current_hunk))
                current_hunk = [line]
            elif current_hunk:
                current_hunk.append(line)
        
        if current_hunk:
            hunks.append('\n'.join(current_hunk))
        
        return hunks
    
    def _analyze_pr_impact(self, repo_id: str, files: List[Any]) -> Dict[str, Any]:
        """
        Analyze the impact of PR across the codebase using graph queries
        Returns metrics about affected areas, coupling, and potential risks
        """
        impact_metrics = {
            "total_affected_callers": 0,
            "high_coupling_files": [],
            "complexity_hotspots": [],
            "unused_code_introduced": [],
            "cross_file_dependencies": 0
        }
        
        try:
            # Get list of changed files
            changed_files = [f.filename for f in files if hasattr(f, 'filename')]
            
            # Check for highly coupled files in the changeset
            coupled_files = self.graph_db.get_highly_coupled_files(repo_id, min_connections=5, limit=10)
            for coupling in coupled_files:
                if coupling["source_file"] in changed_files or coupling["target_file"] in changed_files:
                    impact_metrics["high_coupling_files"].append({
                        "source": coupling["source_file"],
                        "target": coupling["target_file"],
                        "connections": coupling["connections"]
                    })
            
            # Find complexity hotspots in changed files
            hotspots = self.graph_db.get_complexity_hotspots(repo_id, min_calls=5, limit=10)
            for hotspot in hotspots:
                if hotspot["file"] in changed_files:
                    impact_metrics["complexity_hotspots"].append({
                        "file": hotspot["file"],
                        "function": hotspot["name"],
                        "call_count": hotspot["call_count"]
                    })
            
            # Count total affected callers (impact radius)
            for file_path in changed_files[:10]:  # Limit to first 10 files
                entities = self.graph_db.find_related_by_file(repo_id, file_path, limit=20)
                for entity in entities:
                    if entity["type"] == "Function":
                        callers = self.graph_db.find_callers(repo_id, entity["name"], limit=100)
                        impact_metrics["total_affected_callers"] += len(callers)
            
            # Check for unused functions (potential dead code)
            unused = self.graph_db.find_unused_functions(repo_id, limit=20)
            for unused_func in unused:
                if unused_func["file"] in changed_files:
                    impact_metrics["unused_code_introduced"].append({
                        "file": unused_func["file"],
                        "function": unused_func["name"],
                        "line": unused_func["line"]
                    })
            
            # Calculate cross-file dependencies
            for file_path in changed_files:
                deps = self.graph_db.find_file_dependencies(repo_id, file_path)
                impact_metrics["cross_file_dependencies"] += len(deps)
            
            logger.info(f"PR Impact Analysis: {impact_metrics['total_affected_callers']} affected callers, "
                       f"{len(impact_metrics['high_coupling_files'])} coupled files, "
                       f"{len(impact_metrics['complexity_hotspots'])} hotspots")
            
        except Exception as e:
            logger.error(f"Error analyzing PR impact: {e}")
        
        return impact_metrics
