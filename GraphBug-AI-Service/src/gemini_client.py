"""
Google Gemini API Client - Phase 4.3
Handles authentication, model selection, and API calls to Gemini
"""

from typing import Optional, List, Dict, Any, AsyncIterator
import os
import asyncio
from dataclasses import dataclass
from google import genai
from google.genai import types
from .logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GeminiConfig:
    """Configuration for Gemini API"""
    api_key: str
    
    # Model names
    flash_lite_model: str = "gemini-2.5-flash-lite"
    flash_model: str = "gemini-2.5-flash"
    pro_model: str = "gemini-2.5-pro"
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    
    # Rate limiting
    max_requests_per_minute: int = 60
    retry_attempts: int = 3
    retry_delay: float = 2.0
    
    # Safety settings
    enable_safety_filters: bool = True


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

class PromptTemplates:
    """Comprehensive prompt templates for code reviews"""
    
    SYSTEM_PROMPT = """You are an expert code reviewer with deep knowledge of software engineering best practices, security, performance, and maintainability.

**Your Task:**
Provide thorough, actionable code reviews with context-aware insights.

**Guidelines:**
- **Be context-aware**: Reference similar code patterns when suggesting improvements
- **Consider dependencies**: Highlight impact on related files and functions
- **Be constructive and specific** with actionable feedback
- **Prioritize**: Security vulnerabilities > Critical bugs > Performance > Quality
- **Suggest concrete improvements** with code examples
- **Identify refactoring opportunities** (consolidate duplicates, extract common patterns)
- **Format professionally** with markdown, severity badges, and clear sections

**Focus Areas:**
1. 🔒 Security vulnerabilities and dependency impact
2. 🐛 Logic errors and bugs
3. ⚡ Performance issues and optimization opportunities
4. 🏗️ Architecture and design patterns
5. 🔄 Code duplication and refactoring opportunities
6. 📝 Documentation and maintainability
7. ✅ Testing coverage recommendations
"""
    
    QUICK_REVIEW_PROMPT = """## Quick Scan: {pr_title}

**Scope:** {total_files} files | +{additions}/-{deletions}

**GraphRAG:** Entities: {entities} | Dependencies: {dependencies} | Similar: {similar_code}

**Files:** {files_summary}
**Issues:** {issues_summary}

---

## Find Problems Only

Identify actual mistakes:
🔴 Critical (security, crashes)
🟠 High (bugs, logic errors)
🟡 Medium (code smells)

Cite line numbers (L45+). Reference GraphRAG if data provided (not "None"). No generic advice.

**Output:**
### Issues
[List or "✅ None"]

### GraphRAG Matches  
[Cite 1-2 if available]
"""
    
    QUICK_SCAN_PROMPT = """## 🔎 PHASE 3: Quick Security & Critical Issue Scan

**Purpose:** Fast scan to identify CRITICAL issues and security vulnerabilities ONLY.

**File:** {filename}
**Language:** {language}
**Changes:** +{additions} -{deletions}

**Diff with Line Numbers:**
```
{diff}
```

## 🎯 YOUR TASK: Scan for CRITICAL Issues ONLY

**Focus Areas (in priority order):**

1. **🔒 Security Vulnerabilities**
   - SQL injection
   - XSS vulnerabilities
   - Authentication/authorization bypasses
   - Credential exposure
   - Insecure deserialization

2. **💥 Critical Bugs**
   - Null pointer/undefined access
   - Resource leaks (memory, connections, files)
   - Race conditions
   - Data corruption risks

3. **🚫 Immediate Blockers**
   - Breaking API changes without migration
   - Data loss scenarios
   - Production outage risks

## ⌛ SPEED REQUIREMENTS:
- This is a QUICK scan - complete in <5 seconds
- Skip non-critical issues (quality, style, minor optimizations)
- Only flag issues that are:
  - **Urgent** (must fix before merge)
  - **High impact** (security, data integrity, availability)
  - **Well-founded** (not hypothetical)

## 📝 OUTPUT FORMAT:

If critical issues found:
```
🔴 **CRITICAL ISSUES DETECTED**

🔴 **L<line>+/-**: <Brief issue title>
- **Problem:** <What's wrong with evidence from diff>
- **Impact:** <Why this is critical>
- **Fix:** <Quick suggestion>
```

If NO critical issues:
```
✅ **No critical issues detected in quick scan**

Proceed to detailed review for code quality, performance, and best practices.
```

**IMPORTANT:**
- Cite line numbers using L<num>+/- format from the diff
- Be specific - reference actual code from the diff
- Don't flag minor issues or style problems
- If unsure whether issue is critical, skip it (detailed review will catch it)
"""
    
    STANDARD_REVIEW_PROMPT = """## Standard Code Review Request

Review this pull request thoroughly.

**PR Title:** {pr_title}
**Description:** {description}

**Changes Overview:**
- Files: {total_files}
- Additions: {additions} lines
- Deletions: {deletions} lines
- Languages: {languages}
- Risk Level: {risk_level}

**Issues Detected:**
- Critical: {critical_count}
- High: {high_count}
- Medium: {medium_count}

{critical_issues}

**Impact Analysis:**
- Affected callers: {affected_callers}
- Complexity hotspots: {complexity_hotspots}
- High coupling: {coupling_files}

**Files to Review:**
{files_details}

Provide a comprehensive review including:
1. Security Analysis
2. Logic and Correctness
3. Performance Considerations
4. Code Quality
5. Testing Recommendations
6. Overall Assessment

Be specific and reference line numbers where applicable.
"""
    
    DEEP_REVIEW_PROMPT = """## Issue-Focused Code Review with GraphRAG Context

**PR:** {pr_title}
**Scope:** {total_files} files | +{additions}/-{deletions} | Risk: {risk_level}

---

## Context from GraphRAG Analysis

### 🔗 Similar Code Patterns in Codebase
{similar_code}

### 📦 Dependencies & Impact
{dependencies}

### 🏗️ Key Entities & Relationships
{entities}

### 🐛 Static Analysis Pre-Scan
- Critical Issues: {critical_issues}
- High Priority Issues: {high_issues}

---

## Files Changed (with code snippets)

{files_details}

---

## YOUR TASK: Find Issues with Evidence

You are a senior code reviewer. For EACH issue you find:

1. **Cite the line number**: `L45+` (added), `L67-` (removed)
2. **Show the problematic code**: Include the actual code snippet
3. **Explain the issue**: What's wrong and why it matters
4. **Reference GraphRAG**: If similar patterns exist in the codebase, cite them with code
5. **Suggest a fix**: Show how to fix it (code example when possible)

**CRITICAL**: You MUST include actual code snippets in your review. Do not just cite line numbers.

---

## Required Output Format

### 🔴 CRITICAL
[If found:]

**L<line>+**: <Issue title>

**Code:**
```
<actual problematic code from the diff>
```

**Issue**: <Detailed explanation of the problem>

**Evidence**: <GraphRAG reference with similar code if applicable>

**Fix**: <How to fix it, with code example if possible>

---

### 🟠 HIGH  
[Same format as above]

### 🟡 MEDIUM
[Same format as above]

### ✅ No Issues Found
[If no issues in a category, state "None - code looks good"]

---

## GraphRAG Requirement

If similar code, dependencies, or entities are provided above (not "None"), you MUST cite at least 3-5 in your findings with actual code comparisons.

**Example of GOOD usage:**

❌ **BAD**: "SQL injection at L45"

✅ **GOOD**: 
"**L45+**: SQL injection vulnerability

**Code:**
```python
query = "SELECT * FROM users WHERE id=" + user_input
```

**Issue**: Direct string concatenation enables SQL injection attacks

**Evidence**: Similar vulnerable pattern in auth.py:L234 (GraphRAG similarity: 0.89):
```python
query = f"SELECT * FROM sessions WHERE token={{user_token}}"
```
Both use unsafe string operations instead of parameterized queries.

**Fix**:
```python
query = "SELECT * FROM users WHERE id=?"
db.execute(query, [user_input])
```"

---

**Focus**: Find bugs, security issues, and mistakes. Include code context for EVERY issue.
"""
    
    FILE_REVIEW_PROMPT = """## File Review: {filename}

**Changes:** +{additions}/-{deletions} | **Language:** {language}

---

## Code Diff (with line numbers)

```{language}
{diff}
```

---

## GraphRAG Context for This File

### Similar Code in Codebase
{similar_code}

### Dependencies & Impact
{dependencies}

### Related Entities
{entities}

### Pre-Scan Issues Found
{issues}

---

## Find Issues with Code Context

Review ONLY the changes shown above. For EACH issue:

1. **Cite line number**: `L45+` or `L67-`
2. **Show the code**: Include the actual code snippet
3. **Explain the problem**: What's wrong and why
4. **Reference GraphRAG**: Cite similar patterns with code if available
5. **Suggest fix**: Provide code example when possible

**CRITICAL**: Include actual code snippets, not just line numbers.

---

## Required Format

### CRITICAL
[If found:]

**L<line>+**: <Issue title>

**Code:**
```{language}
<actual code>
```

**Issue**: <Explanation>
**Evidence**: <GraphRAG reference with code if applicable>
**Fix**: <Solution with code>

---

### HIGH
[Same format]

### MEDIUM
[Same format]

### No Issues Found
[If clean: "No issues found - code looks good"]

---

**GraphRAG Usage**: If similar code/dependencies are provided (not "None"), cite them with actual code comparisons.
"""
    
    AGGREGATION_PROMPT = """## Summary Review

**PR:** {pr_title}
**Files:** {files_count} | **Total Issues:** {total_issues}

**Breakdown:**
\ud83d\udd34 Critical: {critical_count}
\ud83d\udfe0 High: {high_count}
\ud83d\udfe1 Medium: {medium_count}

**Individual Reviews:**
{file_reviews}

---

## Create Brief Summary

Group by severity. Preserve GraphRAG insights from individual reviews (dependencies, similar code).

**Output:**

### \ud83d\udd34 Critical Issues Summary
[List key issues across files or "None"]

### \ud83d\udfe0 High Priority Summary
[List or "None"]

### \ud83d\udfe1 Medium Priority Summary
[List or "None"]

### Overall Assessment
[Approve / Request Changes / Comment with brief rationale]

Format as a clear, well-structured PR review comment. DO NOT discard GraphRAG context.
"""


# ============================================================================
# GEMINI CLIENT
# ============================================================================

class GeminiClient:
    """
    Client for Google Gemini API with rate limiting and error handling
    """
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or self._load_config()
        self.client = self._configure_api()
        self.templates = PromptTemplates()
        self._request_times: List[float] = []
    
    def _load_config(self) -> GeminiConfig:
        """Load configuration from environment"""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable required"
            )
        
        return GeminiConfig(api_key=api_key)
    
    def _configure_api(self):
        """Configure the Gemini API client"""
        client = genai.Client(api_key=self.config.api_key)
        logger.info("Gemini API configured successfully")
        return client
    
    def select_model(
        self,
        total_files: int,
        total_additions: int,
        risk_level: str,
        review_strategy: str
    ) -> str:
        """
        Select appropriate Gemini model based on PR characteristics
        
        Model Selection Logic:
        - flash-lite: Quick reviews, small PRs (< 3 files, < 100 lines)
        - flash: Standard reviews, medium PRs (< 10 files, < 500 lines)
        - pro: Deep reviews, large/complex PRs or high risk
        """
        if review_strategy == "quick" and total_files <= 3 and total_additions <= 100:
            model = self.config.flash_lite_model
            logger.info(f"Selected {model} for quick review")
        
        elif review_strategy == "deep" or risk_level in ["critical", "high"]:
            model = self.config.pro_model
            logger.info(f"Selected {model} for deep/high-risk review")
        
        elif total_files > 10 or total_additions > 500:
            model = self.config.pro_model
            logger.info(f"Selected {model} for large PR")
        
        else:
            model = self.config.flash_model
            logger.info(f"Selected {model} for standard review")
        
        return model
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting for API calls"""
        import time
        
        now = time.time()
        # Keep only requests from the last minute
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self.config.max_requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                # Clean up old requests after waiting
                now = time.time()
                self._request_times = [t for t in self._request_times if now - t < 60]
        
        self._request_times.append(now)
    
    def _get_generation_config(self) -> types.GenerateContentConfig:
        """Get generation configuration"""
        return types.GenerateContentConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_output_tokens=self.config.max_output_tokens,
        )
    
    def _get_safety_settings(self) -> List[types.SafetySetting]:
        """Get safety settings"""
        if not self.config.enable_safety_filters:
            return [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE"
                ),
            ]
        
        return [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
        ]
    
    async def generate_review(
        self,
        model_name: str,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate code review using Gemini
        
        Args:
            model_name: Gemini model to use
            prompt: Review prompt
            system_instruction: System instruction (optional)
        
        Returns:
            Generated review text
        """
        await self._enforce_rate_limit()
        
        system = system_instruction or self.templates.SYSTEM_PROMPT
        
        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(f"Generating review with {model_name} (attempt {attempt + 1})")
                
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_name,
                    contents=prompt,
                    config=self._get_generation_config()
                )
                
                if response.text:
                    logger.info(f"Review generated successfully ({len(response.text)} chars)")
                    return response.text
                else:
                    logger.warning("Empty response from Gemini")
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    return "Unable to generate review at this time."
                
            except Exception as e:
                logger.error(f"Error generating review (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max retries exceeded")
                    raise
    
    async def stream_review(
        self,
        model_name: str,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream code review generation
        
        Yields chunks of generated text as they're produced
        """
        await self._enforce_rate_limit()
        
        system = system_instruction or self.templates.SYSTEM_PROMPT
        
        try:
            logger.info(f"Starting streaming review with {model_name}")
            
            response = await asyncio.to_thread(
                self.client.models.generate_content_stream,
                model=model_name,
                contents=prompt,
                config=self._get_generation_config(),
                safety_settings=self._get_safety_settings()
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            
            logger.info("Streaming review completed")
            
        except Exception as e:
            logger.error(f"Error streaming review: {e}")
            yield f"Error generating review: {str(e)}"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_gemini_client(api_key: Optional[str] = None) -> GeminiClient:
    """Factory function to create Gemini client"""
    if api_key:
        config = GeminiConfig(api_key=api_key)
        return GeminiClient(config)
    return GeminiClient()
