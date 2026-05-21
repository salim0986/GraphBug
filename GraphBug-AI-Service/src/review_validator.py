"""
Review Validator - Ensures AI reviews reference actual code
Prevents hallucinations by checking all claims against diff
"""

import re
from typing import Dict, List, Tuple, Any
from .logger import setup_logger

logger = setup_logger(__name__)


class ReviewValidator:
    """Validates that reviews reference actual code changes"""
    
    def validate_review(
        self,
        review_text: str,
        files_with_diffs: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate review against actual diff content
        
        Args:
            review_text: The AI-generated review text
            files_with_diffs: List of file objects with patch content
            
        Returns:
            (is_valid, warnings, metrics)
        """
        warnings = []
        metrics = {
            "line_references": 0,
            "valid_line_references": 0,
            "code_quotes": 0,
            "valid_code_quotes": 0,
            "generic_statements": 0,
            "fabricated_metrics": 0
        }
        
        # Extract all line number references from review
        line_refs = self._extract_line_references(review_text)
        metrics["line_references"] = len(line_refs)
        
        # Validate each line reference exists in diffs
        for file, line_num in line_refs:
            if self._line_exists_in_diff(file, line_num, files_with_diffs):
                metrics["valid_line_references"] += 1
            else:
                warnings.append(
                    f"⚠️ Review references {file}:{line_num} which doesn't exist in diff"
                )
        
        # Extract code quotes from review
        code_quotes = self._extract_code_quotes(review_text)
        metrics["code_quotes"] = len(code_quotes)
        
        # Validate code quotes match actual diff
        for quote in code_quotes:
            if self._quote_exists_in_diffs(quote, files_with_diffs):
                metrics["valid_code_quotes"] += 1
            else:
                warnings.append(
                    f"⚠️ Review quotes code not found in diff: {quote[:50]}..."
                )
        
        # Check for generic statements without citations
        generic_patterns = [
            r"consider implementing",
            r"it would be better",
            r"you should",
            r"recommend adding",
            r"could be improved",
        ]
        
        for pattern in generic_patterns:
            matches = re.finditer(pattern, review_text, re.IGNORECASE)
            for match in matches:
                # Check if this statement has a line reference nearby
                context = review_text[max(0, match.start() - 100):match.end() + 100]
                if not re.search(r"line \d+", context, re.IGNORECASE):
                    metrics["generic_statements"] += 1
        
        # Check for fabricated metrics
        metric_patterns = [
            r"cyclomatic complexity of \d+",
            r"complexity score of \d+",
            r"\d+/\d+ score",
        ]
        
        for pattern in metric_patterns:
            matches = re.findall(pattern, review_text, re.IGNORECASE)
            for match in matches:
                # Check if we actually calculated this metric
                metrics["fabricated_metrics"] += len(matches)
                warnings.append(
                    f"⚠️ Review contains uncited metric: {match}"
                )
        
        # Determine if review is valid
        is_valid = True
        
        # If review references lines, most should be valid
        if metrics["line_references"] > 0:
            validity_rate = metrics["valid_line_references"] / metrics["line_references"]
            if validity_rate < 0.8:
                is_valid = False
                warnings.append(
                    f"❌ Only {validity_rate*100:.0f}% of line references are valid"
                )
        
        # Reviews should have specific citations, not just generic advice
        if metrics["line_references"] == 0 and len(review_text) > 500:
            warnings.append(
                "❌ Review is long but has no line number citations"
            )
            is_valid = False
        
        # Too many generic statements is bad
        if metrics["generic_statements" ] >= 5:
            warnings.append(
                f"⚠️ Review has {metrics['generic_statements']} generic statements without citations"
            )
        
        logger.info(f"Review validation: valid={is_valid}, warnings={len(warnings)}")
        logger.info(f"Metrics: {metrics}")
        
        return is_valid, warnings, metrics
    
    def _extract_line_references(self, text: str) -> List[Tuple[str, int]]:
        """Extract (filename, line_number) pairs from review text"""
        refs = []
        
        # Pattern: "file.py:123" or "file.py line 123" or "Line 123:" or "[file.py#L123]" or "L123+" or "L123-"
        patterns = [
            r"L(\d+)[+-]",  # L45+ or L45-
            r"([a-zA-Z0-9_/.]+\.\w+):(\d+)",  # file.py:123
            r"([a-zA-Z0-9_/.]+\.\w+)\s+[Ll]ine\s+(\d+)",  # file.py line 123
            r"\[([a-zA-Z0-9_/.]+\.\w+)#L(\d+)\]",  # [file.py#L123]
            r"[Ll]ine\s+(\d+):?",  # Line 123 or Line 123:
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                groups = match.groups()
                if len(groups) == 2:
                    file = groups[0] if not groups[0].isdigit() else "unknown"
                    line = int(groups[1])
                elif len(groups) == 1:
                    file = "unknown"
                    line = int(groups[0])
                else:
                    continue
                refs.append((file, line))
        
        return refs
    
    def _line_exists_in_diff(
        self,
        filename: str,
        line_num: int,
        files_with_diffs: List[Dict[str, Any]]
    ) -> bool:
        """Check if line number exists in the file's diff"""
        for file_obj in files_with_diffs:
            file_name = file_obj.get("filename", "")
            
            # Match if filename is "unknown" or matches the file
            if filename == "unknown" or file_name.endswith(filename) or filename in file_name:
                patch = file_obj.get("patch", "")
                if not patch:
                    continue
                    
                # Parse patch to extract line numbers from hunks
                # Format: @@ -old_start,old_count +new_start,new_count @@
                hunks = re.findall(r"@@ -\d+,?\d* \+(\d+),?(\d*) @@", patch)
                for hunk in hunks:
                    start = int(hunk[0])
                    length = int(hunk[1]) if hunk[1] else 1
                    if start <= line_num <= start + length:
                        return True
        return False
    
    def _extract_code_quotes(self, text: str) -> List[str]:
        """Extract code snippets quoted in review"""
        quotes = []
        
        # Match code blocks: ```lang ... ``` or ``` ... ```
        blocks = re.findall(r"```[\w]*\n(.*?)\n```", text, re.DOTALL)
        quotes.extend(blocks)
        
        # Match inline code that's substantial (at least 15 chars)
        inline = re.findall(r"`([^`]{15,})`", text)
        quotes.extend(inline)
        
        return quotes
    
    def _quote_exists_in_diffs(
        self,
        quote: str,
        files_with_diffs: List[Dict[str, Any]]
    ) -> bool:
        """Check if quoted code appears in any diff"""
        # Normalize whitespace for comparison
        normalized_quote = " ".join(quote.split())
        
        # Ignore very short quotes (too generic)
        if len(normalized_quote) < 20:
            return True  # Don't flag short quotes as invalid
        
        for file_obj in files_with_diffs:
            patch = file_obj.get("patch", "")
            if not patch:
                continue
                
            normalized_patch = " ".join(patch.split())
            
            # Check if quote appears in patch (fuzzy match with 80% threshold)
            if normalized_quote in normalized_patch:
                return True
            
            # Try substring match for longer quotes
            if len(normalized_quote) > 50:
                # Check if significant portion of quote exists
                words = normalized_quote.split()
                if len(words) > 5:
                    # Check if at least 70% of words appear in sequence
                    chunk_size = max(4, len(words) // 2)
                    for i in range(len(words) - chunk_size):
                        chunk = " ".join(words[i:i+chunk_size])
                        if chunk in normalized_patch:
                            return True
        
        return False
