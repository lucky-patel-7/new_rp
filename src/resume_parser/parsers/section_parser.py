"""
Section-based resume parser - replacement for pattern-based extraction
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


def identify_resume_sections(lines: List[str]) -> Dict[str, List[str]]:
    """Identify different sections in the resume"""

    sections = {
        'header': [],
        'contact': [],
        'summary': [],
        'skills': [],
        'experience': [],
        'education': [],
        'projects': [],
        'other': []
    }

    current_section = 'header'
    section_keywords = {
        'summary': ['summary', 'profile', 'objective', 'about'],
        'skills': ['skills', 'technical', 'technologies', 'tools', 'expertise', 'competencies'],
        'experience': ['experience', 'work', 'employment', 'career', 'professional', 'history'],
        'education': ['education', 'qualifications', 'academic', 'degrees', 'schooling'],
        'projects': ['projects', 'portfolio', 'work samples', 'achievements']
    }

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        # Check if this line is a section header
        section_found = False
        for section_name, keywords in section_keywords.items():
            for keyword in keywords:
                # More flexible matching for section headers
                if (re.search(f'\\b{keyword}\\b', line_lower) and ':' in line) or \
                   re.match(f'^{keyword}s?:?\\s*$', line_lower) or \
                   line_lower == keyword:
                    current_section = section_name
                    section_found = True
                    break
            if section_found:
                break

        # Skip section headers themselves
        if section_found:
            continue

        # Add content to current section
        if line.strip():
            sections[current_section].append(line)

    return sections


MONTHS_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
}


def _parse_partial_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse a partial date string (month/year or year) into a datetime object."""
    if not date_str or not isinstance(date_str, str):
        return None

    date_str = date_str.strip().lower()
    if not date_str:
        return None

    if any(token in date_str for token in ['present', 'current', 'now']):
        return datetime.now()

    # Month + year (e.g., March 2021 or Mar 2021)
    month_year_match = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(\d{4})', date_str)
    if month_year_match:
        month = MONTHS_MAP.get(month_year_match.group(1)[:3])
        year = int(month_year_match.group(2))
        if month and year:
            return datetime(year, month, 1)

    # Numeric month/year formats like 03/2020 or 2020/03
    numeric_match = re.search(r'(\d{1,2})[\-/](\d{4})', date_str)
    if numeric_match:
        month = int(numeric_match.group(1))
        year = int(numeric_match.group(2))
        if 1 <= month <= 12:
            return datetime(year, month, 1)

    # Year only
    year_match = re.search(r'(19|20)\d{2}', date_str)
    if year_match:
        return datetime(int(year_match.group(0)), 1, 1)

    return None


def _parse_duration_range(duration: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Parse a duration string (e.g., "Jan 2019 - Mar 2022") into start/end datetimes."""
    if not duration or not isinstance(duration, str):
        return None, None

    parts = re.split(r'\s*(?:-|to|–|—)\s*', duration)
    if len(parts) >= 2:
        start = _parse_partial_date(parts[0])
        end = _parse_partial_date(parts[1])
        return start, end

    # If only one part, try to parse it as a standalone date
    single = _parse_partial_date(duration)
    return single, None


def _coerce_experience_entry(entry: Any) -> Dict[str, Any]:
    """Convert different experience representations into a dict."""
    if entry is None:
        return {}

    if hasattr(entry, 'model_dump'):
        return entry.model_dump()
    if hasattr(entry, 'dict'):
        return entry.dict()
    if isinstance(entry, dict):
        return entry

    return {}


def calculate_total_experience(experience_data: List[Dict]) -> str:
    """Calculate total years and months of experience from work history."""
    if not experience_data:
        return "0 years"

    total_months = 0
    fallback_text_chunks: List[str] = []

    for entry in experience_data:
        job = _coerce_experience_entry(entry)
        if not job:
            continue

        start = _parse_partial_date(job.get('start_date'))
        end = _parse_partial_date(job.get('end_date'))

        if not start or not end:
            duration_start, duration_end = _parse_duration_range(job.get('duration'))
            start = start or duration_start
            end = end or duration_end

        # Treat missing or ongoing end dates as current date
        if start and not end:
            end = datetime.now()
        if end and not start and job.get('duration'):
            # If only one date present in duration, assume that's start and use current date as end
            start = _parse_duration_range(job.get('duration'))[0]
            end = datetime.now()

        if start and end:
            if end < start:
                start, end = end, start
            months = (end.year - start.year) * 12 + (end.month - start.month)
            if months < 0:
                months = 0
            total_months += months

        # Collect text for fallback pattern search
        responsibilities = job.get('responsibilities', [])
        if isinstance(responsibilities, list):
            responsibilities_text = ' '.join(responsibilities)
        elif isinstance(responsibilities, str):
            responsibilities_text = responsibilities
        else:
            responsibilities_text = ''

        fallback_text_chunks.extend([
            job.get('title', ''),
            job.get('company', ''),
            job.get('duration', ''),
            responsibilities_text
        ])

    if total_months > 0:
        years = total_months // 12
        months = total_months % 12
        parts = []
        if years:
            parts.append(f"{years} year{'s' if years != 1 else ''}")
        if months:
            parts.append(f"{months} month{'s' if months != 1 else ''}")
        if not parts:
            parts.append("0 months")
        return ' '.join(parts)

    # Fallback to textual pattern matching if structured dates unavailable
    fallback_text = ' '.join(chunk for chunk in fallback_text_chunks if chunk)
    exp_patterns = [
        r'(\d+(?:\.\d+)?)\+?\s*years?\s*(?:of\s*)?experience',
        r'(\d+(?:\.\d+)?)\+?\s*yrs?'
    ]

    for pattern in exp_patterns:
        matches = re.findall(pattern, fallback_text.lower())
        if matches:
            max_years = max(float(x) for x in matches)
            whole_years = int(max_years)
            if max_years == whole_years:
                return f"{whole_years} years"
            return f"{max_years} years"

    return "0 years"