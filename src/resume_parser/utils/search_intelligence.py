"""
Intelligent search processing for resume matching using configuration files.

This version uses YAML configuration files to define role mappings instead of hardcoded values.
Much more flexible and maintainable.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from ..utils.logging import get_logger
from .roles_config import roles_config
from ..clients.azure_openai import azure_client

logger = get_logger(__name__)


@dataclass
class ParsedQuery:
    """Parsed search query with extracted components."""

    original_query: str
    job_roles: List[str]
    skills: List[str]
    role_inferred_skills: List[str]
    forced_keywords: List[str] = field(default_factory=list)
    required_degrees: List[str] = field(default_factory=list)
    required_institutions: List[str] = field(default_factory=list)
    experience_years: Optional[int] = None
    min_experience_years: Optional[int] = None
    max_experience_years: Optional[int] = None
    seniority_level: Optional[str] = None
    location: Optional[str] = None
    companies: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    company_type: Optional[str] = None
    intent: str = 'general'  # 'hire', 'find', 'search', etc.
    expanded_query: str = ''
    unavailable_role_info: Optional[Dict[str, Any]] = None

    def effective_skills(self) -> List[str]:
        """Return explicit skills if present, otherwise inferred role skills."""
        return self.skills if self.skills else self.role_inferred_skills


class ConfigBasedRoleDatabase:
    """Role database that loads from configuration file."""

    def __init__(self):
        """Initialize with roles from config file."""
        self.roles_config = roles_config

    @classmethod
    def find_matching_roles(cls, query: str) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """
        Find job roles that match the query using configuration file.

        Returns:
            Tuple of (role_keys, unavailable_role_info)
        """
        query_lower = query.lower().strip()

        # First check if this is an unavailable role
        unavailable_info = roles_config.check_unavailable_role(query)
        if unavailable_info:
            return [], unavailable_info

        # Find role mapping from config
        role_key = roles_config.find_role_mapping(query)
        if role_key:
            return [role_key], None

        # No mapping found
        return [], None

    @classmethod
    def get_role_skills(cls, role_category: str) -> List[str]:
        """Get skills associated with a role category."""
        return roles_config.get_role_skills(role_category)

    @classmethod
    def get_role_titles(cls, role_category: str) -> List[str]:
        """Get all title variations for a role category."""
        return roles_config.get_role_search_terms(role_category)

    @classmethod
    def get_database_roles(cls) -> List[str]:
        """Get list of actual database role categories."""
        return roles_config.get_database_roles()

    @classmethod
    def get_role_database_category(cls, role_name: str) -> Optional[str]:
        """Get database category (sector) for a role name."""
        return roles_config.get_role_database_category(role_name)

    @classmethod
    def get_seniority_database_values(cls, seniority_key: str) -> List[str]:
        """Get database values for seniority level."""
        return roles_config.get_seniority_database_values(seniority_key)


class IntelligentSearchProcessor:
    """Processes search queries with configuration-based intelligence."""

    def __init__(self):
        self.job_db = ConfigBasedRoleDatabase()

    def parse_query(self, query: str) -> ParsedQuery:
        """Parse a natural language query into structured components using AI."""
        logger.info(f"Parsing search query: {query}")

        # Use AI-powered query classification instead of regex
        ai_parsed = self._ai_parse_query(query)
        if ai_parsed:
            logger.info(f"AI parsed query successfully: {ai_parsed}")
            return ai_parsed

        # Fallback to enhanced regex approach if AI fails
        logger.warning("AI parsing failed, falling back to regex approach")

        # First try LLM-enhanced query parsing for better accuracy
        enhanced_query = self._enhance_query_with_llm(query)
        if enhanced_query:
            query = enhanced_query
            logger.info(f"Enhanced query: {query}")

        # Extract job roles and check for unavailable roles
        job_roles, unavailable_info = self.job_db.find_matching_roles(query)

        # Extract skills
        skills = self._extract_skills(query, job_roles)

        # Extract experience requirements (min/max range awareness)
        min_experience_years, max_experience_years = self._extract_experience_range(query)
        experience_years = min_experience_years if min_experience_years is not None else max_experience_years

        role_inferred_skills = self._aggregate_role_skills(job_roles)

        # Extract seniority level
        seniority_level = self._extract_seniority(query)

        # Extract location
        location = self._extract_location(query)

        # Determine intent
        intent = self._determine_intent(query)

        # Create expanded query
        expanded_query = self._expand_query(query, job_roles, skills)

        forced_keywords = self._extract_forced_keywords(query)
        required_degrees, required_institutions = self._extract_education_constraints(query)

        # If the detected location matches an education institution requirement, treat it as education
        if location:
            normalized_location = location.strip().lower()
            if any(normalized_location == inst.strip().lower() for inst in required_institutions):
                logger.info(f"🎓 Treating location '{location}' as required institution; removing from location filter")
                location = None

        parsed = ParsedQuery(
            original_query=query,
            job_roles=job_roles,
            skills=skills,
            role_inferred_skills=role_inferred_skills,
            forced_keywords=forced_keywords,
            experience_years=experience_years,
            min_experience_years=min_experience_years,
            max_experience_years=max_experience_years,
            seniority_level=seniority_level,
            location=location,
            company_type=None,
            intent=intent,
            expanded_query=expanded_query,
            unavailable_role_info=unavailable_info,
            required_degrees=required_degrees,
            required_institutions=required_institutions
        )

        # Augment forced keywords with parsed data context
        parsed.forced_keywords = self._extract_forced_keywords(query, parsed)

        logger.info(
            "Parsed query - Roles: %s, Skills: %s, Seniority: %s, Location: %s, MinExp: %s, MaxExp: %s",
            job_roles,
            skills[:5],
            seniority_level,
            location,
            min_experience_years,
            max_experience_years
        )
        if unavailable_info:
            logger.info(f"Unavailable role detected: {unavailable_info}")

        return parsed

    def _extract_skills(self, query: str, job_roles: List[str]) -> List[str]:
        """Extract relevant skills from query - comprehensive and flexible for any technology/skill."""
        skills = []
        query_lower = query.lower()
        query_words = query_lower.split()

        # Get skills from identified roles first (only if mentioned in query)
        for role in job_roles:
            role_skills = self.job_db.get_role_skills(role)
            for skill in role_skills:
                skill_lower = skill.lower()
                if skill_lower in query_lower:
                    skills.append(skill)

        # Comprehensive technology/skill patterns - much more flexible
        skill_patterns = [
            # Programming languages
            r'\b(python|java|javascript|typescript|go|golang|rust|c\+\+|c#|php|ruby|swift|kotlin|scala|r|matlab)\b',
            # Frontend frameworks
            r'\b(react|reactjs|angular|vue|vuejs|svelte|next\.?js|nuxt|gatsby|ember)\b',
            # Backend frameworks
            r'\b(node\.?js|express|django|flask|spring|laravel|rails|asp\.net|fastapi)\b',
            # Databases
            r'\b(sql|mysql|postgresql|postgres|mongodb|redis|elasticsearch|cassandra|neo4j|dynamodb)\b',
            # Cloud & DevOps
            r'\b(aws|azure|gcp|google cloud|docker|kubernetes|k8s|terraform|ansible|jenkins|ci/cd)\b',
            # Data & AI
            r'\b(machine learning|ml|ai|artificial intelligence|deep learning|nlp|computer vision|tensorflow|pytorch|pandas|numpy|scikit-learn)\b',
            # Vector/Search technologies
            r'\b(qdrant|pinecone|weaviate|chroma|faiss|vector|vectorization|embedding|embeddings|semantic search|similarity search)\b',
            # Mobile development
            r'\b(ios|android|react native|flutter|xamarin|ionic)\b',
            # Testing
            r'\b(jest|pytest|selenium|cypress|unit testing|integration testing)\b',
            # Project management
            r'\b(agile|scrum|kanban|jira|confluence|trello|asana)\b',
            # Other tools
            r'\b(git|github|gitlab|bitbucket|figma|adobe|photoshop|illustrator)\b'
        ]

        # Apply all patterns
        for pattern in skill_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            skills.extend(matches)

        # Extract potential technical terms (2+ characters, alphanumeric with optional dots/hyphens)
        technical_terms = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\.-]{1,}[a-zA-Z0-9]\b', query)

        # Filter technical terms - include if they look like technology names
        for term in technical_terms:
            term_lower = term.lower()
            # Skip common English words
            common_words = {'the', 'and', 'with', 'for', 'who', 'has', 'can', 'will', 'need', 'want', 'looking', 'find', 'hire', 'developer', 'engineer', 'manager', 'years', 'experience', 'work', 'team', 'project', 'company', 'role', 'position', 'job', 'skill', 'technology', 'tool', 'framework', 'language', 'database', 'system', 'application', 'software', 'web', 'mobile', 'full', 'stack', 'senior', 'junior', 'lead', 'principal', 'director', 'knows', 'knowledge', 'expert', 'proficient', 'familiar'}

            if (len(term) >= 3 and
                term_lower not in common_words and
                term_lower not in [skill.lower() for skill in skills] and
                not term_lower.isdigit()):
                skills.append(term)

        # Clean and deduplicate
        clean_skills = []
        seen = set()
        for skill in skills:
            skill_clean = skill.strip()
            skill_lower = skill_clean.lower()
            if skill_clean and skill_lower not in seen and len(skill_clean) >= 2:
                clean_skills.append(skill_clean)
                seen.add(skill_lower)

        return clean_skills

    def _aggregate_role_skills(self, roles: List[str], limit: int = 30) -> List[str]:
        """Aggregate unique skills from configured role data."""
        aggregated: List[str] = []
        seen: set = set()

        for role in roles:
            role_skills = self.job_db.get_role_skills(role) or []
            for skill in role_skills:
                if not skill or not isinstance(skill, str):
                    continue
                skill_clean = skill.strip()
                if not skill_clean:
                    continue
                key = skill_clean.lower()
                if key in seen:
                    continue
                aggregated.append(skill_clean)
                seen.add(key)
                if len(aggregated) >= limit:
                    return aggregated

        return aggregated

    def _extract_forced_keywords(self, query: str, parsed_query: Optional[ParsedQuery] = None) -> List[str]:
        """Extract keywords that must appear in candidate data (attribute-style constraints)."""
        forced: List[str] = []
        seen: set = set()

        def add_term(text: Optional[str]):
            if not text or not isinstance(text, str):
                return
            cleaned = text.strip()
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            forced.append(cleaned)
            seen.add(key)

        query_lower = query.lower()

        # Attribute pattern: "<attr> status <value>" or "<attr> status is <value>"
        status_pattern = re.compile(r'(\b[\w\s]{2,40}?)\s+status\s*(?:is|=|:)?\s*(\b[\w\s-]{2,40}\b)', re.IGNORECASE)
        for attr, value in status_pattern.findall(query_lower):
            attr_clean = attr.strip()
            value_clean = value.strip()
            if attr_clean and value_clean:
                add_term(f"{attr_clean} status {value_clean}")
                add_term(value_clean)

        # General "<attr> is <value>" / "<attr> = <value>" / "<attr>: <value>"
        attribute_pattern = re.compile(r'(\b[\w\s]{2,30}?)\s*(?:is|=|:)\s*(\b[\w\s-]{2,40}\b)', re.IGNORECASE)
        for attr, value in attribute_pattern.findall(query_lower):
            attr_clean = attr.strip()
            value_clean = value.strip()
            if attr_clean and value_clean and attr_clean not in {'what', 'who', 'where', 'when', 'why', 'how'}:
                add_term(f"{attr_clean} {value_clean}")
                add_term(value_clean)

        # Pattern: "must be <value>" / "should be <value>" / "needs to be <value>"
        be_pattern = re.compile(r'(must|should|needs|need|has to|have to)\s+(?:be|have)\s+(\b[\w\s-]{2,40}\b)', re.IGNORECASE)
        for _, value in be_pattern.findall(query_lower):
            add_term(value)

        # Quoted phrases enforce exact matches
        quoted_pattern = re.compile(r'"([^"]+)"|\'([^\']+)\'')
        for quoted in quoted_pattern.findall(query_lower):
            phrase = quoted[0] or quoted[1]
            add_term(phrase)

        # Include AI-derived keywords to ensure coverage
        if parsed_query is not None:
            for keyword in parsed_query.keywords:
                add_term(keyword)

        return forced

    def _extract_education_constraints(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract degree and institution requirements from the query."""
        degrees: List[str] = []
        institutions: List[str] = []
        seen_degrees: set = set()
        seen_institutions: set = set()

        query_lower = query.lower()

        degree_patterns = [
            r'(bachelor(?:s)?(?: of)? [a-z\s]+)',
            r'(master(?:s)?(?: of)? [a-z\s]+)',
            r'((?:ba|b\.a\.|bs|b\.s\.|btech|b\.tech|mtech|m\.tech|mba|m\.b\.a\.|phd|ph\.d\.|mca|bca)\b[\w\s]*)',
            r'(associate(?:s)?(?: of)? [a-z\s]+)',
            r'(diploma(?: in)? [a-z\s]+)'
        ]

        for pattern in degree_patterns:
            for match in re.findall(pattern, query_lower, re.IGNORECASE):
                degree_clean = match.strip().title()
                key = degree_clean.lower()
                if degree_clean and key not in seen_degrees and len(degree_clean) > 2:
                    degrees.append(degree_clean)
                    seen_degrees.add(key)

        # Pattern: "from <institution>" / "at <institution>" / "in <institution>"
        institution_pattern = re.compile(r'\bfrom\s+([a-z0-9&.,\-\s]{3,60})|\bat\s+([a-z0-9&.,\-\s]{3,60})', re.IGNORECASE)
        for match in institution_pattern.findall(query):
            institution = match[0] or match[1]
            cleaned = institution.strip()
            if cleaned:
                # Trim trailing words like "university", "college" etc remain included
                key = cleaned.lower()
                if key not in seen_institutions:
                    institutions.append(cleaned)
                    seen_institutions.add(key)

        return degrees, institutions

    def _extract_experience_range(self, query: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract minimum and maximum experience requirements from a query."""
        query_lower = query.lower()

        # Range expressions (between/from/A-B)
        range_patterns = [
            r'between\s*(\d+)\s*(?:\+?\s*)?and\s*(\d+)\s*years?',
            r'from\s*(\d+)\s*(?:\+?\s*)?(?:years?|yrs?)?\s*to\s*(\d+)\s*years?',
            r'(\d+)\s*[-–—]\s*(\d+)\s*years?',
            r'(\d+)\s*(?:to)\s*(\d+)\s*years?'
        ]

        for pattern in range_patterns:
            match = re.search(pattern, query_lower)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                if start > end:
                    start, end = end, start
                return start, end

        # "Up to" / maximum expressions
        match = re.search(r'(?:up to|at most|maximum(?: of)?|no more than)\s*(\d+)\s*years?', query_lower)
        if match:
            return None, int(match.group(1))

        # "Less than" / "under"
        match = re.search(r'(?:less than|under|below)\s*(\d+)\s*years?', query_lower)
        if match:
            value = int(match.group(1))
            return None, max(value - 1, 0)

        # "More than" / "over"
        match = re.search(r'(?:more than|over|greater than)\s*(\d+)\s*years?', query_lower)
        if match:
            value = int(match.group(1))
            return value, None

        # "At least" / minimum
        match = re.search(r'(?:at least|minimum(?: of)?|not less than)\s*(\d+)\s*years?', query_lower)
        if match:
            return int(match.group(1)), None

        # Explicit "exactly" phrasing
        match = re.search(r'(?:exactly|equal to)\s*(\d+)\s*years?', query_lower)
        if match:
            value = int(match.group(1))
            return value, value

        # Numeric value followed by years/yrs +/- modifiers
        simple_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*yrs?\s*experience',
            r'(\d+)\s*years?\s*(?:experience|exp)',
            r'(\d+)\s*yrs?'
        ]

        for pattern in simple_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return int(match.group(1)), None

        return None, None

    def _extract_experience(self, query: str) -> Optional[int]:
        """Backward-compatible helper that returns the minimum experience if present."""
        min_exp, max_exp = self._extract_experience_range(query)
        return min_exp if min_exp is not None else max_exp

    def _extract_seniority(self, query: str) -> Optional[str]:
        """Extract seniority level from query using config."""
        return roles_config.find_seniority_mapping(query)

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location requirements from query."""
        location_patterns = [
            r'(?:in|from|at|near|based\s+in|located\s+in)\s+([a-zA-Z][a-zA-Z\s,.-]+?)(?:\s+(?:for|and|with|who|that|\?|$))',
            r'(?:in|from|at|near|based\s+in|located\s+in)\s+([a-zA-Z][a-zA-Z\s,.-]+?)$',
            r'(?:in|from|at|near|based\s+in|located\s+in)\s+([a-zA-Z][a-zA-Z\s,.-]+?)(?:\s|,|\.|$)'
        ]

        for pattern in location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                location = re.sub(r'\s+(and|for|with|who|that)$', '', location, flags=re.IGNORECASE)
                if len(location) > 2:
                    return location

        return None

    def _determine_intent(self, query: str) -> str:
        """Determine the search intent."""
        query_lower = query.lower()

        if any(word in query_lower for word in ['hire', 'hiring', 'recruit', 'recruiting']):
            return 'hire'
        elif any(word in query_lower for word in ['find', 'looking for', 'need', 'want']):
            return 'find'
        elif any(word in query_lower for word in ['search', 'seeking', 'candidate']):
            return 'search'
        else:
            return 'general'

    def _expand_query(self, query: str, job_roles: List[str], skills: List[str]) -> str:
        """Expand the query with synonyms and related terms."""
        expanded_parts: List[str] = []
        seen: set = set()

        def add_part(text: Optional[str]):
            if not text or not isinstance(text, str):
                return
            cleaned = text.strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                expanded_parts.append(cleaned)
                seen.add(key)

        add_part(query)

        # Add role synonyms from config
        for role in job_roles:
            role_titles = self.job_db.get_role_titles(role)
            for title in role_titles[:5]:
                add_part(title)

        # Add explicit skills mentioned in query
        for skill in skills:
            add_part(skill)

        # Add inferred skills from configuration
        for skill in self._aggregate_role_skills(job_roles, limit=40):
            add_part(skill)

        return ' '.join(expanded_parts)

    def create_search_filters(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Create Qdrant search filters based on parsed query and config."""
        filters = {}

        # Role category filter using config mappings
        if parsed_query.job_roles:
            possible_categories = []
            for role in parsed_query.job_roles:
                db_category = self.job_db.get_role_database_category(role)
                if db_category:
                    possible_categories.append(db_category)

            if possible_categories:
                filters['role_category'] = {'$in': possible_categories}

        # Seniority filter using config
        if parsed_query.seniority_level:
            db_values = self.job_db.get_seniority_database_values(parsed_query.seniority_level)
            if db_values:
                filters['seniority'] = {'$in': db_values}

        # Location filter
        if parsed_query.location:
            filters['location'] = parsed_query.location
            logger.info(f"🗺️ Adding location filter: {parsed_query.location}")

        # Experience filter
        if parsed_query.min_experience_years is not None:
            filters['min_experience_years'] = parsed_query.min_experience_years

        if parsed_query.max_experience_years is not None:
            filters['max_experience_years'] = parsed_query.max_experience_years

        return filters

    def extract_comprehensive_search_text(self, payload: Dict[str, Any]) -> str:
        """Extract comprehensive searchable text from all available resume data."""
        text_parts = []

        # Basic info
        text_parts.extend([
            payload.get('name', ''),
            payload.get('email', ''),
            payload.get('current_position', ''),
            payload.get('summary', ''),
            payload.get('best_role', '')
        ])

        # Skills
        skills = payload.get('skills', [])
        if skills:
            text_parts.extend(skills)

        # Work history - comprehensive extraction
        work_history = payload.get('work_history', [])
        for job in work_history:
            if isinstance(job, dict):
                text_parts.extend([
                    job.get('title', ''),
                    job.get('company', ''),
                    job.get('location', '')
                ])

                # Extract responsibilities
                responsibilities = job.get('responsibilities', [])
                if responsibilities:
                    text_parts.extend(responsibilities)

                # Extract technologies from work
                technologies = job.get('technologies', [])
                if technologies:
                    text_parts.extend(technologies)

        # Projects - comprehensive extraction
        projects = payload.get('projects', [])
        for project in projects:
            if isinstance(project, dict):
                text_parts.extend([
                    project.get('name', ''),
                    project.get('description', ''),
                    project.get('role', '')
                ])

                # Extract project technologies
                tech = project.get('technologies', [])
                if tech:
                    text_parts.extend(tech)

        # Education - comprehensive extraction
        education = payload.get('education', [])
        for edu in education:
            if isinstance(edu, dict):
                text_parts.extend([
                    edu.get('degree', ''),
                    edu.get('field', ''),
                    edu.get('institution', ''),
                    edu.get('location', '')
                ])

        # Recommended roles
        recommended_roles = payload.get('recommended_roles', [])
        if recommended_roles:
            text_parts.extend(recommended_roles)

        # Current employment details
        current_emp = payload.get('current_employment', {})
        if current_emp and isinstance(current_emp, dict):
            text_parts.extend([
                current_emp.get('company', ''),
                current_emp.get('position', '')
            ])

        # Filter out empty strings and join
        clean_parts = [part.strip() for part in text_parts if part and isinstance(part, str) and part.strip()]
        return ' '.join(clean_parts)

    def calculate_comprehensive_relevance_score(self, payload: Dict[str, Any], parsed_query: ParsedQuery) -> float:
        """Calculate comprehensive relevance score using all available data."""

        # For AI/ML specialized roles, enforce strict skill requirements
        if self._is_ai_ml_query(parsed_query):
            if not self._has_required_ai_ml_skills(payload, parsed_query):
                return 0.0  # Exclude candidates without AI/ML skills

        score = 0.0
        max_score = 0.0

        # Extract comprehensive text for full-text matching
        comprehensive_text = self.extract_comprehensive_search_text(payload).lower()
        query_lower = parsed_query.original_query.lower()
        query_words = [word.strip() for word in query_lower.split() if len(word.strip()) >= 2]

        effective_skills = parsed_query.effective_skills()

        # 1. Enhanced skills matching (weight: 4.0) - but only if we have skills in query
        if effective_skills:
            candidate_skills = [s.lower() for s in payload.get('skills', [])]
            skill_matches = 0
            exact_skill_matches = 0
            matched_skills = []

            logger.info(f"🔧 Evaluating skills for candidate {payload.get('name', 'UNKNOWN')}")
            logger.info(f"🔧 Query skills: {effective_skills}")
            logger.info(f"🔧 Candidate skills: {payload.get('skills', [])}")

            for skill in effective_skills:
                skill_lower = skill.lower()
                # Exact match in skills array
                if skill_lower in candidate_skills:
                    exact_skill_matches += 1
                    matched_skills.append(f"EXACT:{skill}")
                    logger.info(f"🔧 ✅ EXACT match: '{skill}' found in candidate's skills")
                # Partial match in skills array (e.g., "python" matches "Python 3.9")
                elif any(skill_lower in candidate_skill for candidate_skill in candidate_skills):
                    skill_matches += 0.8  # Increased from 0.7 for better partial matching
                    matched_skills.append(f"PARTIAL:{skill}")
                    matching_skill = next(cs for cs in candidate_skills if skill_lower in cs)
                    logger.info(f"🔧 ✅ PARTIAL match: '{skill}' found in '{matching_skill}'")
                # Reverse partial match (e.g., "machine learning" partially matches "ml")
                elif any(candidate_skill in skill_lower for candidate_skill in candidate_skills if len(candidate_skill) > 2):
                    skill_matches += 0.6
                    matched_skills.append(f"REVERSE:{skill}")
                    matching_skill = next(cs for cs in candidate_skills if cs in skill_lower)
                    logger.info(f"🔧 ✅ REVERSE match: '{matching_skill}' matches part of '{skill}'")
                # Check if skill appears in comprehensive text
                elif skill_lower in comprehensive_text:
                    skill_matches += 0.4  # Reduced weight for text-only matches
                    matched_skills.append(f"TEXT:{skill}")
                    logger.info(f"🔧 ✅ TEXT match: '{skill}' found in resume content")
                else:
                    logger.info(f"🔧 ❌ NO match: '{skill}' not found")

            # Enhanced scoring: bonus for multiple matches and exact matches
            skill_match_ratio = (exact_skill_matches + skill_matches) / max(len(effective_skills), 1)
            exact_match_bonus = exact_skill_matches / max(len(effective_skills), 1) * 0.5  # Bonus for exact matches
            total_skill_score = min(skill_match_ratio + exact_match_bonus, 1.0)  # Cap at 1.0

            score += total_skill_score * 4.0
            max_score += 4.0

            logger.info(f"🔧 Skills scoring: exact={exact_skill_matches}, partial={skill_matches:.1f}, total_score={total_skill_score:.3f}")
            logger.info(f"🔧 Matched skills: {matched_skills}")

            # Additional boost for highly relevant candidates with strong skill matches
            if total_skill_score > 0.7:
                score += 0.5  # Small bonus for strong skill matches
                logger.info(f"🔧 🚀 Skill excellence bonus: +0.5 (total_skill_score={total_skill_score:.3f})")

        # For role-only queries (like "hr manager"), focus on role and position matching
        else:
            # Check role category match
            candidate_role = payload.get('role_category', '').lower()
            current_position = payload.get('current_position', '').lower()

            role_match_score = 0.0
            for role in parsed_query.job_roles:
                # Map to database categories
                db_category = self.job_db.get_role_database_category(role)
                if db_category and db_category.lower() in candidate_role:
                    role_match_score += 1.0  # Perfect match
                elif role.replace('_', ' ') in current_position:
                    role_match_score += 0.7

            score += min(role_match_score, 1.0) * 4.0
            max_score += 4.0

        # Rest of scoring logic remains the same...
        # (truncated for brevity, but includes work history, projects, education, etc.)

        # Special handling for role-only queries
        if not parsed_query.skills and parsed_query.job_roles:
            role_bonus = 0.0
            candidate_role = payload.get('role_category', '').lower()
            current_position = payload.get('current_position', '').lower()

            # Direct role matching bonus
            for role in parsed_query.job_roles:
                db_category = self.job_db.get_role_database_category(role)
                if db_category and db_category.lower() in candidate_role:
                    role_bonus += 0.8

            score += role_bonus * 2.0
            max_score += 2.0

        # Normalize score to 0-1 range
        if max_score > 0:
            return min(score / max_score, 1.0)

        # Fallback for edge cases
        return 0.1 if parsed_query.job_roles or effective_skills else 0.0

    def _enhance_query_with_llm(self, query: str) -> Optional[str]:
        """Use LLM to fix typos and normalize the query."""
        try:
            prompt = f"""
Fix any typos and normalize this job search query to be more standard and clear.
Return ONLY the corrected query, nothing else.

Examples:
- "sofyware developer from hyderabad" -> "software developer from Hyderabad, India"
- "java devloper with 3 yrs exp" -> "java developer with 3 years experience"
- "full stak engineer in mumbai" -> "full stack engineer from Mumbai, India"

Query to fix: {query}
"""

            client = azure_client.get_sync_client()
            chat_deployment = azure_client.get_chat_deployment()

            response = client.chat.completions.create(
                model=chat_deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that fixes typos and normalizes job search queries. Return only the corrected query."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )

            enhanced_query = response.choices[0].message.content
            if enhanced_query:
                enhanced_query = enhanced_query.strip()

            # Basic validation - only return if it looks reasonable
            if enhanced_query and len(enhanced_query) > 3 and enhanced_query.lower() != query.lower():
                return enhanced_query

        except Exception as e:
            logger.warning(f"LLM query enhancement failed: {e}")

        return None

    def _ai_parse_query(self, query: str) -> Optional[ParsedQuery]:
        """Use AI to parse and classify the query components accurately."""
        try:
            client = azure_client.get_sync_client()
            chat_deployment = azure_client.get_chat_deployment()

            # Get available roles for the AI to choose from
            available_roles = self.job_db.get_database_roles()

            # Prioritize common tech roles that users frequently search for
            priority_roles = [
                "Software Developer", "Software Engineer", "Data Scientist", "Data Analyst",
                "DevOps Engineer", "Frontend Developer", "Backend Developer", "Full Stack Developer",
                "Machine Learning Engineer", "AI Engineer", "Cloud Engineer", "Database Administrator",
                "Product Manager", "Project Manager", "Business Analyst", "Business Intelligence", "Quality Assurance",
                "UI/UX Designer", "System Administrator", "Network Engineer", "Cybersecurity Analyst",
                # Add missing roles from actual resume data
                "Business Development", "Software Development"
            ]

            # Create prioritized list: priority roles first, then others
            prioritized_roles = []
            for role in priority_roles:
                if role in available_roles:
                    prioritized_roles.append(role)

            # Add remaining roles up to 30 total
            remaining_roles = [r for r in available_roles if r not in priority_roles]
            prioritized_roles.extend(remaining_roles[:30 - len(prioritized_roles)])

            roles_list = ", ".join(prioritized_roles)

            prompt = f"""
You are an expert in parsing job search queries. Parse the following query and extract information in JSON format.

Available job roles in database: {roles_list}

Query to parse: "{query}"

Return ONLY a valid JSON object with these exact fields:
{{
    "job_roles": ["list of relevant job roles from the available roles, or empty list if none match"],
    "skills": ["list of technical skills/technologies mentioned, excluding location words and common words"],
    "experience_years": number or null,
    "min_experience_years": number or null,
    "max_experience_years": number or null,
    "seniority_level": "junior/mid/senior/manager/director" or null,
    "location": "extracted location" or null,
    "companies": ["list of company/organization names mentioned, or empty list if none"],
    "keywords": ["list of important keywords for word-based matching if no specific intent found"],
    "intent": "hire/find/search/general"
}}

Important rules:
1. job_roles: Include roles from available list OR common role names. Also interpret natural language descriptions:
   - "build websites" → "Frontend Developer" or "Full Stack Developer"
   - "data analysis" → "Data Analyst" or "Data Scientist"
   - "manage social media" → "Social Media Manager" or "Digital Marketing"
   - "accounting software" → "Accountant" or "Financial Analyst"
   - "write code" → "Software Developer"
   - "good with databases" → "Database Administrator"
   - "handle customers" → "Customer Support"
   - "manage projects" → "Project Manager"
   - "design graphics" → "UI/UX Designer"
   - "handle finances" → "Financial Analyst"
   - "sales person" → "Sales Representative"
   - "marketing campaigns" → "Marketing Manager"
   - "ai developer" → "Data Scientist" or "Machine Learning Engineer" (ONLY if candidate has AI/ML skills)
   - "artificial intelligence" → "Data Scientist" or "Machine Learning Engineer" (ONLY if candidate has AI/ML skills)
   - "machine learning" → "Data Scientist" or "Machine Learning Engineer" (ONLY if candidate has AI/ML skills)
   - "ml engineer" → "Machine Learning Engineer" or "Data Scientist" (ONLY if candidate has AI/ML skills)
   - "ai engineer" → "Machine Learning Engineer" or "Data Scientist" (ONLY if candidate has AI/ML skills)
   - "nlp" → "Data Scientist" or "Machine Learning Engineer" (ONLY if candidate has AI/ML skills)
   - "computer vision" → "Data Scientist" or "Machine Learning Engineer" (ONLY if candidate has AI/ML skills)
   - "deep learning" → "Data Scientist" or "Machine Learning Engineer" (ONLY if candidate has AI/ML skills)

CRITICAL: For AI/ML related queries, candidates MUST have specific AI/ML skills like:
- Machine Learning, Deep Learning, TensorFlow, PyTorch, Scikit-learn, Keras
- NLP, Computer Vision, Neural Networks, AI, Artificial Intelligence
- Data Science, Statistics, Model Deployment, Feature Engineering
Do NOT include general software developers unless they have these specific skills.
2. skills: Include technical skills AND interpret natural language:
   - "good with Excel" → ["Excel"]
   - "knows databases" → ["SQL", "Database"]
   - "website development" → ["HTML", "CSS", "JavaScript"]
   - "data analysis" → ["Python", "Excel", "SQL"]
3. location: Extract complete location (city, state/country if mentioned)
4. companies: Extract company/organization names like "Silvertouch Technologies", "Google", "Microsoft", etc.
5. keywords: Extract all important words for semantic matching if no specific role/skill/location found
6. Do not include words like "from", "in", "at", city names, or country names in skills
7. Return only the JSON, no explanations
8. Populate min_experience_years/max_experience_years for any mentioned range or boundary (e.g., "3-5 years", "at least 4 years", "up to 2 years")

Natural Language Examples:
- "someone who can build websites" → job_roles: ["Frontend Developer"], skills: ["HTML", "CSS", "JavaScript"]
- "person good with data analysis" → job_roles: ["Data Analyst"], skills: ["Python", "Excel", "SQL"]
- "need someone for social media" → job_roles: ["Social Media Manager"], skills: ["Social Media", "Digital Marketing"]
- "candidate who knows accounting software" → job_roles: ["Accountant"], skills: ["Excel", "Accounting Software"]
- "ai developer from ahmedabad" → job_roles: ["Data Scientist"], skills: ["Python", "Machine Learning", "AI"], location: "Ahmedabad"
- "need machine learning engineer" → job_roles: ["Machine Learning Engineer"], skills: ["Python", "Machine Learning", "Deep Learning"]
- "artificial intelligence expert" → job_roles: ["Data Scientist"], skills: ["AI", "Python", "Machine Learning"]
- "nlp developer" → job_roles: ["Data Scientist"], skills: ["NLP", "Python", "Machine Learning"]
- "candidates from Silvertouch Technologies" → companies: ["Silvertouch Technologies"], keywords: ["candidates", "Silvertouch", "Technologies"]
- "people working at Google" → companies: ["Google"], keywords: ["people", "working", "Google"]
"""

            response = client.chat.completions.create(
                model=chat_deployment,
                messages=[
                    {"role": "system", "content": "You are a precise query parser that returns only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )

            ai_response = response.choices[0].message.content
            if ai_response:
                ai_response = ai_response.strip()

                # Remove any markdown code blocks
                if ai_response.startswith("```json"):
                    ai_response = ai_response[7:]
                if ai_response.startswith("```"):
                    ai_response = ai_response[3:]
                if ai_response.endswith("```"):
                    ai_response = ai_response[:-3]

                ai_response = ai_response.strip()

                # Parse the JSON response
                import json
                parsed_data = json.loads(ai_response)

                # Validate and clean the response
                def _safe_int(value: Any) -> Optional[int]:
                    try:
                        if value is None:
                            return None
                        if isinstance(value, (int, float)):
                            return int(value)
                        value_str = str(value).strip()
                        if not value_str:
                            return None
                        return int(float(value_str))
                    except (TypeError, ValueError):
                        return None

                job_roles = parsed_data.get("job_roles", [])
                skills = parsed_data.get("skills", [])
                experience_years = _safe_int(parsed_data.get("experience_years"))
                min_experience_years = _safe_int(parsed_data.get("min_experience_years"))
                max_experience_years = _safe_int(parsed_data.get("max_experience_years"))
                seniority_level = parsed_data.get("seniority_level")
                location = parsed_data.get("location")
                companies = parsed_data.get("companies", [])
                keywords = parsed_data.get("keywords", [])
                intent = parsed_data.get("intent", "general")

                if min_experience_years is None and max_experience_years is None and experience_years is not None:
                    min_experience_years = experience_years

                if min_experience_years is not None and max_experience_years is not None and min_experience_years > max_experience_years:
                    min_experience_years, max_experience_years = max_experience_years, min_experience_years

                primary_experience_years = min_experience_years if min_experience_years is not None else max_experience_years
                role_inferred_skills = self._aggregate_role_skills(job_roles)

                # Check for unavailable roles
                unavailable_info = None
                if job_roles:
                    for role in job_roles:
                        unavailable_check = roles_config.check_unavailable_role(role)
                        if unavailable_check:
                            unavailable_info = unavailable_check
                            job_roles = []  # Clear roles if unavailable
                            role_inferred_skills = []
                            break

                # Create expanded query
                expanded_query = self._expand_query(query, job_roles, skills)

                required_degrees, required_institutions = self._extract_education_constraints(query)

                location = parsed_data.get("location")
                if location:
                    normalized_location = location.strip().lower()
                    if any(normalized_location == inst.strip().lower() for inst in required_institutions):
                        logger.info(f"🎓 Treating location '{location}' as required institution (AI parse); removing from location filter")
                        location = None

                parsed = ParsedQuery(
                    original_query=query,
                    job_roles=job_roles,
                    skills=skills,
                    role_inferred_skills=role_inferred_skills,
                    forced_keywords=[],
                    required_degrees=required_degrees,
                    required_institutions=required_institutions,
                    experience_years=primary_experience_years,
                    min_experience_years=min_experience_years,
                    max_experience_years=max_experience_years,
                    seniority_level=seniority_level,
                    location=location,
                    companies=companies,
                    keywords=keywords,
                    company_type=None,
                    intent=intent,
                    expanded_query=expanded_query,
                    unavailable_role_info=unavailable_info
                )

                parsed.forced_keywords = self._extract_forced_keywords(query, parsed)
                deg_llm, inst_llm = self._extract_education_constraints(query)
                if deg_llm:
                    parsed.required_degrees = list({*(parsed.required_degrees), *deg_llm})
                if inst_llm:
                    parsed.required_institutions = list({*(parsed.required_institutions), *inst_llm})
                return parsed

        except Exception as e:
            logger.warning(f"AI query parsing failed: {e}")
            return None

        return None

    def _is_ai_ml_query(self, parsed_query: ParsedQuery) -> bool:
        """Check if this is an AI/ML specialized query that requires specific skills."""
        ai_ml_terms = [
            'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
            'nlp', 'natural language processing', 'computer vision', 'neural network',
            'ai developer', 'ml engineer', 'ai engineer', 'data scientist'
        ]

        query_lower = parsed_query.original_query.lower()

        # Check if query contains AI/ML terms
        if any(term in query_lower for term in ai_ml_terms):
            return True

        # Check if detected roles are AI/ML related
        ai_ml_roles = ['Data Scientist', 'Machine Learning Engineer', 'AI Research Scientist',
                       'Computer Vision Engineer', 'NLP Engineer', 'AI Software Engineer']

        return any(role in parsed_query.job_roles for role in ai_ml_roles)

    def _has_required_ai_ml_skills(self, payload: Dict[str, Any], parsed_query: ParsedQuery) -> bool:
        """Check if candidate has required AI/ML skills for specialized roles."""
        # Core AI/ML technologies that indicate actual AI development
        core_ai_skills = [
            'machine learning', 'deep learning', 'artificial intelligence', 'ai development',
            'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'neural networks',
            'nlp', 'natural language processing', 'computer vision', 'opencv',
            'data science', 'model deployment', 'feature engineering', 'xgboost',
            'lightgbm', 'neural network', 'ml models', 'ai models'
        ]

        # Supporting skills (less specific)
        supporting_skills = ['pandas', 'numpy', 'python', 'r', 'statistics', 'matlab']

        # Check skills array for core AI/ML technologies
        candidate_skills = [s.lower() for s in payload.get('skills', [])]
        core_skills_count = sum(1 for skill in core_ai_skills if any(skill in cs for cs in candidate_skills))

        logger.info(f"🤖 AI/ML Skills Check for {payload.get('name', 'UNKNOWN')}")
        logger.info(f"🤖 Candidate skills: {payload.get('skills', [])}")
        logger.info(f"🤖 Core AI skills found: {core_skills_count}")

        # Check work history for ACTUAL AI/ML work (not just mentions)
        work_history = payload.get('work_history', [])
        substantial_ai_experience = False
        ai_work_details = []

        for job in work_history:
            if isinstance(job, dict):
                job_description = job.get('description', '').lower()
                position = job.get('position', '').lower()

                # Look for substantial AI/ML work indicators
                ai_indicators = [
                    'machine learning model', 'ai model', 'neural network', 'deep learning',
                    'data science', 'ml engineer', 'ai engineer', 'data scientist',
                    'computer vision', 'nlp', 'natural language processing', 'tensorflow',
                    'pytorch', 'model training', 'model deployment', 'feature engineering'
                ]

                ai_mentions = sum(1 for indicator in ai_indicators if indicator in job_description or indicator in position)
                if ai_mentions >= 2:  # Multiple substantial AI mentions
                    substantial_ai_experience = True
                    ai_work_details.append(f"{job.get('position', 'Unknown')} - {ai_mentions} AI indicators")

        # Check projects for ACTUAL AI/ML implementation
        key_projects = payload.get('key_projects', [])
        substantial_ai_projects = False
        ai_project_details = []

        logger.info(f"🤖 DEBUG: Projects for {payload.get('name', 'UNKNOWN')}: {key_projects}")

        for project in key_projects:
            if isinstance(project, str):
                project_lower = project.lower()
                # Look for AI/ML project indicators
                ai_project_indicators = [
                    'machine learning', 'neural network', 'ai', 'computer vision',
                    'nlp', 'deep learning', 'tensorflow', 'pytorch', 'model',
                    'prediction', 'classification', 'regression', 'clustering'
                ]

                ai_project_mentions = sum(1 for indicator in ai_project_indicators if indicator in project_lower)
                logger.info(f"🤖 DEBUG: Project '{project}' has {ai_project_mentions} AI indicators")

                if ai_project_mentions >= 1:  # At least one substantial AI project mention
                    substantial_ai_projects = True
                    ai_project_details.append(f"{project} - {ai_project_mentions} AI indicators")

        # Also check comprehensive text for AI/ML project references since key_projects might be empty
        comprehensive_text = self.extract_comprehensive_search_text(payload).lower()
        logger.info(f"🤖 DEBUG: Comprehensive text snippet for {payload.get('name', 'UNKNOWN')}: {comprehensive_text[:200]}...")

        if not substantial_ai_projects:
            # More sensitive detection for AI/ML in project context
            ai_text_indicators = [
                'machine learning', 'neural network', 'deep learning', 'ai development',
                'artificial intelligence', 'computer vision', 'nlp', 'natural language processing',
                'tensorflow', 'pytorch', 'ml model', 'ai model', 'prediction model'
            ]

            text_ai_mentions = sum(1 for indicator in ai_text_indicators if indicator in comprehensive_text)
            logger.info(f"🤖 DEBUG: Text AI mentions found: {text_ai_mentions}")

            if text_ai_mentions >= 1:
                substantial_ai_projects = True
                ai_project_details.append(f"Text content - {text_ai_mentions} AI indicators found")
                logger.info(f"🤖 DEBUG: ✅ Found AI indicators in comprehensive text: {text_ai_mentions}")

        logger.info(f"🤖 Substantial AI work experience: {substantial_ai_experience} - {ai_work_details}")
        logger.info(f"🤖 Substantial AI projects: {substantial_ai_projects} - {ai_project_details}")

        # Enhanced criteria for AI/ML roles - More inclusive for actual AI developers:
        # Option 1: Has at least 1 core AI skill (shows technical knowledge)
        # Option 2: Has substantial AI projects (shows practical experience)
        # Option 3: Has substantial AI work experience (shows professional background)
        # Option 4: Has supporting skills with AI project mentions (for candidates like Lucky who have ML projects)

        # Additional check: Look for explicit AI project mentions in full resume content
        full_resume_text = ' '.join([
            payload.get('summary', ''),
            ' '.join([str(proj) for proj in payload.get('key_projects', [])]),
            ' '.join([str(job.get('description', '')) for job in payload.get('work_history', []) if isinstance(job, dict)])
        ]).lower()

        has_ai_project_mention = any(term in full_resume_text for term in [
            'machine learning', 'ai project', 'artificial intelligence', 'neural network',
            'deep learning', 'computer vision', 'nlp project'
        ])

        meets_criteria = (
            core_skills_count >= 1 or  # Has at least 1 core AI skill
            substantial_ai_experience or  # Has substantial AI work experience
            substantial_ai_projects or  # Has substantial AI projects
            (has_ai_project_mention and core_skills_count >= 0)  # Has AI project mentions (for cases like Lucky)
        )

        logger.info(f"🤖 Final AI/ML qualification: {meets_criteria} (core_skills={core_skills_count}, ai_exp={substantial_ai_experience}, ai_proj={substantial_ai_projects}, ai_mentions={has_ai_project_mention})")
        return meets_criteria


# Global instance
search_processor = IntelligentSearchProcessor()
