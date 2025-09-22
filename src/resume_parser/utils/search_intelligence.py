"""
Intelligent search processing for resume matching.

Provides industry-standard search capabilities with proper role understanding,
query expansion, and intelligent ranking.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedQuery:
    """Parsed search query with extracted components."""

    original_query: str
    job_roles: List[str]
    skills: List[str]
    experience_years: Optional[int]
    seniority_level: Optional[str]
    location: Optional[str]
    company_type: Optional[str]
    intent: str  # 'hire', 'find', 'search', etc.
    expanded_query: str


class JobRoleDatabase:
    """Database of job roles, their synonyms, and related skills."""

    ROLE_CATEGORIES = {
        'software_engineer': {
            'titles': [
                'software engineer', 'software developer', 'developer', 'programmer',
                'full stack developer', 'frontend developer', 'backend developer',
                'web developer', 'mobile developer', 'app developer', 'coding specialist'
            ],
            'skills': [
                'python', 'javascript', 'java', 'react', 'node.js', 'html', 'css',
                'angular', 'vue.js', 'sql', 'mongodb', 'git', 'docker', 'kubernetes'
            ],
            'seniority': ['junior', 'senior', 'lead', 'principal', 'staff']
        },
        'data_scientist': {
            'titles': [
                'data scientist', 'data analyst', 'machine learning engineer',
                'ml engineer', 'ai engineer', 'data engineer', 'analytics specialist',
                'business intelligence analyst', 'statistician'
            ],
            'skills': [
                'python', 'r', 'sql', 'machine learning', 'deep learning', 'tensorflow',
                'pytorch', 'pandas', 'numpy', 'scikit-learn', 'tableau', 'power bi',
                'statistics', 'data visualization', 'big data', 'spark'
            ],
            'seniority': ['junior', 'senior', 'lead', 'principal', 'staff']
        },
        'hr_manager': {
            'titles': [
                'hr manager', 'human resources manager', 'hr specialist', 'hr generalist',
                'people manager', 'talent acquisition', 'recruiter', 'hr business partner',
                'hr coordinator', 'human resources specialist', 'people operations',
                'talent manager', 'hr executive', 'chief people officer'
            ],
            'skills': [
                'recruitment', 'talent acquisition', 'employee relations', 'performance management',
                'compensation', 'benefits', 'hris', 'workday', 'successfactors',
                'interviewing', 'onboarding', 'policy development', 'compliance',
                'diversity and inclusion', 'training and development', 'employee engagement'
            ],
            'seniority': ['coordinator', 'specialist', 'manager', 'senior manager', 'director', 'vp']
        },
        'product_manager': {
            'titles': [
                'product manager', 'product owner', 'senior product manager',
                'principal product manager', 'vp product', 'chief product officer',
                'product marketing manager', 'technical product manager'
            ],
            'skills': [
                'product management', 'agile', 'scrum', 'jira', 'roadmapping',
                'market research', 'user research', 'analytics', 'a/b testing',
                'product strategy', 'stakeholder management', 'wireframing', 'figma'
            ],
            'seniority': ['associate', 'manager', 'senior', 'principal', 'director', 'vp']
        },
        'marketing_manager': {
            'titles': [
                'marketing manager', 'digital marketing manager', 'marketing specialist',
                'brand manager', 'content marketing manager', 'social media manager',
                'marketing coordinator', 'growth manager', 'performance marketing manager'
            ],
            'skills': [
                'digital marketing', 'content marketing', 'social media', 'seo', 'sem',
                'google analytics', 'facebook ads', 'google ads', 'email marketing',
                'marketing automation', 'hubspot', 'salesforce', 'brand management'
            ],
            'seniority': ['coordinator', 'specialist', 'manager', 'senior manager', 'director', 'vp']
        },
        'sales_manager': {
            'titles': [
                'sales manager', 'sales representative', 'account manager',
                'business development manager', 'sales executive', 'account executive',
                'sales director', 'vp sales', 'chief revenue officer'
            ],
            'skills': [
                'sales', 'business development', 'account management', 'crm',
                'salesforce', 'lead generation', 'negotiation', 'pipeline management',
                'client relations', 'revenue growth', 'b2b sales', 'b2c sales'
            ],
            'seniority': ['representative', 'manager', 'senior manager', 'director', 'vp']
        }
    }

    @classmethod
    def find_matching_roles(cls, query: str) -> List[str]:
        """Find job roles that match the query."""
        query_lower = query.lower()
        matching_roles = []

        for category, data in cls.ROLE_CATEGORIES.items():
            for title in data['titles']:
                if title in query_lower or any(word in query_lower for word in title.split()):
                    matching_roles.append(category)
                    break

        return matching_roles

    @classmethod
    def get_role_skills(cls, role_category: str) -> List[str]:
        """Get skills associated with a role category."""
        return cls.ROLE_CATEGORIES.get(role_category, {}).get('skills', [])

    @classmethod
    def get_role_titles(cls, role_category: str) -> List[str]:
        """Get all title variations for a role category."""
        return cls.ROLE_CATEGORIES.get(role_category, {}).get('titles', [])


class IntelligentSearchProcessor:
    """Processes search queries with industry-standard intelligence."""

    def __init__(self):
        self.job_db = JobRoleDatabase()

    def parse_query(self, query: str) -> ParsedQuery:
        """Parse a natural language query into structured components."""
        logger.info(f"Parsing search query: {query}")

        # Extract job roles
        job_roles = self.job_db.find_matching_roles(query)

        # Extract skills
        skills = self._extract_skills(query, job_roles)

        # Extract experience requirements
        experience_years = self._extract_experience(query)

        # Extract seniority level
        seniority_level = self._extract_seniority(query)

        # Extract location
        location = self._extract_location(query)

        # Determine intent
        intent = self._determine_intent(query)

        # Create expanded query
        expanded_query = self._expand_query(query, job_roles, skills)

        parsed = ParsedQuery(
            original_query=query,
            job_roles=job_roles,
            skills=skills,
            experience_years=experience_years,
            seniority_level=seniority_level,
            location=location,
            company_type=None,
            intent=intent,
            expanded_query=expanded_query
        )

        logger.info(f"Parsed query - Roles: {job_roles}, Skills: {skills[:5]}, Seniority: {seniority_level}, Location: {location}")
        return parsed

    def _extract_skills(self, query: str, job_roles: List[str]) -> List[str]:
        """Extract relevant skills from query and role context."""
        skills = []
        query_lower = query.lower()

        # Get skills from identified roles
        for role in job_roles:
            role_skills = self.job_db.get_role_skills(role)
            for skill in role_skills:
                if skill.lower() in query_lower:
                    skills.append(skill)

        # Common skill patterns
        skill_patterns = [
            r'\b(python|java|javascript|react|angular|vue|node\.?js)\b',
            r'\b(sql|mongodb|postgresql|mysql)\b',
            r'\b(aws|azure|gcp|docker|kubernetes)\b',
            r'\b(machine learning|ai|deep learning|nlp)\b',
            r'\b(agile|scrum|jira|confluence)\b',
        ]

        for pattern in skill_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            skills.extend(matches)

        return list(set(skills))

    def _extract_experience(self, query: str) -> Optional[int]:
        """Extract experience requirements from query."""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*yrs?\s*experience',
            r'minimum\s*(\d+)\s*years?',
            r'at least\s*(\d+)\s*years?',
            r'(\d+)\+\s*years?'
        ]

        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))

        return None

    def _extract_seniority(self, query: str) -> Optional[str]:
        """Extract seniority level from query."""
        seniority_map = {
            'junior': ['junior', 'jr', 'entry level', 'entry-level', 'fresher', 'graduate'],
            'mid': ['mid', 'middle', 'intermediate', '2-5 years', '3-6 years'],
            'senior': ['senior', 'sr', 'experienced', 'expert', 'lead'],
            'principal': ['principal', 'staff', 'architect', 'chief'],
            'director': ['director', 'vp', 'vice president', 'head of'],
            'manager': ['manager', 'team lead', 'supervisor']
        }

        query_lower = query.lower()
        for level, keywords in seniority_map.items():
            if any(keyword in query_lower for keyword in keywords):
                return level

        return None

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location requirements from query."""
        location_patterns = [
            # Better patterns to capture full location names
            r'(?:in|from|at|near|based\s+in|located\s+in)\s+([a-zA-Z][a-zA-Z\s,.-]+?)(?:\s+(?:for|and|with|who|that|\?|$))',
            r'(?:in|from|at|near|based\s+in|located\s+in)\s+([a-zA-Z][a-zA-Z\s,.-]+?)$',
            r'(?:in|from|at|near|based\s+in|located\s+in)\s+([a-zA-Z][a-zA-Z\s,.-]+?)(?:\s|,|\.|$)'
        ]

        for pattern in location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Clean up common trailing words
                location = re.sub(r'\s+(and|for|with|who|that)$', '', location, flags=re.IGNORECASE)
                if len(location) > 2:  # Filter out short matches
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
        expanded_parts = [query]

        # Add role synonyms
        for role in job_roles:
            role_titles = self.job_db.get_role_titles(role)
            expanded_parts.extend(role_titles[:3])  # Add top 3 synonyms

        # Add related skills
        for role in job_roles:
            role_skills = self.job_db.get_role_skills(role)
            expanded_parts.extend(role_skills[:5])  # Add top 5 skills

        return ' '.join(expanded_parts)

    def create_search_filters(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Create Qdrant search filters based on parsed query."""
        filters = {}

        # Role category filter
        if parsed_query.job_roles:
            # Map our role categories to the stored role categories
            role_mapping = {
                'software_engineer': ['Software Engineer', 'Developer', 'Programmer', 'Full Stack Developer', 'Frontend Developer', 'Backend Developer', 'Web Developer', 'Mobile Developer'],
                'data_scientist': ['Data Scientist', 'Data Analyst', 'ML Engineer', 'Machine Learning Engineer', 'AI Engineer'],
                'hr_manager': ['HR Manager', 'Human Resources', 'People Manager', 'HR Specialist', 'Human Resources Manager'],
                'product_manager': ['Product Manager', 'Product Owner', 'Senior Product Manager'],
                'marketing_manager': ['Marketing Manager', 'Digital Marketing', 'Marketing Specialist', 'Brand Manager'],
                'sales_manager': ['Sales Manager', 'Account Manager', 'Business Development', 'Sales Representative', 'Account Executive']
            }

            possible_categories = []
            for role in parsed_query.job_roles:
                possible_categories.extend(role_mapping.get(role, []))

            if possible_categories:
                filters['role_category'] = {'$in': possible_categories}

        # Seniority filter
        if parsed_query.seniority_level:
            seniority_mapping = {
                'junior': ['Junior', 'Entry Level', 'Associate'],
                'mid': ['Mid Level', 'Intermediate'],
                'senior': ['Senior', 'Lead'],
                'principal': ['Principal', 'Staff', 'Architect'],
                'director': ['Director', 'VP', 'Head'],
                'manager': ['Manager', 'Team Lead']
            }

            possible_seniorities = seniority_mapping.get(parsed_query.seniority_level, [])
            if possible_seniorities:
                filters['seniority'] = {'$in': possible_seniorities}

        # Location filter
        if parsed_query.location:
            # For location, we'll use a partial match approach
            filters['location'] = parsed_query.location
            logger.info(f"🗺️ Adding location filter: {parsed_query.location}")

        # Experience filter
        if parsed_query.experience_years:
            # This would need custom logic in the search to filter by experience
            filters['min_experience_years'] = parsed_query.experience_years

        return filters

# Global instance
search_processor = IntelligentSearchProcessor()