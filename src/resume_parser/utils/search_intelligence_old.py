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
        """Find job roles that match the query with precision."""
        query_lower = query.lower().strip()
        role_scores = {}

        # For specific role queries like "hr manager", be very focused
        for category, data in cls.ROLE_CATEGORIES.items():
            max_score = 0
            for title in data['titles']:
                title_lower = title.lower()

                # Exact phrase match gets highest score
                if title_lower in query_lower:
                    score = len(title_lower) / len(query_lower)
                    max_score = max(max_score, score)

                # Individual word matching
                title_words = title_lower.split()
                query_words = query_lower.split()

                # If all title words appear in query
                if all(word in query_words for word in title_words):
                    score = len(title_words) / len(query_words) * 0.9
                    max_score = max(max_score, score)

            if max_score > 0:
                role_scores[category] = max_score

        # Only map to roles that actually exist and make sense
        accurate_mappings = {
            'analyst': 'business_analyst',
            'business analyst': 'business_analyst',
            'data scientist': 'data_scientist',
            'software engineer': 'software_engineer',
            'developer': 'software_engineer',
            'programmer': 'software_engineer',
            'business development': 'business_development',
            'sales': 'business_development'
        }

        # Apply accurate mappings only
        for term, mapped_role in accurate_mappings.items():
            if term in query_lower:
                role_scores[mapped_role] = 0.9
                break

        # DO NOT MAP HR TO ANYTHING - be honest about what we don't have
        # HR queries should return empty or low-relevance results
        if 'hr' in query_lower or 'human resources' in query_lower:
            # Don't force any role mapping - let search fail honestly
            pass

        # Return top 1 role for focused search, lower threshold for forced mappings
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
        return [role for role, score in sorted_roles[:1] if score > 0.1]

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
        """Extract relevant skills from query - comprehensive and flexible for any technology/skill."""
        skills = []
        query_lower = query.lower()
        query_words = query_lower.split()

        # Get skills from identified roles first
        for role in job_roles:
            role_skills = self.job_db.get_role_skills(role)
            for skill in role_skills:
                if skill.lower() in query_lower:
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
        # This catches specific tools, frameworks, or technologies not in our patterns
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
                # Likely a technical term
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

        # Role category filter - map to ACTUAL categories in database
        if parsed_query.job_roles:
            # Map to what ACTUALLY exists: Software Engineer, Data Scientist, Business Analyst, Business Development, Full Stack Developer
            role_mapping = {
                'software_engineer': ['Software Engineer', 'Software Development', 'Full Stack Developer'],
                'data_scientist': ['Data Scientist'],
                'business_analyst': ['Business Analyst'],
                'business_development': ['Business Development'],
                # Smart fallbacks for common queries that don't match our database
                'hr_manager': ['Business Analyst'],  # No HR in DB, map to closest
                'product_manager': ['Business Analyst'],  # No PM in DB, map to closest
                'marketing_manager': ['Business Development'],  # No Marketing in DB, map to closest
                'sales_manager': ['Business Development']  # Sales -> Business Development
            }

            possible_categories = []
            for role in parsed_query.job_roles:
                possible_categories.extend(role_mapping.get(role, []))

            # Remove duplicates
            unique_categories = list(dict.fromkeys(possible_categories))

            if unique_categories:
                filters['role_category'] = {'$in': unique_categories}

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
        """Calculate comprehensive relevance score using all available data - optimized for specific technical queries."""
        score = 0.0
        max_score = 0.0

        # Extract comprehensive text for full-text matching
        comprehensive_text = self.extract_comprehensive_search_text(payload).lower()
        query_lower = parsed_query.original_query.lower()
        query_words = [word.strip() for word in query_lower.split() if len(word.strip()) >= 2]

        # 1. Direct skills matching (weight: 4.0) - increased for technical queries
        candidate_skills = [s.lower() for s in payload.get('skills', [])]
        skill_matches = 0
        exact_skill_matches = 0

        for skill in parsed_query.skills:
            skill_lower = skill.lower()
            # Exact match in skills
            if skill_lower in candidate_skills:
                exact_skill_matches += 1
            # Partial match in skills
            elif any(skill_lower in candidate_skill for candidate_skill in candidate_skills):
                skill_matches += 0.7
            # Check if skill appears in comprehensive text
            elif skill_lower in comprehensive_text:
                skill_matches += 0.5

        if parsed_query.skills:
            total_skill_score = (exact_skill_matches + skill_matches) / len(parsed_query.skills)
            score += total_skill_score * 4.0
        max_score += 4.0

        # 2. Work history matching (weight: 3.0) - role titles and context
        work_role_score = 0.0
        work_tech_score = 0.0
        work_context_score = 0.0
        work_history = payload.get('work_history', [])

        for job in work_history:
            if isinstance(job, dict):
                job_title = job.get('title', '').lower()

                # Check job titles for role matches (important for HR manager queries)
                for role in parsed_query.job_roles:
                    role_terms = self.job_db.get_role_titles(role)
                    for role_term in role_terms:
                        if role_term.lower() in job_title:
                            work_role_score += 1.0
                            break

                # Check technologies for skill matches
                if parsed_query.skills:
                    technologies = job.get('technologies', [])
                    for tech in technologies:
                        tech_lower = tech.lower()
                        for skill in parsed_query.skills:
                            if skill.lower() in tech_lower or tech_lower in skill.lower():
                                work_tech_score += 1

                    # Check responsibilities for skills
                    responsibilities = job.get('responsibilities', [])
                    for resp in responsibilities:
                        resp_lower = resp.lower()
                        for skill in parsed_query.skills:
                            if skill.lower() in resp_lower:
                                work_context_score += 0.5

                    # Check job title for skills
                    for skill in parsed_query.skills:
                        if skill.lower() in job_title:
                            work_context_score += 0.7

        # Calculate work history score
        if work_history:
            role_score_norm = work_role_score / len(work_history) if parsed_query.job_roles else 0
            tech_score_norm = (work_tech_score + work_context_score) / (len(parsed_query.skills) * len(work_history)) if parsed_query.skills else 0
            combined_work_score = role_score_norm + tech_score_norm
            score += min(combined_work_score, 1.0) * 3.0

        max_score += 3.0

        # 3. Project technology and description matching (weight: 2.5)
        project_score = 0.0
        projects = payload.get('projects', [])

        for project in projects:
            if isinstance(project, dict):
                project_relevance = 0.0

                # Check technologies
                technologies = project.get('technologies', [])
                for tech in technologies:
                    tech_lower = tech.lower()
                    for skill in parsed_query.skills:
                        if skill.lower() in tech_lower or tech_lower in skill.lower():
                            project_relevance += 1

                # Check project description
                description = project.get('description', '').lower()
                for skill in parsed_query.skills:
                    if skill.lower() in description:
                        project_relevance += 0.7

                # Check project name
                name = project.get('name', '').lower()
                for skill in parsed_query.skills:
                    if skill.lower() in name:
                        project_relevance += 0.5

                project_score += min(project_relevance, 2.0)  # Cap per project

        if projects and parsed_query.skills:
            normalized_project_score = project_score / (len(parsed_query.skills) * len(projects))
            score += min(normalized_project_score, 1.0) * 2.5
        max_score += 2.5

        # 4. Education field matching (weight: 1.5)
        education_score = 0.0
        education = payload.get('education', [])
        for edu in education:
            if isinstance(edu, dict):
                field = edu.get('field', '').lower()
                degree = edu.get('degree', '').lower()
                # Check if query matches education field or degree
                for role in parsed_query.job_roles:
                    role_keywords = self.job_db.get_role_skills(role)
                    if any(keyword.lower() in field or keyword.lower() in degree for keyword in role_keywords):
                        education_score += 1
                        break

        if education:
            score += (education_score / len(education)) * 1.5
        max_score += 1.5

        # 5. Responsibilities matching (weight: 2.0)
        resp_matches = 0
        resp_count = 0
        for job in work_history:
            if isinstance(job, dict):
                responsibilities = job.get('responsibilities', [])
                for resp in responsibilities:
                    resp_count += 1
                    resp_lower = resp.lower()
                    # Check if any query skills or roles mentioned in responsibilities
                    if any(skill.lower() in resp_lower for skill in parsed_query.skills):
                        resp_matches += 1
                    elif any(role.replace('_', ' ') in resp_lower for role in parsed_query.job_roles):
                        resp_matches += 1

        if resp_count > 0:
            score += (resp_matches / resp_count) * 2.0
        max_score += 2.0

        # 6. Project descriptions matching (weight: 1.5)
        proj_desc_matches = 0
        proj_desc_count = 0
        for project in projects:
            if isinstance(project, dict):
                description = project.get('description', '')
                if description:
                    proj_desc_count += 1
                    desc_lower = description.lower()
                    if any(skill.lower() in desc_lower for skill in parsed_query.skills):
                        proj_desc_matches += 1
                    elif any(role.replace('_', ' ') in desc_lower for role in parsed_query.job_roles):
                        proj_desc_matches += 1

        if proj_desc_count > 0:
            score += (proj_desc_matches / proj_desc_count) * 1.5
        max_score += 1.5

        # 7. Full-text query matching with emphasis on technical terms (weight: 2.0)
        # Filter out common words for better matching
        common_words = {'the', 'and', 'with', 'for', 'who', 'has', 'can', 'will', 'need', 'want', 'looking', 'find', 'hire', 'developer', 'engineer', 'manager', 'years', 'experience', 'work', 'team', 'project', 'company', 'role', 'position', 'job', 'knows', 'knowledge'}

        meaningful_words = [word for word in query_words if word not in common_words and len(word) >= 3]

        if meaningful_words:
            text_matches = sum(1 for word in meaningful_words if word in comprehensive_text)
            # Boost score for exact technical term matches
            exact_matches = sum(1 for word in meaningful_words if f' {word} ' in f' {comprehensive_text} ')

            text_score = (text_matches + exact_matches * 0.5) / len(meaningful_words)
            score += text_score * 2.0
        max_score += 2.0

        # 8. Company matching (weight: 1.0)
        company_score = 0.0
        # Check work history companies
        for job in work_history:
            if isinstance(job, dict):
                company = job.get('company', '').lower()
                if any(word in company for word in query_words):
                    company_score += 0.5

        # Check current employment company
        current_emp = payload.get('current_employment', {})
        if isinstance(current_emp, dict):
            current_company = current_emp.get('company', '').lower()
            if any(word in current_company for word in query_words):
                company_score += 0.5

        score += min(company_score, 1.0) * 1.0
        max_score += 1.0

        # 9. Query-specific term boost (weight: 1.5) - for hiring agency flexibility
        query_term_score = 0.0

        # For each significant term in query, check if it appears anywhere in the resume
        significant_terms = [word for word in query_words if len(word) >= 3 and word not in {'the', 'and', 'with', 'for', 'who', 'has', 'can', 'will', 'need', 'want', 'looking', 'find', 'hire', 'developer', 'engineer', 'manager', 'years', 'experience', 'work', 'team', 'project', 'company', 'role', 'position', 'job', 'knows', 'knowledge', 'senior', 'junior', 'lead'}]

        for term in significant_terms:
            if term in comprehensive_text:
                # Exact word boundary match gets higher score
                if f' {term} ' in f' {comprehensive_text} ':
                    query_term_score += 1.0
                else:
                    query_term_score += 0.7

        if significant_terms:
            normalized_term_score = query_term_score / len(significant_terms)
            score += normalized_term_score * 1.5
        max_score += 1.5

        # Special handling for role-only queries (like "hr manager")
        if not parsed_query.skills and parsed_query.job_roles:
            role_bonus = 0.0
            candidate_role = payload.get('role_category', '').lower()
            current_position = payload.get('current_position', '').lower()

            # Direct role matching bonus
            for role in parsed_query.job_roles:
                if role == 'hr_manager' and 'analyst' in candidate_role:
                    role_bonus += 0.5  # HR -> Business Analyst mapping
                elif role == 'business_analyst' and 'analyst' in candidate_role:
                    role_bonus += 0.8  # Direct match
                elif 'manager' in parsed_query.original_query.lower() and 'manager' in current_position:
                    role_bonus += 0.4  # Manager level match
                elif 'senior' in current_position and parsed_query.seniority_level:
                    role_bonus += 0.3  # Seniority match

            score += role_bonus * 2.0  # Boost role matches
            max_score += 2.0

        # Normalize score to 0-1 range
        if max_score > 0:
            return min(score / max_score, 1.0)

        # Fallback for edge cases
        return 0.1 if parsed_query.job_roles or parsed_query.skills else 0.0

# Global instance
search_processor = IntelligentSearchProcessor()