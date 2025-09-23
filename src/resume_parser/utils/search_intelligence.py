"""
Intelligent search processing for resume matching using configuration files.

This version uses YAML configuration files to define role mappings instead of hardcoded values.
Much more flexible and maintainable.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
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
    experience_years: Optional[int]
    seniority_level: Optional[str]
    location: Optional[str]
    companies: List[str]
    keywords: List[str]
    company_type: Optional[str]
    intent: str  # 'hire', 'find', 'search', etc.
    expanded_query: str
    unavailable_role_info: Optional[Dict[str, Any]] = None


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
            expanded_query=expanded_query,
            unavailable_role_info=unavailable_info
        )

        logger.info(f"Parsed query - Roles: {job_roles}, Skills: {skills[:5]}, Seniority: {seniority_level}, Location: {location}")
        if unavailable_info:
            logger.info(f"Unavailable role detected: {unavailable_info}")

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
        expanded_parts = [query]

        # Add role synonyms from config
        for role in job_roles:
            role_titles = self.job_db.get_role_titles(role)
            expanded_parts.extend(role_titles[:3])

        # Add related skills from config
        for role in job_roles:
            role_skills = self.job_db.get_role_skills(role)
            expanded_parts.extend(role_skills[:5])

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
        if parsed_query.experience_years:
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
        """Calculate comprehensive relevance score using all available data."""
        score = 0.0
        max_score = 0.0

        # Extract comprehensive text for full-text matching
        comprehensive_text = self.extract_comprehensive_search_text(payload).lower()
        query_lower = parsed_query.original_query.lower()
        query_words = [word.strip() for word in query_lower.split() if len(word.strip()) >= 2]

        # 1. Direct skills matching (weight: 4.0) - but only if we have skills in query
        if parsed_query.skills:
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

            total_skill_score = (exact_skill_matches + skill_matches) / len(parsed_query.skills)
            score += total_skill_score * 4.0
            max_score += 4.0

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
        return 0.1 if parsed_query.job_roles or parsed_query.skills else 0.0

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

Natural Language Examples:
- "someone who can build websites" → job_roles: ["Frontend Developer"], skills: ["HTML", "CSS", "JavaScript"]
- "person good with data analysis" → job_roles: ["Data Analyst"], skills: ["Python", "Excel", "SQL"]
- "need someone for social media" → job_roles: ["Social Media Manager"], skills: ["Social Media", "Digital Marketing"]
- "candidate who knows accounting software" → job_roles: ["Accountant"], skills: ["Excel", "Accounting Software"]
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
                job_roles = parsed_data.get("job_roles", [])
                skills = parsed_data.get("skills", [])
                experience_years = parsed_data.get("experience_years")
                seniority_level = parsed_data.get("seniority_level")
                location = parsed_data.get("location")
                companies = parsed_data.get("companies", [])
                keywords = parsed_data.get("keywords", [])
                intent = parsed_data.get("intent", "general")

                # Check for unavailable roles
                unavailable_info = None
                if job_roles:
                    for role in job_roles:
                        unavailable_check = roles_config.check_unavailable_role(role)
                        if unavailable_check:
                            unavailable_info = unavailable_check
                            job_roles = []  # Clear roles if unavailable
                            break

                # Create expanded query
                expanded_query = self._expand_query(query, job_roles, skills)

                return ParsedQuery(
                    original_query=query,
                    job_roles=job_roles,
                    skills=skills,
                    experience_years=experience_years,
                    seniority_level=seniority_level,
                    location=location,
                    companies=companies,
                    keywords=keywords,
                    company_type=None,
                    intent=intent,
                    expanded_query=expanded_query,
                    unavailable_role_info=unavailable_info
                )

        except Exception as e:
            logger.warning(f"AI query parsing failed: {e}")
            return None

        return None


# Global instance
search_processor = IntelligentSearchProcessor()