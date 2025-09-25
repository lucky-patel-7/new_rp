"""
Roles Configuration Loader

Loads role mappings and configurations from comprehensive sector-wise JSON file.
This allows easy customization of role mappings based on actual database content.
"""

import json
import os
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RolesConfig:
    """Loads and manages role configuration from comprehensive sector-wise JSON file."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize roles configuration.

        Args:
            config_path: Path to sector-wise roles JSON file. If None, uses default.
        """
        if config_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parents[3]
            config_path = project_root / "config" / "sectorWiseRoles.json"# type: ignore

        self.config_path = Path(config_path)# type: ignore
        self.config = self._load_config()
        self._processed_roles = self._process_roles_data()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            if not self.config_path.exists():
                logger.error(f"Roles config file not found: {self.config_path}")
                return self._get_fallback_config()

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Loaded comprehensive roles config from: {self.config_path}")
                return config

        except Exception as e:
            logger.error(f"Error loading roles config: {e}")
            return self._get_fallback_config()

    def _get_fallback_config(self) -> Dict[str, Any]:
        """Fallback configuration if file loading fails."""
        return {
            "sectors": [
                {
                    "sector_name": "Information Technology",
                    "roles": [
                        {
                            "role_name": "Software Engineer",
                            "sub_roles": ["Frontend Developer", "Backend Developer", "Full Stack Developer"]
                        }
                    ]
                }
            ]
        }

    def _process_roles_data(self) -> Dict[str, Any]:
        """Process the loaded JSON data to create searchable mappings."""
        processed = {
            "all_role_names": set(),
            "all_sub_roles": set(),
            "role_to_sector": {},
            "search_terms_to_role": {},
            "role_skills": {}
        }

        for sector in self.config.get("sectors", []):
            sector_name = sector.get("sector_name", "")

            for role in sector.get("roles", []):
                role_name = role.get("role_name", "")
                sub_roles = role.get("sub_roles", [])
                skills = [skill.strip() for skill in role.get("skills", []) if isinstance(skill, str) and skill.strip()]

                # Add main role name
                processed["all_role_names"].add(role_name)
                processed["role_to_sector"][role_name.lower()] = sector_name
                if skills:
                    processed["role_skills"][role_name] = skills
                    processed["role_skills"][role_name.lower()] = skills

                # Add search terms for main role
                search_terms = [role_name.lower()]
                processed["search_terms_to_role"][role_name.lower()] = role_name

                # Add sub-roles
                for sub_role in sub_roles:
                    processed["all_sub_roles"].add(sub_role)
                    processed["search_terms_to_role"][sub_role.lower()] = role_name
                    if skills:
                        processed["role_skills"][sub_role] = skills
                        processed["role_skills"][sub_role.lower()] = skills

                    # Also map sub-role variations
                    processed["search_terms_to_role"][sub_role.lower().replace(" ", "")] = role_name
                    if skills:
                        processed["role_skills"][sub_role.lower().replace(" ", "")] = skills

                # Add role name variations
                role_variations = self._generate_role_variations(role_name)
                for variation in role_variations:
                    processed["search_terms_to_role"][variation.lower()] = role_name
                    if skills:
                        processed["role_skills"][variation] = skills
                        processed["role_skills"][variation.lower()] = skills

        return processed

    def _generate_role_variations(self, role_name: str) -> List[str]:
        """Generate common variations of a role name."""
        variations = []
        role_lower = role_name.lower()

        # Basic variations
        variations.extend([
            role_lower,
            role_lower.replace(" ", ""),
            role_lower.replace(" ", "_"),
            role_lower.replace("_", " "),
        ])

        # Common abbreviations and terms
        if "engineer" in role_lower:
            variations.append(role_lower.replace("engineer", "developer"))
            variations.append(role_lower.replace("engineer", "eng"))

        if "developer" in role_lower:
            variations.append(role_lower.replace("developer", "dev"))
            variations.append(role_lower.replace("developer", "engineer"))

        if "manager" in role_lower:
            variations.append(role_lower.replace("manager", "mgr"))

        if "analyst" in role_lower:
            variations.append(role_lower.replace("analyst", "analysis"))

        return list(set(variations))

    def get_database_roles(self) -> List[str]:
        """Get list of all available role names."""
        return list(self._processed_roles["all_role_names"])

    def find_role_mapping(self, query: str) -> Optional[str]:
        """
        Find role mapping for a search query with fuzzy matching for typos.

        Args:
            query: Search query string

        Returns:
            Role name if mapping found, None otherwise
        """
        query_lower = query.lower()

        # Direct search in processed search terms
        for search_term, role_name in self._processed_roles["search_terms_to_role"].items():
            if search_term in query_lower:
                return role_name

        # If no exact match, try partial matching with role names
        for role_name in self._processed_roles["all_role_names"]:
            if role_name.lower() in query_lower:
                return role_name

        # Try sub-roles
        for sub_role in self._processed_roles["all_sub_roles"]:
            if sub_role.lower() in query_lower:
                # Find the parent role for this sub-role
                for search_term, role_name in self._processed_roles["search_terms_to_role"].items():
                    if search_term == sub_role.lower():
                        return role_name

        # Fuzzy matching for common typos and variations
        return self._fuzzy_role_match(query_lower)

    def _fuzzy_role_match(self, query: str) -> Optional[str]:
        """Handle common typos and fuzzy matching for role names."""
        # Common typos and variations
        typo_mappings = {
            "marketting": "marketing",
            "devloper": "developer",
            "sofyware": "software",
            "softwar": "software",
            "engineeer": "engineer",
            "analist": "analyst",
            "manageer": "manager",
            "scientest": "scientist",
            "desiner": "designer",
            "arcitect": "architect",
            "programer": "programmer",
            "administator": "administrator"
        }

        # Fix common typos
        corrected_query = query
        for typo, correct in typo_mappings.items():
            corrected_query = corrected_query.replace(typo, correct)

        # If we made corrections, try again with corrected query
        if corrected_query != query:
            # Try direct search with corrected query
            for search_term, role_name in self._processed_roles["search_terms_to_role"].items():
                if search_term in corrected_query:
                    return role_name

            # Try role names with corrected query
            for role_name in self._processed_roles["all_role_names"]:
                if role_name.lower() in corrected_query:
                    return role_name

        # Try partial word matching (at least 4 characters match)
        query_words = query.split()
        for word in query_words:
            if len(word) >= 4:  # Only check words with 4+ characters
                for role_name in self._processed_roles["all_role_names"]:
                    role_words = role_name.lower().split()
                    for role_word in role_words:
                        if len(role_word) >= 4 and (word in role_word or role_word in word):
                            # Simple similarity check - if most of the word matches
                            if self._simple_similarity(word, role_word) > 0.7:
                                return role_name

        return None

    def _simple_similarity(self, word1: str, word2: str) -> float:
        """Calculate simple character-based similarity between two words."""
        if not word1 or not word2:
            return 0.0

        # Simple Jaccard similarity based on character bigrams
        def get_bigrams(word):
            return set(word[i:i+2] for i in range(len(word)-1))

        bigrams1 = get_bigrams(word1.lower())
        bigrams2 = get_bigrams(word2.lower())

        if not bigrams1 and not bigrams2:
            return 1.0
        if not bigrams1 or not bigrams2:
            return 0.0

        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)

        return intersection / union if union > 0 else 0.0

    def check_unavailable_role(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check if query is for an unavailable role (HR, etc.).

        Args:
            query: Search query string

        Returns:
            Unavailable role info if found, None otherwise
        """
        query_lower = query.lower()

        # Common unavailable roles that are often searched but not in our comprehensive database
        unavailable_terms = {
            "hr": "Human Resources",
            "human resources": "Human Resources",
            "hr manager": "Human Resources Manager",
            "recruiter": "Recruiter",
            "talent acquisition": "Talent Acquisition",
            "people manager": "People Manager"
        }

        for term, role_name in unavailable_terms.items():
            if term in query_lower:
                return {
                    "role": term,
                    "message": f"No {role_name} candidates found in database",
                    "suggestions": [
                        "Marketing Manager (for people-focused marketing)",
                        "Business Development (for recruitment-like processes)",
                        "Content Marketing Manager (for employer branding)"
                    ]
                }

        return None

    def get_role_database_category(self, role_name: str) -> Optional[str]:
        """Get database category (sector) for a role name."""
        return self._processed_roles["role_to_sector"].get(role_name.lower())

    def get_role_skills(self, role_name: str) -> List[str]:
        """Get skills associated with a role from configuration, with heuristics fallback."""
        if not role_name:
            return []

        role_lower = role_name.lower()
        skills = self._processed_roles["role_skills"].get(role_name)
        if not skills:
            skills = self._processed_roles["role_skills"].get(role_lower, [])

        if skills:
            return skills

        # Fallback heuristics if config is missing skills
        inferred = []
        if any(term in role_lower for term in ["software", "developer", "engineer", "programming"]):
            inferred.extend(["Python", "JavaScript", "Java", "Git", "SQL", "React", "Node.js", "Docker"])
        if any(term in role_lower for term in ["data", "analyst", "scientist", "ml", "ai"]):
            inferred.extend(["Python", "SQL", "Pandas", "NumPy", "Machine Learning", "Tableau", "Excel"])
        if any(term in role_lower for term in ["marketing", "social media", "content", "seo"]):
            inferred.extend(["Digital Marketing", "SEO", "Content Marketing", "Social Media", "Analytics", "Adobe Creative Suite"])
        if any(term in role_lower for term in ["business", "analyst", "manager", "project"]):
            inferred.extend(["Business Analysis", "Project Management", "Excel", "PowerBI", "Agile", "Scrum"])

        return inferred

    def get_role_search_terms(self, role_name: str) -> List[str]:
        """Get search terms for a role."""
        # Return variations of the role name
        variations = self._generate_role_variations(role_name)

        # Add sub-roles if they exist
        for search_term, mapped_role in self._processed_roles["search_terms_to_role"].items():
            if mapped_role == role_name:
                variations.append(search_term)

        return list(set(variations))

    def find_seniority_mapping(self, query: str) -> Optional[str]:
        """Find seniority level mapping for query."""
        query_lower = query.lower()

        seniority_mappings = {
            "junior": ["junior", "jr", "entry level", "entry-level", "fresher", "graduate", "associate"],
            "mid": ["mid", "middle", "intermediate"],
            "senior": ["senior", "sr", "experienced", "lead"],
            "manager": ["manager", "team lead", "supervisor", "mgr"],
            "director": ["director", "head of", "vp", "vice president", "chief"]
        }

        for seniority, terms in seniority_mappings.items():
            if any(term in query_lower for term in terms):
                return seniority

        return None

    def get_seniority_database_values(self, seniority_key: str) -> List[str]:
        """Get database values for seniority level."""
        mapping = {
            "junior": ["Junior", "Entry Level", "Associate"],
            "mid": ["Mid Level", "Intermediate"],
            "senior": ["Senior", "Lead"],
            "manager": ["Manager", "Team Lead"],
            "director": ["Director", "VP", "Head"]
        }
        return mapping.get(seniority_key, [])

    def reload_config(self) -> bool:
        """Reload configuration from file."""
        try:
            self.config = self._load_config()
            self._processed_roles = self._process_roles_data()
            logger.info("Comprehensive roles configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error reloading config: {e}")
            return False


# Global instance
roles_config = RolesConfig()
