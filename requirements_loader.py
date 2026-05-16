"""
requirements_loader.py
Load and manage requirements from YAML files
"""

import yaml
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class RequirementLoader:
    """Load and manage requirements from YAML files"""
    
    def __init__(self, requirements_dir: str = "data/requirements"):
        """
        Initialize requirement loader.
        
        Args:
            requirements_dir: Directory containing YAML files
        """
        self.requirements_dir = Path(requirements_dir)
        self.requirements_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaded_standards = {}
    
    def load_standard(self, standard_file: str) -> Dict:
        """
        Load requirements from YAML file.
        
        Args:
            standard_file: Filename (e.g., "gri_305.yml")
        
        Returns:
            Parsed requirements dictionary
        """
        file_path = self.requirements_dir / standard_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Requirements file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Cache loaded standard
        standard_id = file_path.stem
        self.loaded_standards[standard_id] = data
        
        return data
    
    def get_requirement(self, standard: str, req_id: str) -> Dict:
        """
        Get specific requirement.
        
        Args:
            standard: Standard name (e.g., "GRI")
            req_id: Requirement ID (e.g., "305-1")
        """
        # Special handling for GRI
        if standard.upper() == "GRI":
            # Extract disclosure number: "305-1" → "305"
            disclosure_num = req_id.split('-')[0]
            standard_file = f"gri_{disclosure_num}.yml"
            cache_key = f"gri_{disclosure_num}"
        else:
            standard_file = f"{standard}.yml"
            cache_key = standard
        
        # Load if not cached
        if cache_key not in self.loaded_standards:
            self.load_standard(standard_file)
        
        # Return requirement
        return self.loaded_standards[cache_key]['requirements'].get(req_id, {})
    
    def get_all_requirements(self, standard: str) -> Dict:
        """Get standard metadata (version, source, etc.)"""
        if standard not in self.loaded_standards:
            self.load_standard(f"{standard}.yml")
        
        data = self.loaded_standards[standard]
        return {
            'version': data.get('version'),
            'standard_id': data.get('standard_id'),
            'last_verified': data.get('last_verified'),
            'source_url': data.get('source_url')
        }
    
    def get_keywords(self, standard: str, req_id: str) -> List[str]:
        """Get search keywords for a requirement"""
        req = self.get_requirement(standard, req_id)
        return req.get('keywords', [])
    
    def verify_requirement(
        self,
        standard: str,
        req_id: str,
        verifier: str = "manual"
    ):
        """
        Mark requirement as verified.
        
        Args:
            standard: Standard name
            req_id: Requirement ID
            verifier: Who verified (manual/rag/hybrid)
        """
        if standard not in self.loaded_standards:
            self.load_standard(f"{standard}.yml")
        
        # Update verification
        req = self.loaded_standards[standard]['requirements'][req_id]
        for elem in req['required_elements']:
            if elem.get('verification_status') != 'verified':
                elem['verification_status'] = 'verified'
                elem['verified_by'] = verifier
                elem['verified_date'] = datetime.now().isoformat()
    
    def export_checklist(self, standard: str, output_path: str = None):
        """Export requirements to JSON for compatibility"""
        import json
        
        if standard not in self.loaded_standards:
            self.load_standard(f"{standard}.yml")
        
        if output_path is None:
            output_path = f"data/checklists/{standard}_checklist.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                self.loaded_standards[standard]['requirements'],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        return output_path

# Test
if __name__ == "__main__":
    print("="*80)
    print(" REQUIREMENTS LOADER V2 (YAML-based)")
    print("="*80)
    
    loader = RequirementLoader()
    
    # Load GRI 305
    print("\n1. Loading GRI 305...")
    print("-"*80)
    
    try:
        gri_305 = loader.load_standard("gri_305.yml")
        print(f" Loaded GRI 305")
        print(f"   Version: {gri_305['version']}")
        print(f"   Last verified: {gri_305['last_verified']}")
        print(f"   Requirements: {len(gri_305['requirements'])}")
        
        # Show metadata
        print("\n2. Standard Metadata:")
        print("-"*80)
        metadata = loader.get_metadata("gri_305")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        
        # Show requirement details
        print("\n3. GRI 305-1 Details:")
        print("-"*80)
        req_305_1 = loader.get_requirement("gri_305", "305-1")
        print(f"Name: {req_305_1['name']}")
        print(f"Source chunks: {len(req_305_1['source_chunks'])}")
        print(f"\nRequired Elements:")
        for i, elem in enumerate(req_305_1['required_elements'], 1):
            status = elem.get('verification_status', 'unverified')
            print(f"  {i}. [{status}] {elem['element']}")
            print(f"      Source: {elem.get('source_chunk', 'N/A')}")
        
        # Show keywords
        print("\n4. Keywords:")
        print("-"*80)
        keywords = loader.get_keywords("gri_305", "305-1")
        print(f"   {', '.join(keywords)}")
        
        # Export to JSON
        print("\n5. Exporting to JSON...")
        print("-"*80)
        json_path = loader.export_checklist("gri_305")
        print(f" Exported to: {json_path}")
        
    except FileNotFoundError as e:
        print(f" Error: {e}")
        print("\n  Please make sure data/requirements/gri_305.yml exists")
    except Exception as e:
        print(f" Unexpected error: {e}")