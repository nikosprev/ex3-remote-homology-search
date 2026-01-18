import re
import requests
import time
import sys

INPUT_FILE = "output"           
OUTPUT_FILE = "detailed_bio_report.txt"  

def parse_output(file_path):
    """
    Reads the raw output text and extracts every neighbor 
    that was NOT found by BLAST (In BLAST? == No).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Make sure it exists.")
        sys.exit(1)

    candidates = []
    current_query = None

    for line in lines:
        line = line.strip()
        
        if line.startswith("Query Protein:"):
            parts = line.split(":")
            if len(parts) > 1:
                current_query = parts[1].strip().split()[0]
        
        elif "|" in line and "Neighbor ID" not in line and "---" not in line:
            cols = [c.strip() for c in line.split('|')]
            if len(cols) >= 5:
                neighbor_id = cols[1]
                dist = cols[2]     
                in_blast = cols[4]
                if in_blast == "No":
                    candidates.append({
                        'query': current_query,
                        'neighbor': neighbor_id,
                        'dist': dist
                    })
    return candidates

def get_uniprot_data(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        r = requests.get(url, headers={"Accept": "application/json"})
        if r.status_code != 200: 
            return None
        
        data = r.json()
        
        # 1. Extract Name
        desc = "Unknown Protein"
        if 'proteinDescription' in data:
            rec = data['proteinDescription'].get('recommendedName', {})
            sub = data['proteinDescription'].get('submissionNames', [{}])[0]
            desc = rec.get('fullName', {}).get('value') or sub.get('fullName', {}).get('value')

        # 2. Extract Keywords
        keywords = set()
        if 'keywords' in data:
            for k in data['keywords']:
                keywords.add(k['name'])

        # 3. Extract Pfam & GO (FIXED LOCATION)
        pfam = set()
        go_terms = set()
        
        # KEY FIX: Check both possible names for the reference list
        refs = data.get('uniProtKBCrossReferences', [])
        if not refs:
            refs = data.get('dbReferences', [])

        for ref in refs:
            db_name = ref['database']
            
            # --- PFAM PARSING ---
            if db_name == 'Pfam':
                if 'id' in ref:
                    pfam.add(ref['id'])
            
            # --- GO PARSING ---
            elif db_name == 'GO':
                term_value = None
                for prop in ref.get('properties', []):
                    if prop['key'] == 'GoTerm':
                        term_value = prop['value']
                        break
                
                if term_value:
                    if term_value.startswith('F:') or term_value.startswith('P:'):
                        clean_term = term_value[2:]
                        go_terms.add(clean_term)

        return {
            "id": accession,
            "name": desc,
            "keywords": keywords,
            "pfam": pfam,
            "go": go_terms
        }
    except Exception as e:
        print(f"Error fetching {accession}: {e}")
        return None

def write_set(f, title, data_set):
    f.write(f"      {title}:\n")
    if not data_set:
        f.write("        (None found)\n")
    else:
        for item in sorted(list(data_set)):
            f.write(f"        - {item}\n")

def analyze_candidates_detailed(candidates):
    print(f"Found {len(candidates)} pairs to analyze. Starting detailed scan...")
    print(f"Results will be written to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("############################################################\n")
        f.write("#          DETAILED REMOTE HOMOLOGY ANALYSIS REPORT        #\n")
        f.write("############################################################\n")
        f.write(f"# Total Candidates (BLAST missed): {len(candidates)}\n")
        f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("############################################################\n\n")

        for i, item in enumerate(candidates):
            q_id = item['query']
            n_id = item['neighbor']
            dist = item['dist']
            
            print(f"[{i+1}/{len(candidates)}] Fetching data for {q_id} vs {n_id}...")
            
            q_data = get_uniprot_data(q_id)
            n_data = get_uniprot_data(n_id)
            
            if not q_data or not n_data:
                f.write(f"ERROR: Could not fetch data for pair {q_id} vs {n_id}\n\n")
                continue

            # Intersections
            common_pfam = q_data['pfam'].intersection(n_data['pfam'])
            common_go = q_data['go'].intersection(n_data['go'])
            
            ignore_list = {'Reference proteome', '3D-structure', 'Direct protein sequencing'}
            q_kw_clean = q_data['keywords'] - ignore_list
            n_kw_clean = n_data['keywords'] - ignore_list
            common_kw = q_kw_clean.intersection(n_kw_clean)

            status = "NO_OBVIOUS_MATCH"
            if common_pfam:
                status = "STRONG_HOMOLOG (Shared Pfam)"
            elif len(common_go) > 0 or len(common_kw) >= 2:
                status = "LIKELY_HOMOLOG (Shared Function)"
            
            # Write Report
            f.write(f"=== PAIR {i+1}: {q_id}  vs  {n_id} =================\n")
            f.write(f"   LSH Distance: {dist}\n")
            f.write(f"   VERDICT:      {status}\n\n")
            
            f.write("   --- [ OVERLAP ANALYSIS ] ---\n")
            if common_pfam:
                f.write(f"   [+] SHARED PFAM FAMILIES: {', '.join(common_pfam)}\n")
            else:
                f.write("   [-] No shared Pfam families.\n")
                
            if common_go:
                f.write(f"   [+] SHARED GO TERMS: {', '.join(list(common_go)[:5])}")
                if len(common_go) > 5: f.write(" ...and more")
                f.write("\n")
            else:
                f.write("   [-] No shared GO terms.\n")
                
            if common_kw:
                f.write(f"   [+] SHARED KEYWORDS: {', '.join(list(common_kw)[:5])}\n")
            else:
                f.write("   [-] No shared Keywords.\n")
            f.write("\n")

            f.write(f"   --- [ QUERY: {q_id} ] ---\n")
            f.write(f"      Name: {q_data['name']}\n")
            write_set(f, "Keywords", q_data['keywords'])
            write_set(f, "Pfam IDs", q_data['pfam'])
            write_set(f, "GO Terms", q_data['go'])
            f.write("\n")

            f.write(f"   --- [ NEIGHBOR: {n_id} ] ---\n")
            f.write(f"      Name: {n_data['name']}\n")
            write_set(f, "Keywords", n_data['keywords'])
            write_set(f, "Pfam IDs", n_data['pfam'])
            write_set(f, "GO Terms", n_data['go'])
            
            f.write("\n" + "="*60 + "\n\n")
            time.sleep(0.3)

    print(f"\nDone! Open '{OUTPUT_FILE}' to see the detailed report.")

if __name__ == "__main__":
    try:
        candidates = parse_output(INPUT_FILE)
        if candidates:
            analyze_candidates_detailed(candidates)
        else:
            print("No candidates found! (Did you set 'In BLAST?' to 'No' in your file?)")
    except KeyboardInterrupt:
        print("\nProcess stopped by user.")