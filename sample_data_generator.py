import csv
import random

def generate_amr_dataset(output_filename="synthetic_amr_data.csv", num_samples=100):
    proteins = [
        ("CLUSTER_001", "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRIDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPVAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"),
        ("CLUSTER_002", "MKKWFPAFLFLSLSFAALASPAQAQPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRIDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"),
        ("CLUSTER_003", "MVKVAIDGKQIKVRAGEIVHLGPKGEEAEVEVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRIDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"),
    ]

    antibiotics = [
        ("ANT_001", "Ampicillin", "Beta-lactam", "CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccccc3)C(=O)N2[C@@H]1C(=O)O", 8.0, 32.0),
        ("ANT_002", "Ciprofloxacin", "Fluoroquinolone", "CCN1C=C(C(=O)O)C(=O)c2cc(F)c(N3CCNCC3)cc21", 1.0, 4.0),
        ("ANT_003", "Levofloxacin", "Fluoroquinolone", "CN1CCN(c2c(F)cc3c(c2F)c(=O)c(C(=O)O)cn3C4CC4)CC1", 1.0, 2.0),
        ("ANT_004", "Gentamicin", "Aminoglycoside", "CC(C1C(C(C(C(O1)OC2C(CC(C(C2O)OC3C(C(C(O3)CO)O)N)N)N)O)NC)O)N", 2.0, 8.0),
    ]

    species_list = ["Escherichia coli", "Klebsiella pneumoniae", "Pseudomonas aeruginosa"]
    mechanisms = ["TEM-1", "KPC-2", "GyrA_S83L", "MexAB-OprM", "Wildtype"]
    operators = ["=", "<=", ">"]

    rows = []
    for i in range(1, num_samples + 1):
        p_cluster, seq = random.choice(proteins)
        ant_id, name, scaffold, smiles, s_bp, r_bp = random.choice(antibiotics)
        species = random.choice(species_list)
        mech = random.choice(mechanisms)
        op = random.choice(operators)
        
        # Random MIC values distributed around breakpoints
        mic = random.choice([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])

        rows.append({
            "protein_sequence": seq,
            "smiles": smiles,
            "mic_value": mic,
            "mic_unit": "mg/L",
            "mic_operator": op,
            "susceptible_breakpoint": s_bp,
            "resistant_breakpoint": r_bp,
            "breakpoint_source": "EUCAST",
            "antibiotic_id": ant_id,
            "antibiotic_name": name,
            "drug_scaffold": scaffold,
            "organism_species": species,
            "resistance_mechanism": mech,
            "protein_cluster": p_cluster,
            "isolate_id": f"ISO_{i:03d}",
        })

    fieldnames = list(rows[0].keys())
    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Generated {num_samples} sample records in '{output_filename}'")

if __name__ == "__main__":
    generate_amr_dataset()
