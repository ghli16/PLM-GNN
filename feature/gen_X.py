import os
import numpy as np
import torch

def get_pdb_xyz(pdb_file):
    """
    Extract the coordinates from a PDB file.
    """
    current_pos = -1000
    X = []
    current_aa = {}  # N, CA, C, O, R
    for line in pdb_file:
        if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
            if current_aa != {}:
                R_group = []
                for atom in current_aa:
                    if atom not in ["N", "CA", "C", "O"]:
                        R_group.append(current_aa[atom])
                if R_group == []:
                    R_group = [current_aa["CA"]]
                R_group = np.array(R_group).mean(0)
                X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], R_group])
                current_aa = {}
            if line[0:4].strip() != "TER":
                current_pos = int(line[22:26].strip())

        if line[0:4].strip() == "ATOM":
            atom = line[13:16].strip()
            if atom != "H":
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                current_aa[atom] = xyz
    return np.array(X)

def process_pdb_files(input_folder, output_folder):
    """
    Process all PDB files in the input folder and save the coordinates as PyTorch tensors in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for pdb_file_name in os.listdir(input_folder):
        if pdb_file_name.endswith('.pdb'):
            pdb_file_path = os.path.join(input_folder, pdb_file_name)
            with open(pdb_file_path, 'r') as file:
                pdb_file = file.readlines()
            
            coord = get_pdb_xyz(pdb_file)
            tensor_file_name = os.path.splitext(pdb_file_name)[0] + '.tensor'
            tensor_file_path = os.path.join(output_folder, tensor_file_name)
            torch.save(torch.tensor(coord, dtype=torch.float32), tensor_file_path)
            print(f"Processed {pdb_file_name} and saved to {tensor_file_path}")

if __name__ == "__main__":
    input_folder = 'TSS' 
    output_folder = 'TSS_xyz' 
    process_pdb_files(input_folder, output_folder)