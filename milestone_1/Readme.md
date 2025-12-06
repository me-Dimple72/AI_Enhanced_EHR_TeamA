# AI-Powered Enhanced EHR Imaging & Documentation System:

An advanced healthcare application that integrates Generative AI with Electronic Health Records (EHR) to enhance medical imaging, automate clinical documentation, and improve ICD-10 coding accuracy. The system leverages Azure OpenAI, Computer Vision, and APIs to streamline clinical workflows, reduce administrative workload, and improve diagnostic efficiency.

## Milestone 1 — Data Collection & Preprocessing

### Steps to Prepare the Dataset
1.Collect datasets

- Search open sources like Kaggle, PhysioNet, NIH.
- Creating Synthetic EHR data using Synthea with MIMIC-III schema
- Download MRI, CT, and EHR datasets that are openly licensed.
- The data sources are mentioned in docs/dataset_sources.md

  
2.Organize into folders

- Create a root project folder.
- Inside it, keep two main folders: images and ehr_notes.
- Name files properly (for example: MRI_001.png, CT_002.png, note_001.txt).

  
3.Data cleaning and Preprocessing
 
- Remove corrupted, duplicate, or unreadable image files.
- Standardize image formats (e.g., convert to PNG/JPEG).
- Organize images into structured directories by modality and Validate image-label mappings.
- Handle missing values in patient demographics, diagnoses, and procedures.
- Standardize column names across files (patient ID, age, sex).
- Normalize categorical values (e.g., M/F vs Male/Female).

### Challenges faced
- Variability in image quality and resolution across datasets (X-ray, CT, MRI, Ultrasound).
- Non-standard naming conventions and folder structures and Labeling inconsistencies.
- Handling sensitive health data responsibly and ensuring compliance.
- Combining structured (EHR) and unstructured (imaging) datasets for multimodal AI models.
- High storage requirements for large imaging datasets.
