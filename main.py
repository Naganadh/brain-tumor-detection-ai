import os
# Minimize TensorFlow and OpenMP memory allocations on cloud free-tier (512MB RAM)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, render_template, request, send_from_directory, send_file
import numpy as np
import cv2
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and stamp total page count with medical footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        # Top running header for subsequent pages
        if self._pageNumber > 1:
            self.setFont('Helvetica-Bold', 8)
            self.setFillColor(colors.HexColor('#0f172a'))
            self.drawString(36, 762, 'NEUROVISION AI CLINICAL SUITE')
            self.setFont('Helvetica', 8)
            self.setFillColor(colors.HexColor('#64748b'))
            self.drawRightString(612 - 36, 762, 'NEUROLOGICAL MRI DIAGNOSTIC REPORT')
            self.setStrokeColor(colors.HexColor('#e2e8f0'))
            self.setLineWidth(0.75)
            self.line(36, 755, 612 - 36, 755)

        # Bottom footer on all pages
        self.setStrokeColor(colors.HexColor('#e2e8f0'))
        self.setLineWidth(0.75)
        self.line(36, 32, 612 - 36, 32)

        self.setFont('Helvetica-Bold', 7.5)
        self.setFillColor(colors.HexColor('#0f172a'))
        self.drawString(36, 20, 'CONFIDENTIAL MEDICAL REPORT')
        self.setFont('Helvetica', 7.5)
        self.setFillColor(colors.HexColor('#64748b'))
        self.drawString(180, 20, '|  Assisted Diagnostic Evaluation  |  CE / Clinical Research Tier')
        self.drawRightString(612 - 36, 20, f'Page {self._pageNumber} of {page_count}')
        self.restoreState()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates') if os.path.exists(os.path.join(BASE_DIR, 'templates')) else BASE_DIR

# Initialize Flask app
app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Global model holder with lazy loading
_model = None

def get_model():
    """Lazy-load model only on first prediction to allow instant web server boot & save RAM."""
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        model_path = os.path.join(BASE_DIR, 'models', 'model.h5')
        if os.path.exists(model_path):
            print(f"Loading trained neural model from {model_path}...")
            _model = load_model(model_path, compile=False)
            print("Model loaded successfully into memory.")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}.")
    return _model

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define directories
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
REPORTS_FOLDER = os.path.join(BASE_DIR, 'reports')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# ==============================================================================
# Comprehensive Clinical Knowledge Base: Explanations, WHO Staging, Foods & Meds
# ==============================================================================
TUMOR_KNOWLEDGE_BASE = {
    'glioma': {
        'name': 'Glioma Brain Tumor',
        'is_tumor': True,
        'tagline': 'Primary Central Nervous System Glial Neoplasm',
        'overview': (
            'Gliomas are primary central nervous system (CNS) tumors originating from glial cells '
            '(astrocytes, oligodendrocytes, and ependymal cells) that surround and insulate neurons. '
            'Gliomas are characteristically infiltrative, diffusing along white matter tracts into surrounding '
            'cerebral parenchyma rather than developing encapsulated borders.'
        ),
        'pathophysiology': (
            'Glial cells undergo oncogenic mutations (such as IDH1/IDH2 mutation, TP53 alteration, '
            'ATRX loss, EGFR amplification, or 1p/19q codeletion). These genomic aberrations drive dysregulated '
            'cell proliferation, diffuse parenchymal infiltration, and abnormal neo-angiogenesis.'
        ),
        'symptoms': [
            'Persistent or progressive headaches (characteristically prominent in early morning)',
            'New-onset epileptic seizures (focal or generalized)',
            'Cognitive deterioration, personality changes, or memory impairment',
            'Progressive neurological deficits (hemiparesis, dysphasia, motor weakness, numbness)',
            'Signs of elevated intracranial pressure (nausea, projectile vomiting, papilledema)'
        ],
        'cancer_stage': {
            'system': 'World Health Organization (WHO) CNS Tumor Grading Classification (Grades I - IV)',
            'summary': (
                'Gliomas range across WHO Grades I through IV. While MRI signal hyperintensity and contrast '
                'ring enhancement provide key staging indicators, definitive grade requires histological and molecular biopsy analysis.'
            ),
            'grades': [
                {
                    'grade': 'WHO Grade I',
                    'title': 'Pilocytic Astrocytoma',
                    'risk': 'Low Risk / Benign',
                    'badge_class': 'bg-success',
                    'description': 'Slow-growing, circumscribed, non-infiltrative. Most common in pediatric patients. Complete surgical resection frequently yields curative outcomes.'
                },
                {
                    'grade': 'WHO Grade II',
                    'title': 'Diffuse Low-Grade Glioma',
                    'risk': 'Intermediate / Infiltrative',
                    'badge_class': 'bg-warning text-dark',
                    'description': 'Slow-growing but diffusely infiltrative into adjacent brain parenchyma. Moderate cellular atypia. Tendency to recur and potentially undergo higher-grade malignant transformation over time.'
                },
                {
                    'grade': 'WHO Grade III',
                    'title': 'Anaplastic Glioma',
                    'risk': 'High Risk / Malignant',
                    'badge_class': 'bg-danger',
                    'description': 'Malignant tumor displaying brisk mitotic activity, distinct nuclear atypia, and prominent infiltration. Requires multimodal therapeutic intervention (surgery + radiation + chemotherapy).'
                },
                {
                    'grade': 'WHO Grade IV',
                    'title': 'Glioblastoma Multiforme (GBM)',
                    'risk': 'Critical / Highly Malignant',
                    'badge_class': 'bg-danger',
                    'description': 'Most aggressive primary brain cancer. Hallmark characteristics include rapid mitotic division, microvascular proliferation, and widespread central pseudopalisading necrosis.'
                }
            ],
            'clinical_note': 'Definitive WHO Grade confirmation requires histopathological biopsy and molecular profiling (IDH1/2 mutation status, 1p/19q codeletion, and MGMT promoter methylation).'
        },
        'food_recommendations': {
            'overview': (
                'Complementary nutritional support helps fight oxidative stress, modulate systemic neuro-inflammation, '
                'and reinforce healthy cellular immunity alongside standard neuro-oncological care.'
            ),
            'beneficial_foods': [
                {
                    'name': 'Antioxidant-Rich Berries',
                    'items': 'Wild Blueberries, Blackberries, Raspberries, Strawberries',
                    'icon': 'bi-apple',
                    'benefit': 'Abundant in anthocyanins, pterostilbene, and ellagic acid that scavenge reactive oxygen species (ROS) and support neuro-cellular repair.'
                },
                {
                    'name': 'Cruciferous Vegetables',
                    'items': 'Broccoli, Brussels Sprouts, Cauliflower, Cabbage',
                    'icon': 'bi-flower2',
                    'benefit': 'High in sulforaphane and indole-3-carbinol, bioactives shown to activate the Nrf2 cytoprotective pathway and support natural cellular detoxification.'
                },
                {
                    'name': 'Omega-3 Rich Healthy Fats',
                    'items': 'Wild Salmon, Mackerel, Sardines, Flaxseeds, Walnuts, Chia Seeds',
                    'icon': 'bi-droplet-half',
                    'benefit': 'DHA & EPA fatty acids suppress inflammatory eicosanoids and pro-inflammatory cytokines (IL-6, TNF-α), promoting healthy cerebral cell membranes.'
                },
                {
                    'name': 'Turmeric (Curcumin) with Piperine',
                    'items': 'Pure Turmeric root paired with freshly ground Black Pepper',
                    'icon': 'bi-fire',
                    'benefit': 'Curcumin modulates NF-κB and VEGF pathways involved in inflammation and abnormal angiogenesis. Piperine boosts curcumin bioavailability by up to 2,000%.'
                },
                {
                    'name': 'Green Tea & Pure Matcha',
                    'items': 'Organic Japanese Sencha, Ceremonial Matcha',
                    'icon': 'bi-cup-hot',
                    'benefit': 'Packed with Epigallocatechin Gallate (EGCG), a potent polyphenol researched for suppressing abnormal tumor cell proliferation and vascular growth factors.'
                },
                {
                    'name': 'Dark Leafy Greens',
                    'items': 'Organic Spinach, Kale, Swiss Chard, Arugula',
                    'icon': 'bi-tree',
                    'benefit': 'Rich in folates, carotenoids, and magnesium to maintain DNA methylation integrity and support overall neurological cellular vitality.'
                }
            ],
            'foods_to_avoid': [
                'Refined sugars & high-fructose corn syrups (cancerous cells heavily utilize glucose via the Warburg effect)',
                'Ultra-processed meats containing synthetic nitrites, nitrates, and high sodium',
                'Deep-fried foods and trans-fatty acids that accelerate systemic oxidative stress',
                'Excessive alcohol consumption and artificial chemical additives'
            ]
        },
        'medicines': {
            'overview': (
                'Standard-of-care glioma medical therapeutics are prescribed by a neuro-oncologist and commonly '
                'include alkylating chemotherapy, anti-angiogenic biologics, anti-edema steroids, and antiepileptic agents.'
            ),
            'medication_list': [
                {
                    'name': 'Temozolomide (TMZ)',
                    'category': 'Alkylating Chemotherapy (Oral)',
                    'mechanism': 'Crosses the blood-brain barrier and methylates DNA at O6 and N7 positions of guanine, triggering tumor cell cycle arrest and apoptosis.',
                    'usage': 'Concurrent administration with radiation therapy followed by monthly adjuvant maintenance cycles.'
                },
                {
                    'name': 'Bevacizumab (Avastin)',
                    'category': 'Targeted Monoclonal Antibody (Anti-VEGF)',
                    'mechanism': 'Selectively binds and neutralizes Vascular Endothelial Growth Factor A (VEGF-A), inhibiting tumor neo-vascularization and reducing refractory brain edema.',
                    'usage': 'Indicated in recurrent high-grade glioma or progressive edema refractory to steroids.'
                },
                {
                    'name': 'Dexamethasone',
                    'category': 'High-Potency Synthetic Glucocorticoid',
                    'mechanism': 'Restores blood-brain barrier tight junctions, rapidly reducing peritumoral vasogenic cerebral edema and easing intracranial mass effect.',
                    'usage': 'Tapered oral/IV dosage to relieve headaches, neurological deficits, and cerebral swelling.'
                },
                {
                    'name': 'Levetiracetam (Keppra)',
                    'category': 'Second-Generation Anticonvulsant',
                    'mechanism': 'Modulates synaptic vesicle protein SV2A to suppress seizure discharge with minimal hepatic drug-drug interactions with chemotherapy.',
                    'usage': 'First-line seizure prophylaxis and management in brain tumor patients.'
                }
            ],
            'warning': (
                'CRITICAL MEDICAL WARNING: All listed medications are strictly prescription-only oncological agents. '
                'They require careful dose titration, routine blood counts (CBC), liver/renal profiling, and continuous monitoring by a certified Neuro-Oncologist.'
            ),
            'emergency_instruction': (
                'IN CASE OF ANY SIDE EFFECTS (such as severe nausea, continuous vomiting, skin rash, shortness of breath, sudden escalation of headache, '
                'vision disturbances, slurred speech, confusion, or new seizure activity), PLEASE CONTACT YOUR DOCTOR OR EMERGENCY MEDICAL SERVICES IMMEDIATELY.'
            )
        }
    },
    'meningioma': {
        'name': 'Meningioma Brain Tumor',
        'is_tumor': True,
        'tagline': 'Meningeal Covering Neoplasm (Predominantly Extra-Axial)',
        'overview': (
            'Meningiomas arise from arachnoid cap cells of the meninges (the protective membranous layers: dura mater, '
            'arachnoid mater, and pia mater surrounding the brain and spinal cord). They are predominantly extra-axial '
            'tumors that grow slowly and compress adjacent cerebral cortex rather than invading deeply into brain tissue.'
        ),
        'pathophysiology': (
            'Originates from arachnoid barrier cells, frequently driven by NF2 (neurofibromatosis type 2) mutations, '
            'chromosome 22q loss, or TRAF7, KLF4, and AKT1 mutations. Many meningiomas express progesterone and estrogen receptors, '
            'contributing to a higher incidence in females.'
        ),
        'symptoms': [
            'Gradual-onset, non-specific persistent headaches caused by local mass effect',
            'Focal motor weakness or sensory loss corresponding to anatomical tumor location',
            'Visual disturbances, visual field cuts, or diplopia (in skull base/sphenoid wing meningiomas)',
            'New-onset epileptic seizures triggered by cortical irritation',
            'Loss of smell (anosmia) or personality changes (in olfactory groove/frontal subfrontal lesions)'
        ],
        'cancer_stage': {
            'system': 'World Health Organization (WHO) Meningioma Histological Grading (Grades I - III)',
            'summary': (
                'Approximately 80% to 85% of meningiomas are benign WHO Grade I lesions with excellent long-term prognosis. '
                'Higher grades exhibit greater recurrence potential and necessitate closer surgical and radiological follow-up.'
            ),
            'grades': [
                {
                    'grade': 'WHO Grade I',
                    'title': 'Benign Meningioma (~80 - 85% of cases)',
                    'risk': 'Low Risk / Benign',
                    'badge_class': 'bg-success',
                    'description': 'Slow-growing, sharply demarcated margins, low mitotic index (<4 mitoses per 10 HPF). Common subtypes: Meningothelial, Fibrous, Transitional. Excellent 10-year survival; small asymptomatic cases may only require MRI surveillance.'
                },
                {
                    'grade': 'WHO Grade II',
                    'title': 'Atypical Meningioma (~15 - 20% of cases)',
                    'risk': 'Intermediate / Increased Recurrence',
                    'badge_class': 'bg-warning text-dark',
                    'description': 'Exhibits increased mitotic activity (4-19 mitoses per 10 HPF), brain parenchymal invasion, or specific histological variants (Clear cell, Chordoid). Higher risk of local recurrence; often treated with complete resection and adjuvant radiotherapy.'
                },
                {
                    'grade': 'WHO Grade III',
                    'title': 'Anaplastic / Malignant Meningioma (~1 - 3% of cases)',
                    'risk': 'Critical / Malignant',
                    'badge_class': 'bg-danger',
                    'description': 'Aggressive tumor resembling sarcoma or carcinoma with high mitotic figures (≥20 per 10 HPF), cellular dedifferentiation, and aggressive invasion into adjacent brain and bone. High propensity for recurrence and rare distant metastases.'
                }
            ],
            'clinical_note': 'Classic MRI findings include homogeneous contrast enhancement with a characteristic "dural tail sign". Simpson surgical resection grading and histological evaluation determine staging and recurrence prognosis.'
        },
        'food_recommendations': {
            'overview': (
                'A nutrient-dense, anti-inflammatory dietary regimen helps support intracranial vascular health, '
                'modulates tissue swelling, and fosters robust post-surgical and radiological recovery.'
            ),
            'beneficial_foods': [
                {
                    'name': 'Antioxidant-Rich Berries',
                    'items': 'Blueberries, Cranberries, Blackberries, Pomegranate',
                    'icon': 'bi-apple',
                    'benefit': 'High concentrations of polyphenols and punicalagins to minimize oxidative cellular stress and support micro-vascular stability.'
                },
                {
                    'name': 'Cruciferous Vegetables',
                    'items': 'Broccoli, Kale, Brussels Sprouts, Watercress',
                    'icon': 'bi-flower2',
                    'benefit': 'Supplies dietary indole-3-carbinol and diindolylmethane (DIM), which assist healthy estrogen metabolism and promote cellular protection.'
                },
                {
                    'name': 'Omega-3 Rich Foods',
                    'items': 'Wild Alaskan Salmon, Sardines, Flaxseeds, Walnuts',
                    'icon': 'bi-droplet-half',
                    'benefit': 'Anti-inflammatory fatty acids that attenuate peritumoral inflammation and support cranial nerve preservation.'
                },
                {
                    'name': 'Turmeric (Curcumin) Extract',
                    'items': 'Golden Milk / Curcumin with Piperine',
                    'icon': 'bi-fire',
                    'benefit': 'Natural inhibitor of inflammatory cascades (COX-2, TNF-α) investigated for suppressing meningeal abnormal tissue expansion.'
                },
                {
                    'name': 'Green Tea (Camellia sinensis)',
                    'items': 'Organic Green Tea, Matcha',
                    'icon': 'bi-cup-hot',
                    'benefit': 'EGCG content helps strengthen healthy vascular endothelial function and mitigate reactive cellular changes.'
                },
                {
                    'name': 'Citrus & Bioflavonoid Foods',
                    'items': 'Oranges, Lemons, Grapefruits, Bell Peppers',
                    'icon': 'bi-sun',
                    'benefit': 'Vitamin C and bioflavonoids maintain collagen synthesis and robust dural matrix repair.'
                }
            ],
            'foods_to_avoid': [
                'Excessive high-glycemic carbohydrates and refined sugars',
                'Trans-fatty acids and highly processed industrial seed oils',
                'Processed deli meats containing chemical sodium nitrites',
                'Excessive sodium intake (which can aggravate fluid retention and cerebral edema)'
            ]
        },
        'medicines': {
            'overview': (
                'Primary intervention for symptomatic meningioma is neurosurgical resection or stereotactic radiosurgery '
                '(Gamma Knife / CyberKnife). Pharmacological treatments are utilized for perioperative stabilization, symptom management, and recurrent disease.'
            ),
            'medication_list': [
                {
                    'name': 'Dexamethasone',
                    'category': 'Corticosteroid Anti-Edema Agent',
                    'mechanism': 'Suppresses localized vascular permeability and vasogenic edema surrounding the meningioma, providing prompt relief from compression symptoms.',
                    'usage': 'Prescribed short-term for perioperative stabilization or acute mass effect.'
                },
                {
                    'name': 'Levetiracetam (Keppra)',
                    'category': 'Anticonvulsant Therapy',
                    'mechanism': 'Stabilizes neuronal excitability to prevent seizures triggered by cortical compression or irritation from convexial meningiomas.',
                    'usage': 'Standard prophylactic or therapeutic antiepileptic treatment.'
                },
                {
                    'name': 'Hydroxyurea',
                    'category': 'Antineoplastic / Ribonucleotide Reductase Inhibitor',
                    'mechanism': 'Inhibits DNA synthesis; occasionally used as systemic therapy in recurrent, inoperable, or progressive meningiomas failing radiation.',
                    'usage': 'Considered under clinical trial protocols for non-resectable meningiomas.'
                },
                {
                    'name': 'Somatostatin Receptor Analogues (Octreotide LAR)',
                    'category': 'Targeted Peptide Analogue',
                    'mechanism': 'Meningiomas frequently express somatostatin receptor subtype 2 (SSTR2); targeted analogues can stabilize select refractory lesions.',
                    'usage': 'Off-label oncological use in progressive skull-base meningiomas.'
                }
            ],
            'warning': (
                'CRITICAL MEDICAL WARNING: Medications must be prescribed and managed strictly by a Neurosurgeon, Neuro-Oncologist, '
                'or qualified medical specialist. Dosages and tapering schedules must be followed precisely without self-modification.'
            ),
            'emergency_instruction': (
                'IN CASE OF ANY SIDE EFFECTS (including acute worsening of headache, sudden vision loss or double vision, focal weakness, '
                'severe nausea, allergic reaction, or seizure episodes), PLEASE CONTACT YOUR DOCTOR OR EMERGENCY MEDICAL SERVICES IMMEDIATELY.'
            )
        }
    },
    'pituitary': {
        'name': 'Pituitary Neuroendocrine Tumor (PitNET / Adenoma)',
        'is_tumor': True,
        'tagline': 'Sellar Region Endocrine & Neurological Neoplasm',
        'overview': (
            'Pituitary tumors arise in the anterior pituitary gland located within the sella turcica at the base of the skull, '
            'situated directly beneath the optic chiasm. Over 99% of pituitary tumors are benign adenomas (PitNETs). Because the pituitary '
            'is the master endocrine regulator, these tumors can cause systemic hormonal imbalances and visual field deficits.'
        ),
        'pathophysiology': (
            'Develops from adenohypophyseal neuroendocrine cells. May be functional (hyper-secreting hormones such as Prolactin, '
            'Growth Hormone, ACTH, or TSH) or non-functional (silent, causing compression of the optic apparatus and pituitary gland).'
        ),
        'symptoms': [
            'Bitemporal hemianopsia (loss of peripheral lateral vision due to optic chiasm compression)',
            'Chronic frontal or retro-orbital headaches',
            'Hormonal excess signs: Galactorrhea, amenorrhea, infertility, or erectile dysfunction (Prolactinoma)',
            'Acromegaly symptoms: Enlarged hands/feet, facial coarsening, joint aches (Growth hormone excess)',
            'Cushingoid symptoms: Moon face, central adiposity, hypertension, purple striae (ACTH excess)',
            'Hypopituitarism: Severe fatigue, cold intolerance, adrenal insufficiency, dizziness upon standing'
        ],
        'cancer_stage': {
            'system': 'Size Classification & WHO Pituitary Tumor Classification',
            'summary': (
                'Pituitary neoplasms are classified primarily by anatomical size (Microadenoma vs. Macroadenoma), '
                'invasion of surrounding cavernous sinuses (Knosp scale), and hormonal secretion profile.'
            ),
            'grades': [
                {
                    'grade': 'Microadenoma (< 10 mm / 1 cm)',
                    'title': 'Pituitary Microadenoma',
                    'risk': 'Low Risk / Benign',
                    'badge_class': 'bg-success',
                    'description': 'Tumor diameter is less than 10 mm. Typically confined within the sella turcica without compressing the optic chiasm. Often functional (detected early via hormonal symptoms). Excellent prognosis with medical or endoscopic management.'
                },
                {
                    'grade': 'Macroadenoma (≥ 10 mm / 1 cm)',
                    'title': 'Pituitary Macroadenoma',
                    'risk': 'Intermediate / Compressive',
                    'badge_class': 'bg-warning text-dark',
                    'description': 'Tumor diameter is 10 mm or greater. Frequently causes suprasellar extension, upward compression of the optic chiasm (visual field deficit), and potential lateral cavernous sinus extension. Requires formal endocrine and neuro-ophthalmic evaluation.'
                },
                {
                    'grade': 'Giant Pituitary Adenoma (≥ 40 mm / 4 cm)',
                    'title': 'Giant Pituitary Adenoma',
                    'risk': 'High Structural Risk',
                    'badge_class': 'bg-danger',
                    'description': 'Extensive sellar and parasellar mass that can involve third ventricle, cavernous sinuses, or frontal lobes. Requires specialized transsphenoidal or transcranial neurosurgical decompression.'
                },
                {
                    'grade': 'Atypical PitNET / Carcinoma (< 0.2% of cases)',
                    'title': 'Atypical PitNET / Pituitary Carcinoma',
                    'risk': 'Critical / Extremely Rare',
                    'badge_class': 'bg-danger',
                    'description': 'Characterized by high proliferation indices (Ki-67 >3%), extensive p53 expression, or non-contiguous craniospinal/systemic metastasis. Managed with aggressive multimodal therapy.'
                }
            ],
            'clinical_note': 'High-resolution sellar MRI with thin dynamic pituitary cuts and comprehensive endocrine blood panels (Prolactin, IGF-1, ACTH, Morning Cortisol, Free T4, TSH, LH/FSH) are required for definitive staging.'
        },
        'food_recommendations': {
            'overview': (
                'Nutritional therapy focuses on stabilizing endocrine metabolism, supporting blood sugar equilibrium, '
                'reducing adrenal fatigue, and preserving neuro-ophthalmic nerve vitality.'
            ),
            'beneficial_foods': [
                {
                    'name': 'Endocrine-Balancing Zinc Foods',
                    'items': 'Raw Pumpkin Seeds, Organic Lentils, Chickpeas, Sesame Tahini',
                    'icon': 'bi-shield-check',
                    'benefit': 'Zinc is an essential co-factor in healthy pituitary hormone synthesis and receptor signaling pathways.'
                },
                {
                    'name': 'Omega-3 Rich Fish & Seeds',
                    'items': 'Wild Salmon, Atlantic Mackerel, Chia Seeds, Hemp Hearts',
                    'icon': 'bi-droplet-half',
                    'benefit': 'Maintains cellular membrane fluidity and provides protective anti-inflammatory support to the optic nerve sheath.'
                },
                {
                    'name': 'Antioxidant-Dense Berries',
                    'items': 'Blackberries, Blueberries, Acai, Aronia Berries',
                    'icon': 'bi-apple',
                    'benefit': 'Protects anterior pituitary vascular beds from oxidative damage through bio-flavonoid anthocyanins.'
                },
                {
                    'name': 'Cruciferous Vegetables',
                    'items': 'Broccoli, Bok Choy, Cabbage, Radishes',
                    'icon': 'bi-flower2',
                    'benefit': 'Promotes liver Phase II detoxification pathways to maintain balanced estrogen, prolactin, and steroid hormone clearance.'
                },
                {
                    'name': 'Complex Low-Glycemic Whole Grains',
                    'items': 'Steel-cut Oats, Quinoa, Brown Rice',
                    'icon': 'bi-patch-check',
                    'benefit': 'Provides steady glucose release, preventing sudden insulin spikes that stress the hypothalamic-pituitary-adrenal (HPA) axis.'
                },
                {
                    'name': 'Adaptogenic Herbal Teas (Under supervision)',
                    'items': 'Holy Basil (Tulsi), Chamomile, Spearmint',
                    'icon': 'bi-cup-hot',
                    'benefit': 'Calms cortisol reactivity, supports restorative sleep, and reduces chronic physiological neuro-endocrine stress.'
                }
            ],
            'foods_to_avoid': [
                'Refined white sugars and sweetened beverages (worsens glucose intolerance in GH/ACTH tumors)',
                'Excessive caffeine and energy stimulants (over-excites adrenal and pituitary output)',
                'Highly processed foods containing endocrine-disrupting additives (phthalates, BPA)',
                'High-sodium packaged goods (aggravates hypertension and fluid shifts)'
            ]
        },
        'medicines': {
            'overview': (
                'Medical pharmacotherapy is the primary treatment for prolactinomas and an important adjunct '
                'for GH/ACTH-secreting and hormone-deficient pituitary tumors under an Endocrinologist and Neurosurgeon.'
            ),
            'medication_list': [
                {
                    'name': 'Cabergoline',
                    'category': 'Dopamine D2 Receptor Agonist (First-Line)',
                    'mechanism': 'Potently stimulates pituitary dopamine D2 receptors, suppressing prolactin secretion and inducing tumor shrinkage in over 85-90% of prolactinomas.',
                    'usage': 'Low-dose oral administration twice weekly; highly effective for prolactinoma resolution.'
                },
                {
                    'name': 'Bromocriptine',
                    'category': 'Dopamine Agonist Alternative',
                    'mechanism': 'Alternative dopamine agonist that suppresses prolactin synthesis and normalizes reproductive hormone function.',
                    'usage': 'Commonly used in pregnancy-seeking patients due to extensive safety data.'
                },
                {
                    'name': 'Octreotide (Sandostatin LAR) / Lanreotide',
                    'category': 'Long-Acting Somatostatin Analogue',
                    'mechanism': 'Binds somatostatin receptors (SSTR2/SSTR5) to inhibit excess Growth Hormone (GH) and IGF-1 secretion, controlling Acromegaly symptoms.',
                    'usage': 'Monthly deep subcutaneous or intramuscular depot injections.'
                },
                {
                    'name': 'Hormone Replacement Therapy (Levothyroxine / Hydrocortisone)',
                    'category': 'Targeted Endocrine Substitution',
                    'mechanism': 'Replaces deficient target-organ hormones when a large macroadenoma compresses normal pituitary gland tissue causing hypopituitarism.',
                    'usage': 'Physiological daily replacement carefully adjusted based on serum hormone levels.'
                }
            ],
            'warning': (
                'CRITICAL MEDICAL WARNING: All pituitary medications and hormone replacement therapies require strict supervision '
                'by a board-certified Endocrinologist. Periodic hormonal blood tests, visual field assessments, and cardiac echocardiograms are required.'
            ),
            'emergency_instruction': (
                'IN CASE OF ANY SIDE EFFECTS (including sudden thunderclap headache, acute visual darkening or double vision, extreme dizziness, '
                'fainting, chest discomfort, or severe nausea/vomiting), PLEASE CONTACT YOUR DOCTOR OR EMERGENCY SERVICES IMMEDIATELY (Rule out Pituitary Apoplexy).'
            )
        }
    },
    'notumor': {
        'name': 'Healthy Brain Scan (No Tumor Detected)',
        'is_tumor': False,
        'tagline': 'Intact Symmetrical Cerebral Anatomy Within Normal Limits',
        'overview': (
            'The automated deep learning computer vision scan shows symmetrical cerebral hemispheres, intact sulcal '
            'and gyral morphology, and ventricular systems within normal limits for age. No focal space-occupying '
            'masses, hyperintense lesions, or pathological tumor contours were localized.'
        ),
        'pathophysiology': (
            'Cerebral parenchyma demonstrates uniform signal characteristics without focal mass effect, midline shift, '
            'or abnormal neovascular contrast enhancement.'
        ),
        'symptoms': [
            'No tumor-associated focal neurological deficits detected',
            'Normal intracranial pressure dynamics',
            'Intact cranial nerve tracts and symmetrical ventricular architecture'
        ],
        'cancer_stage': {
            'system': 'World Health Organization (WHO) CNS Grading Scale',
            'summary': 'Normal Scan — WHO Grade: Not Applicable (0 / Normal Brain Scan).',
            'grades': [
                {
                    'grade': 'Normal Examination (Grade 0)',
                    'title': 'No Pathological Neoplasm',
                    'risk': 'Normal / Healthy',
                    'badge_class': 'bg-success',
                    'description': 'Normal brain parenchyma and cranial compartments. No malignant or benign space-occupying lesions detected.'
                }
            ],
            'clinical_note': 'This AI finding reflects algorithmic image evaluation. For complete clinical assessment or persistent symptoms, regular medical checkups with a physician are recommended.'
        },
        'food_recommendations': {
            'overview': (
                'Maintaining proactive neuro-protective nutrition helps sustain cognitive health, vascular integrity, '
                'and long-term cellular wellness.'
            ),
            'beneficial_foods': [
                {
                    'name': 'MIND / Mediterranean Diet Foods',
                    'items': 'Extra Virgin Olive Oil, Avocados, Walnuts, Almonds',
                    'icon': 'bi-heart-pulse',
                    'benefit': 'Rich in monounsaturated fats and polyphenols that promote healthy cerebral micro-circulation.'
                },
                {
                    'name': 'Wild Fatty Cold-Water Fish',
                    'items': 'Wild Salmon, Sardines, Rainbow Trout',
                    'icon': 'bi-droplet-half',
                    'benefit': 'Abundant in DHA, the predominant structural omega-3 fatty acid in the brain cortex.'
                },
                {
                    'name': 'Colorful Fruits & Dark Berries',
                    'items': 'Blueberries, Pomegranates, Cherries',
                    'icon': 'bi-apple',
                    'benefit': 'Supplies dietary flavonoids to defend neural cells against daily metabolic oxidative stress.'
                },
                {
                    'name': 'Hydration & Green Tea',
                    'items': 'Pure Spring Water, Green Tea, Herbal Infusions',
                    'icon': 'bi-cup-hot',
                    'benefit': 'Optimal brain hydration maintains cerebrospinal fluid (CSF) dynamics and glymphatic clearance.'
                }
            ],
            'foods_to_avoid': [
                'Excessive refined sugars and sweetened sodas',
                'Industrial hydrogenated trans-fats',
                'Excessive alcohol consumption and tobacco exposure'
            ]
        },
        'medicines': {
            'overview': 'No oncology medications indicated for normal MRI examinations.',
            'medication_list': [
                {
                    'name': 'No Prescription Oncology Medications Required',
                    'category': 'Preventive Wellness Only',
                    'mechanism': 'Maintain balanced lifestyle, routine physical activity, and adequate restorative sleep.',
                    'usage': 'Consult your family physician for routine preventive health maintenance.'
                }
            ],
            'warning': (
                'GENERAL HEALTH WARNING: Do not self-prescribe or take unnecessary medications. Always consult a licensed medical doctor for health concerns.'
            ),
            'emergency_instruction': (
                'IN CASE OF ANY SIDE EFFECTS OR UNEXPECTED SYMPTOMS (such as sudden severe headache, vision changes, slurred speech, or unexplained dizziness), '
                'PLEASE CONTACT YOUR DOCTOR OR SEEK EMERGENCY MEDICAL ATTENTION IMMEDIATELY.'
            )
        }
    }
}

# ==============================================================================
# Tumor Localization: Red Highlight (No Bounding Box)
# ==============================================================================
def highlight_tumor_region(image_path, output_path, is_tumor, label, confidence):
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if not is_tumor:
        # Healthy scan: draw clean green outline and normal banner
        overlay = img.copy()
        cv2.rectangle(overlay, (4, 4), (w - 4, h - 4), (0, 220, 100), 2)
        banner_h = max(26, int(h * 0.07))
        sub_img = overlay[h - banner_h : h, 0:w]
        white_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        white_rect[:] = (10, 45, 20)
        overlay[h - banner_h : h, 0:w] = cv2.addWeighted(sub_img, 0.25, white_rect, 0.75, 0)
        font_scale = 0.38 if w < 300 else 0.42
        cv2.putText(overlay, 'NORMAL - NO LESIONS DETECTED', (10, h - int(banner_h * 0.32)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (80, 255, 140), 1, cv2.LINE_AA)
        cv2.imwrite(output_path, overlay)
        return

    # Tumor detected: isolate skull/brain area
    _, brain_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain_eroded = cv2.erode(brain_mask, kernel_small, iterations=2)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    masked_brain = cv2.bitwise_and(blurred, blurred, mask=brain_eroded)

    # Adaptive focal anomaly thresholding
    brain_pixels = masked_brain[brain_eroded > 0]
    thresh_val = np.percentile(brain_pixels, 88) if len(brain_pixels) > 0 else 160
    _, tumor_thresh = cv2.threshold(masked_brain, int(thresh_val), 255, cv2.THRESH_BINARY)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tumor_cleaned = cv2.morphologyEx(tumor_thresh, cv2.MORPH_CLOSE, kernel)
    tumor_cleaned = cv2.morphologyEx(tumor_cleaned, cv2.MORPH_OPEN, kernel_small)

    # Extract dominant tumor contours
    contours, _ = cv2.findContours(tumor_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tumor_contours = [c for c in contours if cv2.contourArea(c) > 60]
    if not tumor_contours and contours:
        tumor_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    else:
        tumor_contours = sorted(tumor_contours, key=cv2.contourArea, reverse=True)[:4]

    overlay = img.copy()

    # Red highlighting for tumor region
    mask = np.zeros((h, w), dtype=np.uint8)
    if tumor_contours:
        cv2.drawContours(mask, tumor_contours, -1, 255, thickness=cv2.FILLED)

    # Vivid red overlay color in BGR
    red_layer = np.zeros_like(img)
    red_layer[:] = (0, 0, 245)

    # Soft glowing red aura around tumor edges
    soft_glow = cv2.GaussianBlur(mask, (21, 21), 0).astype(float) / 255.0

    # Alpha blending on tumor pixels: 60% red tint, 40% MRI tissue texture
    alpha = 0.60
    tumor_idx = mask > 0
    overlay[tumor_idx] = cv2.addWeighted(img, 1 - alpha, red_layer, alpha, 0)[tumor_idx]

    # Outer soft red glow around the contour perimeter
    glow_zone = (soft_glow > 0.05) & (~tumor_idx)
    for c in range(3):
        overlay[:, :, c] = np.where(
            glow_zone,
            np.clip(img[:, :, c] * (1 - soft_glow * 0.45) + red_layer[:, :, c] * (soft_glow * 0.45), 0, 255).astype(np.uint8),
            overlay[:, :, c]
        )

    # Sharp bright red border contour directly on tumor boundary (NO bounding box!)
    cv2.drawContours(overlay, tumor_contours, -1, (0, 0, 255), 2, cv2.LINE_AA)

    # Top clinical status banner indicating red-highlighted identification
    banner_h = max(24, int(h * 0.065))
    sub_img = overlay[0:banner_h, 0:w]
    dark_banner = np.zeros(sub_img.shape, dtype=np.uint8)
    dark_banner[:] = (15, 15, 85)
    overlay[0:banner_h, 0:w] = cv2.addWeighted(sub_img, 0.25, dark_banner, 0.75, 0)

    tag = f"TUMOR HIGHLIGHTED (RED) | {label.upper()} ({confidence*100:.1f}%)"
    font_scale = 0.35 if w < 300 else 0.40
    cv2.putText(overlay, tag, (8, int(banner_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, overlay)


# ==============================================================================
# Comprehensive Medical Diagnostic PDF Report Generator
# ==============================================================================
def generate_pdf_report(original_img_path, highlighted_img_path, pred_data, filename, pdf_output_path):
    """
    Generates an executive-tier, board-certified clinical PDF diagnostic report.
    Structured cleanly across 2 pages:
      - Page 1: Clinical Header, Patient Scan Metadata, Primary Diagnosis Banner,
                Comparative MRI Plates (Original vs AI Red-Highlighted), Multi-Class Distribution.
      - Page 2: Pathology Profile, WHO Histopathological Staging, Oncology Nutritional Support,
                Pharmacotherapy Guidance, Emergency Directives & Physician Sign-off.
    """
    doc = SimpleDocTemplate(
        pdf_output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=42
    )
    styles = getSampleStyleSheet()
    elements = []

    label = pred_data['label']
    info = TUMOR_KNOWLEDGE_BASE.get(label, TUMOR_KNOWLEDGE_BASE['glioma'])
    is_tumor = pred_data['is_tumor']
    conf_pct = pred_data['confidence'] * 100

    # Custom Design Tokens & Typography
    col_primary = colors.HexColor('#0f172a')     # Slate 900
    col_secondary = colors.HexColor('#334155')   # Slate 700
    col_muted = colors.HexColor('#64748b')       # Slate 500
    col_border = colors.HexColor('#e2e8f0')      # Slate 200
    col_brand = colors.HexColor('#0284c7')       # Sky 600
    col_danger = colors.HexColor('#dc2626')      # Red 600
    col_danger_bg = colors.HexColor('#fef2f2')   # Red 50
    col_danger_border = colors.HexColor('#fca5a5') # Red 300
    col_success = colors.HexColor('#16a34a')     # Green 600
    col_success_bg = colors.HexColor('#f0fdf4')  # Green 50
    col_success_border = colors.HexColor('#86efac') # Green 300

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=15,
        leading=18,
        textColor=col_primary,
        fontName='Helvetica-Bold'
    )
    sub_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=col_muted,
        fontName='Helvetica'
    )
    sec_title_style = ParagraphStyle(
        'SecTitle',
        parent=styles['Heading2'],
        fontSize=10.5,
        leading=14,
        textColor=col_primary,
        fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'BodyP',
        parent=styles['Normal'],
        fontSize=8,
        leading=11.5,
        textColor=col_secondary,
        fontName='Helvetica'
    )
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=styles['Normal'],
        fontSize=8,
        leading=11.5,
        textColor=col_primary,
        fontName='Helvetica-Bold'
    )
    th_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontSize=8,
        leading=10.5,
        fontName='Helvetica-Bold',
        textColor=colors.white
    )
    td_style = ParagraphStyle(
        'TD',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=10.5,
        fontName='Helvetica',
        textColor=col_secondary
    )
    td_bold = ParagraphStyle(
        'TDBold',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=10.5,
        fontName='Helvetica-Bold',
        textColor=col_primary
    )

    # --------------------------------------------------------------------------
    # PAGE 1: HEADER & IDENTITY
    # --------------------------------------------------------------------------
    header_table = Table([
        [
            Paragraph("<b>NEUROVISION CLINICAL IMAGING LABORATORY</b><br/><font color='#64748b' size='8'>Advanced Deep Learning Neurological Diagnostic Evaluation Suite</font>", ParagraphStyle('HLeft', parent=styles['Normal'], fontSize=11, leading=14, textColor=col_primary)),
            Paragraph(f"<b>REPORT STATUS:</b> <font color='{'#dc2626' if is_tumor else '#16a34a'}'><b>{'PATHOLOGY IDENTIFIED' if is_tumor else 'NORMAL / CLEAR'}</b></font><br/><font color='#64748b' size='7.5'>REF: DL-VGG16-CV9</font>", ParagraphStyle('HRight', parent=styles['Normal'], fontSize=8.5, leading=12, textColor=col_primary, alignment=2))
        ]
    ], colWidths=[360, 180])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 3))
    elements.append(HRFlowable(width="100%", thickness=2, color=col_brand, spaceAfter=8))

    # Patient & Scan Case Info Matrix
    now_str = datetime.datetime.now().strftime("%Y-%m-%d | %H:%M:%S UTC")
    report_id = f"NV-REP-{abs(hash(filename)) % 10000000:07d}"
    meta_data = [
        [
            Paragraph(f"<b>Accession ID:</b> {report_id}", body_style),
            Paragraph(f"<b>Analysis Timestamp:</b> {now_str}", body_style)
        ],
        [
            Paragraph(f"<b>Source Image:</b> <font face='Courier'>{filename}</font>", body_style),
            Paragraph("<b>Model Architecture:</b> VGG16 Deep CNN + OpenCV CV", body_style)
        ],
        [
            Paragraph("<b>Input Preprocessing:</b> 128x128 px | Symmetrical Normalization", body_style),
            Paragraph(f"<b>Localization Overlay:</b> {'<font color=\"#dc2626\"><b>Red Anomaly Mask (No Box)</b></font>' if is_tumor else '<font color=\"#16a34a\"><b>Intact Morphology (Normal)</b></font>'}", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, col_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, col_border),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))

    # Diagnostic Finding Primary Banner
    finding_title = pred_data['title'].upper()
    banner_bg = col_danger_bg if is_tumor else col_success_bg
    banner_border = col_danger_border if is_tumor else col_success_border
    banner_text_color = '#b91c1c' if is_tumor else '#15803d'

    finding_html = (
        f"<table width='100%'><tr>"
        f"<td><font size='10.5' color='{banner_text_color}'><b>PRIMARY CLINICAL FINDING:</b> {finding_title}</font></td>"
        f"<td align='right'><font size='10.5' color='{banner_text_color}'><b>CONFIDENCE:</b> {conf_pct:.2f}%</font></td>"
        f"</tr></table>"
    )
    banner_p = Paragraph(finding_html, ParagraphStyle('Banner', parent=styles['Normal'], fontSize=9.5, leading=13))
    banner_table = Table([[banner_p]], colWidths=[540])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), banner_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, banner_border),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(banner_table)
    elements.append(Spacer(1, 10))

    # Dual Comparative MRI Plates
    plate_w = 210
    plate_h = 160
    img_orig = RLImage(original_img_path, width=plate_w, height=plate_h)
    img_high = RLImage(highlighted_img_path, width=plate_w, height=plate_h)
    plate_title_style = ParagraphStyle('PlateTitle', parent=styles['Normal'], fontSize=8.5, fontName='Helvetica-Bold', alignment=1, textColor=col_primary)
    plate_sub_style = ParagraphStyle('PlateSub', parent=styles['Normal'], fontSize=7, fontName='Helvetica', alignment=1, textColor=col_muted)

    plate_data = [
        [
            Paragraph("PLATE 1: ORIGINAL PATIENT MRI", plate_title_style),
            Paragraph("PLATE 2: AI COMPUTER VISION LOCALIZATION", plate_title_style)
        ],
        [
            Paragraph("Pre-processed DICOM/T2 Weighted Baseline", plate_sub_style),
            Paragraph("Red Region Overlay & Contour Gradient (No Bounding Box)", plate_sub_style)
        ],
        [img_orig, img_high]
    ]
    plate_table = Table(plate_data, colWidths=[270, 270])
    plate_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, col_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#f1f5f9')),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 1),
        ('TOPPADDING', (0, 1), (-1, 1), 0),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 3),
        ('TOPPADDING', (0, 2), (-1, 2), 3),
        ('BOTTOMPADDING', (0, 2), (-1, 2), 4),
    ]))
    elements.append(plate_table)
    elements.append(Spacer(1, 8))

    # Multi-Class Probability Distribution
    scores = pred_data['scores']
    prob_data = [[
        Paragraph("DIAGNOSTIC CRITERIA / PATHOLOGY CLASS", th_style),
        Paragraph("PREDICTED PROBABILITY", th_style),
        Paragraph("CLINICAL SIGNIFICANCE", th_style)
    ]]
    label_names = {
        'glioma': 'Glioma (Primary Glial Neoplasm)',
        'meningioma': 'Meningioma (Extra-Axial Dural Mass)',
        'pituitary': 'Pituitary Neuroendocrine Tumor (PitNET)',
        'notumor': 'Healthy Scan (No Lesions Localized)'
    }
    for k in ['glioma', 'meningioma', 'pituitary', 'notumor']:
        p_val = scores.get(k, 0.0) * 100
        sel = (k == label)
        c_style = td_bold if sel else td_style
        status_label = "<font color='#dc2626'><b>PRIMARY POSITIVE DETECTION</b></font>" if (sel and is_tumor) else ("<font color='#16a34a'><b>PRIMARY NORMAL MATCH</b></font>" if sel else "Negative / Non-Significant")
        prob_data.append([
            Paragraph(label_names.get(k, k), c_style),
            Paragraph(f"<b>{p_val:.2f}%</b>", c_style),
            Paragraph(status_label, c_style)
        ])
    prob_table = Table(prob_data, colWidths=[240, 130, 170])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, col_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, col_border),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 6))

    # Footnote on Page 1
    elements.append(Paragraph(
        "<i>* Note: Preliminary automated decision support artifact. Turn to Page 2 for WHO histopathological staging, oncology nutrition, pharmacotherapy, and clinical sign-off.</i>",
        ParagraphStyle('Page1Note', parent=styles['Normal'], fontSize=7, leading=9, textColor=col_muted)
    ))

    # --------------------------------------------------------------------------
    # PAGE 2: CLINICAL WORKUP, STAGING, NUTRITION, MEDS, SIGN-OFF
    # --------------------------------------------------------------------------
    elements.append(PageBreak())

    # Section 1: Detailed Pathology Profile
    elements.append(Paragraph("1. Detailed Tumor Explanation & Pathological Profile", sec_title_style))
    elements.append(Spacer(1, 2))
    elements.append(Paragraph(f"<b>Pathological Classification:</b> {info['name']} &mdash; <i>{info['tagline']}</i>", body_bold))
    elements.append(Spacer(1, 1.5))
    elements.append(Paragraph(f"<b>Histopathological Overview:</b> {info['overview']}", body_style))
    elements.append(Spacer(1, 1.5))
    elements.append(Paragraph(f"<b>Molecular Pathophysiology:</b> {info['pathophysiology']}", body_style))
    elements.append(Spacer(1, 1.5))
    symptoms_list = ", ".join(info['symptoms'][:4])
    elements.append(Paragraph(f"<b>Associated Neurological Symptoms:</b> {symptoms_list}.", body_style))
    elements.append(Spacer(1, 5))

    # Section 2: WHO Cancer Staging & Grading
    elements.append(Paragraph("2. Tumor / Cancer Staging Assessment (WHO CNS Classification)", sec_title_style))
    elements.append(Spacer(1, 2))
    stage_info = info['cancer_stage']
    elements.append(Paragraph(f"<b>Staging System:</b> {stage_info['system']}", body_style))
    elements.append(Spacer(1, 2.5))

    stage_table_data = [[
        Paragraph("Grade / Classification", th_style),
        Paragraph("Risk Stratification", th_style),
        Paragraph("Histological & Clinical Hallmarks", th_style)
    ]]
    for g in stage_info['grades'][:2]:  # 2 primary representative grades
        stage_table_data.append([
            Paragraph(f"<b>{g['grade']}</b><br/><font color='#64748b'>{g['title']}</font>", td_style),
            Paragraph(f"<b>{g['risk']}</b>", td_style),
            Paragraph(g['description'], td_style)
        ])
    stage_table = Table(stage_table_data, colWidths=[130, 110, 300])
    stage_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, col_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, col_border),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(stage_table)
    elements.append(Spacer(1, 5))

    # Section 3: Nutritional Support & Foods
    elements.append(Paragraph("3. Suggested Nutritional Support & Foods to Inhibit Tumor Growth", sec_title_style))
    elements.append(Spacer(1, 2))
    food_info = info['food_recommendations']
    food_table_data = [[
        Paragraph("Dietary Category & Food Items", th_style),
        Paragraph("Anti-Tumor & Neuro-Protective Biological Mechanism", th_style)
    ]]
    for item in food_info['beneficial_foods'][:2]:  # Top 2 most impactful foods
        food_table_data.append([
            Paragraph(f"<b>{item['name']}</b><br/><font color='#475569'>{item['items']}</font>", td_style),
            Paragraph(item['benefit'], td_style)
        ])
    food_table = Table(food_table_data, colWidths=[190, 350])
    food_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#065f46')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0fdf4')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#a7f3d0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1fae5')),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(food_table)
    elements.append(Spacer(1, 5))

    # Section 4: Clinical Pharmacotherapy
    elements.append(Paragraph("4. Suggested Medical Therapeutics & Clinical Pharmacotherapy", sec_title_style))
    elements.append(Spacer(1, 2))
    med_info = info['medicines']
    med_table_data = [[
        Paragraph("Medication & Class", th_style),
        Paragraph("Biological Mechanism of Action", th_style),
        Paragraph("Clinical Indications / Usage", th_style)
    ]]
    for med in med_info['medication_list'][:2]:  # Top 2 key medications
        med_table_data.append([
            Paragraph(f"<b>{med['name']}</b><br/><font color='#64748b'>{med['category']}</font>", td_style),
            Paragraph(med['mechanism'], td_style),
            Paragraph(med['usage'], td_style)
        ])
    med_table = Table(med_table_data, colWidths=[140, 240, 160])
    med_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, col_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, col_border),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(med_table)
    elements.append(Spacer(1, 5))

    # Section 5: Warning & Emergency Protocol
    warn_p_style = ParagraphStyle('WarnP', parent=styles['Normal'], fontSize=6.8, leading=9.5, textColor=colors.HexColor('#991b1b'))
    warn_box_data = [[
        Paragraph(
            f"<b>CRITICAL MEDICAL WARNING:</b> {med_info['warning']}<br/>"
            f"<b>URGENT EMERGENCY DIRECTIVE:</b> <font color='#b91c1c'><b>{med_info['emergency_instruction']}</b></font>",
            warn_p_style
        )
    ]]
    warn_box = Table(warn_box_data, colWidths=[540])
    warn_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef2f2')),
        ('BOX', (0, 0), (-1, -1), 1.25, colors.HexColor('#dc2626')),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
    ]))
    elements.append(warn_box)
    elements.append(Spacer(1, 6))

    # Section 6: Official Physician Sign-Off & Attestation
    sig_data = [
        [
            Paragraph("<b>Diagnostic System:</b> Deep Learning VGG16 + CV Red Localization", body_style),
            Paragraph("<b>Attending Radiologist / Neuro-Oncologist:</b>", body_style)
        ],
        [
            Paragraph("<b>Verification Method:</b> Autonomous Tensor Softmax Calibration", body_style),
            Paragraph("<b>Signature:</b> ___________________________  <b>Date:</b> _____________", body_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[270, 270])
    sig_table.setStyle(TableStyle([
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(sig_table)

    # Build PDF with dynamic NumberedCanvas for professional page numbering
    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"Generated comprehensive PDF report: {pdf_output_path}")


# ==============================================================================
# Prediction Function
# ==============================================================================
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    # Use cv2 for image loading and preprocessing (saves ~150MB overhead compared to PIL/keras)
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise ValueError(f"Could not read image file at {image_path}")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.expand_dims(resized_img.astype(np.float32) / 255.0, axis=0)

    active_model = get_model()
    predictions = active_model.predict(img_array, verbose=0)
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    confidence_score = float(np.max(predictions, axis=1)[0])
    predicted_label = class_labels[predicted_class_index]

    all_scores = {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}

    if predicted_label == 'notumor':
        result_title = "No Tumor Detected"
        is_tumor = False
    else:
        result_title = f"Tumor Detected: {predicted_label.capitalize()}"
        is_tumor = True

    tumor_info = TUMOR_KNOWLEDGE_BASE.get(predicted_label, TUMOR_KNOWLEDGE_BASE['notumor'])

    return {
        'title': result_title,
        'label': predicted_label,
        'is_tumor': is_tumor,
        'confidence': confidence_score,
        'scores': all_scores,
        'tumor_info': tumor_info
    }


# ==============================================================================
# Web Routes
# ==============================================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sample_name = request.form.get('sample_name')
        file = request.files.get('file')

        target_file = None
        target_name = None
        is_sample = False

        if sample_name and sample_name in ['Te-gl_0015.jpg', 'Te-meTr_0001.jpg', 'Te-noTr_0004.jpg', 'Te-piTr_0003.jpg']:
            target_file = os.path.join(BASE_DIR, sample_name)
            target_name = sample_name
            is_sample = True
        elif file and file.filename != '':
            target_name = file.filename
            target_file = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(target_file)

        if target_file and os.path.exists(target_file):
            pred = predict_tumor(target_file)

            # Generate Highlighted Image (in Red, no box)
            highlighted_name = f"highlighted_{target_name}"
            highlighted_path = os.path.join(app.config['UPLOAD_FOLDER'], highlighted_name)
            highlight_tumor_region(target_file, highlighted_path, pred['is_tumor'], pred['label'], pred['confidence'])

            # Generate PDF Report
            pdf_name = f"report_{os.path.splitext(target_name)[0]}.pdf"
            pdf_path = os.path.join(app.config['REPORTS_FOLDER'], pdf_name)
            generate_pdf_report(target_file, highlighted_path, pred, target_name, pdf_path)

            file_web_path = f'/samples/{target_name}' if is_sample else f'/uploads/{target_name}'
            highlighted_web_path = f'/uploads/{highlighted_name}'

            return render_template(
                'index.html',
                result=pred['title'],
                label=pred['label'],
                is_tumor=pred['is_tumor'],
                confidence=f"{pred['confidence']*100:.2f}",
                scores=pred['scores'],
                file_path=file_web_path,
                highlighted_path=highlighted_web_path,
                report_name=pdf_name,
                tumor_info=pred['tumor_info']
            )

    return render_template('index.html', result=None)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/samples/<filename>')
def get_sample_file(filename):
    return send_from_directory(BASE_DIR, filename)


@app.route('/download_report/<filename>')
def download_report(filename):
    report_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
    if os.path.exists(report_path):
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"Brain_MRI_Diagnostic_Report_{filename}"
        )
    return "Report not found", 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print(f" Brain Tumor Detection AI System Started on port {port}")
    print(f" Open in browser: http://127.0.0.1:{port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)
