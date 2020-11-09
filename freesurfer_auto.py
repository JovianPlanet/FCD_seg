import os
import subprocess

import nibabel as nib
import numpy as np

from pathlib import Path

from nipype.interfaces.freesurfer import ReconAll, MRIConvert
from nipype.interfaces.fsl import BET, FAST

from dipy.io.image import load_nifti_data, load_nifti
from dipy.segment.tissue import TissueClassifierHMRF

np.set_printoptions(precision=2, suppress=True)

heads_path = '/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/MRBrainS13DataNii/TrainingData/'
head = 'T1.nii'
brain_path = '/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/skullStrippedMRBrainS13DataNii/'
freesurfer_brain = 'freesurfer_brainmask.nii'
fsl_brain = 'fsl_brainmask.nii.gz'
fsl_fast_root_name = 'fsl_fast_'

def freesurfer_reconall():
	reconall = ReconAll()
	reconall.inputs.subject_id = '1'
	reconall.inputs.directive = 'autorecon1'
	reconall.inputs.flags = ['nomotioncor', 'nonuintensitycor', 'notalairach', 'nonormalization']
	reconall.inputs.subjects_dir = '/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/skullStrippedMRBrainS13DataNii/'
	reconall.inputs.T1_files = '/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/MRBrainS13DataNii/TestData/1/T1.nii'
	reconall.inputs.openmp = 4 # Number of processors to use in parallel

	reconall.run()


def freesurfer_mriconvert():
	mriconvert = MRIConvert()
	mriconvert.inputs.in_file = '/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/skullStrippedMRBrainS13DataNii/1/mri/brainmask.mgz'
	mriconvert.inputs.out_file = '/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/skullStrippedMRBrainS13DataNii/1/mri/fs_brainmask.nii'
	mriconvert.inputs.out_type = 'nii'

	mriconvert.run()

def fsl_bet(input_file, out_file):
	skullstrip = BET()
	skullstrip.inputs.in_file = input_file #os.path.join(head_path, head)
	skullstrip.inputs.out_file = out_file #os.path.join(brain_path, fsl_brain)
	skullstrip.run()

def fsl_fast(input_file, out_file):
	fslfast = FAST()
	fslfast.inputs.in_files = input_file #os.path.join(brain_path, 'fsl_brainmask.nii.gz')
	fslfast.inputs.out_basename = out_file #'/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/skullStrippedMRBrainS13DataNii/1/mri/fsl_fast_'
	fslfast.inputs.img_type = 1 # 1=T1, 2=T2, 3=PD
	fslfast.inputs.number_classes = 3 # WM, GM, CSF
	fslfast.inputs.segments = True
	#fslfast.inputs.verbose = True
	fslfast.run()

def dipy_segmentation(brain_path):
	nclass = 3
	beta = 0.1
	img, static_affine = load_nifti(os.path.join(brain_path, 'fsl_brainmask.nii.gz'))
	print(static_affine)
	hmrf = TissueClassifierHMRF()
	initial_segmentation, final_segmentation, PVE = hmrf.classify(img, nclass, beta, max_iter=20)
	print(initial_segmentation.shape, final_segmentation.shape, PVE.shape)
	nii_CSF = nib.Nifti1Image(PVE[:,:,:,0], static_affine)
	nib.save(nii_CSF, os.path.join(brain_path, 'fsl_dipy_csf.nii'))
	nii_GM = nib.Nifti1Image(PVE[:,:,:,1], static_affine)
	nib.save(nii_GM, os.path.join(brain_path, 'fsl_dipy_gm.nii'))
	nii_WM = nib.Nifti1Image(PVE[:,:,:,2], static_affine)
	nib.save(nii_WM, os.path.join(brain_path, 'fsl_dipy_wm.nii'))

def apply_affine_to_freesurfer_brain():
	img_orig, static_affine = load_nifti(os.path.join(head_path, head))
	img_brain, fs_static_affine = load_nifti(os.path.join(brain_path, 'brainmask.nii'))
	#new_orientation = nib.orientations.ornt_transform(fs_static_affine, static_affine)
	print(static_affine)
	print(fs_static_affine)
	#print(new_orientation)
	affined_nii = nib.Nifti1Image(img_brain, np.eye(4))
	nib.save(affined_nii, os.path.join(brain_path, 'fs_affined_brainmask.nii'))

subjects = next(os.walk(heads_path))[1]

for subject in subjects:
    #img_dir = sorted(train_ids)[mri]
    Path(os.path.join(brain_path, subject)).mkdir(parents=True, exist_ok=True)
    fsl_bet(os.path.join(heads_path, subject, head), os.path.join(brain_path, subject, fsl_brain))
    print(os.path.join(brain_path, subject, fsl_brain))
    fsl_fast(os.path.join(brain_path, subject, fsl_brain), os.path.join(brain_path, subject, fsl_fast_root_name))
    dipy_segmentation(os.path.join(brain_path, subject))


print(subjects)

#freesurfer_reconall()
#freesurfer_mriconvert()


#os.chdir('/home/david/freesurfer')
#subprocess.run(['pwd'])

#subprocess.run(['./set_freesurfer_env.sh'])

'''

os.environ['FREESURFER_HOME']='/home/david/freesurfer'
os.environ['SUBJECTS_DIR']='/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/freesurfer_subjects'

subprocess.run(['pwd'])
#subprocess.run(['chmod', '777', '/home/david/freesurfer/SetUpFreeSurfer.sh'])
#subprocess.run(['sh', './home/david/freesurfer/SetUpFreeSurfer.sh'])
#subprocess.run(['/bin/bash', './SetUpFreeSurfer.sh'])
os.system('sh /home/david/freesurfer/SetUpFreeSurfer.sh')

#output = subprocess.check_output(['which'], shell=True, text=True, input='freeview')
#print(output)

os.environ['MRBRAIN_DATASET']='/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/MRBrainS13DataNii'
os.chdir('/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/MRBrainS13DataNii')

subprocess.run(['freeview', '-v', 'T1.nii'], shell=True)

'''


#text=True, input=

#export MRBRAIN_DATASET="/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/MRBrainS13DataNii"
#cd $MRBRAIN_DATASET/TestData/1
#pwd
#freeview -v T1.nii
#recon-all -subject 2 -i T1.nii -autorecon1

# Instalacion inicial
'''os.chdir('/home/david/freesurfer')
#os.system('cd $HOME/freesurfer')
os.environ['FREESURFER_HOME']='$HOME/freesurfer'
#os.system('export FREESURFER_HOME=$HOME/freesurfer')
os.environ['SUBJECTS_DIR']='$FRESURFER_HOME/subjects'
#os.system('export SUBJECTS_DIR=$FRESURFER_HOME/subjects')
os.system('source $FREESURFER_HOME/SetUpFreeSurfer.sh')
os.system('which freeview')'''

# Cargar base de datos (Directorio con las imagenes)
'''os.system('export MRBrain=/home/david/Documents/MRBrainTest/')
os.system('export CABEZAS_1=$MRBrain/TestData/1')
os.system('cd $CABEZAS_1')'''

# Cargar una imagen
'''os.system('freeview -v T1.nii ')'''