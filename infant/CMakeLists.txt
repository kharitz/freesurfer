project(infant)

install(PROGRAMS
  atlassubjects.csh
  check_infant_recon_success.csh
  create_pial_surfaces_mprage_subject.csh
  create_wm_surfaces_mprage_subject.csh
  find_cerebral_wm_labels.csh
  find_neighbors_byAge.csh
  find_neighbors_byAge_extended.csh
  find_neighbors_byMI.csh
  find_neighbors_byMI_extended.csh
  run_label_fusion.csh
  set_babydev_packages.csh
  set_infant_recon_params.csh
  DESTINATION bin
)

if (FSPYTHON_INSTALL_TREE)
   set(infant_dest "\.\./fspython/python/packages")
   set(pydir "${CMAKE_INSTALL_PREFIX}/\.\./fspython/python")
else()
   set(infant_dest "python/packages")
   set(pydir "${CMAKE_INSTALL_PREFIX}/python")
endif()

if (FSPYTHON_INSTALL_TREE)
   install_pyscript_fspython_tree(infant_recon_py)
   install_pyscript_fspython_tree(mri_label_fusion)
   install(FILES freesurfer_pipeline.py DESTINATION ${infant_dest})
else()
   install_pyscript(infant_recon_py)
   install_pyscript(mri_label_fusion)
   install(FILES freesurfer_pipeline.py DESTINATION ${infant_dest})
endif()

install(FILES CNYBCH.yaml DESTINATION average)

install_symlinks(niftyreg/license.niftyreg.txt TYPE files DESTINATION docs)

install_append_help(infant_recon_all infant_recon_all.help.xml bin)

install_symlinks(all_c_robtemplate_affine_mean.05.01.2018.nii.gz TYPE files DESTINATION average)

add_subdirectory(labelfusion)

# Entrypoint for containers
install(PROGRAMS docker/infant-container-entrypoint.bash DESTINATION bin)
install(PROGRAMS docker/infant-container-entrypoint-aws.bash DESTINATION bin)

# install external niftyreg binaries
if(APPLE)
  if(Apple_ARM64)
    set(NIFTY_REG_DIR niftyreg/mac_arm64)
  else()
    set(NIFTY_REG_DIR niftyreg/mac_x86)
  endif()
else()
  set(NIFTY_REG_DIR niftyreg/linux_x86)
endif()
foreach(CMD reg_resample reg_f3d reg_aladin)
  if(APPLE)
    install_symlinks(${NIFTY_REG_DIR}/${CMD} TYPE PROGRAMS DESTINATION bin)
  else()
    install_symlinks(${NIFTY_REG_DIR}/${CMD} TYPE PROGRAMS DESTINATION bin)
  endif()
endforeach()

# install the model file for skullstripping
install_symlinks(infant_synthstrip_01012025.pt TYPE files DESTINATION average/sscnn_skullstripping)
# install externally-stored skullstripping models
#foreach(MODEL ax_sscnn.h5 cor_sscnn.h5 sag_sscnn.h5 12600.pt)
#  install(PROGRAMS /autofs/cluster/freesurfer/build/misc/infant/sscnn_skullstripping/${MODEL} #DESTINATION average/sscnn_skullstripping)
#endforeach()
#endif()

# install any package requirements
#if(INSTALL_PYTHON_DEPENDENCIES)
#  set(PY_PKGS "numpy==1.16 tensorflow==1.5.0 keras==2.2.2 h5py==2.10.0 sklearn scikit-image tables psutil nibabel")
#  if(NOT DISTRIBUTE_FSPYTHON)
#    set(PKG_TARGET "--target=${pydir}/packages")
#  endif()
#  install(CODE "
#    message(STATUS \"Installing infant-freesurfer python package dependencies\")
#    execute_process(COMMAND bash -c \"${pydir}/bin/python3 -m pip install ${PKG_TARGET} ${PY_PKGS}\" RESULT_VARIABLE retcode)
#    if(NOT \${retcode} STREQUAL 0)
#      message(FATAL_ERROR \"Could not install infant-freesurfer python dependencies\")
#    endif()
#    execute_process(COMMAND bash -c \"find ${pydir}/bin ! -name 'python3*' -type f -exec rm -f {} +\" RESULT_VARIABLE retcode)
#    if(NOT \${retcode} STREQUAL 0)
#      message(FATAL_ERROR \"Could not remove pip-installed scripts\")
#    endif()"
#  )
#endif()

