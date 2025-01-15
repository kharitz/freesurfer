# Notes on building third party niftyreg utils

Source code for these utils is available on SourceForge [here](https://sourceforge.net/p/niftyreg/git/ci/master/tree/).
Only the master branch seemed to respect the `-voff` flag used in the original
infant pipeline.
Commit hash: 4e4525b84223c182b988afaa85e32ac027774c42

Executables were compiled on a CentOS 7 machine, using the default build options

Only the three executables needed by the infant pipeline have been moved into
the source tree: `reg_aladin`, `reg_f3d` & `reg_resample`.