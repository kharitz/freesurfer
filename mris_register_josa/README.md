# Usage

## 1. Register left hemisphere for subject bert using model file lh.model.h5, result is saved to bert/surf/lh.sphere.reg
`mris_register_josa -h lh -s bert -m lh.model.h5`

## 2. Same as 1 but saving result to a custom file
`mris_register_josa -h lh -s bert -m lh.model.h5 -o my.sphere`

## 3. Same as 2 but instead of using the default files under bert/surf, we can specify each file separately
`mris_register_josa -h lh -S lh.sulc -C lh.curv -H lh.inflated.H -t lh.sphere.rot -m lh.model.h5 -o my.sphere`
